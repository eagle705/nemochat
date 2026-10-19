"""CPU-only pinned-image gate: actual native module + worker-local O_DIRECT."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import types
import warnings
import zipfile

import torch
import zstandard
from torch.utils.data import Dataset, DataLoader, get_worker_info
from .binding import install
from .transport import DirectTransport, Expectation


def native_module(manifest):
    # Import real native files, not AST-isolated decoder replicas or stubs.
    sys.path.insert(0, manifest['sdk_root'])
    # msc_utils has only stdlib / optional MSC imports. Load that real file directly
    # to avoid megatron.core.__init__ eagerly importing CUDA Transformer Engine.
    # No fake feature flags or decoder modules are supplied.
    fullname = 'megatron.core.msc_utils'
    if fullname not in sys.modules:
        msc_path = Path(manifest['sdk_root']) / 'megatron/core/msc_utils.py'
        spec = importlib.util.spec_from_file_location(fullname, msc_path)
        msc_module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = msc_module
        spec.loader.exec_module(msc_module)
    src = Path(manifest['native_source'])
    package = types.ModuleType('_directio_native')
    package.__path__ = [str(src.parent)]
    sys.modules[package.__name__] = package
    for name in ('runtime', 'utils_logits'):
        fullname = package.__name__ + '.' + name
        spec = importlib.util.spec_from_file_location(fullname, src.parent/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = module
        spec.loader.exec_module(module)
    return module


class RecordingTransport(DirectTransport):
    def open(self, *args, **kwargs):
        result = super().open(*args, **kwargs)
        self.last = result.getvalue()  # Immutable verified bytes, no second file read.
        return result


class NativeDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = manifest
        self.native = None

    def __len__(self):
        return len(self.manifest['shards'])

    def __getitem__(self, index):
        row = self.manifest['shards'][index]
        result = {'rank': row['rank'], 'iteration': row['iteration'], 'path': row['path'], 'ok': False}
        try:
            if get_worker_info() is None:
                raise RuntimeError('direct reads must execute inside the DataLoader worker')
            if self.native is None:
                torch.set_num_threads(1)
                self.native = native_module(self.manifest)
                self.transport = RecordingTransport([Expectation(x['path'], x['sha256'], tuple(x['stat'])) for x in self.manifest['shards']])
                install(self.native, self.manifest['native_contract'], self.transport)
            entries = list(self.native.iter_logprobs_tar_entries(row['path'], expected_hash=self.manifest['metadata_hash']))
            data = self.transport.last
            assert hashlib.sha256(data).hexdigest() == row['sha256']
            result['sha256'] = row['sha256']; result['direct_sha_match'] = True
            with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tar:
                members = tar.getmembers()
                assert [m.name for m in members] == ['_meta.json', f"{row['iteration']}.pt.zst"]
                assert all(m.isreg() for m in members)
                metadata = json.load(tar.extractfile(members[0]))
                assert metadata == self.manifest['metadata']
            assert len(entries) == 1 and entries[0].iteration == row['iteration']
            payload = entries[0].data
            assert hashlib.sha256(payload).hexdigest() == row['payload_sha256']
            # Independent zstd frame/ZIP member CRC validation, separate from native decoder.
            raw = zstandard.ZstdDecompressor().decompress(payload)
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                assert z.testzip() is None
            stored = torch.load(io.BytesIO(raw), weights_only=True)
            assert set(stored) == {'values', 'indices_low', 'bit_17'}
            assert all(len(stored[k]) == 16 for k in stored)
            assert all(x.dtype == torch.uint16 for x in stored['indices_low'])
            assert all(x.dtype == torch.bool for x in stored['bit_17'])
            del stored, raw
            values, indices = self.native.decode_logprobs_payload(payload)
            assert len(values) == len(indices) == 16
            assert all(tuple(x.shape) == (4096, 1, 32) and x.dtype == torch.float16 and x.device.type == 'cpu' for x in values)
            assert all(tuple(x.shape) == (4096, 1, 32) and x.dtype == torch.int64 and x.device.type == 'cpu' for x in indices)
            assert all(bool(torch.isfinite(x).all()) for x in values)
            lo = min(int(x.min()) for x in indices); hi = max(int(x.max()) for x in indices)
            assert 0 <= lo <= hi <= 131071
            result.update(ok=True, metadata=True, zstd=True, zip_crc=True, native=True,
                          microbatches=16, shape=[4096,1,32], values_dtype='float16',
                          indices_storage='uint16+bool17', index_min=lo, index_max=hi,
                          worker_pid=os.getpid(), worker_id=get_worker_info().id,
                          rss_kib=int(Path('/proc/self/status').read_text().split('VmRSS:')[1].split()[0]),
                          open_fds=len(list(Path('/proc/self/fd').iterdir())))
            self.transport.last = None
        except Exception as exc:
            result.update(error_type=type(exc).__name__, error=str(exc))
        return result


def counterproof(manifest):
    native = native_module(manifest)
    result = []
    for iteration in (0, 70):
        row = next(x for x in manifest['shards'] if (x['rank'], x['iteration']) == (7, iteration))
        data = Path(row['path']).read_bytes()  # Deliberate counterproof only, never transport fallback.
        digest = hashlib.sha256(data).hexdigest()
        error = None
        try:
            entries = list(native.iter_logprobs_tar_entries(row['path'], expected_hash=manifest['metadata_hash']))
            native.decode_logprobs_payload(entries[0].data)
        except Exception as exc:
            error = repr(exc)
        assert digest != row['sha256'] and error is not None
        result.append({'iteration': iteration, 'rank': 7, 'buffered_sha256': digest,
                       'expected_direct_sha256': row['sha256'], 'buffered_native_error': error})
    return result


def main():
    manifest_path, expected_sha, output = sys.argv[1:]
    out = Path(output); data = Path(manifest_path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_sha
    manifest = json.loads(data)
    assert not list(Path('/dev').glob('nvidia*')), 'GPU device unexpectedly exposed'
    torch.set_num_threads(1)
    counter = counterproof(manifest)
    (out/'counterproof.json').write_text(json.dumps(counter, indent=2))
    dataset = NativeDataset(manifest)
    loader = DataLoader(dataset, batch_size=None, num_workers=1, pin_memory=True,
                        persistent_workers=False, prefetch_factor=1,
                        multiprocessing_context='fork')
    started = time.monotonic(); seen = []; bad = []; pids = set()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with (out/'shards.jsonl').open('x') as f:
            for row in loader:
                f.write(json.dumps(row, sort_keys=True)+'\n'); f.flush()
                seen.append((row['rank'], row['iteration']))
                if not row['ok']: bad.append(row)
                if 'worker_pid' in row: pids.add(row['worker_pid'])
                if len(seen) % 500 == 0:
                    print(json.dumps({'checked':len(seen),'bad':len(bad),'seconds':time.monotonic()-started}), flush=True)
    expected = {(r,i) for r in range(8) for i in range(1143)}
    complete = len(seen) == 9144 and len(set(seen)) == 9144 and set(seen) == expected and not bad and len(pids) == 1
    summary = {'status': 'COMPLETE' if complete else 'BLOCKED', 'clean': complete,
               'expected':9144, 'checked':len(seen), 'good':len(seen)-len(bad), 'bad':bad,
               'all_iterations':1143 if set(seen)==expected else None,
               'coverage_exact':set(seen)==expected and len(seen)==len(set(seen)),
               'worker_pids':sorted(pids), 'loader':{'num_workers':1,'pin_memory_configured':True,
               'pin_memory_effective':False,'reason':'no accelerator; native PyTorch CPU-supported automatic disable',
               'warnings':[str(x.message) for x in caught]},
               'manifest_sha256':expected_sha,'image':manifest['image'],
               'torch':torch.__version__,'zstandard':zstandard.__version__,
               'native_scope':'full real native utils_logits module loaded via private package; real SDK msc_utils; only open_logit_file replaced',
               'seconds':time.monotonic()-started,'gpu_devices':[], 'checkpoint_weights_read':False}
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary),flush=True)
    if not complete: raise SystemExit(1)


if __name__ == '__main__': main()
