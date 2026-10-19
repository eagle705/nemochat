"""Opt-in direct transport: actual historical buffered-vs-direct symptom first."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PACKAGE = ROOT / 'midtraining/cache/directio/transport.py'
CACHE = ROOT / 'results/midtraining/offline2b-production1920-01/cache'


@pytest.fixture
def tmp_path():
    # Same statx-DIO-capable filesystem as the actual cache, never system /tmp.
    import tempfile
    parent = ROOT / 'results/midtraining/offline1920-parent-01/directio-acceptance-01/test-fixtures'
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        yield Path(directory)


def transport():
    assert PACKAGE.exists(), 'missing fail-closed O_DIRECT transport'
    from midtraining.cache.directio import transport as module
    return module


@pytest.mark.parametrize('iteration,expected', [
    (0, '792412157de7b5528927a57647270ef8de7aa8a68428b49afee591e82cc58880'),
    (70, 'd450e523a6bcf1900edf511529f232c43c88c29ce379c65782a096616896718a'),
])
def test_real_buffered_corrupt_direct_authenticated(iteration, expected):
    path = CACHE / f'cp0_dp7__{iteration}.tar'
    buffered = path.read_bytes()
    assert hashlib.sha256(buffered).hexdigest() != expected, 'historical symptom no longer present'
    module = transport()
    item = module.expectation(path, expected)
    data = module.DirectTransport([item]).read_bytes(str(path))
    assert isinstance(data, bytes)
    assert hashlib.sha256(data).hexdigest() == expected
    assert data != buffered


@pytest.mark.parametrize('kind', ['relative', 'uri', 'symlink', 'sha', 'empty', 'directory', 'duplicate', 'oversize'])
def test_reject_invalid_allowlist(tmp_path, kind):
    m = transport()
    p = tmp_path / 'cp0_dp0__0.tar'
    p.write_bytes(b'a' * 4096)
    item = m.expectation(p, hashlib.sha256(p.read_bytes()).hexdigest())
    from dataclasses import replace
    if kind == 'relative': item = replace(item, path='relative.tar')
    if kind == 'uri': item = replace(item, path='msc://bucket/file.tar')
    if kind == 'symlink':
        link = tmp_path / 'cp0_dp0__1.tar'; link.symlink_to(p)
        item = m.expectation(link, item.sha256)
    if kind == 'sha': item = replace(item, sha256='not-a-sha')
    if kind == 'empty':
        p.write_bytes(b''); item = m.expectation(p, item.sha256)
    if kind == 'directory': item = m.expectation(tmp_path, item.sha256)
    if kind == 'oversize':
        with p.open('wb') as f: f.truncate(1024**3)
        item = m.expectation(p, item.sha256)
    with pytest.raises((ValueError, RuntimeError)):
        m.DirectTransport([item, item] if kind == 'duplicate' else [item])


@pytest.mark.parametrize('mode,kwargs', [('wb', {}), ('r+b', {}), ('rb', {'surprise': True}), ('rb', {'disable_read_cache': False})])
def test_open_rejects_non_native_options(mode, kwargs):
    m = transport(); path = CACHE / 'cp0_dp7__0.tar'
    t = m.DirectTransport([m.expectation(path, '792412157de7b5528927a57647270ef8de7aa8a68428b49afee591e82cc58880')])
    with pytest.raises(ValueError): t.open(str(path), mode, **kwargs)


@pytest.mark.parametrize('fault', ['sha', 'stat', 'alignment', 'unsupported', 'zero', 'unaligned', 'excess', 'poststat', 'unapproved'])
def test_fail_closed_transport_faults(tmp_path, monkeypatch, fault):
    import os
    m = transport(); p = tmp_path / 'cp0_dp0__0.tar'; p.write_bytes(b'x' * 10241)
    e = m.expectation(p, hashlib.sha256(p.read_bytes()).hexdigest())
    from dataclasses import replace
    if fault == 'sha': e = replace(e, sha256='0' * 64)
    t = m.DirectTransport([e])
    if fault == 'stat': os.utime(p, ns=(1, 1))
    if fault == 'alignment': monkeypatch.setattr(m, 'dio_alignment', lambda fd: (_ for _ in ()).throw(RuntimeError('alignment')))
    if fault == 'unsupported': monkeypatch.setattr(m.os, 'preadv', lambda *a: (_ for _ in ()).throw(OSError('unsupported')))
    if fault in ('zero', 'unaligned', 'excess'): monkeypatch.setattr(m.os, 'preadv', lambda *a: {'zero': 0, 'unaligned': 1, 'excess': 999999}[fault])
    if fault == 'poststat':
        real = os.preadv
        def changed(fd, buffers, offset):
            n = real(fd, buffers, offset); os.utime(p, ns=(1, 1)); return n
        monkeypatch.setattr(m.os, 'preadv', changed)
    with pytest.raises((ValueError, RuntimeError, OSError, KeyError)):
        t.read_bytes(str(p) if fault != 'unapproved' else str(p) + '.evil')


def test_aligned_short_read_loop_and_non_block_eof(tmp_path, monkeypatch):
    m = transport(); p = tmp_path / 'cp0_dp0__0.tar'; data = b'z' * 10241; p.write_bytes(data)
    t = m.DirectTransport([m.expectation(p, hashlib.sha256(data).hexdigest())])
    real = m.os.preadv; calls = []
    def short(fd, buffers, offset):
        calls.append(offset)
        view = buffers[0][:4096]
        try: return real(fd, [view], offset)
        finally: view.release()
    monkeypatch.setattr(m.os, 'preadv', short)
    assert t.read_bytes(str(p)) == data
    assert calls == [0, 4096, 8192]


def test_native_source_guard_and_process_local_binding():
    import ast, types
    path = Path('/home/yoon/.hermes/cache/scratch/offline1920-prefix03-student-stt7nda8/staged/src/megatron/bridge/training/cached_kd/utils_logits.py')
    source = path.read_bytes(); tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'open_logit_file')
    ns = {'_msc_if_needed': lambda p: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    from midtraining.cache.directio import binding
    module = types.SimpleNamespace(open_logit_file=ns['open_logit_file'], __file__=str(path))
    contract = binding.source_contract(path)
    m = transport(); p = CACHE / 'cp0_dp7__0.tar'
    t = m.DirectTransport([m.expectation(p, '792412157de7b5528927a57647270ef8de7aa8a68428b49afee591e82cc58880')])
    wrong = dict(contract, source_sha256='0' * 64)
    with pytest.raises(RuntimeError): binding.install(module, wrong, t)
    module.open_logit_file = lambda *args: None
    with pytest.raises(RuntimeError): binding.install(module, contract, t)
    module.open_logit_file = ns['open_logit_file']
    binding.install(module, contract, t)
    with module.open_logit_file(str(p), 'rb', prefetch_file=True) as f:
        assert hashlib.sha256(f.read()).hexdigest() == t.entries[str(p)].sha256
    with pytest.raises(RuntimeError): binding.install(module, contract, t)


def test_full_audit_manifest_coverage_and_override():
    from midtraining.cache.directio import acceptance
    manifest = acceptance.build_manifest()
    rows = manifest['shards']
    assert len(rows) == 9144
    assert {(x['rank'], x['iteration']) for x in rows} == {(r, i) for r in range(8) for i in range(1143)}
    override = next(x for x in rows if x['rank'] == 7 and x['iteration'] == 70)
    assert override['sha256'] == 'd450e523a6bcf1900edf511529f232c43c88c29ce379c65782a096616896718a'
    assert manifest['budget']['train_samples'] == 468736
    assert manifest['budget']['train_iters'] == 3662
    assert manifest['budget']['warmup_iters'] == 37
    assert manifest['budget']['decay_iters'] == 366
    assert manifest['metadata_hash'] == 'a73bc4fd805c51bf8596cf62cc88f66b'
    assert manifest['checkpoint_weights_read'] is False


def test_gate_native_dataset_is_available():
    path = ROOT / 'midtraining/cache/directio/cpu_gate.py'
    assert path.exists(), 'missing exact-image all-shard native CPU gate'


def test_opt_in_integration_rejects_unbound_manifest():
    from midtraining.cache.directio import integration
    path = ROOT / 'results/midtraining/offline1920-parent-01/directio-acceptance-01/expectations.json'
    data = path.read_bytes()
    with pytest.raises(RuntimeError): integration.load_transport(path, '0' * 64)
    manifest, t = integration.load_transport(path, hashlib.sha256(data).hexdigest())
    assert len(t.entries) == 9144
    assert manifest['metadata']['identifiers']['train_samples'] == 468736
    with pytest.raises(TypeError): t.entries['forbidden'] = None


def test_manifest_rank_to_filename_mapping_cannot_be_swapped(tmp_path):
    from midtraining.cache.directio import integration
    path = ROOT / 'results/midtraining/offline1920-parent-01/directio-acceptance-01/expectations.json'
    manifest = json.loads(path.read_text())
    manifest['shards'][0]['rank'], manifest['shards'][1]['rank'] = manifest['shards'][1]['rank'], manifest['shards'][0]['rank']
    p = tmp_path / 'bad-manifest.json'; data = json.dumps(manifest).encode(); p.write_bytes(data)
    with pytest.raises(RuntimeError, match='mapping'):
        integration.load_transport(p, hashlib.sha256(data).hexdigest())


def test_repeated_real_direct_read_does_not_leak_fds():
    import os
    m = transport(); p = CACHE / 'cp0_dp7__0.tar'
    t = m.DirectTransport([m.expectation(p, '792412157de7b5528927a57647270ef8de7aa8a68428b49afee591e82cc58880')])
    before = len(os.listdir('/proc/self/fd'))
    for _ in range(12):
        with t.open(str(p)) as f: assert len(f.read()) == t.entries[str(p)].stat[2]
    assert len(os.listdir('/proc/self/fd')) == before


def test_fork_waits_for_direct_fd_and_mapping_to_close(tmp_path, monkeypatch):
    import os, threading, time
    m = transport(); p = tmp_path / 'cp0_dp0__0.tar'; data = b'z' * 10241; p.write_bytes(data)
    t = m.DirectTransport([m.expectation(p, hashlib.sha256(data).hexdigest())])
    entered = threading.Event(); real = m.os.preadv
    def slow(fd, buffers, offset):
        entered.set(); time.sleep(0.15); return real(fd, buffers, offset)
    monkeypatch.setattr(m.os, 'preadv', slow)
    failures = []
    def read():
        try: assert t.read_bytes(str(p)) == data
        except Exception as exc: failures.append(repr(exc))
    thread = threading.Thread(target=read); thread.start(); assert entered.wait(5)
    pid = os.fork()
    if pid == 0:
        leaked = []
        for name in os.listdir('/proc/self/fd'):
            try:
                if os.readlink('/proc/self/fd/' + name) == str(p): leaked.append(name)
            except FileNotFoundError: pass
        os._exit(1 if leaked else 0)
    _, status = os.waitpid(pid, 0); thread.join(5)
    assert os.waitstatus_to_exitcode(status) == 0
    assert not thread.is_alive() and not failures


def test_activate_requires_opt_in():
    from midtraining.cache.directio.integration import activate
    with pytest.raises(RuntimeError, match='opt-in'):
        activate(None, '/not/read', '0' * 64)


def test_activate_real_original_native_opener_opt_in():
    import ast, types
    from midtraining.cache.directio.integration import activate
    path = ROOT / 'results/midtraining/offline1920-parent-01/directio-acceptance-01/expectations.json'
    data = path.read_bytes(); manifest = json.loads(data)
    source = Path(manifest['native_source'])
    node = next(n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'open_logit_file')
    ns = {'_msc_if_needed': lambda p: None}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    module = types.SimpleNamespace(__file__=str(source), open_logit_file=ns['open_logit_file'])
    receipt = activate(module, path, hashlib.sha256(data).hexdigest(), opt_in=True)
    assert receipt['shards'] == 9144 and receipt['native_codec_unchanged']
    with module.open_logit_file(str(CACHE / 'cp0_dp7__70.tar'), 'rb', prefetch_file=True) as f:
        assert hashlib.sha256(f.read()).hexdigest() == 'd450e523a6bcf1900edf511529f232c43c88c29ce379c65782a096616896718a'


@pytest.mark.parametrize('rc,mask,mem,off', [(-1,0,0,0), (0,0,512,512), (0,0x2000,0,512), (0,0x2000,8192,512), (0,0x2000,512,513)])
def test_actual_statx_alignment_fail_closed(monkeypatch, rc, mask, mem, off):
    import ctypes, struct, types
    m = transport()
    def statx(fd, path, flags, requested, pointer):
        buf = bytearray(256); struct.pack_into('I',buf,0,mask); struct.pack_into('II',buf,152,mem,off)
        ctypes.memmove(pointer, bytes(buf), len(buf)); return rc
    monkeypatch.setattr(m.ctypes, 'CDLL', lambda *args, **kwargs: types.SimpleNamespace(statx=statx))
    with pytest.raises(RuntimeError, match='alignment'):
        m.dio_alignment(0)
