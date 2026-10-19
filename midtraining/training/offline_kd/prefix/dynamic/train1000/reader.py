"""Synchronous verified bytes -> untouched native decoder -> native KD reader."""
import hashlib
import io
import json
from pathlib import Path
from .broker import CacheRejected

def quarantinable_transport(exc):
    import errno
    if isinstance(exc,OSError):return exc.errno in (errno.EIO,errno.ENOENT,errno.ESTALE,errno.ELOOP)
    message=str(exc)
    return any(message.startswith(prefix) for prefix in ('size/SHA mismatch','pre-read stat mismatch',
        'post-read stat/path mismatch','invalid/zero short read','unaligned short read',
        'invalid/duplicate closed-tar allowlist entry'))

class DynamicDataset:
    def __init__(self, source, broker, rank):
        if len(source)<468736:raise ValueError('original full 468736 sampler budget required; native surplus is preserved')
        self.source=source;self.broker=broker;self.rank=rank
    def __len__(self):return len(self.source)
    def __getitem__(self, i):
        logical,offset=divmod(i,128)
        if (logical!=self.broker.calls or len(self.broker.mapping)!=logical+1
            or offset%8!=self.rank):raise IndexError(f'unresolved or wrong-rank logical sample {i}')
        return self.broker.current[1][2][offset//8]

class ImmutableOpener:
    def __init__(self):self.path=None;self.data=None
    def open(self,path,mode='rb',**kwargs):
        if str(path)!=self.path or self.data is None or mode!='rb' or kwargs not in ({},{'prefetch_file':True}):
            raise RuntimeError('unverified native cache read is forbidden')
        return io.BytesIO(self.data)

def validate_source_digest(physical,micro,rank,source_index,expected,actual,masked_tokens):
    if (not 0<=physical<1143 or not 0<=micro<16 or not 0<=rank<8
        or source_index!=physical*128+micro*8+rank or not 0<=source_index<468736
        or set(expected)!={'micro','iteration','rank','sha256','masked_tokens'}
        or (expected['micro'],expected['iteration'],expected['rank'])!=(physical*16+micro,physical,rank)
        or not isinstance(expected['sha256'],str) or len(expected['sha256'])!=64):
        raise ValueError('structural source index/budget/teacher schema disagreement')
    if actual!=expected['sha256'] or masked_tokens!=expected['masked_tokens']:
        raise CacheRejected(dict(error_type='SourceInputDigestMismatch',error='pre-forward native source input differs from historical teacher bytes',
            candidate_physical=physical,micro=micro,source_index=source_index,rank=rank,
            expected_sha256=expected['sha256'],observed_sha256=actual,
            expected_masked_tokens=expected['masked_tokens'],observed_masked_tokens=masked_tokens))

class VerifiedReader:
    def __init__(self, full, rank, native, opener, source, expected_samples, out):
        self.full=full;self.rank=rank;self.native=native;self.opener=opener
        self.source=source;self.expected_samples=expected_samples;self.out=out
        self.rows={x['iteration']:x for x in full['shards'] if x['rank']==rank}
    def __call__(self,physical):
        import torch
        from midtraining.training.offline_kd.prefix.filtered.transport import DirectTransport,Expectation
        row=self.rows[physical];path=row['path']
        observation=dict(path=path,expected_sha256=row['sha256'],expected_bytes=row['stat'][2],
                         observed_sha256=None,observed_bytes=None)
        # Classify only this input read boundary, never model/collective/optimizer errors.
        try:
            transport=DirectTransport([Expectation(path,row['sha256'],tuple(row['stat']))])
            data=transport.read_bytes(path)
        except (OSError,RuntimeError,ValueError) as exc:
            if not quarantinable_transport(exc):raise
            import re
            message=str(exc)
            for key in ('observed_sha256','observed_bytes'):
                match=re.search(key+r'=([^ ]+)',message)
                if match:observation[key]=int(match[1]) if key=='observed_bytes' else match[1]
            observation.update(error_type=type(exc).__name__,error=message)
            raise CacheRejected(observation) from exc
        observation.update(observed_sha256=hashlib.sha256(data).hexdigest(),observed_bytes=len(data))
        self.opener.path=path;self.opener.data=data
        import pickle, tarfile, zstandard
        try:
            entries=list(self.native.iter_logprobs_tar_entries(path,expected_hash=self.full['metadata_hash']))
            if len(entries)!=1 or entries[0].iteration!=physical:raise ValueError('native tar physical member mismatch')
            if hashlib.sha256(entries[0].data).hexdigest()!=row['payload_sha256']:raise ValueError('native payload SHA mismatch')
            values,indices=self.native.decode_logprobs_payload(entries[0].data)
            if len(values)!=16 or len(indices)!=16:raise ValueError('native microbatch count mismatch')
            for v,i in zip(values,indices):
                if (tuple(v.shape)!=(4096,1,32) or v.dtype!=torch.float16 or v.device.type!='cpu'
                    or tuple(i.shape)!=(4096,1,32) or i.dtype!=torch.int64 or i.device.type!='cpu'
                    or not bool(torch.isfinite(v).all()) or int(i.min())<0 or int(i.max())>131071):
                    raise ValueError('native tensor shape/dtype/finite/index validation failed')
        except (ValueError,RuntimeError,EOFError,tarfile.TarError,zstandard.ZstdError,pickle.UnpicklingError) as exc:
            observation.update(error_type=type(exc).__name__,error=str(exc))
            raise CacheRejected(observation) from exc
        finally:self.opener.path=None;self.opener.data=None
        # Only a correctly mapped pre-forward candidate digest rejection is DATA.
        # Structural mapping/schema exceptions remain fatal; accepted inputs are
        # checked again in audited_common and any mutation there remains fatal.
        samples=[]
        for micro in range(16):
            source_index=physical*128+micro*8+self.rank
            if len(self.source)<468736 or not 0<=source_index<468736:
                raise ValueError('structural source sampler budget/index mismatch')
            sample=self.source[source_index]
            if any(key not in sample or tuple(sample[key].shape)!=(4096,)
                   or sample[key].device.type!='cpu' for key in ('tokens','labels','loss_mask','position_ids')):
                raise ValueError('structural native sample schema mismatch')
            h=hashlib.sha256()
            for key in ('tokens','labels','loss_mask','position_ids'):
                h.update(sample[key].cpu().contiguous().numpy().tobytes())
            expected=self.expected_samples[physical*16+micro]
            validate_source_digest(physical,micro,self.rank,source_index,expected,
                                   h.hexdigest(),int(sample['loss_mask'].sum()))
            samples.append(sample)
        with (self.out/f'cache-reads-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(dict(observation,physical_iteration=physical,rank=self.rank,
                transport='O_DIRECT',codec='unaltered_native',samples_verified=16))+'\n')
        return values,indices,samples

def install_loss_iterator(native,broker,expected_sha256):
    if hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()!=expected_sha256:
        raise RuntimeError('native cached loss source changed')
    original=native.CachedLogitsKDLoss
    class SynchronousCachedLoss(original):
        def _init_dataloader(self,start_iteration):
            def verified():
                while True:
                    values,indices,_=broker.take()
                    yield values,indices
            self._dataloader_iter=verified()
    native.CachedLogitsKDLoss=SynchronousCachedLoss
