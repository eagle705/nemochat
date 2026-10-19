"""New-output-only native cache acceptance, exact producer payload + direct read."""
import hashlib
import io
import json
import os
import tarfile
import zipfile
from midtraining.training.offline_kd.prefix.filtered.transport import DirectTransport, Expectation, identity

def audit_closed(path,physical,rank,full,native,opener,emitted_sha):
    import torch
    import zstandard
    # New outputs have no historical tar SHA. Establish their expectation only
    # with producer-emitted RAW payload equality and native metadata validation.
    # Require both read paths to agree; never refresh an OLD expectation here.
    before=identity(path.stat());buffered=path.read_bytes();tarsha=hashlib.sha256(buffered).hexdigest()
    data=DirectTransport([Expectation(str(path),tarsha,before)]).read_bytes(str(path))
    if path.name!=f'cp0_dp{rank}__{physical}.tar':raise ValueError('native cache filename identity mismatch')
    opener.path=str(path);opener.data=data
    try:
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tar:
            members=tar.getmembers()
            if [m.name for m in members]!=['_meta.json',f'{physical}.pt.zst'] or not all(m.isreg() for m in members):
                raise ValueError('native tar member structure mismatch')
            if json.load(tar.extractfile(members[0]))!=full['metadata']:raise ValueError('original native metadata mismatch')
        entries=list(native.iter_logprobs_tar_entries(str(path),expected_hash=full['metadata_hash']))
        if len(entries)!=1 or entries[0].iteration!=physical:raise ValueError('native physical iteration mismatch')
        payload=entries[0].data;raw=zstandard.ZstdDecompressor().decompress(payload)
        if hashlib.sha256(raw).hexdigest()!=emitted_sha:raise ValueError('new native emitted payload mismatch')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if archive.testzip() is not None:raise ValueError('new teacher serialized data CRC failed')
        values,indices=native.decode_logprobs_payload(payload)
        if len(values)!=16 or len(indices)!=16:raise ValueError('native microbatch count mismatch')
        for v,i in zip(values,indices):
            if (tuple(v.shape)!=(4096,1,32) or tuple(i.shape)!=(4096,1,32)
                or v.dtype!=torch.float16 or i.dtype!=torch.int64
                or v.device.type!='cpu' or i.device.type!='cpu'
                or not bool(torch.isfinite(v).all()) or int(i.min())<0 or int(i.max())>=131072):
                raise ValueError('native tensor dtype/shape/finite/index mismatch')
        fd=os.open(path,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return dict(rank=rank,iteration=physical,path=str(path),sha256=tarsha,
            stat=list(identity(path.stat())),payload_sha256=hashlib.sha256(payload).hexdigest(),
            emitted_raw_payload_sha256=emitted_sha,native_codec=True,direct_sha=True,finite=True)
    finally:opener.path=None;opener.data=None
