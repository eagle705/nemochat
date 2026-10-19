"""Activate only accepted kept shard SHA allowlist and sparse cursor adapter."""
import hashlib
import json
import os
from pathlib import Path
from .mapping import INVERSE,identity
from .transport import DirectTransport,Expectation
from .reader import install as install_reader
from midtraining.cache.directio.binding import install as install_opener

def activate(manifest,out):
    from megatron.bridge.training.cached_kd import utils_logits,cached_logits_loss
    assert manifest['filtered_mapping_sha256']==identity()
    full=Path(manifest['full_expectations_path']).read_bytes()
    if hashlib.sha256(full).hexdigest()!=manifest['full_expectations_sha256']:raise ValueError('full original expectations changed')
    class AuditedTransport(DirectTransport):
        def read_bytes(self,path):
            try:
                data=super().read_bytes(path)
            except BaseException as exc:
                with (out/f'cache-read-rejected-rank{os.environ["RANK"]}.jsonl').open('a') as f:
                    f.write(json.dumps(dict(path=path,error=str(exc)))+'\n')
                raise
            physical=int(Path(path).stem.split('__')[1])
            row=dict(rank=int(os.environ['RANK']),physical_iteration=physical,logical_iteration=INVERSE[physical],sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),transport='O_DIRECT')
            with (out/f'cache-reads-rank{os.environ["RANK"]}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            return data
    t=AuditedTransport([Expectation(x['path'],x['sha256'],tuple(x['stat'])) for x in manifest['shards']])
    install_opener(utils_logits,manifest['native_contract'],t)
    install_reader(cached_logits_loss,manifest['native_reader_sha256'])
