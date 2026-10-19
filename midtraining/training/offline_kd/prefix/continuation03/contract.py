"""Separate bounded execution from immutable full-run identity."""
import hashlib
import json

def segment_plan(config, stop_step):
    if not 0 < stop_step <= 1143:
        raise ValueError('stop must be inside authenticated completed prefix')
    return dict(stop_step=stop_step,available_iterations=1143,
                sampler_budget=[config['train_samples'],(config['train_iters']//250+1)*8,8],
                schedule_horizon=config['train_iters'],warmup_iters=config['warmup_iters'],
                decay_iters=config['decay_iters'],
                cache_id=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest())

def build_full_datasets(config, context, builder, dataset):
    dataset.tokenizer=context.tokenizer
    dataset.finalize()
    return builder(segment_plan(config,1143)['sampler_budget'],dataset)

def authenticate_shard(path, config, iteration):
    import os
    import tarfile
    with tarfile.open(path,'r:') as tar:
        members=tar.getmembers()
        if [m.name for m in members]!=['_meta.json',f'{iteration}.pt.zst']:
            raise ValueError('closed shard payload ownership mismatch')
        if not all(m.isfile() and m.size>0 for m in members):
            raise ValueError('invalid closed shard members')
        raw=tar.extractfile(members[0]).read()
        meta=json.loads(raw)
        ids=meta['identifiers']
        if any(ids[k]!=v for k,v in {'seed':config['seed'],'sequence_length':4096,'train_samples':config['train_samples']}.items()):
            raise ValueError('full sampler metadata mismatch')
        if meta['saver']!={'k':32,'p':None,'min_k':1,'save_dtype':'fp16'}:
            raise ValueError('cache codec mismatch')
    st=path.stat()
    if st.st_uid!=os.getuid():raise ValueError('foreign shard owner')
    return dict(name=path.name,bytes=st.st_size,mtime_ns=st.st_mtime_ns,metadata_sha256=hashlib.sha256(raw).hexdigest())



def continuation_plan(config,resume_step,stop_step):
    plan=segment_plan(config,stop_step)
    if not 0 < resume_step < stop_step:
        raise ValueError('continuation cursor must precede bounded stop')
    plan.update(resume_step=resume_step,validation_interval=250,
                remaining_validation_samples=((stop_step//250-resume_step//250)+1)*8)
    return plan


def create_local_compiler_cache(target):
    # Inductor generated Python embeds absolute kernel/cache filenames.
    # Copying another workspace can write outside this owned writable mount.
    target.mkdir(parents=True,exist_ok=False)
    return target
