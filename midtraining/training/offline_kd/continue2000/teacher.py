"""Fresh bounded frozen native teacher. No historical prefix is replayed.

Full native dataset/sample budget is unchanged. Each whole global batch is
schema/index/direct-input accepted on all8 BEFORE its first model forward.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'midtraining/training/offline_kd'))
from midtraining.training.offline_kd.continue2000.contract import CACHE_ID,plan,SOURCE,OUT as HOST_OUT
from midtraining.training.offline_kd.continue2000.input_gate import verified_sample,sample_digest,SourceDigestMismatch
from midtraining.training.offline_kd.continue2000.broker import durable
OUT=Path('/output');RANK=int(os.environ.get('RANK','0'))

def main():
    import torch
    import torch.distributed as dist
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.training.cached_kd.logits_saver import LogitsSaverHooks
    from megatron.bridge.training.cached_kd.runtime import bind_runtime
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
    from megatron.bridge.data.samplers import build_pretraining_data_loader
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model
    spec=importlib.util.spec_from_file_location('full_original',REPO/'midtraining/training/offline_kd/native.py')
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    original.C=json.loads((SOURCE/'run-config.json').read_text());original.OUT=OUT
    p=plan();assert original.cache_id()==CACHE_ID and original.dataset_budget()==[468736,120,8]
    kd=original.kd_helpers()
    if RANK==0:original.publish('tokenizer',kd.tokenizer_gate(REPO/original.STUDENT,REPO/original.TEACHER))
    provider=AutoBridge.from_hf_pretrained(str(REPO/original.TEACHER),trust_remote_code=False,
        local_files_only=True).to_megatron_provider(load_weights=True)
    kd.configure_provider(provider,dict(parallel=original.PARALLEL,sequence_length=4096))
    provider.hidden_dropout=0.;provider.attention_dropout=0.;provider.finalize()
    model=provider.provide_distributed_model(wrap_with_ddp=False,bf16=True)[0]
    model.requires_grad_(False);model.train();kd.check_teacher(model)
    assert parallel_state.get_data_parallel_world_size()==parallel_state.get_expert_model_parallel_world_size()==8
    assert all(x.dtype==torch.bfloat16 for x in model.parameters())
    ds=original.dataset_config();ds.tokenizer=build_tokenizer(original.tokenizer_config());ds.finalize()
    train,_,_=pretrain_train_valid_test_datasets_provider(original.dataset_budget(),ds)
    spec=importlib.util.spec_from_file_location('full_unique_gate',REPO/'midtraining/training/longrun_native.py')
    longrun=importlib.util.module_from_spec(spec);spec.loader.exec_module(longrun)
    unique=longrun.no_repeat_gate(train)
    original.publish(f'teacher-rank{RANK}-dataset',dict(full_budget=original.dataset_budget(),
        actual_length=len(train),no_repeat=unique,start_consumed=p['consumed_teacher_start']))
    loader=build_pretraining_data_loader(train,consumed_samples=p['consumed_teacher_start'],dataloader_type='single',
        micro_batch_size=1,num_workers=0,data_sharding=True,persistent_workers=False,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),data_parallel_size=8)
    iterator=iter(loader);scored=[];quarantined=[];rows=[];rejections=[]
    for suffix in ('samples','shards'):
        with (OUT/f'teacher-rank{RANK}-{suffix}.jsonl').open('x'):pass
    def gather(row):
        ranks=[None]*8;dist.all_gather_object(ranks,row)
        if [r['rank'] for r in ranks]!=list(range(8)) or any(r['iteration']!=row['iteration'] for r in ranks):
            raise RuntimeError('teacher allrank source identity mismatch')
        return ranks
    full=json.loads((OUT/'preparation/old-expectations.json').read_text())
    from megatron.bridge.training.cached_kd import utils_logits
    from midtraining.cache.directio.binding import install
    from midtraining.training.offline_kd.prefix.dynamic.restart01.reader import ImmutableOpener
    from midtraining.training.offline_kd.continue2000.new_cache import audit_closed
    opener=ImmutableOpener();install(utils_logits,full['native_contract'],opener)
    for physical in range(p['teacher_start'],p['teacher_stop']):
        samples=[];errors=[];local=dict(rank=RANK,iteration=physical,status='valid')
        for micro in range(16):
            index=physical*128+micro*8+RANK
            try:
                batch=next(iterator);given={k:v[0] for k,v in batch.items()}
                sample=verified_sample(train,index,given=given);samples.append(sample)
            except SourceDigestMismatch as exc:
                errors.append(dict(micro=micro,**exc.observation))
                if local['status']!='fatal':local['status']='quarantine'
            except OSError as exc:
                import errno
                if exc.errno in (errno.EIO,errno.ESTALE,errno.ENOENT):
                    errors.append(dict(micro=micro,source_index=index,error=str(exc)))
                    if local['status']!='fatal':local['status']='quarantine'
                else:local.update(status='fatal',error_type=type(exc).__name__,error=str(exc))
            except Exception as exc:local.update(status='fatal',error_type=type(exc).__name__,error=str(exc))
        local['observations']=errors
        ranks=gather(local)
        if any(r['status']=='fatal' for r in ranks):raise RuntimeError(f'teacher structural/raw constructor failure: {ranks}')
        if any(r['status']!='valid' for r in ranks):
            quarantined.append(physical);rejections.append(dict(iteration=physical,ranks=ranks,no_forward=True,optimizer_calls=0))
            if RANK==0:durable(OUT/'teacher-input-quarantine.json',rejections)
            gather(dict(rank=RANK,iteration=physical,status='ledger_committed'))
            continue
        if len(samples)!=16:raise RuntimeError('teacher accepted incomplete native batch')
        state=SimpleNamespace(cfg=SimpleNamespace(model=provider),
            train_state=SimpleNamespace(step=physical,consumed_train_samples=physical*128))
        with bind_runtime(original.runtime(state)):
            saver=LogitsSaverHooks(str(OUT/'cache'),k=32,p=None,min_k=1,save_dtype='fp16')
            saver.attach_hooks(unwrap_model(model))
            try:
                for micro,sample in enumerate(samples):
                    row=dict(micro=physical*16+micro,iteration=physical,rank=RANK,
                        sha256=sample_digest(sample),masked_tokens=int(sample['loss_mask'].sum()))
                    with (OUT/f'teacher-rank{RANK}-samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    gpu={k:v.unsqueeze(0).cuda(non_blocking=True) for k,v in sample.items()}
                    with torch.no_grad():model(input_ids=gpu['tokens'],position_ids=gpu['position_ids'],attention_mask=None,labels=gpu['labels'])
                torch.cuda.synchronize()
                path,writes,meta,msc=saver.take_pending_data()
                if list(writes)!=[physical] or msc:raise ValueError('native writer global cursor/storage mismatch')
                emitted_sha=hashlib.sha256(writes[physical]).hexdigest()
                saver._write_batched_tar(path,writes,meta,msc)
            finally:saver.remove_hooks()
        row=audit_closed(Path(path),physical,RANK,full,utils_logits,opener,emitted_sha)
        row['path']=str(HOST_OUT/'cache'/Path(path).name)
        rows.append(row);scored.append(physical)
        with (OUT/f'teacher-rank{RANK}-shards.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        kd.check_teacher(model)
        gather(dict(rank=RANK,iteration=physical,status='scored_closed_validated'))
        if RANK==0:original.publish('teacher-progress',dict(last_physical=physical,scored=len(scored),quarantined=len(quarantined),student_optimizer_calls=0))
    durable(OUT/f'teacher-rank{RANK}-complete.json',dict(rank=RANK,scored=scored,quarantined=quarantined,
        frozen=True,optimizer_calls=0,cache_id=CACHE_ID,source_budget=original.dataset_budget(),
        direct_input_checked=True,native_cache_validated=True))
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
