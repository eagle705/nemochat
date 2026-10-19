"""Native Core-first cached KD. Same full GPTDataset budget on dump/train/resume."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[4] / 'midtraining/training/offline_kd'))
from launch import REPO,STUDENT,TEACHER,KEYS,PARALLEL
from midtraining.training.offline_kd.prefix.contract import build_full_datasets
sys.path.insert(0,str(REPO))
from midtraining.training.offline_kd.continue2000.broker import Broker, durable
from midtraining.training.offline_kd.continue2000.contract import OLD, extend_state, sha
from midtraining.training.offline_kd.continue2000.input_gate import VerifiedSource
from midtraining.training.offline_kd.continue2000.binding import adapter_class
from midtraining.training.offline_kd.prefix.dynamic.restart01.reader import DynamicDataset, VerifiedReader, ImmutableOpener, install_loss_iterator
BROKER=None
SOURCE=Path('/raid/yoon/nemochat/results/midtraining/offline2b-production1920-01')
CACHE=SOURCE/'cache'
STOP=2000
OUT=Path('/output')
C=None
RANK=int(os.environ.get('RANK','0'))
STAGE=None

def publish(name,value):
    durable(OUT/(name+'.json'),value)
    print(name,json.dumps(value,allow_nan=False),flush=True)

def kd_helpers():
    spec=importlib.util.spec_from_file_location('offline2b_accepted_kd',REPO/'midtraining/training/native_kd.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def cache_id():
    return hashlib.sha256(json.dumps(C,sort_keys=True).encode()).hexdigest()

def tokenizer_config():
    from megatron.bridge.training.config import TokenizerConfig
    return TokenizerConfig(tokenizer_type='NullTokenizer',vocab_size=131072,padded_vocab_size=131072,
                           use_tokenizer_vocab_size=False,null_tokenizer_eod_id=11)

def dataset_config():
    from megatron.bridge.training.config import GPTDatasetConfig
    manifest=json.loads(Path(C['manifest_path']).read_text())
    if hashlib.sha256(Path(C['manifest_path']).read_bytes()).hexdigest()!=C['manifest_sha256']:
        raise ValueError('corpus manifest changed')
    paths=[[str(REPO/manifest['sources'][key][split]['prefix']) for key in KEYS] for split in ('train','validation')]
    return GPTDatasetConfig(seq_length=4096,random_seed=C['seed'],blend=None,
        blend_per_split=[(paths[0],C['weights']),(paths[1],C['weights']),None],split=None,
        num_dataset_builder_threads=1,num_workers=0,dataloader_type='single',persistent_workers=False,
        eod_mask_loss=False,reset_position_ids=False,reset_attention_mask=False,
        skip_getting_attention_mask_from_dataset=True,add_extra_token_to_sequence=True,
        path_to_cache=str(OUT/'dataset-cache'))

def runtime(state):
    from megatron.bridge.training.cached_kd.runtime import CoreRuntime
    ds=dataset_config()
    return CoreRuntime(seed=C['seed'],seq_length=4096,train_samples=C['train_samples'],
        global_batch_size=128,micro_batch_size=1,blend=ds.blend,blend_per_split=ds.blend_per_split,
        iteration=lambda:int(state.train_state.step),num_microbatches=lambda:16,
        data_parallel_size=8,expert_parallel_size=8,dataset_identity=cache_id())

def dataset_budget():
    interval=2 if C['mode']=='gate' else 250
    return [C['train_samples'],(C['train_iters']//interval+1)*8,8]

def audited_common(teacher=False):
    """Bind actual sample bytes, not only a seed or assumed index equivalence."""
    import torch
    from megatron.bridge.training.gpt_step import _forward_step_common
    seen={'micro':0,'masked_tokens':0}
    expected={}
    if not teacher:
        expected={r['micro']:r for r in map(json.loads,(OUT/f'merged-samples-rank{RANK}.jsonl').read_text().splitlines())}
    def common(state,iterator,model,return_schedule_plan=False):
        batch=next(iterator)
        if model.training:
            logical=int(state.train_state.step)
            physical=BROKER.mapping[logical]
            i=physical*16+seen['micro']%16
            h=hashlib.sha256()
            for key in ('tokens','labels','loss_mask','position_ids'):
                h.update(batch[key].cpu().contiguous().numpy().tobytes())
            row=dict(micro=i,iteration=physical,rank=RANK,sha256=h.hexdigest(),
                     masked_tokens=int(batch['loss_mask'].sum()))
            if not teacher and row!=expected[i]:raise ValueError(f'actual teacher/student sample bytes differ logical={logical} physical={physical} micro={i}')
            row.update(logical_iteration=logical,physical_iteration=physical,logical_micro=logical*16+seen['micro']%16)
            with (OUT/f'{STAGE}-rank{RANK}-samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            seen['micro']+=1;seen['masked_tokens']+=row['masked_tokens']
        if teacher:
            gpu={k:(v.cuda(non_blocking=True) if v is not None else None) for k,v in batch.items()}
            output=model(input_ids=gpu['tokens'],position_ids=gpu['position_ids'],attention_mask=None,labels=gpu['labels'])
            return output,gpu['loss_mask']
        return _forward_step_common(state,iter([batch]),model,return_schedule_plan)
    return common,seen

def student(resume=False):
    import torch
    import torch.distributed as dist
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.config import (ConfigContainer,TrainingConfig,OptimizerConfig,SchedulerConfig,
        LoggerConfig,CheckpointConfig,DistributedDataParallelConfig,ValidationConfig,RNGConfig,runtime_config_update)
    from megatron.bridge.training.cached_kd.integration import CachedKDConfig,CachedKDForwardStep
    from megatron.core.utils import unwrap_model
    manifest=json.loads((OUT/'prefix-manifest.json').read_text())
    assert manifest['cache_id']==cache_id()
    full_bytes=Path(manifest['full_expectations_path']).read_bytes()
    assert hashlib.sha256(full_bytes).hexdigest()==manifest['full_expectations_sha256']
    full=json.loads(full_bytes)
    assert manifest['sampler_budget']==dataset_budget()==[468736,120,8]
    assert full['budget']==C
    assert len(full['shards'])==manifest['merged_shard_count']
    from megatron.bridge.training.cached_kd import utils_logits,cached_logits_loss
    from midtraining.cache.directio.binding import install
    opener=ImmutableOpener()
    install(utils_logits,full['native_contract'],opener)
    def gather(row):
        ranks=[None]*8;dist.all_gather_object(ranks,row);return ranks
    global BROKER
    BROKER=Broker(RANK,None,gather,OUT,limit=2064,historical=manifest['historical_global_exclusions'])
    BROKER.historical_details={(r['rank'],r['source_iteration']):r['observation'] for r in manifest['historical_observations']}
    install_loss_iterator(cached_logits_loss,BROKER,manifest['native_reader_sha256'])
    start=16
    if not resume:raise ValueError('continue2000 requires FULL checkpoint16 resume; no model-only reset')
    binding=json.loads((OUT/'resume-ancestry.json').read_text())['binding']
    old_manifest=json.loads((OLD/'prefix-manifest.json').read_text())
    extended_state=extend_state(binding,old_manifest,sha(OLD/'prefix-manifest.json'),2064)
    if RANK==0:publish('resume-lineage',dict(full_state=True,optimizer_reset=False,rng_reset=False,
        cursor_reset=False,restored_step=start,ancestry_manifest_sha256=binding['dynamic_manifest_sha256'],
        expanded_manifest_sha256=sha(OUT/'prefix-manifest.json'),schedule_horizon=C['train_iters'],bounded_stop=STOP))
    kd=kd_helpers();provider=kd.provider_from_hf(REPO/STUDENT,load_weights=not resume)
    kd.configure_provider(provider,dict(parallel=PARALLEL,sequence_length=4096))
    provider.hidden_dropout=0.;provider.attention_dropout=0.;kd.check_provider_shapes(provider)
    interval=32
    from megatron.bridge.data.base import DatasetProvider
    from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
    class FullBudgetProvider(DatasetProvider):
        seq_length=4096
        def build_datasets(self,context):
            publish(f'{STAGE}-rank{RANK}-dataset-budget',dict(full=dataset_budget(),
                segment_requested=[context.train_samples,context.valid_samples,context.test_samples]))
            train,valid,test=build_full_datasets(C,context,pretrain_train_valid_test_datasets_provider,dataset_config())
            if train is None:raise RuntimeError('DP8 requires full native dataset on every rank')
            expected={r['micro']:r for r in map(json.loads,(OUT/f'merged-samples-rank{RANK}.jsonl').read_text().splitlines())}
            BROKER.reader=VerifiedReader(full,RANK,utils_logits,opener,VerifiedSource(train,expected),expected,OUT)
            return DynamicDataset(train,BROKER,RANK),valid,test
    cfg=ConfigContainer(model=provider,
        train=TrainingConfig(train_iters=STOP,micro_batch_size=1,global_batch_size=128,
                             exit_duration_in_mins=None),
        optimizer=OptimizerConfig(lr=1e-5,min_lr=1e-6,weight_decay=.01,bf16=True,params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,overlap_param_gather=False,clip_grad=1.,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-8),
        ddp=DistributedDataParallelConfig(grad_reduce_in_fp32=True,overlap_grad_reduce=False,
            use_distributed_optimizer=False,overlap_param_gather=False,average_in_collective=False),
        scheduler=SchedulerConfig(lr_decay_style='WSD',lr_decay_iters=C['train_iters'],
            lr_warmup_iters=C['warmup_iters'],lr_wsd_decay_iters=C['decay_iters'],
            start_weight_decay=.01,end_weight_decay=.01,weight_decay_incr_style='constant',
            use_checkpoint_opt_param_scheduler=resume),
        dataset=FullBudgetProvider(dataloader_type='single',num_workers=0,persistent_workers=False),tokenizer=tokenizer_config(),
        logger=LoggerConfig(log_interval=1,wandb_project=None,wandb_entity=None,wandb_exp_name=None,
            wandb_save_dir=str(OUT/'sdk-logs'),save_config_filepath=str(OUT/f'{STAGE}-final-config.yaml')),
        checkpoint=CheckpointConfig(save=str(OUT/'checkpoint'),
            load=str(OLD/'checkpoint'),pretrained_checkpoint=None,save_interval=interval,
            save_optim=True,save_rng=True,load_optim=resume,load_rng=resume,finetune=False,
            exit_on_missing_checkpoint=True,async_save=False,fully_parallel_save=False,ckpt_format='torch_dist',
            save_tokenizer_assets=False,verify_integrity=False,most_recent_k=3),
        validation=ValidationConfig(eval_iters=1,eval_interval=250,eval_global_batch_size=8),
        rng=RNGConfig(seed=C['seed']),mixed_precision='bf16_mixed')
    runtime_config_update(cfg)
    if RANK==0:cfg.to_yaml(str(OUT/f'{STAGE}-config.yaml'))
    common,seen=audited_common()
    DynamicForwardStep=adapter_class(CachedKDForwardStep,binding,BROKER,
        binding['dynamic_manifest_sha256'],sha(OUT/'prefix-manifest.json'))
    adapter=DynamicForwardStep(CachedKDConfig('read',str(CACHE),cache_id(),msc_prefetch_depth=0),runtime,
                                resume_binding=binding,common_forward=common)
    updates=[];saves=[];timer={};held=[]
    class Audit(Callback):
        def on_data_init_start(self,ctx):
            assert int(ctx.state.train_state.step)==start
            assert int(ctx.state.train_state.consumed_train_samples)==start*128
            if resume:
                adapter.validate_resume(runtime(ctx.state))
                if not math.isclose(float(ctx.optimizer.param_groups[0]['lr']),1e-5*16/37,rel_tol=1e-8,abs_tol=1e-15):
                    raise ValueError('native checkpoint16 scheduler LR not restored')
                publish(f'resume-rank{RANK}-restored',dict(step=start,consumed_samples=start*128,
                    optimizer_loaded=True,rng_loaded=True,lr=float(ctx.optimizer.param_groups[0]['lr']),binding=binding))
        def on_train_start(self,ctx):
            kd.check_provider_shapes(unwrap_model(ctx.model[0]).config)
            BROKER.restore(extended_state)
            BROKER.prime()
            if BROKER.current is None:
                ctx.state.cfg.train.train_iters=start
                if RANK==0:durable(OUT/'no-data.json',dict(status='NoData',optimizer_calls=0,dynamic_state=BROKER.snapshot()))
            # Inspect actual native optimizer groups before deciding on the TE handle-pool cap.
            counts=[len(g['params']) for g in ctx.optimizer.param_groups]
            if max(counts,default=0)*4>20000:
                from optimizer_compat import install_and_verify
                compatibility=install_and_verify()
            else:compatibility=dict(installed=False,reason='EP8 native groups within TE handle pool')
            publish(f'{STAGE}-rank{RANK}-loaded',dict(optimizer_group_tensors=counts,compatibility=compatibility,
                hf_import=getattr(provider,'_nemochat_hf_import_receipt',{}),teacher_present=False))
        def on_train_step_start(self,ctx):
            logical=int(ctx.state.train_state.step)
            physical=BROKER.begin(logical)
            if BROKER.is_last:ctx.state.cfg.train.train_iters=logical+1
            torch.cuda.synchronize();timer.update(start=time.perf_counter(),lr=float(ctx.optimizer.param_groups[0]['lr']),physical=physical)
        def on_train_step_end(self,ctx):
            torch.cuda.synchronize();step=int(ctx.state.train_state.step)+1
            local=seen['masked_tokens'];seen['masked_tokens']=0
            counts=torch.tensor([local],device='cuda',dtype=torch.int64);dist.all_reduce(counts)
            row=dict(rank=RANK,update=step,metrics={str(k):float(v) for k,v in ctx.loss_dict.items()},
                grad_norm=None if ctx.grad_norm is None else float(ctx.grad_norm),lr_used=timer['lr'],
                skipped=bool(ctx.skipped_iter),seconds=time.perf_counter()-timer['start'],
                local_masked_tokens=local,allrank_masked_tokens=int(counts.item()),
                logical_iteration=step-1,physical_iteration=timer['physical'],
                global_batch_size=128,sequence_length=4096,peak_memory_bytes=torch.cuda.max_memory_allocated())
            with (OUT/f'{STAGE}-rank{RANK}-updates.jsonl').open('a') as s:s.write(json.dumps(row)+'\n')
            updates.append(row)
            if row['skipped'] or row['grad_norm'] is None or row['grad_norm']<=0 or row['lr_used']<0 or not all(
                math.isfinite(v) for v in [*row['metrics'].values(),row['grad_norm'],row['lr_used']]):
                raise FloatingPointError('actual nonfinite/zero-gradient/skipped update')
            assert row['allrank_masked_tokens']==128*4096
            BROKER.complete()
            assert BROKER.calls==step
            if RANK==0:
                durable(OUT/'dynamic-state.json',BROKER.snapshot())
                publish('student-progress',dict(stage=STAGE,update=step,total=C['train_iters'],physical=timer['physical'],quarantines=len(BROKER.exclusions)))
        def on_checkpoint_save(self,ctx):
            step=int(ctx.state.train_state.step);b=adapter.checkpoint_binding()
            assert BROKER.calls==step and len(BROKER.mapping)==step
            if step==32:ctx.state.cfg.checkpoint.save_interval=64
            checkpoint=OUT/'checkpoint'/f'iter_{step:07d}'
            assert checkpoint.is_dir() and any(checkpoint.rglob('*.distcp'))
            if RANK==0:durable(OUT/f'dynamic-checkpoint-step{step}.json',BROKER.snapshot())
            assert b['next_iteration']==step
            # Native callback is after synchronous save. Allrank acknowledgement
            # precedes the only atomic host-readable cursor commit.
            ranks=[None]*8;dist.all_gather_object(ranks,dict(rank=RANK,step=step,binding=b))
            assert all(r['step']==step and r['binding']==b for r in ranks)
            if RANK==0:
                publish(f'{STAGE}-checkpoint-step{step}-transaction',dict(binding=b,ranks=ranks,
                    native_synchronous_save_completed=True,save_optim=True,save_rng=True,
                    checkpoint_integrity=C['checkpoint_integrity']))
                if not resume:publish(f'checkpoint-step{step}-transaction',dict(binding=b,ranks=ranks,
                    native_synchronous_save_completed=True,save_optim=True,save_rng=True,
                    checkpoint_integrity=C['checkpoint_integrity']))
            saves.append(step);dist.barrier()
        def on_eval_end(self,ctx):
            held.append(int(ctx.state.train_state.step))
            publish(f'{STAGE}-rank{RANK}-heldout',dict(step=int(ctx.state.train_state.step),
                mode='native-disjoint-GPTDataset-LM-only',evaluations=len(held)))
    try:pretrain(cfg,adapter,callbacks=[Audit()])
    finally:adapter.close(require_complete=True)
    completed=BROKER.calls
    assert [r['update'] for r in updates]==list(range(start+1,completed+1))
    assert completed<=STOP
    if completed:assert completed in saves
    if RANK==0:durable(OUT/'final-dynamic-state.json',BROKER.snapshot())
    publish(f'{STAGE}-rank{RANK}-complete',dict(rank=RANK,completed=completed,
        optimizer_calls=len(updates),total_optimizer_calls=completed,
        positive_updates=sum(r['lr_used']>0 for r in updates),total_positive_updates=15+sum(r['lr_used']>0 for r in updates),
        target=STOP,shortfall=STOP-completed,source_exhausted=BROKER.current is None,
        skipped=sum(r['skipped'] for r in updates),nonfinite=0,save_steps=saves,heldout_steps=held,
        operational_resume=resume,restored_step=start,checkpoint_integrity=C['checkpoint_integrity']))


def main():
    global C,STAGE
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['student','resume']);a=p.parse_args()
    STAGE=a.stage;C=json.loads((OUT/'run-config.json').read_text())
    student(resume=STAGE=='resume')

if __name__=='__main__':main()
