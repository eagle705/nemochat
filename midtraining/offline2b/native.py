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

from launch import REPO,STUDENT,TEACHER,KEYS,PARALLEL
OUT=Path('/output')
C=None
RANK=int(os.environ.get('RANK','0'))
STAGE=None

def publish(name,value):
    target=OUT/(name+'.json');temp=target.with_suffix('.json.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(target)
    print(name,json.dumps(value,allow_nan=False),flush=True)

def kd_helpers():
    spec=importlib.util.spec_from_file_location('offline2b_accepted_kd',REPO/'midtraining/native_kd.py')
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
        expected={r['micro']:r for r in map(json.loads,(OUT/f'teacher-rank{RANK}-samples.jsonl').read_text().splitlines())}
    def common(state,iterator,model,return_schedule_plan=False):
        batch=next(iterator)
        if model.training:
            i=int(state.train_state.step)*16+seen['micro']%16
            h=hashlib.sha256()
            for key in ('tokens','labels','loss_mask','position_ids'):
                h.update(batch[key].cpu().contiguous().numpy().tobytes())
            row=dict(micro=i,iteration=int(state.train_state.step),rank=RANK,sha256=h.hexdigest(),
                     masked_tokens=int(batch['loss_mask'].sum()))
            if not teacher and row!=expected[i]:raise ValueError('actual teacher/student sample bytes differ')
            with (OUT/f'{STAGE}-rank{RANK}-samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            seen['micro']+=1;seen['masked_tokens']+=row['masked_tokens']
        if teacher:
            gpu={k:(v.cuda(non_blocking=True) if v is not None else None) for k,v in batch.items()}
            output=model(input_ids=gpu['tokens'],position_ids=gpu['position_ids'],attention_mask=None,labels=gpu['labels'])
            return output,gpu['loss_mask']
        return _forward_step_common(state,iter([batch]),model,return_schedule_plan)
    return common,seen

def teacher():
    import torch
    import torch.distributed as dist
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.training.cached_kd.integration import CachedKDConfig,CachedKDForwardStep
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from megatron.bridge.data.utils import pretrain_train_valid_test_datasets_provider
    from megatron.bridge.data.samplers import build_pretraining_data_loader
    from megatron.core import parallel_state
    kd=kd_helpers()
    if RANK==0:publish('tokenizer',kd.tokenizer_gate(REPO/STUDENT,REPO/TEACHER))
    # Plain AutoBridge for teacher: never apply student embedding-name alias.
    provider=AutoBridge.from_hf_pretrained(str(REPO/TEACHER),trust_remote_code=False,
        local_files_only=True).to_megatron_provider(load_weights=True)
    kd.configure_provider(provider,dict(parallel=PARALLEL,sequence_length=4096))
    provider.hidden_dropout=0.;provider.attention_dropout=0.;provider.finalize()
    model=provider.provide_distributed_model(wrap_with_ddp=False,bf16=True)[0]
    model.requires_grad_(False);model.train()
    assert parallel_state.get_data_parallel_world_size()==8
    assert parallel_state.get_expert_model_parallel_world_size()==8
    frozen=kd.check_teacher(model)
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
    publish(f'teacher-rank{RANK}-loaded',dict(frozen=frozen,ddp=False,optimizer=False,
        parameter_tensors=len(list(model.parameters())),dropout=0,params_dtype='torch.bfloat16'))
    ds=dataset_config();ds.tokenizer=build_tokenizer(tokenizer_config());ds.finalize()
    train,valid,_=pretrain_train_valid_test_datasets_provider(dataset_budget(),ds)
    # Reject extra epochs from actual native indices, not capacity metadata alone.
    spec=importlib.util.spec_from_file_location('offline2b_longrun',REPO/'midtraining/longrun_native.py')
    longrun=importlib.util.module_from_spec(spec);spec.loader.exec_module(longrun)
    no_repeat=longrun.no_repeat_gate(train)
    publish(f'teacher-rank{RANK}-dataset',dict(budget=dataset_budget(),no_repeat=no_repeat,
        native_dataset_type=type(train).__name__,native_sampler='MegatronPretrainingSampler'))
    loader=build_pretraining_data_loader(train,consumed_samples=0,dataloader_type='single',
        micro_batch_size=1,num_workers=0,data_sharding=True,persistent_workers=False,
        data_parallel_rank=parallel_state.get_data_parallel_rank(),data_parallel_size=8)
    iterator=iter(loader);state=SimpleNamespace(cfg=SimpleNamespace(model=provider),
        train_state=SimpleNamespace(step=0,consumed_train_samples=0))
    common,seen=audited_common(teacher=True)
    with CachedKDForwardStep(CachedKDConfig('write',str(OUT/'cache'),cache_id()),runtime,common_forward=common) as adapter:
        for i in range(C['train_iters']):
            state.train_state.step=i;state.train_state.consumed_train_samples=i*128
            for _ in range(16):adapter.cache_forward(state,iterator,model)
            torch.cuda.synchronize()
            if RANK==0:publish('teacher-progress',dict(iterations=i+1,total=C['train_iters']))
        binding=adapter.checkpoint_binding()
    kd.check_teacher(model);dist.barrier()
    if RANK==0:
        shards=sorted((OUT/'cache').glob('*.tar'))
        if len(shards)!=8*C['train_iters']:raise ValueError('exact allrank ready cache shards required')
        inventory=[dict(name=p.name,bytes=p.stat().st_size) for p in shards]
        if sum(r['bytes'] for r in inventory)>C['cache_limit_bytes']:raise ValueError('cache cap exceeded')
        with (OUT/'cache-manifest.json').open('x') as stream:
            json.dump(dict(cache_id=cache_id(),config=C,shards=inventory,binding=binding,
                sampler_budget=dataset_budget(),immutable_ready=True),stream,indent=2)
        for p in shards:p.chmod(0o444)
    dist.barrier()
    publish(f'teacher-rank{RANK}-complete',dict(rank=RANK,completed=C['train_iters'],optimizer_calls=0,
        frozen=True,binding=binding,local_masked_tokens=seen['masked_tokens']))
    dist.destroy_process_group()

def student(resume=False):
    import torch
    import torch.distributed as dist
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.config import (ConfigContainer,TrainingConfig,OptimizerConfig,SchedulerConfig,
        LoggerConfig,CheckpointConfig,DistributedDataParallelConfig,ValidationConfig,RNGConfig,runtime_config_update)
    from megatron.bridge.training.cached_kd.integration import CachedKDConfig,CachedKDForwardStep
    from megatron.core.utils import unwrap_model
    manifest=json.loads((OUT/'cache-manifest.json').read_text())
    if manifest['cache_id']!=cache_id() or not manifest['immutable_ready'] or manifest['sampler_budget']!=dataset_budget():
        raise ValueError('ready immutable teacher cache does not match full sampler contract')
    for r in manifest['shards']:
        if (OUT/'cache'/r['name']).stat().st_size!=r['bytes']:raise ValueError('ready shard missing/changed length')
    start=(C['resume_step'] or C['train_iters']-1) if resume else 0
    binding=json.loads((OUT/f'checkpoint-step{start}-transaction.json').read_text())['binding'] if resume else None
    kd=kd_helpers();provider=kd.provider_from_hf(REPO/STUDENT,load_weights=not resume)
    kd.configure_provider(provider,dict(parallel=PARALLEL,sequence_length=4096))
    provider.hidden_dropout=0.;provider.attention_dropout=0.;kd.check_provider_shapes(provider)
    interval=2 if C['mode']=='gate' else 250
    cfg=ConfigContainer(model=provider,
        train=TrainingConfig(train_iters=C['train_iters'],micro_batch_size=1,global_batch_size=128,
                             exit_duration_in_mins=None),
        optimizer=OptimizerConfig(lr=1e-5,min_lr=1e-6,weight_decay=.01,bf16=True,params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,overlap_param_gather=False,clip_grad=1.,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-8),
        ddp=DistributedDataParallelConfig(grad_reduce_in_fp32=True,overlap_grad_reduce=False,
            use_distributed_optimizer=False,overlap_param_gather=False,average_in_collective=False),
        scheduler=SchedulerConfig(lr_decay_style='WSD',lr_decay_iters=C['train_iters'],
            lr_warmup_iters=C['warmup_iters'],lr_wsd_decay_iters=C['decay_iters'],
            start_weight_decay=.01,end_weight_decay=.01,weight_decay_incr_style='constant',
            use_checkpoint_opt_param_scheduler=resume),
        dataset=dataset_config(),tokenizer=tokenizer_config(),
        logger=LoggerConfig(log_interval=1,wandb_project=None,wandb_entity=None,wandb_exp_name=None,
            wandb_save_dir=str(OUT/'sdk-logs'),save_config_filepath=str(OUT/f'{STAGE}-final-config.yaml')),
        checkpoint=CheckpointConfig(save=str(OUT/('resume-checkpoint' if resume else 'checkpoint')),
            load=str(OUT/'resume-load') if resume else None,pretrained_checkpoint=None,save_interval=interval,
            save_optim=True,save_rng=True,load_optim=resume,load_rng=resume,finetune=False,
            exit_on_missing_checkpoint=True,async_save=False,fully_parallel_save=False,ckpt_format='torch_dist',
            save_tokenizer_assets=False,verify_integrity=False,most_recent_k=3),
        validation=ValidationConfig(eval_iters=1,eval_interval=interval,eval_global_batch_size=8),
        rng=RNGConfig(seed=C['seed']),mixed_precision='bf16_mixed')
    runtime_config_update(cfg)
    if RANK==0:cfg.to_yaml(str(OUT/f'{STAGE}-config.yaml'))
    common,seen=audited_common()
    adapter=CachedKDForwardStep(CachedKDConfig('read',str(OUT/'cache'),cache_id()),runtime,
                                resume_binding=binding,common_forward=common)
    updates=[];saves=[];timer={};held=[]
    class Audit(Callback):
        def on_data_init_start(self,ctx):
            assert int(ctx.state.train_state.step)==start
            assert int(ctx.state.train_state.consumed_train_samples)==start*128
            if resume:
                adapter.validate_resume(runtime(ctx.state))
                publish(f'resume-rank{RANK}-restored',dict(step=start,consumed_samples=start*128,
                    optimizer_loaded=True,rng_loaded=True,lr=float(ctx.optimizer.param_groups[0]['lr']),binding=binding))
        def on_train_start(self,ctx):
            kd.check_provider_shapes(unwrap_model(ctx.model[0]).config)
            # Inspect actual native optimizer groups before deciding on the TE handle-pool cap.
            counts=[len(g['params']) for g in ctx.optimizer.param_groups]
            if max(counts,default=0)*4>20000:
                from optimizer_compat import install_and_verify
                compatibility=install_and_verify()
            else:compatibility=dict(installed=False,reason='EP8 native groups within TE handle pool')
            publish(f'{STAGE}-rank{RANK}-loaded',dict(optimizer_group_tensors=counts,compatibility=compatibility,
                hf_import=getattr(provider,'_nemochat_hf_import_receipt',{}),teacher_present=False))
        def on_train_step_start(self,ctx):
            torch.cuda.synchronize();timer.update(start=time.perf_counter(),lr=float(ctx.optimizer.param_groups[0]['lr']))
        def on_train_step_end(self,ctx):
            torch.cuda.synchronize();step=int(ctx.state.train_state.step)+1
            local=seen['masked_tokens'];seen['masked_tokens']=0
            counts=torch.tensor([local],device='cuda',dtype=torch.int64);dist.all_reduce(counts)
            row=dict(rank=RANK,update=step,metrics={str(k):float(v) for k,v in ctx.loss_dict.items()},
                grad_norm=None if ctx.grad_norm is None else float(ctx.grad_norm),lr_used=timer['lr'],
                skipped=bool(ctx.skipped_iter),seconds=time.perf_counter()-timer['start'],
                local_masked_tokens=local,allrank_masked_tokens=int(counts.item()),
                global_batch_size=128,sequence_length=4096,peak_memory_bytes=torch.cuda.max_memory_allocated())
            with (OUT/f'{STAGE}-rank{RANK}-updates.jsonl').open('a') as s:s.write(json.dumps(row)+'\n')
            updates.append(row)
            if row['skipped'] or row['grad_norm'] is None or row['grad_norm']<=0 or row['lr_used']<0 or not all(
                math.isfinite(v) for v in [*row['metrics'].values(),row['grad_norm'],row['lr_used']]):
                raise FloatingPointError('actual nonfinite/zero-gradient/skipped update')
            assert row['allrank_masked_tokens']==128*4096
            if RANK==0:publish('student-progress',dict(stage=STAGE,update=step,total=C['train_iters']))
        def on_checkpoint_save(self,ctx):
            step=int(ctx.state.train_state.step);b=adapter.checkpoint_binding()
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
    assert [r['update'] for r in updates]==list(range(start+1,C['train_iters']+1))
    assert C['train_iters'] in saves and C['train_iters'] in held
    publish(f'{STAGE}-rank{RANK}-complete',dict(rank=RANK,completed=C['train_iters'],
        optimizer_calls=len(updates),positive_updates=sum(r['lr_used']>0 for r in updates),
        skipped=sum(r['skipped'] for r in updates),nonfinite=0,save_steps=saves,heldout_steps=held,
        operational_resume=resume,restored_step=start,checkpoint_integrity=C['checkpoint_integrity']))


def main():
    global C,STAGE
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['teacher','student','resume']);a=p.parse_args()
    STAGE=a.stage;C=json.loads((OUT/'run-config.json').read_text())
    if STAGE=='teacher':teacher()
    else:student(resume=STAGE=='resume')

if __name__=='__main__':main()
