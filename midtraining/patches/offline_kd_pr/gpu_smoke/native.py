"""Real teacher dump / native Bridge cached KD smoke. No private optimizer loop."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
from control import KEYS, TEACHER, STUDENT, contract, select_windows, exposure_order

ROOT=Path('/repo')
OUT=Path('/output')

def publish(name,value):
    target=OUT/(name+'.json')
    temp=target.with_suffix('.json.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temp.replace(target)
    print(name,json.dumps(value,allow_nan=False),flush=True)

def kd_helpers():
    spec=importlib.util.spec_from_file_location('accepted_native_kd',ROOT/'midtraining/training/native_kd.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def prepare():
    source=ROOT/'datasets/midtraining/super35-kd-01/pilot-10m-v1/manifest.json'
    manifest=json.loads(source.read_text())
    rows=[];a=[];b=[]
    for key in KEYS:
        x,y=[np.memmap(manifest['sources'][key][s]['prefix']+'.bin',dtype='<i4',mode='r') for s in ('train','validation')]
        train,valid=select_windows(x,y,source=key)
        a.extend(train);b.extend(valid)
        rows.append(dict(source=key,train_prefix=manifest['sources'][key]['train']['prefix'],
                         validation_prefix=manifest['sources'][key]['validation']['prefix'],
                         train_token_offsets=[0,512],heldout_token_offsets=[0],
                         producer_train_sha256=manifest['sources'][key]['train']['bin_sha256'],
                         producer_heldout_sha256=manifest['sources'][key]['validation']['bin_sha256']))
    train=np.stack(a);held=np.stack(b)
    assert not any(np.array_equal(x,y) for x in train for y in held)
    np.savez(OUT/'windows.npz',train=train,heldout=held)
    import hashlib
    digest=hashlib.sha256((OUT/'windows.npz').read_bytes()).hexdigest()
    publish('data',dict(sources=rows,windows_sha256=digest,shape_train=list(train.shape),
                       shape_heldout=list(held.shape),exposure_order=exposure_order(),
                       reserved_ids_18_through_29={str(i):0 for i in range(18,30)},
                       train_heldout_separate_producer_prefixes=True,
                       checkpoint_payload_hashing=False))
    return train,held

def runtime(state):
    from megatron.bridge.training.cached_kd.runtime import CoreRuntime
    return CoreRuntime(seed=43,seq_length=512,train_samples=32,global_batch_size=1,
                       micro_batch_size=1,blend=[['offline-kd-smoke-01-real-eight-windows'],[1.0]],
                       blend_per_split=None,iteration=lambda:int(state.train_state.step),num_microbatches=lambda:1)

def teacher_common(state,iterator,model,return_schedule_plan=False):
    import torch
    values=next(iterator)
    t=torch.as_tensor(values,dtype=torch.long,device='cuda').reshape(1,513)
    output=model(input_ids=t[:,:-1],position_ids=torch.arange(512,device='cuda').reshape(1,-1),
                 attention_mask=None,labels=t[:,1:])
    return output,torch.ones_like(output)

def teacher():
    import torch
    import torch.distributed as dist
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.training.cached_kd.integration import CachedKDConfig,CachedKDForwardStep
    from megatron.core.utils import unwrap_model
    start=time.perf_counter()
    train,held=prepare()
    kd=kd_helpers()
    publish('tokenizer',kd.tokenizer_gate(ROOT/STUDENT,ROOT/TEACHER))
    publish('phase',dict(phase='teacher_provider_loading',time=time.time(),contract=contract()))
    provider=AutoBridge.from_hf_pretrained(str(ROOT/TEACHER),trust_remote_code=False,
                   local_files_only=True).to_megatron_provider(load_weights=True)
    kd.configure_provider(provider,dict(parallel=contract()['parallel'],sequence_length=512))
    provider.hidden_dropout=0.;provider.attention_dropout=0.
    provider.finalize()
    assert provider.hidden_size==4096 and provider.num_layers==88
    publish('teacher-provider',dict(architecture=kd.architecture(provider),params_dtype=str(provider.params_dtype),
                                   ddp=False,optimizer=False,dropout=0.,time=time.time()))
    model=provider.provide_distributed_model(wrap_with_ddp=False,bf16=True)[0]
    model.requires_grad_(False);model.train()
    bare=unwrap_model(model)
    assert all(p.dtype==torch.bfloat16 for p in bare.parameters())
    frozen=kd.check_teacher(bare)
    loaded=time.perf_counter()
    publish('teacher-loaded',dict(**frozen,seconds=loaded-start,memory_allocated=torch.cuda.memory_allocated(),
                                 memory_reserved=torch.cuda.memory_reserved(),ddp=False,optimizer=False))
    state=SimpleNamespace(cfg=SimpleNamespace(model=provider),
                          train_state=SimpleNamespace(step=0,consumed_train_samples=0))
    with CachedKDForwardStep(CachedKDConfig(mode='write',cache_dir=str(OUT/'cache'),cache_id='offline-kd-smoke-01'),
                             runtime,common_forward=teacher_common) as writer:
        # All 32 explicit repeated exposures are represented in native iteration files.
        for i,window in enumerate(exposure_order()):
            state.train_state.step=i;state.train_state.consumed_train_samples=i
            loss=writer.cache_forward(state,iter([train[window]]),model)
            torch.cuda.synchronize()
            publish('phase',dict(phase='teacher_dump',iteration=i+1,sample=window,
                                 writer_zero_loss_placeholder=float(loss.float().mean()),peak_memory_bytes=torch.cuda.max_memory_allocated()))
        binding=writer.checkpoint_binding()
    kd.check_teacher(bare)
    tars=list((OUT/'cache').glob('*.tar'))
    publish('teacher-complete',dict(iterations=32,cache_files=len(tars),cache_bytes=sum(p.stat().st_size for p in tars),
                     binding=binding,load_seconds=loaded-start,dump_seconds=time.perf_counter()-loaded,
                     frozen_no_grad=True,optimizer_calls=0,ddp=False,source_data=json.loads((OUT/'data.json').read_text()),
                     teacher=TEACHER,contract=contract()))
    dist.destroy_process_group()

def student(reload=False):
    import math
    import torch
    from dataclasses import dataclass
    from megatron.bridge.data.base import DatasetProvider
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.config import (ConfigContainer, TrainingConfig, OptimizerConfig,
        SchedulerConfig, LoggerConfig, TokenizerConfig, CheckpointConfig,
        DistributedDataParallelConfig, ValidationConfig, RNGConfig, runtime_config_update)
    from megatron.bridge.training.gpt_step import make_cached_kd_forward_step
    from megatron.bridge.training.cached_kd.integration import CachedKDConfig
    from megatron.core.utils import unwrap_model
    started=time.perf_counter()
    from optimizer_compat import install_and_verify
    publish('reload-optimizer-compat' if reload else 'optimizer-compat',install_and_verify())
    data=np.load(OUT/'windows.npz');train=data['train'];held=data['heldout']
    teacher_receipt=json.loads((OUT/'teacher-complete.json').read_text())
    from control import verify_producer_binding
    verify_producer_binding(OUT/'windows.npz',json.loads((OUT/'data.json').read_text()),teacher_receipt)
    kd=kd_helpers()
    publish('phase',dict(phase='student_reload' if reload else 'student_native_setup',time=time.time()))
    provider=kd.provider_from_hf(ROOT/STUDENT,load_weights=not reload)
    kd.configure_provider(provider,dict(parallel=contract()['parallel'],sequence_length=512))
    provider.hidden_dropout=0.;provider.attention_dropout=0.
    kd.check_provider_shapes(provider)

    class Windows(torch.utils.data.Dataset):
        def __len__(self):return 32
        def __getitem__(self,i):return kd.make_sample(train[exposure_order()[i]],512,131072)
    @dataclass
    class Data(DatasetProvider):
        def build_datasets(self,context):
            if not reload:assert context.train_samples==32
            return (None if reload else Windows()),None,None

    cfg=ConfigContainer(model=provider,
        train=TrainingConfig(train_iters=32,micro_batch_size=1,global_batch_size=1,
                             exit_duration_in_mins=30),
        optimizer=OptimizerConfig(lr=1e-5,min_lr=1e-5,weight_decay=.01,bf16=True,
            params_dtype=torch.bfloat16,use_distributed_optimizer=False,overlap_param_gather=False,
            clip_grad=1.,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-8),
        ddp=DistributedDataParallelConfig(grad_reduce_in_fp32=True,overlap_grad_reduce=False,
            use_distributed_optimizer=False,overlap_param_gather=False,average_in_collective=False),
        scheduler=SchedulerConfig(lr_decay_style='constant',lr_decay_iters=32,lr_warmup_iters=0,
            start_weight_decay=.01,end_weight_decay=.01,weight_decay_incr_style='constant',
            use_checkpoint_opt_param_scheduler=reload),
        dataset=Data(dataloader_type='single',num_workers=0,persistent_workers=False),
        tokenizer=TokenizerConfig(tokenizer_type='NullTokenizer',vocab_size=131072,
            padded_vocab_size=131072,use_tokenizer_vocab_size=False,null_tokenizer_eod_id=11),
        logger=LoggerConfig(log_interval=1,wandb_project=None,wandb_entity=None,wandb_exp_name=None,
            wandb_save_dir=str(OUT/'sdk-logs'),save_config_filepath=str(OUT/('reload-final-config.yaml' if reload else 'student-final-config.yaml'))),
        checkpoint=CheckpointConfig(save=None if reload else str(OUT/'checkpoint'),
            load=str(OUT/'checkpoint') if reload else None,pretrained_checkpoint=None,save_interval=32,
            save_optim=True,save_rng=True,load_optim=reload,load_rng=reload,finetune=False,
            exit_on_missing_checkpoint=True,async_save=False,fully_parallel_save=False,
            ckpt_format='torch_dist',save_tokenizer_assets=False,verify_integrity=False,most_recent_k=1),
        validation=ValidationConfig(eval_iters=0,eval_interval=1000,skip_train=reload),
        rng=RNGConfig(seed=43),mixed_precision='bf16_mixed')
    runtime_config_update(cfg)
    cfg.to_yaml(str(OUT/('reload-config.yaml' if reload else 'student-config.yaml')))
    binding=json.loads((OUT/'checkpoint-cache-binding.json').read_text()) if reload else None
    adapter=make_cached_kd_forward_step(CachedKDConfig(mode='read',cache_dir=str(OUT/'cache'),
               cache_id='offline-kd-smoke-01'),runtime,resume_binding=binding)
    updates=[];saves=[];timer={};evaluations={}

    def fixed_eval(ctx,name):
        # Native model CE on identical inputs and all-one masks. LM-only,
        # deliberately NOT described as cached-KD validation.
        model=ctx.model[0];old=model.training;model.eval()
        rows=[]
        with torch.no_grad():
            for split,windows in (('train',train),('heldout',held)):
                for i,w in enumerate(windows):
                    loss,_=teacher_common(ctx.state,iter([w]),model)
                    value=float(loss.float().mean())
                    if not math.isfinite(value):raise FloatingPointError('nonfinite fixed-window LM')
                    rows.append(dict(split=split,window=i,lm_loss=value,loss_tokens=512))
        model.train(old)
        result=dict(mode='native-model-forward-no-grad-LM-only',rows=rows,
                    train_mean=float(np.mean([r['lm_loss'] for r in rows if r['split']=='train'])),
                    heldout_mean=float(np.mean([r['lm_loss'] for r in rows if r['split']=='heldout'])),
                    update=int(ctx.state.train_state.step),seconds=time.perf_counter()-started)
        publish(name,result);evaluations[name]=result

    class Audit(Callback):
        def on_data_init_start(self,ctx):
            if reload:
                assert int(ctx.state.train_state.step)==32
                assert int(ctx.state.train_state.consumed_train_samples)==32
                adapter.validate_resume(runtime(ctx.state))
                fixed_eval(ctx,'reload-fixed-lm')
                publish('reload-restored',dict(step=32,consumed_samples=32,cache_binding=binding,
                    optimizer_loaded=True,rng_loaded=True,lr_actual=float(ctx.optimizer.param_groups[0]['lr']),
                    checkpoint_integrity='not_checked_explicit_user_waiver'))
        def on_train_start(self,ctx):
            kd.check_provider_shapes(unwrap_model(ctx.model[0]).config)
            assert int(ctx.state.train_state.step)==0
            publish('student-loaded',dict(seconds=time.perf_counter()-started,
                hf_import=getattr(provider,'_nemochat_hf_import_receipt',{}),teacher_present=False))
            fixed_eval(ctx,'initial-fixed-lm')
        def on_train_step_start(self,ctx):
            torch.cuda.synchronize();timer['start']=time.perf_counter()
            timer['lr']=float(ctx.optimizer.param_groups[0]['lr'])
        def on_train_step_end(self,ctx):
            torch.cuda.synchronize()
            index=int(ctx.state.train_state.step)+1
            row=dict(update=index,window=exposure_order()[index-1],epoch=(index-1)//8+1,
                metrics={str(k):float(v) for k,v in ctx.loss_dict.items()},
                grad_norm=None if ctx.grad_norm is None else float(ctx.grad_norm),
                lr_used=timer['lr'],skipped=bool(ctx.skipped_iter),
                seconds=time.perf_counter()-timer['start'],peak_memory_bytes=torch.cuda.max_memory_allocated())
            # Persist observed values before rejecting them.
            with (OUT/'updates.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            updates.append(row)
            if row['skipped'] or row['grad_norm'] is None or row['grad_norm']<=0 or not all(math.isfinite(v) for v in list(row['metrics'].values())+[row['grad_norm'],row['lr_used']]):
                raise FloatingPointError('nonfinite/skipped/zero-gradient native update: '+repr(row))
            publish('phase',dict(phase='student_updating',update=index,lr_used=row['lr_used'],time=time.time()))
        def on_checkpoint_save(self,ctx):
            step=int(ctx.state.train_state.step);saves.append(step)
            if step==32:
                value=adapter.checkpoint_binding()
                assert value['next_iteration']==32
                publish('checkpoint-cache-binding',value)
                publish('checkpoint-save',dict(step=step,native_synchronous_save_completed=True,
                    save_optim=True,save_rng=True,checkpoint_integrity='not_checked_explicit_user_waiver'))
        def on_train_end(self,ctx):
            fixed_eval(ctx,'final-fixed-lm')
    try:
        pretrain(cfg,adapter,callbacks=[Audit()])
    finally:
        adapter.close(require_complete=True)
    if reload:
        assert 'reload-fixed-lm' in evaluations and not updates
        publish('reload-complete',dict(native_exit_code=0,updates=0,operational_reload=True,
                                       checkpoint_integrity='not_checked_explicit_user_waiver'))
    else:
        assert len(updates)==32 and saves==[32]
        positive=sum(r['lr_used']>0 for r in updates)
        epoch_means={str(e):{k:float(np.mean([r['metrics'][k] for r in updates if r['epoch']==e]))
                     for k in updates[0]['metrics']} for e in range(1,5)}
        publish('student-complete',dict(native_exit_code=0,optimizer_calls=len(updates),positive_lr_updates=positive,
            zero_lr_calls=32-positive,skipped_updates=sum(r['skipped'] for r in updates),save_steps=saves,
            epoch_means=epoch_means,seconds=time.perf_counter()-started,
            fixed_lm=evaluations,checkpoint_persisted=True,
            checkpoint_integrity='not_checked_explicit_user_waiver'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['teacher','student','reload'])
    args=parser.parse_args()
    try:
        if args.stage=='teacher':teacher()
        else:student(reload=args.stage=='reload')
    except BaseException:
        (OUT/(args.stage+'-error.log')).write_text(traceback.format_exc())
        raise

if __name__=='__main__':main()
