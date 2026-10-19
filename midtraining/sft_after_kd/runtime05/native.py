#!/usr/bin/env python3
"""Small runtime adapter over frozen native_sft_v1; never an optimizer loop.

Full 34135-call native sampler, LR horizon and validation allocation are kept
for calibration and main. Native exit_interval limits actual optimizer calls.
Calibration weights/state are operationally reloaded for validation only; main
always reimports original C and resets optimizer/scheduler/RNG/cursor.
"""
import argparse
from dataclasses import dataclass
import functools
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path('/raid/yoon/nemochat')
HERE=ROOT/'midtraining/sft_after_kd/runtime05'
sys.path.insert(0,str(HERE))
import controller as control


def base():
    p=ROOT/'midtraining/sft_after_kd/native_sft_v1.py'
    spec=importlib.util.spec_from_file_location('runtime05_frozen_native',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def runtime_config(cfg,c):
    control.require(c['updates']==control.CAPACITY==cfg.train.train_iters,'full one-epoch native allocation required')
    control.require(1<=c['execution_update_cap']<=control.CAPACITY,'invalid native exit cap')
    cfg.train.exit_interval=c['execution_update_cap']
    cfg.checkpoint.save_interval=16
    return cfg


def install_runtime(v,c):
    """Only callback/config controls; stock native data/model/optimizer unchanged."""
    original=v.make_config
    def make(*args,**kwargs):
        cfg,parent=original(*args,**kwargs)
        runtime_config(cfg,c)
        from import_guard import attach
        attach(cfg.model)
        from source_guard import attach as attach_source_guard
        attach_source_guard(parent,entry,pretrain,original_pretrain,control)
        return cfg,parent
    v.make_config=make
    next_lr=v.next_update_lr
    def lr(update,horizon,remaining,seconds,scheduled,minimum):
        return next_lr(update,c['execution_update_cap'],remaining,seconds,scheduled,minimum)
    v.next_update_lr=lr
    # v1 already registers its numerical audit. Add only source and retention
    # guards through the public native Callback interface, not a custom loop.
    import megatron.bridge.training.pretrain as entry
    # importlib avoids package re-export ambiguity in different Bridge versions.
    entry=__import__('importlib').import_module('megatron.bridge.training.pretrain')
    from megatron.bridge.training.callbacks import Callback
    original_pretrain=entry.pretrain
    class RuntimeAudit(Callback):
        def on_train_start(self,ctx):
            receipt=getattr(ctx.state.cfg.model,'_nemochat_hf_import_receipt',{})
            control.require(receipt.get('source_key_coverage')=='all_index_keys_have_native_mappings',
                            'all C HF source tensor keys must have native mappings')
            control.require(receipt.get('native_embedding_transform_calls',0)>0,'learned C embedding was not imported')
            control.require(receipt.get('native_embedding_shape')==[131072,1024],'C embedding shape mismatch')
            coverage=getattr(ctx.state.cfg.model,'_runtime05_import_guard',{})
            control.require(coverage.get('destination_mapping_complete') is True and
                coverage.get('native_transform_outputs',0)>=coverage.get('destination_required_tensors',1),
                'all local native destinations require actual native transforms; random omissions forbidden')
            control.write(Path(ctx.state.cfg.checkpoint.save).parent/f'rank-{os.environ.get("RANK","0")}-destination-import.json',coverage)
        def on_train_step_start(self,ctx):
            cfg=ctx.state.cfg
            update=int(ctx.state.train_state.step)+1
            cfg.checkpoint.save_interval=16 if update<=16 else 250
            # Explicit absolute wall stop; do not begin an extra step after stop.
            # v1's terminal-step callback requests the native duration stop one
            # conservative step ahead. Large overruns are caught by outer Engine.
            if time.time()>=control.TRAIN_STOP:
                raise TimeoutError('native loop crossed absolute 11:30 training stop')
    @functools.wraps(original_pretrain)
    def pretrain(cfg,forward,callbacks=None,**kwargs):
        from sealed_input import guarded_forward
        import torch, torch.distributed as dist
        forward=guarded_forward(forward,v._sealed_snapshot,dist,torch)
        return original_pretrain(cfg,forward,callbacks=list(callbacks or [])+[RuntimeAudit()],**kwargs)
    entry.pretrain=pretrain


def reload_native(args,c,m,v):
    """Native operational load of model+optimizer+scheduler+RNG; eval, no update."""
    import torch
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.data.base import DatasetProvider
    from megatron.bridge.training.pretrain import pretrain
    rank=int(os.environ.get('RANK','0'))
    c=dict(c,load_hf_weights=False)
    @dataclass
    class Data(DatasetProvider):
        def build_datasets(self,ctx):
            return (None,v.PreparedRows(args.manifest.parent,'heldout',ctx.valid_samples,8,
                    window=128,sequence_length=16384),None)
    cfg,parent=v.make_config(c,m,Data,args.output,False)
    runtime_config(cfg,c)
    cfg.validation.skip_train=True
    cfg.checkpoint.load=str(args.reload_checkpoint)
    cfg.checkpoint.save=None
    cfg.checkpoint.load_optim=True
    cfg.checkpoint.load_rng=True
    cfg.checkpoint.finetune=False
    cfg.checkpoint.load_main_params_from_ckpt=True
    cfg._checkpoint_load_required=True
    cfg.train.exit_duration_in_mins=None
    if rank==0:
        control.write(args.output/'requested-config.json',parent.config_snapshot(cfg.to_dict()))
    def forward(iterator,model):
        item=next(iterator);device=torch.cuda.current_device()
        out=model(input_ids=item['tokens'].to(device),labels=item['labels'].to(device),
                  position_ids=item['position_ids'].to(device),attention_mask=None)
        return out,functools.partial(v.masked_sum,mask=item['loss_mask'].to(device))
    class Reload(Callback):
        def on_train_start(self,ctx):
            raise RuntimeError('reload-only stage must not enter native training')
        def on_eval_start(self,ctx):
            control.require(int(ctx.state.train_state.step)==16,'reload checkpoint step is not 16')
            control.require(ctx.scheduler.num_steps==16*128,'native scheduler state not restored')
            control.require(int(ctx.state.train_state.consumed_train_samples)==16*128,'native sample cursor not restored')
            control.require(ctx.optimizer is not None,'native optimizer restore missing')
        def on_eval_end(self,ctx):
            loss=float(ctx.total_loss_dict['lm loss'])
            control.require(math.isfinite(loss),'reload validation nonfinite')
            control.write(args.output/f'rank-{rank}-reload.json',dict(status='native_full_state_reload_eval_complete',
                rank=rank,step=int(ctx.state.train_state.step),scheduler_num_steps=ctx.scheduler.num_steps,
                optimizer_rng_native_load_required=True,optimizer_updates=0,heldout_ce=loss,
                checkpoint_integrity='not_checked_explicit_user_waiver'))
    from sealed_input import guarded_forward
    import torch.distributed as dist
    pretrain(cfg,guarded_forward(forward,v._sealed_snapshot,dist,torch),callbacks=[Reload()])


def verify_runtime_metadata(manifest,migration):
    control.require(migration.get('status')=='passed' and migration.get('vocab_size')==131072
        and migration.get('allowed_changed_ids')==list(range(18,30))
        and manifest.get('tokenizer_revision')==migration.get('new_revision')
        and manifest.get('tokenizer_path')==migration.get('tokenizer_path'),
        'exact official migrated tokenizer lineage required')
    return dict(tokenizer_revision=migration['new_revision'],checkpoint_bytes_read=False)


def main():
    import yaml
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--release',type=Path)
    p.add_argument('--config-only',action='store_true')
    p.add_argument('--reload-checkpoint',type=Path)
    args=p.parse_args()
    c=yaml.safe_load(args.config.read_text());m=control.read(args.manifest)
    v=base();plan=v.stage_plan(c,m)
    plan['load_hf_weights']=not args.config_only and args.reload_checkpoint is None
    control.require(plan['updates']==control.CAPACITY,'full native sampler allocation required')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.config_only:
        # Register the real deferred HF import hook; config-only returns before
        # native pretrain/model construction so no weight payload is opened.
        plan['load_hf_weights']=True
        install_runtime(v,plan)
        v.run(SimpleNamespace(manifest=args.manifest,output=args.output,online=False,config_only=True),plan,m)
        from megatron.bridge.data.base import DatasetProvider
        @dataclass
        class ConfigData(DatasetProvider):
            def build_datasets(self,ctx):
                raise RuntimeError('config-only must not build datasets')
        cfg,parent=v.make_config(plan,m,ConfigData,args.output,False)
        parent.guard_native_sources()
        from guard_regression import prove
        prove(parent,control)
        runtime_config(cfg,plan)
        control.write(args.output/'requested-config.json',parent.config_snapshot(cfg.to_dict()))
        # Read-only installed-source evidence; no package edits or GPU calls.
        import shutil
        from megatron.bridge.models.conversion import auto_bridge
        sdk=args.output/'sdk-conversion-source';sdk.mkdir(exist_ok=True)
        for source in Path(auto_bridge.__file__).parent.glob('*.py'):
            shutil.copyfile(source,sdk/source.name)
        control.write(args.output/'config-only.json',dict(status='NATIVE_CONFIG_CONSTRUCTED_NO_WEIGHTS_NO_GPU',
            native_updates=0,weights_loaded=False,native_full_epoch_updates=cfg.train.train_iters,
            execution_update_cap=cfg.train.exit_interval,eval_iters=cfg.validation.eval_iters,
            producer_source_sha256={str(p.relative_to(ROOT)):control.sha(p) for p in HERE.glob('*.py')}))
        return
    from readiness import validate_native_release
    release=control.read(args.release or control.OUT/'native-release.json')
    ready=validate_native_release(release,control,host=False)
    control.require(args.manifest==control.MANIFEST,'exact manifest required')
    verification=verify_runtime_metadata(m,control.read(c['tokenizer_manifest']))
    from sealed_input import Snapshot, rows_class
    snapshot=Snapshot(ready,m)
    v._sealed_snapshot=snapshot
    v.PreparedRows=rows_class(v.PreparedRows,snapshot)
    verification.update(input_mode='sealed_scm_rights',snapshot_id=ready['snapshot_id'],
        ready_sha256=release['ready_sha256'],per_row_authentication=True)
    control.write(args.output/f'rank-{os.environ.get("RANK","0")}-inputs.json',verification)
    if args.reload_checkpoint:
        control.require(args.reload_checkpoint==control.OUT/'calibration/checkpoint','reload scope must be calibration only')
        reload_native(args,plan,m,v)
        return
    control.require(time.time()<control.TRAIN_STOP,'training stop deadline exhausted')
    plan['max_hours']=min(plan['max_hours'],(control.TRAIN_STOP-time.time())/3600)
    install_runtime(v,plan)
    v.run(SimpleNamespace(manifest=args.manifest,output=args.output,online=False,config_only=False),plan,m)

if __name__=='__main__':
    main()
