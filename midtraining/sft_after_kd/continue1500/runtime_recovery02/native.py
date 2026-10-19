#!/usr/bin/env python3
"""Native Bridge SFT905 full-state continuation. Parent RAM release required."""
import argparse, functools, importlib, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/raid/yoon/nemochat')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import controller as control
import contract
import resume_sft as v

def runtime_config(cfg,c):
 contract.require(c['updates']==contract.HORIZON==cfg.train.train_iters,'native one-epoch horizon must remain34135')
 contract.require(contract.BASE<c['execution_update_cap']<=contract.HORIZON,'cumulative native cap must exceed905')
 cfg.train.exit_interval=c['execution_update_cap']
 cfg.checkpoint.load=contract.CHECKPOINT
 cfg.checkpoint.pretrained_checkpoint=None
 cfg.checkpoint.finetune=False
 cfg.checkpoint.load_optim=True
 cfg.checkpoint.load_rng=True
 cfg.checkpoint.load_main_params_from_ckpt=False
 cfg.checkpoint.finalize()  # Full optimizer restore includes master parameters.
 cfg.checkpoint.exit_on_missing_checkpoint=True
 cfg.checkpoint.dist_ckpt_strictness='raise_all'
 cfg.checkpoint.save_interval=250
 cfg.scheduler.use_checkpoint_opt_param_scheduler=True
 cfg.scheduler.override_opt_param_scheduler=False
 cfg._checkpoint_load_required=True
 cfg._continuation_load_proof={}
 return cfg

def install_runtime(c):
 original=v.make_config
 entry=importlib.import_module('megatron.bridge.training.pretrain')
 original_pretrain=entry.pretrain
 @functools.wraps(original_pretrain)
 def pretrain(cfg,forward,callbacks=None,**kwargs):
  from sealed_input import guarded_forward
  import torch,torch.distributed as dist
  return original_pretrain(cfg,guarded_forward(forward,v._sealed_snapshot,dist,torch),callbacks=callbacks,**kwargs)
 def make(*args,**kwargs):
  cfg,parent=original(*args,**kwargs)
  runtime_config(cfg,c)
  from source_guard import attach
  attach(parent,entry,pretrain,original_pretrain,control)
  from load_observer import install
  install(cfg)
  return cfg,parent
 v.make_config=make
 entry.pretrain=pretrain

def main():
 import yaml
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--release',type=Path)
 p.add_argument('--config-only',action='store_true')
 a=p.parse_args();c=yaml.safe_load(a.config.read_text());m=control.read(a.manifest)
 c=v.stage_plan(c,m);c['load_hf_weights']=False
 contract.require(c['source_load_mode']=='native_full_state_continuation','model-only reset forbidden')
 a.output.mkdir(parents=True,exist_ok=True)
 if not a.config_only:
  from readiness import validate_native_release
  ready=validate_native_release(control.read(a.release),control,False)
  from sealed_input import Snapshot,rows_class
  v._sealed_snapshot=Snapshot(ready,m);v.PreparedRows=rows_class(v.PreparedRows,v._sealed_snapshot)
  control.write(a.output/f'rank-{os.environ.get("RANK","0")}-inputs.json',dict(input_mode='sealed_scm_rights',snapshot_id=ready['snapshot_id'],start_source_sample=115840,per_row_authentication=True))
  contract.require(time.time()<contract.TRAIN_STOP,'training stop exhausted before native entrypoint')
  c['max_hours']=(contract.TRAIN_STOP-time.time())/3600
 install_runtime(c)
 v.run(SimpleNamespace(manifest=a.manifest,output=a.output,online=False,config_only=a.config_only),c,m)
 if a.config_only:
  # Exercise actual installed post-wrapper source guard, including negative controls.
  from megatron.bridge.data.base import DatasetProvider
  @dataclass
  class ConfigData(DatasetProvider):
   def build_datasets(self,ctx):raise RuntimeError('config-only cannot build data')
  cfg,parent=v.make_config(c,m,ConfigData,a.output,False)
  control.write(a.output/'actual-post-wrapper-guard.json',parent.guard_native_sources())
  from guard_regression import prove
  prove(parent,control)
  control.write(a.output/'config-only.json',dict(status='NATIVE_FULLSTATE_CONFIG_CONSTRUCTED_NO_GPU_NO_WEIGHTS',load_hf_weights=False,checkpoint_load=cfg.checkpoint.load,load_optim=cfg.checkpoint.load_optim,load_rng=cfg.checkpoint.load_rng,load_main_params_from_ckpt=cfg.checkpoint.load_main_params_from_ckpt,finetune=cfg.checkpoint.finetune,checkpoint_load_required=cfg._checkpoint_load_required,native_horizon=cfg.train.train_iters,cumulative_exit_cap=cfg.train.exit_interval))

if __name__=='__main__':main()
