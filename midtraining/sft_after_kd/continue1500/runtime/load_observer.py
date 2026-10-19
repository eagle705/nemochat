"""Observe successful native full-state restore; no alternate loader/fallback."""
import functools, random
from contextlib import ExitStack
from unittest.mock import patch
import contract as c

def install(cfg):
 import numpy as np
 import torch
 from megatron.core import tensor_parallel
 from megatron.bridge.training.checkpointing import DefaultCheckpointManager
 original=DefaultCheckpointManager.load
 @functools.wraps(original)
 def load(manager,ctx):
  ck=ctx.state.cfg.checkpoint
  c.require(str(ck.load)==c.CHECKPOINT and ck.load_optim and ck.load_rng and not ck.finetune and ck.load_main_params_from_ckpt and ctx.strict and not ctx.skip_load_to_model_and_opt,'strict native full-state restore mandatory')
  seen={}
  tracker=tensor_parallel.get_cuda_rng_tracker()
  with ExitStack() as stack:
   def observe(obj,name,key):
    fn=getattr(obj,name)
    def call(*args,**kwargs):
     result=fn(*args,**kwargs)
     seen[key]=True
     return result
    stack.enter_context(patch.object(obj,name,call))
   observe(ctx.optimizer,'load_state_dict','optimizer')
   observe(ctx.opt_param_scheduler,'load_state_dict','scheduler')
   for obj,name,key in ((random,'setstate','python_rng'),(np.random,'set_state','numpy_rng'),(torch,'set_rng_state','torch_rng'),(torch.cuda,'set_rng_state','cuda_rng'),(tracker,'set_states','tracker_rng')):
    observe(obj,name,key)
   result=original(manager,ctx)
  proof=dict(seen,loaded=True,step=int(result[0]),rng=all(seen.get(k) for k in ('python_rng','numpy_rng','torch_rng','cuda_rng','tracker_rng')),native_loader='DefaultCheckpointManager.load -> load_checkpoint',strict=True,source=c.CHECKPOINT)
  c.validate_restored(ctx.state.train_state,ctx.opt_param_scheduler,ctx.optimizer,proof)
  cfg._continuation_load_proof=proof
  cfg._continuation_checkpoint_manager=manager
  return result
 DefaultCheckpointManager.load=load
 return load
