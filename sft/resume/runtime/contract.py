"""Continuation invariants. No training loop or weight payload reads."""
import math
from datetime import datetime
BASE=905
CURSOR=115840
HORIZON=34135
TRAIN_STOP=datetime.fromisoformat('2026-10-19T14:50:00+09:00').timestamp()
DEADLINE=TRAIN_STOP+600
CHECKPOINT='/raid/yoon/nemochat/results/midtraining/sft-after-kd-runtime05/main/checkpoint/iter_0000905'
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def validate_restored(state,scheduler,optimizer,proof):
 require(int(state.step)==BASE and int(state.consumed_train_samples)==CURSOR,'native step905/cursor115840 restore required')
 require(scheduler is not None and scheduler.num_steps==CURSOR,'native scheduler115840 restore required')
 require(optimizer is not None and bool(optimizer.param_groups),'native optimizer missing')
 require(proof.get('loaded') is True and proof.get('step')==BASE and all(proof.get(k) is True for k in ('optimizer','scheduler','rng')),'successful actual full-state native load provenance missing')
 return BASE

def budget(now):
 seconds=14.5/.85
 extra=math.floor((TRAIN_STOP-now-300)/seconds)
 require(extra>=1,'insufficient clock for full-state continuation startup/save')
 return dict(execution_update_cap=min(HORIZON,BASE+extra),segment_update_cap=extra,conservative_iteration_seconds=seconds,startup_allowance_seconds=300,assumed_overhead_fraction=.15,proven_recent_iteration_seconds=14.5,training_stop_timestamp=TRAIN_STOP,save_cleanup_reserve_seconds=600,projection_not_guarantee=True)
def termination(code,native_exit,updates,saves,cap):
 final=BASE+updates
 if code==0 and updates>0 and final in saves:
  if final==cap:return 'native_execution_cap_saved'
  if native_exit:return 'native_deadline_saved'
 return 'unexpected_exit'

def stop_due(now,seconds):
 return now>=TRAIN_STOP-seconds
