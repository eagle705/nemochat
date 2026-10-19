"""Inert plans and parent-reviewed durable start. Never releases production implicitly."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[3]
CODE=REPO/'midtraining/training/offline_kd'
PATCH=REPO/'midtraining/patches/offline_kd_pr'
IMAGE='nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
STUDENT='checkpoints/exports/nemochat_baby_base_hf_bf16_02'
REVISION='4ba52e8ffa79e9693936c1a819ac89f551ceeef6'
TEACHER=f'checkpoints/teachers/nemotron35-super-base/{REVISION}-text'
KEYS=('code_concepts','algorithmic','formal_logic','climbmix_replay')
WEIGHTS=[.67,.10,.03,.20]
PARALLEL=dict(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,
              context_parallel_size=1,expert_model_parallel_size=8,
              expert_tensor_parallel_size=1,sequence_parallel=False)

def plan(mode):
    if mode not in ('gate','production','production1920'):raise ValueError('gate or approved production profile required')
    n=4 if mode=='gate' else (3662 if mode=='production1920' else 3815)
    return dict(mode=mode,train_iters=n,train_samples=n*128,positions=n*128*4096,
                sequence_length=4096,global_batch_size=128,micro_batch_size=1,
                num_microbatches=16,world_size=8,parallel=PARALLEL,seed=43,compile_threads=1,
                warmup_iters=0 if mode=='gate' else (37 if mode=='production1920' else 38),
                decay_iters=1 if mode=='gate' else (366 if mode=='production1920' else 382),
                resume_step=2 if mode=='gate' else None,student_hf=STUDENT,teacher_hf=TEACHER,
                teacher_revision=REVISION,weights=WEIGHTS,cache_limit_bytes=600*1024**3,
                data_limit_bytes=64*1024**3,watchdog_seconds=7200 if mode=='gate' else 72*3600,
                checkpoint_integrity='not_checked_explicit_user_waiver')

def source_files():
    return sorted([CODE/'launch.py',CODE/'supervisor.py',CODE/'native.py',
        REPO/'midtraining/training/native_kd.py',REPO/'midtraining/training/longrun_native.py',
        REPO/'midtraining/patches/student_embedding_import.py',
        PATCH/'gpu_smoke/optimizer_compat.py',PATCH/'megatron-bridge-0.6.1-cached-kd.patch',
        *sorted((PATCH/'bridge_cached_kd').glob('*.py'))])

def freeze():
    return {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files()}

def verify_freeze(expected):
    if expected!=freeze():raise ValueError('reviewed exact source freeze mismatch')

def validate_gate_review(review, gate, mode):
    """Reuse executed native gate only across an explicitly reviewed budget delta."""
    if gate.get('status') != 'gate_accepted':
        raise ValueError('accepted real GPU gate required')
    original = gate.get('source_sha256', {})
    current = review['source_sha256']
    if current == original:
        return
    expected = dict(
        status='parent_accepted_budget_only_adjustment', gate_source_sha256=original,
        old_production_calls=3815, new_production_calls=3662, original_base_reset=True,
        same_teacher_tokenizer_precision_sequence_batch_topology_objective=True,
        user_approved_unique_budget=True, gate_execution_reused_not_reexecuted=True)
    if mode != 'production1920' or review.get('budget_compatibility') != expected:
        raise ValueError('explicit approved budget-only compatibility required')
    if set(current) != set(original):
        raise ValueError('exact native gate source allowlist required')
    allowed = {'midtraining/training/offline_kd/launch.py', 'midtraining/training/offline_kd/supervisor.py'}
    changed = {name for name in original if original[name] != current[name]}
    if not changed <= allowed:
        raise ValueError('native model/loss/adapter source changed since GPU gate')
    reference = plan('gate')
    if any(gate.get('plan', {}).get(key) != value for key, value in reference.items()):
        raise ValueError('executed gate model/precision/batch/topology contract changed')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['gate','production','production1920'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start',action='store_true')
    p.add_argument('--review',type=Path,help='Parent-authored JSON: source_sha256 and gate_output')
    a=p.parse_args(); output=a.output.resolve()
    if output.parent!=REPO/'results/midtraining' or not output.name.startswith('offline2b-'):
        raise ValueError('fresh offline2b results path required')
    argv=[str(REPO/'.venv/bin/python'),'-u',str(CODE/'supervisor.py'),'--mode',a.mode,'--output',str(output)]
    if a.mode!='gate':
        if a.review is None:raise ValueError('parent production review receipt required')
        review=json.loads(a.review.read_text());verify_freeze(review['source_sha256'])
        gate=json.loads((Path(review['gate_output'])/'report.json').read_text())
        validate_gate_review(review,gate,a.mode)
        argv+=['--review',str(a.review.resolve())]
    unit='nemochat-'+output.name
    cmd=['systemd-run','--user','--unit='+unit,'--property=Type=exec',
         '--property=WorkingDirectory='+str(REPO),'--property=RuntimeMaxSec='+str(plan(a.mode)['watchdog_seconds']+600),
         '--property=StandardOutput=append:'+str(output/'supervisor.log'),
         '--property=StandardError=append:'+str(output/'supervisor.log'),
         '--property=ExecStopPost='+str(REPO/'.venv/bin/python')+' '+str(CODE/'supervisor.py')+' --cleanup --output '+str(output),
         '--setenv=OFFLINE2B_UNIT='+unit,*argv]
    print(json.dumps(dict(plan=plan(a.mode),argv=cmd,source_sha256=freeze()),indent=2))
    if a.start:
        output.mkdir(parents=True,exist_ok=False)
        (output/'reviewed-source.json').write_text(json.dumps(freeze(),indent=2)+'\n')
        subprocess.run(cmd,check=True)
        print(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','-p','ActiveState','-p','ExecStart'],text=True))

if __name__=='__main__':main()
