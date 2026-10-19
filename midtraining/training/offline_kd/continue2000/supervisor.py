"""Parent-owned durable bounded teacher -> accepted window -> FULL learner.

No retry, old controller restart, acquisition, publication or tracking side effect.
Engine wait is a single blocking exact-container wait (not artifact polling).
"""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
REPO=Path(__file__).resolve().parents[4];sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/'midtraining/training/offline_kd'))
from launch import IMAGE
from midtraining.training.offline_kd.continue2000.contract import OUT,plan,sha,clean_engine
from midtraining.training.offline_kd.continue2000.broker import durable
from midtraining.training.offline_kd.continue2000.start import UNIT,verify_review
from midtraining.training.offline_kd.continue2000.merge import accepted_policy,merge,validate_ready
HERE=Path(__file__).resolve().parent
LABEL='nemochat.offline2000.owner'

def engine():
    spec=importlib.util.spec_from_file_location('owned_engine',REPO/'results/pretrain_followup/engine_helper.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def owned(e,cid,stage):
    a=e.request('GET',f'/containers/{cid}/json')
    if a['Id']!=cid or a['Config']['Labels'].get(LABEL)!=stage:raise ValueError('exact owned container identity mismatch')
    return a

def engine_logs(e,cid,stage):
    c=e.Connection('localhost',timeout=120);c.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1')
    raw=c.getresponse().read();c.close();parts=[]
    while len(raw)>=8:
        size=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+size]);raw=raw[8+size:]
    (OUT/f'{stage}-engine.log').write_bytes(b''.join(parts))

def cleanup():
    e=engine()
    for stage in ('teacher','student'):
        receipt=OUT/f'{stage}-engine-created.json'
        if not receipt.exists():continue
        cid=json.loads(receipt.read_text())['id'];a=owned(e,cid,stage)
        if a['State']['Running']:e.request('POST',f'/containers/{cid}/stop?t=20')
        durable(OUT/f'{stage}-cleanup-readback.json',dict(id=cid,state=owned(e,cid,stage)['State']))

def ownership(review_path,review_sha):
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True))
    expected=[str(REPO/'.venv/bin/python'),'-u',str(HERE/'supervisor.py'),'--review',str(review_path),'--review-sha256',review_sha]
    actual=Path(f'/proc/{os.getpid()}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    if pid!=os.getpid() or Path(f'/proc/{pid}').stat().st_uid!=os.getuid() or actual!=expected:
        raise ValueError('supervisor MainPID/UID/full argv guard failed')
    return dict(unit=UNIT,MainPID=pid,uid=os.getuid(),argv=actual,
        starttime=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])

def idle_lock(stage):
    lock=(REPO/'results/pretraining/nemochat_baby/training/gpu-all8.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        rows=[list(map(int,line.replace(' ','').split(','))) for line in text.splitlines()]
        if (sorted(row[0] for row in rows)!=list(range(8)) or apps.strip()
            or any(memory>1024 or utilization>0 for _,memory,utilization in rows)):
            raise ValueError('all8 not actually idle; no foreign job termination allowed')
        durable(OUT/f'{stage}-idle-preflight.json',dict(gpus=rows,apps=apps,lock=lock.name))
        return lock
    except BaseException:lock.close();raise

def run_stage(stage,review_path,review_sha,deadline):
    verify_review(review_path,review_sha)
    if stage=='student':validate_ready(json.loads(review_path.read_text()))
    limit=min(plan()['stage_seconds'],int(deadline-time.monotonic())-90)
    if limit<=0:raise TimeoutError('controller hardcap exhausted')
    lock=idle_lock(stage)
    try:
        snapshot=Path(json.loads((OUT/'preparation/prepared.json').read_text())['snapshot'])
        scratch=Path(tempfile.mkdtemp(prefix=f'offline2000-{stage}-',dir='/home/yoon/.hermes/cache/scratch'))
        for name in ('home','hfhome','inductor','triton'):(scratch/name).mkdir()
        durable(OUT/f'{stage}-scratch.json',dict(path=str(scratch),snapshot=str(snapshot),compiler_caches_initially_empty=True))
        native=str(HERE/('teacher.py' if stage=='teacher' else 'native.py'))
        cmd=['-m','torch.distributed.run','--nproc_per_node=8','--nnodes=1','--node_rank=0',
            '--master_addr=127.0.0.1','--master_port='+('29724' if stage=='teacher' else '29725'),native]
        if stage=='student':cmd+=['resume']
        env=[f'PYTHONPATH={snapshot}/staged/src:{snapshot}/sdk:/repo/midtraining/patches/offline_kd_pr/gpu_smoke:/repo/src:/repo',
            'PYTHONDONTWRITEBYTECODE=1',f'TMPDIR={scratch}',f'HOME={scratch}/home',f'HF_HOME={scratch}/hfhome',
            'USER=yoon','LOGNAME=yoon','WANDB_MODE=disabled','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
            'TOKENIZERS_PARALLELISM=false','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
            'NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
            'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','TORCHINDUCTOR_COMPILE_THREADS=1',
            f'TORCHINDUCTOR_CACHE_DIR={scratch}/inductor',f'TRITON_CACHE_DIR={scratch}/triton',
            'PYTORCH_ALLOC_CONF=expandable_segments:True','NCCL_IB_DISABLE=1']
        payload=dict(Image=IMAGE,User=f'{os.getuid()}:{os.getgid()}',WorkingDir='/repo',Entrypoint=['python'],Cmd=cmd,
            Labels={LABEL:stage},Env=env,HostConfig=dict(Runtime='runc',NetworkMode='none',ReadonlyRootfs=True,
                CapDrop=['ALL'],SecurityOpt=['no-new-privileges'],ShmSize=32*1024**3,Memory=1500*1024**3,
                NanoCpus=64*10**9,PidsLimit=4096,
                DeviceRequests=[dict(Driver='nvidia',DeviceIDs=[str(i) for i in range(8)],Capabilities=[['gpu']])],
                Binds=[f'{REPO}:/repo:ro',f'{REPO}:{REPO}:ro',f'{OUT}:/output:rw',
                    f'{snapshot}:{snapshot}:ro',f'{scratch}:{scratch}:rw']))
        durable(OUT/f'{stage}-engine-request.json',payload)
        e=engine();cid=e.request('POST',f'/containers/create?name=nemochat-offline2000-{stage}-01',payload)['Id']
        a=owned(e,cid,stage)
        durable(OUT/f'{stage}-engine-created.json',dict(id=cid,image=IMAGE,argv=a['Config']['Cmd'],mounts=a['Mounts']))
        try:
            e.request('POST',f'/containers/{cid}/start')
            durable(OUT/f'{stage}-engine-start-readback.json',owned(e,cid,stage))
            durable(OUT/'phase.json',dict(stage=stage,container_id=cid,time=time.time(),student_learning_confirmed=False))
            # Event-driven Engine container exit. No external PID/artifact wait
            # is used, so there is no unsafe PID polling or prerequisite restart.
            e.request('POST',f'/containers/{cid}/wait?condition=not-running',timeout=limit)
        finally:
            a=owned(e,cid,stage)
            if a['State']['Running']:
                e.request('POST',f'/containers/{cid}/stop?t=20');a=owned(e,cid,stage)
            engine_logs(e,cid,stage)
            final=dict(id=cid,state=a['State']);durable(OUT/f'{stage}-engine-final.json',final)
        clean_engine(final,cid);return final
    finally:lock.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cleanup',action='store_true');p.add_argument('--review',type=Path);p.add_argument('--review-sha256');a=p.parse_args()
    if a.cleanup:cleanup();return
    review=verify_review(a.review,a.review_sha256);accepted_policy(review)
    owner=ownership(a.review,a.review_sha256)
    # Exclusive durable generation claim; no automatic failed-writer retry.
    with (OUT/'controller-started.json').open('x') as f:json.dump(owner,f);f.flush();os.fsync(f.fileno())
    deadline=time.monotonic()+plan()['controller_seconds']-90
    teacher=run_stage('teacher',a.review,a.review_sha256,deadline)
    ready=merge(review,teacher);durable(OUT/'accepted-window.json',ready)
    validate_ready(review)
    run_stage('student',a.review,a.review_sha256,deadline)
    ranks=[json.loads((OUT/f'resume-rank{r}-complete.json').read_text()) for r in range(8)]
    if ([r['rank'] for r in ranks]!=list(range(8)) or any(r['completed']!=ranks[0]['completed']
        or r['skipped'] or r['nonfinite'] or not r['operational_resume'] or r['restored_step']!=16 for r in ranks)):
        raise ValueError('student native allrank resume completion invalid')
    calls=ranks[0]['completed']
    if not 16<=calls<=2000:raise ValueError('student optimizer call budget exceeded')
    if calls>16:
        tx=json.loads((OUT/f'resume-checkpoint-step{calls}-transaction.json').read_text())
        if tx['binding']['next_iteration']!=calls or not tx['save_optim'] or not tx['save_rng']:raise ValueError('final full-state save missing')
    durable(OUT/'complete.json',dict(status='target_complete' if calls==2000 else 'source_exhausted_shortfall',
        total_optimizer_calls=calls,shortfall=2000-calls,total_positive_updates=ranks[0]['total_positive_updates'],
        ranks=ranks,checkpoint_integrity='not_checked_explicit_user_waiver',no_repetition=True))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        durable(OUT/'controller-failure.json',dict(type=type(exc).__name__,error=str(exc),automatic_retry=False));raise
