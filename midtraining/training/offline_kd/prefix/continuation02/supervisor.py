"""Durable single student segment; never starts or resumes a teacher."""
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(Path(__file__).resolve().parents[5] / 'midtraining/training/offline_kd'))
from launch import REPO,IMAGE,freeze
from contract import continuation_plan,authenticate_shard,create_local_compiler_cache
OUT=REPO/'results/midtraining/offline2b-prefixtrain-01/continuation-02'
SOURCE=REPO/'results/midtraining/offline2b-production1920-01'
UNIT='nemochat-offline2b-prefixtrain-01-continuation02.service'
LABEL='nemochat.prefix.continuation02.owner'

def publish(name,value):
    p=OUT/(name+'.json');tmp=p.with_suffix('.json.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
    print(name,json.dumps(value),flush=True)

def engine():
    spec=importlib.util.spec_from_file_location('prefix_engine',REPO/'results/pretrain_followup/engine_helper.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def owned(e,cid):
    a=e.request('GET',f'/containers/{cid}/json')
    if a['Id']!=cid or a['Config']['Labels'].get(LABEL)!=OUT.name:raise ValueError('foreign container')
    return a

def logs(e,cid):
    conn=e.Connection('localhost',timeout=60);conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1')
    raw=conn.getresponse().read();conn.close();parts=[]
    while len(raw)>=8:
        n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
    (OUT/'student-engine.log').write_bytes(b''.join(parts))

def cleanup():
    p=OUT/'engine-created.json'
    if p.exists():
        e=engine();cid=json.loads(p.read_text())['id'];a=owned(e,cid)
        if a['State']['Running']:e.request('POST',f'/containers/{cid}/stop?t=20')
        publish('engine-cleanup',dict(id=cid,state=owned(e,cid)['State']))

def main():
    if '--cleanup' in sys.argv:cleanup();return
    expected=json.loads((OUT/'source-freeze.json').read_text())
    actual={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    if expected['prefix']!=actual or expected['original']!=freeze():raise ValueError('source freeze changed')
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True))
    if pid!=os.getpid() or Path(f'/proc/{pid}').stat().st_uid!=os.getuid():raise ValueError('wrong systemd owner')
    publish('ownership',dict(unit=UNIT,MainPID=pid,uid=os.getuid(),argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[:-1],started=time.time()))
    c=json.loads((SOURCE/'run-config.json').read_text());plan=continuation_plan(c,64,1143)
    publish('run-config',c);publish('segment-plan',plan)
    lock=(REPO/'results/pretraining/nemochat_baby/training/gpu-all8.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    rows=[list(map(int,x.replace(' ','').split(','))) for x in gpu.splitlines()]
    if len(rows)!=8 or apps.strip() or any(mem>1024 or util>0 for _,mem,util in rows):raise ValueError('all8 not idle')
    publish('gpu-preflight',dict(gpus=rows,apps=apps,lock=str(lock.name)))
    shards=[]
    for rank in range(8):
        for iteration in range(1143):shards.append(authenticate_shard(SOURCE/'cache'/f'cp0_dp{rank}__{iteration}.tar',c,iteration))
        samples=[json.loads(x) for x in (SOURCE/f'teacher-rank{rank}-samples.jsonl').read_text().splitlines()]
        required=samples[:1143*16]
        if len(required)!=1143*16 or any(r['rank']!=rank or r['micro']!=i or r['iteration']!=i//16 for i,r in enumerate(required)):
            raise ValueError('teacher microbatch proof incomplete')
    publish('prefix-manifest',dict(cache_id=plan['cache_id'],coverage=[0,1142],full_teacher_ready=False,
        producer_intentionally_stopped=True,sampler_budget=plan['sampler_budget'],shards=shards,
        source=str(SOURCE),config=c,closed_payload_ownership_authenticated=True))
    scratch=Path(tempfile.mkdtemp(prefix='offline1920-prefix-',dir='/home/yoon/.hermes/cache/scratch'))
    old=Path(json.loads((OUT.parent/'scratch.json').read_text())['path'])
    shutil.copytree(old/'sdk',scratch/'sdk');shutil.copytree(old/'staged',scratch/'staged')
    create_local_compiler_cache(scratch/'inductor')
    load=OUT/'resume-load';load.mkdir()
    (load/'iter_0000064').symlink_to(OUT.parent/'checkpoint/iter_0000064',target_is_directory=True)
    (load/'latest_checkpointed_iteration.txt').write_text('64\n')
    if (SOURCE/'dataset-cache').exists():shutil.copytree(SOURCE/'dataset-cache',OUT/'dataset-cache')
    publish('scratch',dict(path=str(scratch),sdk_source=str(old),installed_sdk_unchanged=True))
    cmd=['-m','torch.distributed.run','--nproc_per_node=8','--nnodes=1','--node_rank=0','--master_addr=127.0.0.1','--master_port=29685',str(HERE/'native.py'),'resume']
    payload=dict(Image=IMAGE,User=f'{os.getuid()}:{os.getgid()}',WorkingDir='/repo',Entrypoint=['python'],Cmd=cmd,
        Labels={LABEL:OUT.name},Env=[f'PYTHONPATH={scratch}/staged/src:{scratch}/sdk:{HERE}:{HERE.parent}:/repo/midtraining/patches/offline_kd_pr/gpu_smoke:/repo/src',
        'PYTHONDONTWRITEBYTECODE=1',f'TMPDIR={scratch}',f'HOME={scratch}','USER=yoon','LOGNAME=yoon','WANDB_MODE=disabled',
        'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','TOKENIZERS_PARALLELISM=false','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
        'NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','NVIDIA_DRIVER_CAPABILITIES=compute,utility','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4',
        'TORCHINDUCTOR_COMPILE_THREADS=1',f'TORCHINDUCTOR_CACHE_DIR={scratch}/inductor','PYTORCH_ALLOC_CONF=expandable_segments:True','NCCL_IB_DISABLE=1'],
        HostConfig=dict(Runtime='runc',NetworkMode='none',ReadonlyRootfs=True,CapDrop=['ALL'],SecurityOpt=['no-new-privileges'],
        ShmSize=32*1024**3,Memory=1500*1024**3,NanoCpus=64*10**9,PidsLimit=4096,
        DeviceRequests=[dict(Driver='nvidia',DeviceIDs=[str(i) for i in range(8)],Capabilities=[['gpu']])],
        Binds=[f'{OUT.parent}:/initial:ro',f'{REPO}:/repo:ro',f'{REPO}:{REPO}:ro',f'{OUT}:/output:rw',f'{scratch}:{scratch}:rw']))
    e=engine();cid=e.request('POST','/containers/create?name=nemochat-offline1920-prefix-continuation02',payload)['Id']
    a=owned(e,cid);publish('engine-created',dict(id=cid,image=IMAGE,argv=a['Config']['Cmd'],mounts=a['Mounts']))
    e.request('POST',f'/containers/{cid}/start');deadline=time.monotonic()+12*3600
    try:
        while True:
            os.utime(scratch,None);a=owned(e,cid);logs(e,cid)
            publish('phase',dict(id=cid,state=a['State'],observed=time.time()))
            if not a['State']['Running']:break
            if time.monotonic()>deadline:raise TimeoutError('12h prefix watchdog')
            time.sleep(10)
    finally:
        if owned(e,cid)['State']['Running']:e.request('POST',f'/containers/{cid}/stop?t=20')
        logs(e,cid);publish('engine-final',dict(id=cid,state=owned(e,cid)['State']))
    if a['State']['ExitCode']!=0:raise RuntimeError('native student failed')
    ranks=[json.loads((OUT/f'resume-rank{r}-complete.json').read_text()) for r in range(8)]
    if any(r['completed']!=1143 or r['skipped'] or r['nonfinite'] for r in ranks):raise ValueError('incomplete native prefix')
    publish('complete',dict(ranks=ranks,full_schedule_unchanged=True,teacher_restarted=False))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        publish('failure',dict(type=type(exc).__name__,message=str(exc)));cleanup();raise
