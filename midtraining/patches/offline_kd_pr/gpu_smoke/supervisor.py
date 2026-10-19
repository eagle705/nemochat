"""Task-owned durable Engine supervisor; exact GPU0, cooperative all8 lock."""
import fcntl
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

REPO=Path('/raid/yoon/nemochat')
OUT=REPO/'results/midtraining/offline-kd-smoke-01'
CODE=REPO/'midtraining/patches/offline_kd_pr/gpu_smoke'
IMAGE='nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
OWNER='offline-kd-smoke-01'
RUNTIME='runc'  # NVIDIA device-request hook; daemon has no named nvidia runtime.
spec=importlib.util.spec_from_file_location('engine',REPO/'results/pretrain_followup/engine_helper.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)

def receipt(name,value):
    (OUT/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(name,json.dumps(value),flush=True)

def run(stage,scratch):
    name='nemochat-offline-kd-smoke-01-'+stage+'-'+str(int(time.time()))
    payload=dict(Image=IMAGE,User='1001:985',WorkingDir='/repo',Entrypoint=['python'],
                 Cmd=['-u',str(CODE/'native.py'),stage],
                 Labels={'nemochat.offline-kd-smoke.owner':OWNER,'nemochat.offline-kd-smoke.stage':stage},
                 Env=[f'PYTHONPATH={scratch}/staged/src:{scratch}/sdk:{CODE}:/repo/src',
                      'PYTHONDONTWRITEBYTECODE=1',f'TMPDIR={scratch}',f'HOME={scratch}',
                      'USER=yoon','LOGNAME=yoon','WANDB_MODE=disabled','HF_HUB_OFFLINE=1',
                      'TRANSFORMERS_OFFLINE=1','TOKENIZERS_PARALLELISM=false',
                      'CUDA_VISIBLE_DEVICES=0','NVIDIA_VISIBLE_DEVICES=0',
                      'NVIDIA_DRIVER_CAPABILITIES=compute,utility','OMP_NUM_THREADS=8','MKL_NUM_THREADS=8',
                      f'TORCHINDUCTOR_CACHE_DIR={scratch}/inductor','MASTER_ADDR=127.0.0.1',
                      'MASTER_PORT=29671','RANK=0','LOCAL_RANK=0','WORLD_SIZE=1',
                      'PYTORCH_ALLOC_CONF=expandable_segments:True',
                      'NCCL_P2P_DISABLE=1','NCCL_IB_DISABLE=1'],
                 HostConfig=dict(Runtime=RUNTIME,NetworkMode='none',ReadonlyRootfs=True,
                     CapDrop=['ALL'],SecurityOpt=['no-new-privileges'],ShmSize=8*1024**3,
                     Memory=500*1024**3,NanoCpus=16*10**9,PidsLimit=1024,
                     DeviceRequests=[dict(Driver='nvidia',DeviceIDs=['0'],Capabilities=[['gpu']])],
                     Binds=[f'{REPO}:/repo:ro',f'{REPO}:{REPO}:ro',f'{OUT}:/output:rw',
                            f'{scratch}:{scratch}:rw']))
    created=engine.request('POST','/containers/create?name='+name,payload)
    cid=created['Id'];actual=engine.request('GET',f'/containers/{cid}/json')
    assert actual['Config']['Labels']['nemochat.offline-kd-smoke.owner']==OWNER
    assert actual['HostConfig']['NetworkMode']=='none'
    assert actual['HostConfig']['DeviceRequests'][0]['DeviceIDs']==['0']
    assert all(not m['RW'] or m['Source'] in (str(OUT),str(scratch)) for m in actual['Mounts'])
    receipt(stage+'-engine-created',dict(id=cid,name=name,image=IMAGE,scratch=str(scratch),scope=actual['HostConfig'],mounts=actual['Mounts']))
    engine.request('POST',f'/containers/{cid}/start')
    deadline=time.monotonic()+2400
    while True:
        actual=engine.request('GET',f'/containers/{cid}/json')
        receipt(stage+'-engine-state',dict(id=cid,state=actual['State'],observed_time=time.time()))
        if not actual['State']['Running']:break
        if time.monotonic()>deadline:
            engine.request('POST',f'/containers/{cid}/stop?t=10')
            raise TimeoutError(cid)
        time.sleep(10)
    conn=engine.Connection('localhost',timeout=60)
    conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1')
    response=conn.getresponse();raw=response.read();conn.close();parts=[]
    while len(raw)>=8:
        n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
    (OUT/(stage+'-engine.log')).write_bytes(b''.join(parts))
    receipt(stage+'-engine-final',dict(id=cid,state=actual['State']))
    if actual['State']['ExitCode']!=0:raise RuntimeError(stage+' failed '+cid)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(REPO/'results/pretraining/nemochat_baby/training/gpu-all8.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    row=check.splitlines()[0].replace(' ','').split(',')
    if apps.strip() or int(row[1])>1024 or int(row[2])>0:raise RuntimeError('GPU occupied; stop and ask parent: '+check+'\n'+apps)
    reuse='--reuse-teacher' in sys.argv
    if reuse:
        from control import accept_teacher_reuse
        accept_teacher_reuse(json.loads((OUT/'teacher-complete.json').read_text()),
                             json.loads((OUT/'teacher-engine-final.json').read_text()))
        scratch=Path(json.loads((OUT/'request.json').read_text())['scratch'])
        assert scratch.is_dir()
        receipt('student-retry-request',dict(owner=OWNER,pid=os.getpid(),scratch=str(scratch),
            gpu_preflight=check,compute_apps=apps,cooperative_all8_lock=str(lock.name),
            stage='student_retry_from_original_BASE_same_accepted_cache',started=time.time()))
    else:
        scratch=Path(tempfile.mkdtemp(prefix='offline-kd-gpu-',dir='/home/yoon/.hermes/cache/scratch'))
        shutil.copytree('/home/yoon/.hermes/cache/scratch/bridge-kd-echzges_/sdk',scratch/'sdk')
        target=scratch/'staged/src/megatron/bridge';target.parent.mkdir(parents=True)
        shutil.copytree(scratch/'sdk/megatron/bridge',target)
        subprocess.run(['git','apply','--check',str(CODE.parent/'megatron-bridge-0.6.1-cached-kd.patch')],cwd=scratch/'staged',check=True)
        subprocess.run(['git','apply',str(CODE.parent/'megatron-bridge-0.6.1-cached-kd.patch')],cwd=scratch/'staged',check=True)
        receipt('request',dict(owner=OWNER,pid=os.getpid(),scratch=str(scratch),gpu_preflight=check,compute_apps=apps,
                               cooperative_all8_lock=str(lock.name),stage='teacher_then_student_no_main',started=time.time()))
        run('teacher',scratch)
    # Teacher process must fully exit before student weights/optimizer allocation.
    assert json.loads((OUT/'teacher-complete.json').read_text())['iterations']==32
    receipt('teacher-released',dict(process_exited=True,time=time.time()))
    run('student',scratch)
    run('reload',scratch)
    receipt('supervisor-complete',dict(clean_exit=True,time=time.time()))

if __name__=='__main__':main()
