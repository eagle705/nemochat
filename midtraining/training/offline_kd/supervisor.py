"""Finite user-systemd owner for sequential native DP8 teacher/student phases."""
import argparse
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
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
from launch import REPO,CODE,PATCH,IMAGE,plan,verify_freeze,validate_gate_review
LABEL='nemochat.offline2b.owner'

def helper(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def receipt(out,name,value):
    path=out/(name+'.json');temp=path.with_suffix('.json.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
    print(name,json.dumps(value,allow_nan=False),flush=True)

def stages(mode):
    if mode not in ('gate','production','production1920'):raise ValueError('unknown pipeline mode')
    return ('teacher','student','resume') if mode=='gate' else ('teacher','student')


def claim(path,owner):
    with path.open('x') as s:json.dump(dict(owner=owner,pid=os.getpid(),started=time.time()),s)

def accept_rank_rows(rows,horizon):
    if len(rows)!=8 or sorted(r['rank'] for r in rows)!=list(range(8)):
        raise ValueError('exactly eight rank receipts required')
    if any(r['completed']!=horizon or r['positive_updates']<1 or r['skipped'] or r['nonfinite'] for r in rows):
        raise ValueError('allrank positive finite nonskipped updates and exact horizon required')
    return dict(all_rank_completed=horizon,ranks=rows)

def engine():return helper('offline2b_engine',REPO/'results/pretrain_followup/engine_helper.py')

def owned(e,cid,owner):
    a=e.request('GET',f'/containers/{cid}/json')
    if a['Id']!=cid or a['Config']['Labels'].get(LABEL)!=owner:
        raise ValueError('refuse foreign container')
    return a

def cleanup(out):
    if not (out/'controller.json').exists():return
    owner=json.loads((out/'controller.json').read_text())['owner'];e=engine()
    for p in out.glob('*-engine-created.json'):
        cid=json.loads(p.read_text())['id'];a=owned(e,cid,owner)
        if a['State']['Running']:e.request('POST',f'/containers/{cid}/stop?t=20')
        receipt(out,p.stem+'-cleanup',dict(id=cid,state=owned(e,cid,owner)['State']))

def verify_corpus_approval(review, manifest_path, manifest_sha, e):
    """A success JSON alone cannot authorize a running or failed CPU producer."""
    proof = review.get('corpus_approval', {})
    if (proof.get('status') != 'parent_verified_native_corpus'
            or proof.get('manifest_path') != str(manifest_path)
            or proof.get('manifest_sha256') != manifest_sha):
        raise ValueError('parent-reviewed corpus manifest identity required')
    labels = proof.get('task_labels', {})
    if not labels or any(not name.startswith('nemochat.') for name in labels):
        raise ValueError('exact owned producer labels required')
    a = e.request('GET', '/containers/' + proof['producer_id'] + '/json')
    state = a['State']
    actual_labels = {k: v for k, v in a['Config'].get('Labels', {}).items() if k.startswith('nemochat.')}
    if (a['Id'] != proof['producer_id'] or a['Name'] != proof['producer_name']
            or a['Config']['Image'] != IMAGE or actual_labels != labels
            or a['HostConfig'].get('DeviceRequests')
            or a['HostConfig']['NetworkMode'] != 'none'
            or state.get('Running') is not False or state.get('ExitCode') != 0
            or state.get('OOMKilled') is not False):
        raise ValueError('clean exact owned CPU producer required before GPU release')

def prepare_manifest(mode,out,c):
    native=helper('offline2b_longrun',REPO/'midtraining/training/longrun_native.py')
    rel='pilot-10m-v1' if mode=='gate' else ('offline1920-v1/data' if mode=='production1920' else 'offline2b-v1/data')
    path=REPO/'datasets/midtraining/super35-kd-01'/rel/'manifest.json'
    manifest,digest=native.load_prepared_manifest(path)
    if mode!='gate' and list(native.manifest_weights(manifest))!=c['weights']:
        raise ValueError('production 67/10/3/20 source weights required')
    audit=native.audit_reserved_ids(manifest)
    native.corpus_tokenizer_gate(manifest,audit)
    # Gate consumes a bounded prefix of the pilot with the approved new weights.
    capacity=native.capacity_gate(dict(manifest,weights=dict(zip(native.KEYS,c['weights']))),
                                  c['train_iters'],128,4096)
    data_bytes=sum(manifest['sources'][k][s]['bin_bytes']+manifest['sources'][k][s]['idx_bytes']
                   for k in native.KEYS for s in ('train','validation'))
    if data_bytes>c['data_limit_bytes']:raise ValueError('data byte cap exceeded')
    c.update(manifest_path=str(path),manifest_sha256=digest)
    receipt(out,'data-accepted',dict(manifest_path=str(path),manifest_sha256=digest,
        capacity=capacity,reserved_ids=audit,data_bytes=data_bytes,gate_only=mode=='gate'))
    return c

def keep_scratch_alive(scratch):
    """Refresh only this live owner's workspace against idle scratch pruning."""
    if scratch.stat().st_uid != os.getuid():
        raise ValueError('refuse foreign scratch heartbeat')
    os.utime(scratch, None)

def run_stage(e,out,scratch,c,stage,deadline):
    verify_freeze(json.loads((out/'reviewed-source.json').read_text()))
    owner=out.name;name='nemochat-'+owner+'-'+stage
    cmd=['-m','torch.distributed.run','--nproc_per_node=8','--nnodes=1','--node_rank=0',
         '--master_addr=127.0.0.1','--master_port=29683',str(CODE/'native.py'),stage]
    payload=dict(Image=IMAGE,User=f'{os.getuid()}:{os.getgid()}',WorkingDir='/repo',Entrypoint=['python'],Cmd=cmd,
        Labels={LABEL:owner,'nemochat.offline2b.stage':stage},
        Env=[f'PYTHONPATH={scratch}/staged/src:{scratch}/sdk:{CODE}:{PATCH}/gpu_smoke:/repo/src',
             'PYTHONDONTWRITEBYTECODE=1',f'TMPDIR={scratch}',f'HOME={scratch}',
             'USER=yoon','LOGNAME=yoon','WANDB_MODE=disabled','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
             'TOKENIZERS_PARALLELISM=false','CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7',
             'NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
             'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4',f'TORCHINDUCTOR_COMPILE_THREADS={c["compile_threads"]}',
             f'TORCHINDUCTOR_CACHE_DIR={scratch}/inductor',
             'PYTORCH_ALLOC_CONF=expandable_segments:True','NCCL_IB_DISABLE=1'],
        HostConfig=dict(Runtime='runc',NetworkMode='none',ReadonlyRootfs=True,CapDrop=['ALL'],
            SecurityOpt=['no-new-privileges'],ShmSize=32*1024**3,Memory=1500*1024**3,
            NanoCpus=64*10**9,PidsLimit=4096,
            DeviceRequests=[dict(Driver='nvidia',DeviceIDs=[str(i) for i in range(8)],Capabilities=[['gpu']])],
            Binds=[f'{REPO}:/repo:ro',f'{REPO}:{REPO}:ro',f'{out}:/output:rw',f'{scratch}:{scratch}:rw']))
    cid=e.request('POST','/containers/create?name='+name,payload)['Id']
    a=owned(e,cid,owner)
    assert a['HostConfig']['DeviceRequests'][0]['DeviceIDs']==[str(i) for i in range(8)]
    assert all(not m['RW'] or m['Source'] in (str(out),str(scratch)) for m in a['Mounts'])
    receipt(out,stage+'-engine-created',dict(id=cid,name=name,image=IMAGE,argv=a['Config']['Cmd'],mounts=a['Mounts']))
    e.request('POST',f'/containers/{cid}/start')
    try:
        while True:
            keep_scratch_alive(scratch)
            a=owned(e,cid,owner)
            receipt(out,'phase',dict(stage=stage,id=cid,state=a['State'],observed=time.time()))
            if not a['State']['Running']:break
            if time.monotonic()>deadline:raise TimeoutError('combined finite watchdog')
            cache=out/'cache'
            if cache.exists() and sum(p.stat().st_size for p in cache.glob('*') if p.is_file())>c['cache_limit_bytes']:
                raise ValueError('cache hard cap exceeded')
            time.sleep(10)
    finally:
        a=owned(e,cid,owner)
        if a['State']['Running']:
            e.request('POST',f'/containers/{cid}/stop?t=20');a=owned(e,cid,owner)
        conn=e.Connection('localhost',timeout=60)
        conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1')
        raw=conn.getresponse().read();conn.close();parts=[]
        while len(raw)>=8:
            n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
        (out/(stage+'-engine.log')).write_bytes(b''.join(parts))
        receipt(out,stage+'-engine-final',dict(id=cid,state=a['State']))
    if a['State']['ExitCode']!=0:raise RuntimeError(stage+' native failed '+cid)
    rows=[json.loads((out/f'{stage}-rank{i}-complete.json').read_text()) for i in range(8)]
    if stage!='teacher':accept_rank_rows(rows,c['train_iters'])
    elif any(r['completed']!=c['train_iters'] or r['optimizer_calls']!=0 or not r['frozen'] for r in rows):
        raise ValueError('frozen complete teacher allrank proof missing')
    receipt(out,stage+'-accepted',dict(id=cid,ranks=rows,clean_exit=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['gate','production','production1920']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cleanup',action='store_true');p.add_argument('--review',type=Path);a=p.parse_args();out=a.output.resolve()
    if a.cleanup:cleanup(out);return
    owner=out.name;claim(out/'controller.json',owner)
    try:
        verify_freeze(json.loads((out/'reviewed-source.json').read_text()))
        unit=os.environ['OFFLINE2B_UNIT']
        pid=int(subprocess.check_output(['systemctl','--user','show',unit,'-p','MainPID','--value'],text=True))
        actual=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
        expected=[sys.executable,'-u',str(CODE/'supervisor.py'),'--mode',a.mode,'--output',str(out)]
        if a.review:expected+=['--review',str(a.review)]
        if pid!=os.getpid() or Path(f'/proc/{pid}').stat().st_uid!=os.getuid() or [x.decode() for x in actual]!=expected:
            raise ValueError('exact systemd MainPID UID argv ownership mismatch')
        receipt(out,'ownership',dict(unit=unit,MainPID=pid,uid=os.getuid(),argv=expected))
        c=plan(a.mode)
        if a.mode!='gate':
            review=json.loads(a.review.read_text());verify_freeze(review['source_sha256'])
            gate=json.loads((Path(review['gate_output'])/'report.json').read_text())
            validate_gate_review(review,gate,a.mode)
            if a.mode=='production1920':
                manifest_path=REPO/'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json'
                verify_corpus_approval(review,manifest_path,hashlib.sha256(manifest_path.read_bytes()).hexdigest(),engine())
        c=prepare_manifest(a.mode,out,c)
        lock=(REPO/'results/pretraining/nemochat_baby/training/gpu-all8.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        rows=[list(map(int,x.replace(' ','').split(','))) for x in gpu.splitlines()]
        if len(rows)!=8 or apps.strip() or any(mem>1024 or util>0 for _,mem,util in rows):
            raise ValueError('all eight GPUs must be idle; refuse foreign jobs')
        receipt(out,'gpu-preflight',dict(gpus=rows,compute_apps=apps,lock=str(lock.name)))
        scratch=Path(tempfile.mkdtemp(prefix='offline2b-',dir='/home/yoon/.hermes/cache/scratch'))
        shutil.copytree('/home/yoon/.hermes/cache/scratch/bridge-kd-echzges_/sdk',scratch/'sdk')
        target=scratch/'staged/src/megatron/bridge';target.parent.mkdir(parents=True)
        shutil.copytree(scratch/'sdk/megatron/bridge',target)
        patch=str(PATCH/'megatron-bridge-0.6.1-cached-kd.patch')
        subprocess.run(['git','apply','--check',patch],cwd=scratch/'staged',check=True)
        subprocess.run(['git','apply',patch],cwd=scratch/'staged',check=True)
        receipt(out,'run-config',c);receipt(out,'scratch',dict(path=str(scratch),installed_sdk_unchanged=True))
        deadline=time.monotonic()+c['watchdog_seconds'];e=engine()
        for stage in stages(a.mode):
            if stage=='resume':
                # Metadata-only root; never copy/hash model or optimizer payloads.
                step=c['resume_step'] or c['train_iters']-1
                load=out/'resume-load';load.mkdir()
                (load/f'iter_{step:07d}').symlink_to(out/'checkpoint'/f'iter_{step:07d}',target_is_directory=True)
                (load/'latest_checkpointed_iteration.txt').write_text(str(step)+'\n')
                receipt(out,'resume-selection',dict(step=step,payload_copied=False))
            run_stage(e,out,scratch,c,stage,deadline)
        receipt(out,'report',dict(status='gate_accepted' if a.mode=='gate' else 'production_complete',
            source_sha256=json.loads((out/'reviewed-source.json').read_text()),plan=c,
            checkpoint_integrity=c['checkpoint_integrity'],gate_resumed_extra_updates=2 if a.mode=='gate' else 0))
    except BaseException as exc:
        receipt(out,'failure',dict(type=type(exc).__name__,message=str(exc),time=time.time()))
        cleanup(out);raise

if __name__=='__main__':main()
