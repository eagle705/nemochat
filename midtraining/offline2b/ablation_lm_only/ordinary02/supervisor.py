"""Durable ordinary LM-only B1.0, exact frozen C accepted replay."""
import fcntl,hashlib,importlib.util,json,os,shutil,struct,subprocess,sys,tempfile,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=Path('/raid/yoon/nemochat')
OUT=REPO/'results/midtraining/offline2b-lmonly-1000-02'
UNIT='nemochat-offline2b-lmonly-1000-02.service'
LABEL='nemochat.ablation.lmonly1000.ordinary02.owner'
sys.path.insert(0,str(REPO/'midtraining/offline2b'))
from launch import IMAGE,freeze

def publish(name,data):
    p=OUT/(name+'.json');t=p.with_suffix('.json.partial');t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');t.replace(p)
    print(name,json.dumps(data),flush=True)
def engine():
    spec=importlib.util.spec_from_file_location('engine',REPO/'results/pretrain_followup/engine_helper.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def owned(e,cid):
    a=e.request('GET',f'/containers/{cid}/json')
    if a['Id']!=cid or a['Config']['Labels'].get(LABEL)!='student':raise ValueError('container identity mismatch')
    return a
def logs(e,cid):
    c=e.Connection('localhost',timeout=60);c.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1')
    raw=c.getresponse().read();c.close();parts=[]
    while len(raw)>=8:
        n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
    (OUT/'student-engine.log').write_bytes(b''.join(parts))
def sources():
    paths=[*HERE.glob('*.py'),HERE/'README.md',HERE/'README.ko.md',
           REPO/'tests/test_offline2b_lmonly_ordinary02.py',REPO/'midtraining/offline2b/prefix/contract.py']
    return dict(original=freeze(),ablation={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
def cleanup():
    r=OUT/'engine-created.json'
    if r.exists():
        e=engine();cid=json.loads(r.read_text())['id'];a=owned(e,cid)
        if a['State']['Running']:e.request('POST',f'/containers/{cid}/stop?t=20')
        publish('cleanup',dict(id=cid,state=owned(e,cid)['State']))
def main():
    if '--cleanup' in sys.argv:cleanup();return
    if sources()!=json.loads((OUT/'source-freeze.json').read_text()):raise ValueError('source freeze changed')
    pid=int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value'],text=True))
    if pid!=os.getpid() or Path(f'/proc/{pid}').stat().st_uid!=os.getuid():raise ValueError('wrong supervisor owner')
    publish('ownership',dict(unit=UNIT,MainPID=pid,uid=os.getuid(),argv=sys.argv,time=time.time()))
    lock=(REPO/'results/pretraining/nemochat_baby/training/gpu-all8.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    parsed=[list(map(int,l.replace(' ','').split(','))) for l in rows.splitlines()]
    if len(parsed)!=8 or apps.strip() or any(m>1024 or u>0 for _,m,u in parsed):raise ValueError('all8 not idle')
    publish('gpu-preflight',dict(rows=parsed,apps=apps,lock=str(lock.name)))
    manifest=json.loads((OUT/'replay-manifest.json').read_text())
    if len(manifest['accepted_physical_mapping'])!=1000:raise ValueError('fixed C replay mapping required')
    scratch=Path(tempfile.mkdtemp(prefix='offline2b-lmonly1000-',dir='/home/yoon/.hermes/cache/scratch'))
    old=Path(manifest['approved_scratch_source'])
    shutil.copytree(old/'sdk',scratch/'sdk');shutil.copytree(old/'staged',scratch/'staged')
    for n in ['home','hfhome','inductor','triton']:(scratch/n).mkdir()
    publish('scratch',dict(path=str(scratch),sdk_source=str(old),compiler_caches_initially_empty=True,installed_sdk_unchanged=True))
    dataset=REPO/'results/midtraining/offline2b-prefixtrain-1000-01/dataset-cache'
    if dataset.exists():shutil.copytree(dataset,OUT/'dataset-cache')
    cmd=['-m','torch.distributed.run','--nproc_per_node=8','--nnodes=1','--node_rank=0','--master_addr=127.0.0.1','--master_port=29696',str(HERE/'native.py'),'student']
    payload=dict(Image=IMAGE,User=f'{os.getuid()}:{os.getgid()}',WorkingDir='/repo',Entrypoint=['python'],Cmd=cmd,Labels={LABEL:'student'},
        Env=[f'PYTHONPATH={scratch}/staged/src:{scratch}/sdk:{HERE.parent.parent}:/repo/midtraining/patches/offline_kd_pr/gpu_smoke:/repo/src:/repo',
            'PYTHONDONTWRITEBYTECODE=1',f'TMPDIR={scratch}',f'HOME={scratch}/home',f'HF_HOME={scratch}/hfhome','USER=yoon','LOGNAME=yoon',
            'WANDB_MODE=disabled','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','TOKENIZERS_PARALLELISM=false',
            'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7','NVIDIA_DRIVER_CAPABILITIES=compute,utility',
            'OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','TORCHINDUCTOR_COMPILE_THREADS=1',f'TORCHINDUCTOR_CACHE_DIR={scratch}/inductor',f'TRITON_CACHE_DIR={scratch}/triton',
            'PYTORCH_ALLOC_CONF=expandable_segments:True','NCCL_IB_DISABLE=1'],
        HostConfig=dict(Runtime='runc',NetworkMode='none',ReadonlyRootfs=True,CapDrop=['ALL'],SecurityOpt=['no-new-privileges'],
            ShmSize=32*1024**3,Memory=1500*1024**3,NanoCpus=64*10**9,PidsLimit=4096,
            DeviceRequests=[dict(Driver='nvidia',DeviceIDs=[str(i) for i in range(8)],Capabilities=[['gpu']])],
            Binds=[f'{REPO}:/repo:ro',f'{REPO}:{REPO}:ro',f'{OUT}:/output:rw',f'{scratch}:{scratch}:rw']))
    publish('engine-request',payload)
    e=engine();cid=e.request('POST','/containers/create?name=nemochat-offline2b-lmonly-1000-02',payload)['Id']
    a=owned(e,cid);publish('engine-created',dict(id=cid,image=IMAGE,argv=a['Config']['Cmd'],mounts=a['Mounts']))
    e.request('POST',f'/containers/{cid}/start');publish('engine-start-readback',owned(e,cid));deadline=time.monotonic()+12*3600
    try:
        while True:
            os.utime(scratch,None);a=owned(e,cid);logs(e,cid);publish('phase',dict(id=cid,state=a['State'],time=time.time()))
            if not a['State']['Running']:break
            if time.monotonic()>deadline:raise TimeoutError('12h watchdog')
            time.sleep(15)
    finally:
        if owned(e,cid)['State']['Running']:e.request('POST',f'/containers/{cid}/stop?t=20')
        logs(e,cid);publish('engine-final',dict(id=cid,state=owned(e,cid)['State']))
    if a['State']['ExitCode']!=0:raise RuntimeError('native student failed')
    ranks=[json.loads((OUT/f'student-rank{r}-complete.json').read_text()) for r in range(8)]
    if any(r['completed']!=1000 or r['optimizer_calls']!=1000 or r['positive_updates']!=999 or r['skipped'] or r['nonfinite'] for r in ranks):raise ValueError('incomplete ordinary LM-only native run')
    publish('complete',dict(ranks=ranks,lm_coefficient=1.0,kd_disabled=True,checkpoint=str(OUT/'checkpoint/iter_0001000'),reserved_for_parent_evaluation=True))
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        publish('failure',dict(type=type(exc).__name__,message=str(exc)));raise
