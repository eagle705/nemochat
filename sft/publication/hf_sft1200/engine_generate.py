"""Bounded foreground one-GPU Engine runner holding both cooperative locks."""
import sys,json,time,struct,fcntl,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'results/midtraining/hf-sft1200-publication-01'
sys.path.insert(0,str(ROOT/'results/pretrain_followup'))
from engine_helper import request,Connection
MODEL=ROOT/'checkpoints/exports/nemochat_sft1248_hf_bf16_01'
IMAGE='nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
if __name__=='__main__':
    locks=[]
    for p in [ROOT/'logs/gpu_execution.lock',ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock']:
        h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    assert not apps.strip(),apps
    assert all(int(r.split(',')[1].strip())==0 and int(r.split(',')[2].strip())==0 for r in query.strip().splitlines()),query
    (OUT/'gpu-preflight.json').write_text(json.dumps({'gpus':query,'compute_apps':apps,'locks':[str(h.name) for h in locks],'time':time.time(),'physical_device_ids':['0']},indent=2))
    for p in ['home','triton','inductor']:(OUT/p).mkdir(exist_ok=True)
    spec={'Image':IMAGE,'Cmd':['/usr/bin/timeout','--signal=TERM','--kill-after=30','1800','python','sft/publication/hf_sft1200/generate.py'],'Entrypoint':[],'WorkingDir':str(ROOT),'User':'1001:1001','Env':['CUDA_VISIBLE_DEVICES=0','NVIDIA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','TOKENIZERS_PARALLELISM=false','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PYTHONDONTWRITEBYTECODE=1',f'HOME={OUT}/home',f'TRITON_CACHE_DIR={OUT}/triton',f'TORCHINDUCTOR_CACHE_DIR={OUT}/inductor','TMPDIR=/nct','TMP=/nct','TEMP=/nct'],'HostConfig':{'Binds':[f'{ROOT}:{ROOT}:ro',f'{OUT}:{OUT}:rw'],'NetworkMode':'none','NanoCpus':4000000000,'Memory':137438953472,'ShmSize':2147483648,'Tmpfs':{'/nct':'rw,exec,nosuid,nodev,size=4g,mode=1777'},'DeviceRequests':[{'Driver':'nvidia','Count':0,'DeviceIDs':['0'],'Capabilities':[['gpu']],'Options':{}}]},'Labels':{'nemochat.task':'sft1248-six-examples-01'}}
    name='nemochat-sft1248-six-examples-01';cid=request('POST','/containers/create?name='+name,spec)['Id']
    (OUT/'generation-engine.create.json').write_text(json.dumps({'id':cid,'name':name,'spec':spec},indent=2))
    try:
        request('POST',f'/containers/{cid}/start');deadline=time.monotonic()+1860
        while True:
            state=request('GET',f'/containers/{cid}/json')['State']
            if not state['Running']:break
            if time.monotonic()>deadline:raise TimeoutError('bounded runner deadline')
            time.sleep(2)
    finally:
        live=request('GET',f'/containers/{cid}/json')['State']
        if live['Running']:request('POST',f'/containers/{cid}/stop?t=10')
        inspection=request('GET',f'/containers/{cid}/json');state=inspection['State']
        conn=Connection('localhost',timeout=120);conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1');raw=conn.getresponse().read();conn.close();parts=[]
        while len(raw)>=8:
            n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
        text=b''.join(parts).decode(errors='replace');(OUT/'generation-engine.log').write_text(text)
        (OUT/'generation-engine.state.json').write_text(json.dumps({'id':cid,'state':state,'device_requests':inspection['HostConfig']['DeviceRequests']},indent=2))
        print(text);print(json.dumps(state))
    sys.exit(state['ExitCode'])
