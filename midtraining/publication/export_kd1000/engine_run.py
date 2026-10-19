"""Foreground bounded pinned CPU-only Engine runner; no GPU/network."""
import argparse, json, sys, time, struct
from pathlib import Path
sys.path.insert(0,'/raid/yoon/nemochat/results/pretrain_followup')
from engine_helper import request, Connection
ROOT=Path('/raid/yoon/nemochat')
OUT=ROOT/'results/midtraining/hf-kd1000-conversion-01'
CODE=ROOT/'midtraining/publication/export_kd1000'
EXPORT=ROOT/'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
IMAGE='nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
def run(name,command,export_dir=EXPORT,receipt_dir=OUT):
    global EXPORT,OUT
    EXPORT=Path(export_dir).resolve();OUT=Path(receipt_dir).resolve()
    assert EXPORT.is_relative_to(ROOT) and OUT.is_relative_to(ROOT)
    assert EXPORT.is_dir(), "Create a fresh user-owned output directory before running"
    OUT.mkdir(parents=True,exist_ok=True)
    spec={'Image':IMAGE,'Cmd':['/usr/bin/timeout','--signal=TERM','--kill-after=30','1200',*command],'Entrypoint':[],'WorkingDir':str(ROOT),'Env':['CUDA_VISIBLE_DEVICES=','NVIDIA_VISIBLE_DEVICES=void','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PYTHONDONTWRITEBYTECODE=1'],'User':'0:0','HostConfig':{'Binds':[f'{ROOT}:{ROOT}:ro',f'{OUT}:{OUT}:rw',f'{EXPORT}:{EXPORT}:rw'],'NetworkMode':'none','NanoCpus':4000000000,'Memory':137438953472,'ShmSize':1073741824},'Labels':{'nemochat.task':'hf-kd1000-conversion'}}
    cid=request('POST','/containers/create?name='+name,spec)['Id']
    (OUT/(name+'.create.json')).write_text(json.dumps({'id':cid,'spec':spec},indent=2))
    state={}
    try:
        request('POST',f'/containers/{cid}/start')
        deadline=time.monotonic()+1250
        while True:
            state=request('GET',f'/containers/{cid}/json')['State']
            if not state['Running']:break
            if time.monotonic()>deadline:raise TimeoutError('runner deadline')
            time.sleep(2)
    finally:
        live=request('GET',f'/containers/{cid}/json')['State']
        if live['Running']:request('POST',f'/containers/{cid}/stop?t=10')
        state=request('GET',f'/containers/{cid}/json')['State']
        conn=Connection('localhost',timeout=120);conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1');raw=conn.getresponse().read();conn.close()
        parts=[]
        while len(raw)>=8:
            n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
        text=b''.join(parts).decode(errors='replace')
        (OUT/(name+'.log')).write_text(text)
        (OUT/(name+'.state.json')).write_text(json.dumps(state,indent=2))
        print(text);print(json.dumps(state))
    return state['ExitCode']
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export-dir',type=Path,default=EXPORT);p.add_argument('--receipt-dir',type=Path,default=OUT);p.add_argument('name');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();assert a.command;sys.exit(run(a.name,a.command,a.export_dir,a.receipt_dir))
