"""Bounded pinned-image CPU-only launcher; no child background jobs."""
import argparse,json,sys,time,struct,os
from pathlib import Path
sys.path.insert(0,'/raid/yoon/nemochat/results/pretrain_followup')
from engine_helper import request,Connection
ROOT=Path('/raid/yoon/nemochat')
OUT=ROOT/'results/midtraining/offline2b-ablation-eval-01'
IMAGE='nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
def run(name,command,network=False,seconds=1800):
 OUT.mkdir(parents=True,exist_ok=True)
 receipt=OUT/(name+'.create.json')
 if receipt.exists():raise FileExistsError(receipt)
 spec={'Image':IMAGE,'Cmd':['/usr/bin/timeout','--signal=TERM',str(seconds),*command],'Entrypoint':[],'WorkingDir':str(ROOT),'User':f'{os.getuid()}:{os.getgid()}','Env':['HOME='+str(OUT/'home'),'TORCHINDUCTOR_CACHE_DIR='+str(OUT/'cache/inductor'),'TRITON_CACHE_DIR='+str(OUT/'cache/triton'),'CUDA_VISIBLE_DEVICES=','NVIDIA_VISIBLE_DEVICES=void','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','HF_HOME='+str(OUT/'hf-cache'),'HF_DATASETS_CACHE='+str(OUT/'hf-cache/datasets'),'TOKENIZERS_PARALLELISM=false'],'Labels':{'nemochat.ablation-eval':'cpu-only'},'HostConfig':{'Binds':[f'{ROOT}:{ROOT}:ro',f'{OUT}:{OUT}:rw'],'NetworkMode':'bridge' if network else 'none','NanoCpus':4000000000,'Memory':137438953472,'ShmSize':1073741824,'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges']}}
 cid=request('POST','/containers/create?name='+name,spec)['Id'];receipt.write_text(json.dumps({'id':cid,'spec':spec},indent=2));request('POST',f'/containers/{cid}/start')
 while request('GET',f'/containers/{cid}/json')['State']['Running']:time.sleep(3)
 inspect=request('GET',f'/containers/{cid}/json');assert not inspect['HostConfig'].get('DeviceRequests');assert inspect['HostConfig']['Memory']==137438953472
 conn=Connection('localhost',timeout=120);conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1');raw=conn.getresponse().read();conn.close();parts=[]
 while len(raw)>=8:
  n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
 text=b''.join(parts).decode(errors='replace');(OUT/(name+'.log')).write_text(text);(OUT/(name+'.inspect.json')).write_text(json.dumps(inspect,indent=2));print(text);print(inspect['State']);return inspect['State']['ExitCode']
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name',required=True);p.add_argument('--network',action='store_true');p.add_argument('--seconds',type=int,default=1800);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();sys.exit(run(a.name,a.command,a.network,a.seconds))
