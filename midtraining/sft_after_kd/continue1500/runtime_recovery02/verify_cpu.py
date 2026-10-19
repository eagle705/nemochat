#!/usr/bin/env python3
"""Exercise pinned config/source/CPU contracts with no GPU DeviceRequests."""
import json,time
from pathlib import Path
import controller as c

def main():
 c.OUT.mkdir(parents=True,exist_ok=True);cfg,b=c.config(time.time())
 import yaml
 path=c.OUT/'cpu-config.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
 p=c.payload(path,240,True)
 home=c.OUT/'cpu-proof-home'
 for part in ('inductor','triton','cuda'):(home/part).mkdir(parents=True,exist_ok=False)
 p['Env']=c.process_env(home,1)+['NVIDIA_VISIBLE_DEVICES=void','LD_LIBRARY_PATH=/opt/host-driver']
 for lib in ('libcuda.so.1','libnvidia-ml.so.1'):
  source=Path('/usr/lib/x86_64-linux-gnu')/lib
  p['HostConfig']['Binds'].append(str(source.resolve())+':/opt/host-driver/'+lib+':ro')
 p['Cmd']=['--signal=TERM','--kill-after=20','240','python',str(c.HERE/'cpu_probe.py'),'--config',str(path),'--output',str(c.OUT/'cpu-proof')]
 p['Labels']={c.OWNER:'cpu-proof','nemochat.sealed03':'v1'}
 c.require(not p['HostConfig']['DeviceRequests'],'CPU probe must have no GPU request')
 c.write(c.OUT/'cpu-proof-request.json',p)
 e=c.engine();cid=e.request('POST','/containers/create?name=nemochat-sft-sealed03-continue1500-cpu-proof02',p)['Id']
 c.write(c.OUT/'cpu-proof-container.json',{'id':cid});e.request('POST','/containers/'+cid+'/start');started=time.monotonic()
 try:
  while True:
   a=e.request('GET','/containers/'+cid+'/json')
   c.require(a['Id']==cid and not a['HostConfig'].get('DeviceRequests'),'CPU scope changed')
   if not a['State']['Running']:break
   if time.monotonic()-started>260:raise TimeoutError('no-device CPU probe timeout')
   time.sleep(2)
 finally:
  a=e.request('GET','/containers/'+cid+'/json')
  if a['State']['Running']:e.request('POST','/containers/'+cid+'/stop?t=20')
  a=e.request('GET','/containers/'+cid+'/json');c.capture_logs(e,cid,'cpu-proof');c.write(c.OUT/'cpu-proof-final.json',{'id':cid,'state':a['State'],'gpu_requests':a['HostConfig'].get('DeviceRequests')})
 c.require(a['State']['ExitCode']==0,'actual pinned CPU probe failed; inspect cpu-proof-engine.log')
 print(json.dumps(c.read(c.OUT/'cpu-proof/proof.json'),indent=2))
if __name__=='__main__':main()
