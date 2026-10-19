#!/usr/bin/env python3
"""Bounded no-device storage/Manager proof; never starts native learning."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import controller as c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',choices=['inspect','01','02','03'],required=True)
    args=parser.parse_args();segment='storage-cpu-'+args.attempt
    directory=c.OUT/segment;directory.mkdir(parents=True,exist_ok=False)
    home=c.OUT/(segment+'-home')
    for part in ('inductor','triton','cuda'):(home/part).mkdir(parents=True,exist_ok=False)
    payload=c.container_payload('config',c.OUT/'unused',300,config_only=True)
    payload['Env']=c.process_env(home,1)+['NVIDIA_VISIBLE_DEVICES=void','NVIDIA_DRIVER_CAPABILITIES=compute,utility']
    payload['HostConfig']['Memory']=24*1024**3
    for library in ('libcuda.so.1','libnvidia-ml.so.1'):
        source=Path('/usr/lib/x86_64-linux-gnu')/library
        payload['HostConfig']['Binds'].append(str(source.resolve())+':/opt/host-driver/'+library+':ro')
    payload['Env'].append('LD_LIBRARY_PATH=/opt/host-driver')
    payload['Labels']={c.OWNER:segment,'nemochat.sealed03':'v1'}
    payload['Cmd']=['--signal=TERM','--kill-after=30','300','python',str(c.HERE/'cpu_storage.py'),'--output',str(directory)]
    if args.attempt=='inspect':payload['Cmd'].append('--inspect')
    e=c.engine();name='nemochat-sft-sealed03-r05-'+segment
    c.write(directory/'request.json',payload)
    cid=e.request('POST','/containers/create?name='+name,payload)['Id'];c.write(directory/'container.json',dict(id=cid,name=name))
    def exact():
        a=e.request('GET','/containers/'+cid+'/json')
        c.require(a['Id']==cid and a['Name']=='/'+name and a['Config']['Image']==c.IMAGE and a['Config']['Labels'].get(c.OWNER)==segment,'storage Engine ownership mismatch')
        c.require(not a['HostConfig'].get('DeviceRequests') and a['HostConfig']['ReadonlyRootfs'] and a['HostConfig']['CapDrop']==['ALL'],'storage CPU isolation mismatch')
        c.require(dict(x.split('=',1) for x in a['Config']['Env'])['TMPDIR']=='/nct' and '/nct' in a['HostConfig']['Tmpfs'],'short temp not active at process start')
        return a
    exact();e.request('POST','/containers/'+cid+'/start');started=time.monotonic()
    try:
        while exact()['State']['Running']:
            if time.monotonic()-started>340:raise TimeoutError('bounded CPU storage proof')
            time.sleep(2)
    finally:
        a=exact()
        if a['State']['Running']:e.request('POST','/containers/'+cid+'/stop?t=30')
        a=exact();c.write(directory/'final.json',dict(id=cid,state=a['State'],device_requests=a['HostConfig'].get('DeviceRequests'),tmpfs=a['HostConfig']['Tmpfs']))
        c.capture_logs(e,cid,segment)
    c.require(a['State']['ExitCode']==0 and not a['State']['OOMKilled'],'CPU storage proof failed; retained attempt '+segment)
    if args.attempt!='inspect':
        proof=c.read(directory/'proof.json')
        c.require(proof['status']=='PINNED_CPU_MANAGER_AND_NATIVE_TORCH_DIST_STORAGE_PASS','actual storage success required')
        c.write(c.OUT/'storage-cpu-proof.json',dict(proof,container_id=cid,attempt=args.attempt,source_freeze_sha256=c.sha(c.OUT/'source-freeze.json')))
        print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
