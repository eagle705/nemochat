"""Pinned CPU-only native full-budget config construction; no weights/devices."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import controller as c

def main():
    m=c.read(c.MANIFEST);cfg=c.stage_config('calibration',m,time.time());path=c.write_config('cpu-sdk-calibration',cfg)
    p=c.container_payload('config',path,180,config_only=True)
    p['Cmd'][p['Cmd'].index('--output')+1]=str(c.OUT/'config-sdk02')
    p['Labels']={c.OWNER:'config-sdk02','nemochat.sealed03':'v1'}
    # Same no-device library mounts as the accepted runtime02 config03 probe.
    for library in ('libcuda.so.1','libnvidia-ml.so.1'):
        source=Path('/usr/lib/x86_64-linux-gnu')/library
        c.require(source.is_file(),'host driver library absent: '+library)
        p['HostConfig']['Binds'].append(str(source.resolve())+':/opt/host-driver/'+library+':ro')
    p['Env'].append('LD_LIBRARY_PATH=/opt/host-driver')
    e=c.engine();cid=e.request('POST','/containers/create?name=nemochat-sft-sealed03-config-sdk02',p)['Id']
    c.write(c.OUT/'config-sdk02-request.json',p);c.write(c.OUT/'config-sdk02-container.json',{'id':cid})
    e.request('POST','/containers/'+cid+'/start');start=time.monotonic()
    try:
        while True:
            a=e.request('GET','/containers/'+cid+'/json')
            assert a['Id']==cid and a['Name']=='/nemochat-sft-sealed03-config-sdk02' and not a['HostConfig'].get('DeviceRequests')
            if not a['State']['Running']:break
            if time.monotonic()-start>210:raise TimeoutError('CPU SDK timeout')
            time.sleep(2)
    finally:
        a=e.request('GET','/containers/'+cid+'/json')
        if a['State']['Running']:e.request('POST','/containers/'+cid+'/stop?t=20')
        a=e.request('GET','/containers/'+cid+'/json')
        c.capture_logs(e,cid,'config-sdk02');c.write(c.OUT/'config-sdk02-final.json',dict(id=cid,state=a['State'],device_requests=a['HostConfig'].get('DeviceRequests')))
    print(json.dumps(a['State'],indent=2))
    c.require(a['State']['ExitCode']==0,'pinned SDK CPU configuration failed')
    print(json.dumps(c.read(c.OUT/'config-sdk02/config-only.json'),indent=2))

if __name__=='__main__':main()
