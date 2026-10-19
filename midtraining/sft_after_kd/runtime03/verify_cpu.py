#!/usr/bin/env python3
"""Explicit CPU-only pinned Engine proof. Never invokes production start/service."""
import importlib.util,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import controller as c

def main():
    c.OUT.mkdir(parents=True,exist_ok=True)
    directory=c.OUT/'cpu-collective'
    directory.mkdir(exist_ok=False)
    e=c.engine();name='nemochat-sft-sealed03-cpu-proof'
    payload=c.container_payload('config',c.OUT/'unused.yaml',300,config_only=True)
    payload['Labels']={c.OWNER:'cpu-proof','nemochat.sealed03':'v1'}
    payload['HostConfig']['Memory']=24*1024**3
    payload['Cmd']=['--signal=TERM','--kill-after=30','240','python','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=29629',str(HERE/'cpu_collective_fixture.py'),str(directory)]
    payload['Env']=[x for x in payload['Env'] if not x.startswith('OMP_NUM_THREADS=')]+['OMP_NUM_THREADS=1']
    home=c.OUT/'config-home'
    for part in ('tmp','inductor','triton','cuda'):(home/part).mkdir(parents=True,exist_ok=True)
    c.write(c.OUT/'cpu-proof-request.json',payload)
    cid=e.request('POST','/containers/create?name='+name,payload)['Id']
    c.write(c.OUT/'cpu-proof-container.json',dict(id=cid,name=name))
    def exact():
        a=e.request('GET','/containers/'+cid+'/json')
        c.require(a['Id']==cid and a['Name']=='/'+name and a['Config']['Labels'].get(c.OWNER)=='cpu-proof' and a['Config']['Labels'].get('nemochat.sealed03')=='v1','CPU proof Engine ownership mismatch')
        c.require(not a['HostConfig'].get('DeviceRequests'),'CPU proof must have no GPU request')
        return a
    exact();e.request('POST','/containers/'+cid+'/start');started=time.monotonic()
    try:
        while exact()['State']['Running']:
            if time.monotonic()-started>280:raise TimeoutError('bounded CPU proof')
            time.sleep(2)
    finally:
        a=exact()
        if a['State']['Running']:e.request('POST','/containers/'+cid+'/stop?t=30')
        a=exact();c.write(c.OUT/'cpu-proof-final.json',dict(id=cid,state=a['State'],device_requests=a['HostConfig'].get('DeviceRequests')))
        c.capture_logs(e,cid,'cpu-proof')
    c.require(a['State']['ExitCode']==0 and not a['State']['OOMKilled'],'CPU fixture Engine failed; inspect preserved log')
    rows=[c.read(directory/f'rank-{rank}.json') for rank in range(8)]
    c.require(all(row['rank']==rank and len(row['rejected_faults'])==9 for rank,row in enumerate(rows)),'all eight CPU receipts required')
    receipt=dict(status='PINNED_CPU_8PROCESS_SEALED_FIXTURE_PASS',container_id=cid,ranks=rows,gpu_requests=0,producer_client='TEST_LOCAL_PUBLIC_API_STUB',actual_full03_integration=False)
    c.write(c.OUT/'cpu-collective-proof.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
