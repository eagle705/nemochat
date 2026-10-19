"""Start once via parent-owned user systemd. Never hold RAM here."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from .holder import ROOT,CODE,OUT,CONTRACT,sha,save,freeze_source,inventory,memory_available,process_identity
from .client import require

def start():
    contract_bytes=CONTRACT.read_bytes();contract=json.loads(contract_bytes)
    require(os.getuid()==1001,'approved_uid')
    require(not (OUT/'STARTED.json').exists(),'single_generation_already_started_no_retry')
    require(not (OUT/'READY.sealed.json').exists(),'ready_already_exists')
    fixture=json.loads((OUT/'FIXTURE.json').read_bytes());require(fixture['status']=='PASS','fixture_gate')
    status=subprocess.run(['systemctl','--user','show',contract['holder_unit'],'-p','ActiveState','-p','MainPID'],capture_output=True,text=True,check=True).stdout
    require('MainPID=0' in status and 'ActiveState=active' not in status,'holder_unit_already_used')
    mem=memory_available();require(mem>=contract['minimum_mem_available_before_start_bytes'],'MemAvailable_below_1TiB')
    _,jobs=inventory(contract);freeze=freeze_source()
    request={'schema':'nemochat-sealed03-request-v1','snapshot_id':uuid.uuid4().hex,'contract_sha256':sha(contract_bytes),'source_freeze':freeze,'inventory':jobs,'fixture_sha256':sha((OUT/'FIXTURE.json').read_bytes()),'mem_available_before_systemd_start':mem,'absolute_capture_timeout_seconds':3600,'holder_not_after':contract['holder_not_after']}
    path=OUT/'request.json';save(path,request);save(OUT/'source-freeze.json',freeze)
    runtime=int((datetime.fromisoformat(contract['holder_not_after'])-datetime.now().astimezone()).total_seconds())
    require(runtime>0,'expired_holder_deadline')
    python=str(ROOT/'.venv/bin/python')
    argv=[python,'-m','sft.archive.sealed_snapshot03.holder','hold','--request',str(path)]
    command=['systemd-run','--user','--unit='+contract['holder_unit'],'--description=Bounded CPU-only immutable full03 RAM holder','--property=MemoryMax='+contract['holder_memory_max'],'--property=MemoryHigh='+contract['holder_memory_high'],'--property=CPUQuota='+contract['holder_cpu_quota'],'--property=RuntimeMaxSec='+str(runtime),'--property=TimeoutStopSec=30','--property=Restart=no','--property=MemoryAccounting=yes','--property=CPUAccounting=yes','--property=UMask=0077','--property=NoNewPrivileges=yes','--working-directory='+str(ROOT),'--setenv=PYTHONDONTWRITEBYTECODE=1','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1',*argv]
    receipt={'command':command,'argv':argv,'request_sha256':sha(path.read_bytes()),'snapshot_id':request['snapshot_id'],'source_freeze_sha256':freeze['sha256'],'created_at':datetime.now().astimezone().isoformat()}
    save(OUT/'STARTED.json',receipt)
    execution=subprocess.run(command,capture_output=True,text=True);receipt.update(systemd_run_exit_code=execution.returncode,stdout=execution.stdout,stderr=execution.stderr);save(OUT/'STARTED.json',receipt)
    require(execution.returncode==0,'systemd_run_failed')
    time.sleep(.3)
    output=subprocess.run(['systemctl','--user','show',contract['holder_unit'],'-p','MainPID','-p','ExecStart','-p','MemoryMax','-p','MemoryHigh','-p','CPUQuotaPerSecUSec','-p','ControlGroup','-p','ActiveState','-p','User','-p','Group','-p','RuntimeMaxUSec'],capture_output=True,text=True,check=True).stdout
    fields=dict(line.split('=',1) for line in output.splitlines() if '=' in line)
    receipt['systemd_readback']=fields
    pid=int(fields['MainPID']);require(pid>0 and fields['ActiveState']=='active','systemd_not_active')
    identity=process_identity(pid);receipt['holder_identity']=identity
    require(identity['argv']==argv and identity['uid']==1001,'MainPID_argv_uid')
    require(fields['MemoryMax']==str(768*1024**3) and fields['MemoryHigh']==str(640*1024**3) and fields['CPUQuotaPerSecUSec']=='8s','cgroup_limits')
    require(not fields['User'] and not fields['Group'],'no_explicit_user_group')
    save(OUT/'STARTED.json',receipt)
    print(json.dumps({'holder_identity':identity,'systemd_readback':fields,'source_freeze_sha256':freeze['sha256'],'snapshot_id':request['snapshot_id']},indent=2))

def main():
    p=argparse.ArgumentParser(description='One authorized CPU-only sealed03 generation; durable bounded systemd holder')
    p.add_argument('command',choices=['start','status'])
    a=p.parse_args()
    if a.command=='start':start()
    else:
        for name in ('STARTED.json','progress.json','holder-state.json','BLOCKED.json','READY.sealed.json'):
            path=OUT/name
            if path.exists():print(name+': '+path.read_text())

if __name__=='__main__':main()
