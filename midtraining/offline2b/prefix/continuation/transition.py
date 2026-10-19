"""One-time checkpoint-64 handoff; retains immutable initial source and metrics."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path('/raid/yoon/nemochat')
BASE=REPO/'results/midtraining/offline2b-prefixtrain-01'
CODE=REPO/'midtraining/offline2b/prefix/continuation'
OLD='nemochat-offline2b-prefixtrain-01.service'
NEW='nemochat-offline2b-prefixtrain-01-continuation.service'
PYTHON=str(REPO/'.venv/bin/python')


def receipt(name,value):
    path=BASE/(name+'.json');tmp=path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    print(name,json.dumps(value),flush=True)


def main():
    expected=json.loads((BASE/'continuation-source-freeze.json').read_text())
    actual={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.glob('*.py')}
    if expected!=actual:raise ValueError('continuation source freeze changed')
    own=json.loads((BASE/'ownership.json').read_text());created=json.loads((BASE/'engine-created.json').read_text())
    spec=importlib.util.spec_from_file_location('transition_engine',REPO/'results/pretrain_followup/engine_helper.py')
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    deadline=time.monotonic()+45*60
    receipt('continuation-armed',dict(controller_pid=os.getpid(),old_unit=OLD,old_pid=own['MainPID'],
        old_container=created['id'],resume_step=64,stop_step=1143,reason='checkpoint-64 full-state continuation; retain original validation budget with eval interval 250'))
    path=BASE/'checkpoint-step64-transaction.json'
    while not path.exists():
        a=e.request('GET',f'/containers/{created["id"]}/json')
        if a['Config']['Labels'].get('nemochat.prefix.owner')!=BASE.name or not a['State']['Running']:
            raise RuntimeError('initial owned native student stopped before checkpoint64')
        if time.monotonic()>deadline:raise TimeoutError('checkpoint64 deadline')
        time.sleep(2)
    txn=json.loads(path.read_text());binding=txn['binding'];ranks=txn['ranks']
    if (not txn['native_synchronous_save_completed'] or not txn['save_optim'] or not txn['save_rng'] or
        sorted(r['rank'] for r in ranks)!=list(range(8)) or
        any(r['step']!=64 or r['binding']!=binding for r in ranks) or
        binding['next_iteration']!=64 or binding['consumed_samples']!=8192 or binding['train_samples']!=468736):
        raise ValueError('checkpoint64 transaction mismatch')
    subprocess.run([PYTHON,str(CODE.parent/'charts/render.py'),'--output',str(BASE),'--limit','64'],check=True)
    pid=int(subprocess.check_output(['systemctl','--user','show',OLD,'-p','MainPID','--value'],text=True))
    if pid!=own['MainPID']:raise ValueError('initial unit replaced; refuse stop')
    receipt('intentional-checkpoint64-transition',dict(old_unit=OLD,old_pid=pid,old_container=created['id'],
        selected_checkpoint=str(BASE/'checkpoint/iter_0000064'),selected_step=64,
        reason='explicit 64-call initial tranche then full-state resume; no model reset',
        discarded_unsaved_calls_possible=True,checkpoint_integrity='not_checked_explicit_user_waiver'))
    subprocess.run(['systemctl','--user','stop',OLD],check=True)
    a=e.request('GET',f'/containers/{created["id"]}/json')
    if a['State']['Running']:raise RuntimeError('old owned container still running')
    receipt('initial-engine-stopped',dict(id=created['id'],state=a['State'],intentional=True))
    out=BASE/'continuation';out.mkdir()
    spec=importlib.util.spec_from_file_location('launch',CODE.parent.parent/'launch.py')
    launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
    freeze=dict(prefix=actual,original=launch.freeze())
    (out/'source-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    cmd=['systemd-run','--user','--unit='+NEW.removesuffix('.service'),'--property=Type=exec',
         '--property=WorkingDirectory='+str(REPO),'--property=RuntimeMaxSec=43800',
         '--property=StandardOutput=append:'+str(out/'supervisor.log'),
         '--property=StandardError=append:'+str(out/'supervisor.log'),
         '--property=ExecStopPost='+PYTHON+' '+str(CODE/'supervisor.py')+' --cleanup',
         PYTHON,'-u',str(CODE/'supervisor.py')]
    subprocess.run(cmd,check=True)
    state=subprocess.check_output(['systemctl','--user','show',NEW,'-p','MainPID','-p','ActiveState','-p','ExecStart'],text=True)
    receipt('continuation-started',dict(command=cmd,readback=state,initial_checkpoint_transaction=txn))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        receipt('continuation-transition-failure',dict(type=type(exc).__name__,message=str(exc)));raise
