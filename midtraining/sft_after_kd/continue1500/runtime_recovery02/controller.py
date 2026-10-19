#!/usr/bin/env python3
"""Parent-only durable continuation launch; preview/freeze never allocate GPUs."""
import argparse,fcntl,hashlib,importlib.util,json,math,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/raid/yoon/nemochat');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import contract
OUT=ROOT/'results/midtraining/sft-continue1500-02'
MANIFEST=ROOT/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json'
MANIFEST_SHA='78ab81fbbd9c0b13c55392b7d87c9e7035ad75778abad0dd3795d64121125c21'
SOURCE='checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
DEADLINE=contract.DEADLINE;TRAIN_STOP=contract.TRAIN_STOP;CAPACITY=contract.HORIZON
UNIT='nemochat-sft-sealed03-continue1500-r02-main.service'
NAME='nemochat-sft-sealed03-continue1500-r02-main'
OWNER='nemochat.sft.continue1500.owner'
def require(ok,message):
 if not ok:raise PermissionError(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
 with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def load_module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
legacy=load_module('continue1500_runtime05_helpers',ROOT/'midtraining/sft_after_kd/runtime05/controller.py')
legacy.OUT=OUT;legacy.HERE=HERE;legacy.OWNER=OWNER
legacy.DEADLINE=DEADLINE;legacy.TRAIN_STOP=TRAIN_STOP
IMAGE=legacy.IMAGE;FILES=legacy.FILES
engine=legacy.engine
process_env=legacy.process_env
capture_logs=legacy.capture_logs

def source_paths():
 paths=[str(p.relative_to(ROOT)) for p in HERE.glob('*') if p.is_file() and p.suffix in ('.py','.sh','.md')]
 paths+=legacy.source_paths()
 paths=[p for p in paths if Path(ROOT/p).is_file()]
 paths+=['midtraining/sft_after_kd/runtime05/controller.py','midtraining/sft_after_kd/sealed_snapshot03/client.py','midtraining/sft_after_kd/continue1500/request.json']
 return sorted(set(paths))
def freeze():
 OUT.mkdir(parents=True,exist_ok=True)
 p=OUT/'source-freeze.json';require(not p.exists(),'freeze already exists; do not overwrite reviewed source')
 write(p,{p:sha(ROOT/p) for p in source_paths()});return {'source_freeze_sha256':sha(p),'path':str(p)}
def config(now):
 import yaml
 b=contract.budget(now)
 c=yaml.safe_load((ROOT/'midtraining/sft_after_kd/recipe-v1.yaml').read_text())
 c.update(name='nemochat SFT905 full-state continuation until 2026-10-19 15:00 KST',source_load_mode='native_full_state_continuation',resume_checkpoint=contract.CHECKPOINT,load_hf_weights=False,
  max_updates=CAPACITY,max_hours=(TRAIN_STOP-now)/3600,execution_mode='native_continuation',calibration_required=False,calibration_receipt=None,
  execution_update_cap=b['execution_update_cap'],training_stop_timestamp=TRAIN_STOP,conservative_iteration_seconds=b['conservative_iteration_seconds'],
  hard_deadline='2026-10-19T15:00:00+09:00',save_eval_buffer_seconds=600,launch_ready=True,require_final_validation=False)
 return c,b
def payload(config_path,limit,config_only=False):
 p=legacy.container_payload('config' if config_only else 'main',config_path,limit,config_only)
 p['Labels']={OWNER:'main','nemochat.sealed03':'v1'}
 if not config_only:
  p['Cmd'][p['Cmd'].index('--release')+1]=str(OUT/'native-release.json')
 return p

def owned(e,cid):
 a=e.request('GET','/containers/'+cid+'/json');labels=a['Config'].get('Labels',{})
 require(a['Id']==cid and a['Name']=='/'+NAME and labels.get(OWNER)=='main' and labels.get('nemochat.sealed03')=='v1','exact continuation Engine identity required')
 return a

def cleanup():
 p=OUT/'main-container.json'
 if not p.exists():
  if not (OUT/'execution-claim.json').exists():return {}
  e=engine()
  try:a=e.request('GET','/containers/'+NAME+'/json')
  except RuntimeError as exc:
   if ': 404:' in str(exc):return {}
   raise
  write(p,{'id':a['Id'],'recovered_create_identity':True})
 e=engine();cid=read(p)['id'];a=owned(e,cid)
 if a['State']['Running']:e.request('POST','/containers/'+cid+'/stop?t=60',timeout=90)
 a=owned(e,cid);require(not a['State']['Running'],'owned container still running');write(OUT/'cleanup.json',a['State']);return a['State']

def accept(state,c):
 require(state.get('Running') is False and state.get('ExitCode')==0 and not state.get('OOMKilled') and not state.get('Error'),'clean Engine exit required')
 counts=[];finals=[]
 for rank in range(8):
  d=OUT/'main';rows=[json.loads(x) for x in (d/f'rank-{rank}-updates.jsonl').read_text().splitlines() if x]
  n=len(rows);final=905+n
  require(n>0 and final<=c['execution_update_cap'] and [x['update'] for x in rows]==list(range(906,final+1)) and [x['segment_update'] for x in rows]==list(range(1,n+1)),'contiguous native continuation905->final required')
  for row in rows:
   require(row['native_optimizer'] and row['finite_positive_gradient'] and math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['grad_norm']>0 and all(math.isfinite(v) and v>0 for v in row['optimizer_step_param_group_lrs']),'finite native learning evidence required')
  load=read(d/f'rank-{rank}-load.json')
  proof=load['actual_native_restore_provenance']
  require(load['source']==contract.CHECKPOINT and load['stage_step']==905 and load['optimizer_scheduler_rng_cursor_reset'] is False and proof['loaded'] and proof['optimizer'] and proof['scheduler'] and proof['rng'],'all-rank actual native full-state restore required')
  save=read(d/f'rank-{rank}-save.json');entry=read(d/f'rank-{rank}-entrypoint.json')
  require(final in save['save_steps'] and save['save_optim'] and save['save_rng'] and save['native_synchronous_save_completed'] and entry['native_exit_code']==0 and entry['updates']==n and entry['termination_reason'] in ('native_execution_cap_saved','native_deadline_saved'),'all-rank native final full-state save/exit required')
  counts.append(n);finals.append(final)
 require(len(set(finals))==1,'rank final-step disagreement')
 checkpoint=OUT/'main/checkpoint'/f'iter_{finals[0]:07d}'
 inventory=[{'path':str(p.relative_to(checkpoint)),'bytes':p.stat().st_size} for p in checkpoint.rglob('*') if p.is_file()]
 names={x['path'] for x in inventory}
 require({'train_state.pt','run_config.yaml','.metadata','metadata.json',*[f'__{r}_0.distcp' for r in range(8)]}<=names and all(x['bytes']>0 for x in inventory),'actual native eight-rank checkpoint inventory incomplete')
 write(OUT/'final-checkpoint-inventory.json',inventory)
 return dict(status='native_fullstate_continuation_saved',base_step=905,final_step=finals[0],segment_updates=counts[0],rank_counts=counts,checkpoint=str(checkpoint),checkpoint_integrity='not_checked_explicit_user_waiver',optional_final_evaluation='not_required_unrun_unless_native_receipt_exists')

STOP=False
def stop(signum,frame):
 global STOP
 STOP=True

def execute(release):
 from readiness import validate_parent_release
 r=validate_parent_release(release,sys.modules[__name__]);OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'execution-claim.json').open('x') as f:json.dump({'pid':os.getpid(),'release':str(release),'time':time.time()},f)
 props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','-p','ActiveState'],text=True).splitlines())
 require(props.get('MainPID')==str(os.getpid()) and props.get('ActiveState') in ('active','activating'),'exact parent-owned service MainPID required')
 controls=load_module('continue1500_controls',ROOT/'midtraining/sft_after_kd/controls.py')
 identity=controls.process_identity(os.getpid())
 require(identity['uid']==os.getuid() and identity['argv']==[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'controller.py'),'--execute','--release',str(release)],'supervisor UID/full argv mismatch')
 write(OUT/'supervisor-identity.json',identity)
 lock=(ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
 try:
  write(OUT/'gpu-idle.json',legacy.idle())
  validate_parent_release(release,sys.modules[__name__])
  write(OUT/'native-release.json',dict(r,status='continue1500_verified_parent_release'))
  c,b=config(time.time());write(OUT/'timing-budget.json',b);write(OUT/'main-config.json',c)
  import yaml
  cp=OUT/'main-config.yaml';cp.write_text(yaml.safe_dump(c,sort_keys=False))
  (OUT/'main').mkdir(exist_ok=False)
  for p in ('inductor','triton','cuda'):(OUT/'main-home'/p).mkdir(parents=True,exist_ok=False)
  p=payload(cp,DEADLINE-time.time()-90);write(OUT/'main-request.json',p)
  e=engine();cid=e.request('POST','/containers/create?name='+NAME,p)['Id'];write(OUT/'main-container.json',{'id':cid})
  owned(e,cid);e.request('POST','/containers/'+cid+'/start');a=owned(e,cid);write(OUT/'main-start-readback.json',{'id':cid,'state':a['State']})
  while a['State']['Running']:
   write(OUT/'main-engine-state.json',{'id':cid,'state':a['State'],'time':time.time()})
   if STOP or time.time()>=DEADLINE-65:raise TimeoutError('owned stop/final-save watchdog expired; no auto retry')
   time.sleep(5);a=owned(e,cid)
  write(OUT/'main-engine-final.json',{'id':cid,'state':a['State']});capture_logs(e,cid,'main')
  result=accept(a['State'],c);require(time.time()<DEADLINE,'completion past15:00 deadline');write(OUT/'complete.json',result)
 except BaseException as exc:
  write(OUT/'failure.json',{'type':type(exc).__name__,'message':str(exc),'no_retry':True});raise
 finally:
  cleanup();lock.close()

def start(release):
 from readiness import validate_parent_release
 validate_parent_release(release,sys.modules[__name__]);require(not (OUT/'execution-claim.json').exists(),'already attempted; no automatic restart')
 python=str(ROOT/'.venv/bin/python')
 argv=['systemd-run','--user','--unit='+UNIT,'--service-type=exec','--working-directory='+str(ROOT),'--property=Restart=no','--property=KillMode=control-group','--property=TimeoutStopSec=120','--property=RuntimeMaxSec='+str(max(1,int(DEADLINE-time.time()-30))),'--property=ExecStopPost='+python+' '+str(HERE/'controller.py')+' --cleanup','--property=StandardOutput=append:'+str(OUT/'supervisor.log'),'--property=StandardError=append:'+str(OUT/'supervisor.log'),python,'-u',str(HERE/'controller.py'),'--execute','--release',str(release)]
 subprocess.run(argv,check=True)
 props=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','-p','ActiveState','-p','ExecStart'],text=True);write(OUT/'service-start-readback.json',{'properties':props});require('MainPID=0' not in props,'new service MainPID missing');return {'status':'supervisor_started_not_learning_evidence','unit':UNIT,'properties':props}

def main():
 p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
 for verb in ('preview','freeze','start','execute','cleanup'):g.add_argument('--'+verb,action='store_true')
 p.add_argument('--release',type=Path,default=OUT/'parent-release.json');a=p.parse_args()
 if a.freeze:r=freeze()
 elif a.execute:execute(a.release.resolve());return
 elif a.start:r=start(a.release.resolve())
 elif a.cleanup:r=cleanup()
 else:r={'status':'INERT_PARENT_FRESH_RAM_RELEASE_REQUIRED','checkpoint':contract.CHECKPOINT,'unit':UNIT,'next_global_update':906,'next_segment_update':1,'next_consumed_sample':115840,'horizon':CAPACITY,'training_stop':'2026-10-19T14:50:00+09:00','deadline':'2026-10-19T15:00:00+09:00','budget_preview':contract.budget(time.time())}
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
