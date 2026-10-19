"""GPU-only same-suite evaluation. Independent systemd ownership; never trains."""
import argparse,concurrent.futures,fcntl,hashlib,importlib.util,json,math,os,signal,struct,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path('/raid/yoon/nemochat')
HERE=ROOT/'midtraining/offline2b/ablation_gpu'
OUT=ROOT/'results/midtraining/offline2b-ablation-gpu-01'
EVAL=HERE
SUITE=ROOT/'results/midtraining/offline2b-ablation-eval-01/suite.json'
SUITE_SHA='07609a0cf53e4a63d555a3e50938adfed1cb05471d02b37fb93461e7879af19e'
UNIT='nemochat-offline2b-ablation-gpu-01.service'
OWNER='nemochat.ablation-gpu.owner'
IMAGE='nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
MODELS={a:ROOT/'checkpoints/exports'/n for a,n in [('A','nemochat_baby_base_hf_bf16_02'),('C','nemochat_mid_super35_distill_hf_bf16_01'),('B','nemochat_mid_lmonly_hf_bf16_02')]}
GPUS={'A':'0','C':'1','B':'2'}
MUTEX=threading.RLock()
STOP=threading.Event()
def require(x,message):
 if not x:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def module(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def engine():return module(ROOT/'results/pretrain_followup/engine_helper.py','gpu_engine')
def clean_engine(x,cid):
 s=x['State'];require(x['Id']==cid and not s['Running'] and s['ExitCode']==0 and not s['OOMKilled'] and not s.get('Error'),'GPU Engine not clean Exit0')
def validate_devices(actual,expected):
 require(len(actual)==len(expected)==1,'GPU request count mismatch')
 require(all(actual[0].get(k)==v for k,v in expected[0].items()) and actual[0].get('Count',0)==0 and not actual[0].get('Options'),'GPU device scope mismatch')
def load_suite():
 s=read(SUITE);h=s.pop('suite_sha256');require(hashlib.sha256(json.dumps(s,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()==h==SUITE_SHA,'suite changed');s['suite_sha256']=h;return s
def hashes():
 paths=[*sorted(HERE.glob('*.py')),HERE/'start.sh',ROOT/'results/pretrain_followup/engine_helper.py',SUITE]
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
def verify_freeze():require(read(OUT/'source-freeze.json')==hashes(),'executed source allowlist changed');load_suite()
def spec(arm,phase):
 require(arm in GPUS and phase in ['smoke','full'],'invalid job')
 cache=OUT/(arm+'-'+phase+'-runtime')
 cmd=['python','-u',str(HERE/'eval.py'),'score','--model',str(MODELS[arm]),'--suite',str(SUITE),'--out',str(OUT/(arm+'-'+('smoke01' if phase=='smoke' else 'full01')))]+(['--smoke'] if phase=='smoke' else [])
 return {'Image':IMAGE,'Entrypoint':[],'Cmd':['/usr/bin/timeout','--signal=TERM','21000',*cmd],'WorkingDir':str(ROOT),'User':f'{os.getuid()}:{os.getgid()}','Env':['HOME='+str(cache),'HF_HOME='+str(cache/'hf'),'TORCHINDUCTOR_CACHE_DIR='+str(cache/'inductor'),'TRITON_CACHE_DIR='+str(cache/'triton'),'CUDA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','TOKENIZERS_PARALLELISM=false','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8'],'Labels':{OWNER:UNIT,'nemochat.ablation-gpu.arm':arm,'nemochat.ablation-gpu.phase':phase},'HostConfig':{'Binds':[f'{ROOT}:{ROOT}:ro',f'{OUT}:{OUT}:rw'],'NetworkMode':'none','NanoCpus':4000000000,'Memory':137438953472,'ShmSize':1073741824,'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges'],'DeviceRequests':[{'Driver':'nvidia','DeviceIDs':[GPUS[arm]],'Capabilities':[['gpu']]}]}}
def name(arm,phase):return UNIT[:-8]+'-run03-'+arm.lower()+'-'+phase

def cleanup():
 STOP.set();out=[];e=engine()
 with MUTEX:
  for arm in GPUS:
   for phase in ['smoke','full']:
    p=OUT/(arm+'-'+phase+'.owned.json')
    if not p.exists():continue
    receipt=read(p);x=e.request('GET',f"/containers/{receipt['id']}/json")
    require(x['Name']=='/'+name(arm,phase) and x['Config']['Labels'].get(OWNER)==UNIT and x['Id']==receipt['id'],'cleanup ownership mismatch')
    if x['State']['Running']:e.request('POST',f"/containers/{x['Id']}/stop?t=20")
    x=e.request('GET',f"/containers/{x['Id']}/json");require(not x['State']['Running'],'owned GPU still running');out.append({'arm':arm,'phase':phase,'id':x['Id'],'state':x['State']})
 write(OUT/'cleanup.json',out)

def job(arm,phase):
 verify_freeze();e=engine();sp=spec(arm,phase);cache=OUT/(arm+'-'+phase+'-runtime');cache.mkdir(exist_ok=False)
 with MUTEX:
  require(not STOP.is_set(),'stopping')
  cid=e.request('POST','/containers/create?name='+name(arm,phase),sp)['Id'];write(OUT/(arm+'-'+phase+'.owned.json'),{'id':cid,'spec':sp});e.request('POST',f'/containers/{cid}/start')
 while True:
  x=e.request('GET',f'/containers/{cid}/json')
  if not x['State']['Running']:break
  if STOP.wait(2):raise InterruptedError('controller stopped')
 conn=e.Connection('localhost',timeout=120);conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1');raw=conn.getresponse().read();conn.close();parts=[]
 while len(raw)>=8:
  n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
 (OUT/(arm+'-'+phase+'.log')).write_bytes(b''.join(parts));write(OUT/(arm+'-'+phase+'.final.json'),x);clean_engine(x,cid)
 require(x['Config']['Labels'].get(OWNER)==UNIT,'GPU owner drift');validate_devices(x['HostConfig']['DeviceRequests'],sp['HostConfig']['DeviceRequests'])
 r=read(OUT/(arm+'-'+('smoke01' if phase=='smoke' else 'full01'))/'report.json');validate_backend(r)
 if phase=='smoke':
  require(r['status']=='passed_smoke' and r['question_count']==2 and r['heldout_window_count']==1 and r['heldout_target_count']==1024,'partial smoke')
  s=load_suite();require([q['id'] for q in r['questions']]==[s['questions'][0]['id'],next(q['id'] for q in s['questions'] if q['task']=='arc_challenge')],'smoke question order drift')
 else:validate_report(r,load_suite(),MODELS[arm])
 require(r['model']==str(MODELS[arm]),'model substitution');return r

def validate_backend(r):
 require(r['device']=='cuda:0' and r['dtype']=='bfloat16' and r['transformers']=='5.12.1' and r['torch']=='2.13.0a0+8145d630e8.nv26.06','GPU backend/version drift')
 require(r['gpu_name']=='NVIDIA B300 SXM6 AC' and r['gpu_uuid'] and r['attention']=='eager','GPU hardware/backend drift')
 require(all(r[k] is False for k in ['tf32_matmul','tf32_cudnn','deterministic_algorithms','cudnn_deterministic','cudnn_benchmark','official_harness_backend']),'GPU precision flag drift')
 require(r['all_loaded_tensors_finite'] and r['parameters_unchanged'] and r['no_padding'],'GPU finite/no-mutation failure')
 require(not any(r['loading_info'].get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),'loading key errors')

def run():
 verify_freeze();require(os.getuid()==1001,'wrong UID');require(int(subprocess.check_output(['systemctl','--user','show',UNIT,'-p','MainPID','--value']))==os.getpid(),'not independent owned service')
 require(not any((OUT/p).exists() for p in ['state.json','complete.json','failure.json']),'fresh controller required')
 write(OUT/'ownership.json',{'unit':UNIT,'pid':os.getpid(),'uid':os.getuid(),'argv':sys.argv,'starttime':int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),'final_marker':str(OUT/'complete.json'),'failure_marker':str(OUT/'failure.json')})
 def stopped(signum,frame):raise InterruptedError('GPU controller stopped')
 signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
 lock=ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock'
 with lock.open('a') as held:
  fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   live=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],text=True);require(not live.strip(),'GPUs not idle; no other jobs touched')
   gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv'],text=True);write(OUT/'gpu-preflight.json',{'time':time.time(),'gpus':gpu,'compute_processes':live,'lock':str(lock)})
   with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for phase in ['smoke','full']:
     write(OUT/'state.json',{'phase':phase,'time':time.time()});jobs={a:pool.submit(job,a,phase) for a in GPUS}
     try:reports={a:f.result() for a,f in jobs.items()}
     except BaseException:cleanup();raise
     if phase=='smoke':write(OUT/'smoke-accepted.json',{'status':'all_three_real_GPU_smokes_passed','reports':{a:str(OUT/(a+'-smoke01/report.json')) for a in GPUS},'suite_sha256':SUITE_SHA})
   verify_freeze();write(OUT/'state.json',{'phase':'reporting','time':time.time()});summary=module(HERE/'report.py','gpu_report').render(reports,load_suite(),OUT)
   for a,r in reports.items():
    for q in r['questions']:require(read(OUT/(a+'-full01')/(q['id'].replace('/','__')+'.json'))==q,'per-row receipt mismatch')
   cleanup();write(OUT/'state.json',{'phase':'complete','time':time.time()})
   write(OUT/'complete.json',{'status':'verified_all_three_complete','all_three_complete':True,'unit':UNIT,'suite_sha256':SUITE_SHA,'verified_reports':{a:str(OUT/(a+'-full01/report.json')) for a in GPUS},'images':summary['images'],'coverage':summary['coverage'],'reports':[str(OUT/'FINAL_REPORT.md'),str(OUT/'FINAL_REPORT.ko.md')],'summary':str(OUT/'summary.json'),'checkpoint_integrity':'not_checked_explicit_user_waiver'})
  except BaseException as exc:
   write(OUT/'failure.json',{'status':'failed','error':str(exc),'type':type(exc).__name__,'no_CPU_fallback':True,'no_retry':True});raise
  finally:
   if not (OUT/'complete.json').exists():cleanup()
   fcntl.flock(held,fcntl.LOCK_UN)
def validate_report(r,s,model):
    require(r['status']=='complete' and r['suite_sha256']==s['suite_sha256']==SUITE_SHA and r['model']==str(model),'partial/wrong suite/model report')
    validate_backend(r)
    require(r['parameters_unchanged'] is True and r['no_padding'] is True,'scoring changed parameters or padding')
    require(r['model_class']=='transformers.models.nemotron_h.modeling_nemotron_h.NemotronHForCausalLM','wrong scoring class')
    qs=r['questions'];hs=r['heldout']
    require(len(qs)==r['question_count']==len(s['questions'])==400 and len(hs)==r['heldout_window_count']==len(s['heldout'])==64,'incomplete suite coverage')
    require([q['id'] for q in qs]==[q['id'] for q in s['questions']] and len({q['id'] for q in qs})==400,'question IDs/order/duplicates mismatch')
    require([w['id'] for w in hs]==[w['id'] for w in s['heldout']] and len({w['id'] for w in hs})==64,'heldout IDs/order/duplicates mismatch')
    for q,expected in zip(qs,s['questions']):
        vals=q['choice_logp_full_vocab_sum'];gold=expected['gold']
        require(q['task']==expected['task'] and q['gold']==gold and len(vals)==len(expected['choices'])==4,'question gold/task/choices mismatch')
        require(all(math.isfinite(x) for x in [*vals,q['gold_logp'],q['gold_choice_conditional_logp']]),'nonfinite question score')
        pred=max(range(4),key=lambda i:vals[i]);predn=max(range(4),key=lambda i:vals[i]/len(expected['choices'][i]))
        require(q['prediction']==pred and q['prediction_char_norm']==predn and q['acc']==int(pred==gold) and q['acc_norm']==int(predn==gold) and q['gold_logp']==vals[gold],'score aggregation mismatch')
        mx=max(vals);conditional=vals[gold]-(mx+math.log(sum(math.exp(v-mx) for v in vals)))
        require(math.isclose(conditional,q['gold_choice_conditional_logp'],rel_tol=1e-10,abs_tol=1e-10),'conditional choice probability mismatch')
    for w,expected in zip(hs,s['heldout']):
        require(w['source']==expected['source'] and w['target_count']==expected['target_count']==1024,'heldout source/targets mismatch')
        require(math.isfinite(w['nll_sum']) and math.isfinite(w['nll_mean']) and math.isclose(w['nll_sum']/1024,w['nll_mean'],rel_tol=1e-10),'heldout nonfinite/normalization mismatch')
    require(sum(w['target_count'] for w in hs)==r['heldout_target_count']==65536,'heldout target count mismatch')
    nll=sum(w['nll_sum'] for w in hs)/65536
    require(math.isclose(nll,r['heldout_full_vocab_nll'],rel_tol=1e-10) and math.isclose(math.exp(nll),r['heldout_ppl'],rel_tol=1e-10),'heldout NLL/PPL mismatch')
    require(set(r['metrics'])==set(s['counts']),'benchmark metric groups mismatch')
    for task,count in s['counts'].items():
        subset=[q for q in qs if q['task']==task];metric=r['metrics'][task]
        require(len(subset)==metric['n']==count and metric['correct']==sum(q['acc'] for q in subset),'task denominator mismatch')
        require(math.isclose(metric['acc'],sum(q['acc'] for q in subset)/count) and math.isclose(metric['acc_norm'],sum(q['acc_norm'] for q in subset)/count),'task metric mismatch')
    return {'questions':400,'choices':sum(len(q['choice_logp_full_vocab_sum']) for q in qs),'heldout_windows':64,'heldout_targets':65536,'model':str(model),'suite_sha256':SUITE_SHA}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['run','freeze','cleanup']);a=p.parse_args()
 if a.action=='freeze':require(not (OUT/'source-freeze.json').exists(),'freeze exists');write(OUT/'source-freeze.json',hashes())
 elif a.action=='cleanup':
  if (OUT/'complete.json').exists():sys.exit(0)
  cleanup()
  if not (OUT/'complete.json').exists() and not (OUT/'failure.json').exists():write(OUT/'failure.json',{'status':'failed','error':'service stopped without full acceptance','no_retry':True})
 else:run()
