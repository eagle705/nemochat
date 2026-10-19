"""One bounded foreground SFT905-only GPU benchmark; never rerun prior arms."""
import sys,json,time,struct,fcntl,subprocess,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=ROOT/'results/midtraining/sft905-benchmark-gpu-01';MODEL=ROOT/'checkpoints/exports/nemochat_sft905_hf_bf16_01'
HERE=ROOT/'sft/archive/examples905/benchmark';OLD=ROOT/'midtraining/evaluation/ablation_gpu'
sys.path.insert(0,str(ROOT/'results/pretrain_followup'));from engine_helper import request,Connection
s=importlib.util.spec_from_file_location('old_controller',OLD/'controller.py');control=importlib.util.module_from_spec(s);s.loader.exec_module(control)
if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    suite=control.load_suite();assert len(suite['questions'])==400 and len(suite['heldout'])==64
    for arm in ['A','C','B']:
        report=control.read(control.OUT/(arm+'-full01/report.json'));control.validate_report(report,suite,control.MODELS[arm])
    accepted=json.loads((ROOT/'results/midtraining/sft905-examples-01/final-acceptance.json').read_text());assert accepted['source_step']==905
    locks=[]
    for path in [ROOT/'logs/gpu_execution.lock',ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock']:
        h=path.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    assert not apps.strip() and all(int(r.split(',')[1])==0 and int(r.split(',')[2])==0 for r in gpus.strip().splitlines())
    (OUT/'gpu-preflight.json').write_text(json.dumps({'time':time.time(),'gpus':gpus,'apps':apps,'locks':[str(x.name) for x in locks]},indent=2))
    for p in ['home','triton','inductor']:(OUT/p).mkdir()
    files=[OLD/'eval.py',OLD/'controller.py',HERE/'score_sft905.py',Path(__file__),control.SUITE]
    (OUT/'source-freeze.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    spec={'Image':control.IMAGE,'Cmd':['/usr/bin/timeout','--signal=TERM','--kill-after=30','1800','python','-u',str(HERE/'score_sft905.py'),'score','--model',str(MODEL),'--suite',str(control.SUITE),'--out',str(OUT/'D-full01')],'Entrypoint':[],'WorkingDir':str(ROOT),'User':'1001:1001','Env':['CUDA_VISIBLE_DEVICES=0','NVIDIA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','TOKENIZERS_PARALLELISM=false','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8',f'HOME={OUT}/home',f'HF_HOME={OUT}/home/hf',f'TRITON_CACHE_DIR={OUT}/triton',f'TORCHINDUCTOR_CACHE_DIR={OUT}/inductor','TMPDIR=/nct','TMP=/nct','TEMP=/nct'],'HostConfig':{'Binds':[f'{ROOT}:{ROOT}:ro',f'{OUT}:{OUT}:rw'],'NetworkMode':'none','ReadonlyRootfs':True,'NanoCpus':4000000000,'Memory':137438953472,'ShmSize':1073741824,'Tmpfs':{'/nct':'rw,exec,nosuid,nodev,size=4g,mode=1777'},'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges'],'DeviceRequests':[{'Driver':'nvidia','Count':0,'DeviceIDs':['0'],'Capabilities':[['gpu']],'Options':{}}]},'Labels':{'nemochat.task':'sft905-benchmark-gpu-01'}}
    cid=request('POST','/containers/create?name=nemochat-sft905-benchmark-gpu-01',spec)['Id']
    (OUT/'engine.create.json').write_text(json.dumps({'id':cid,'spec':spec},indent=2))
    try:
        request('POST',f'/containers/{cid}/start');deadline=time.monotonic()+1860
        while True:
            state=request('GET',f'/containers/{cid}/json')['State']
            if not state['Running']:break
            if time.monotonic()>deadline:raise TimeoutError('bounded benchmark runner deadline')
            time.sleep(2)
    finally:
        live=request('GET',f'/containers/{cid}/json')['State']
        if live['Running']:request('POST',f'/containers/{cid}/stop?t=10')
        actual=request('GET',f'/containers/{cid}/json');state=actual['State']
        conn=Connection('localhost',timeout=120);conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1');raw=conn.getresponse().read();conn.close();parts=[]
        while len(raw)>=8:
            n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
        text=b''.join(parts).decode(errors='replace');(OUT/'engine.log').write_text(text)
        (OUT/'engine.final.json').write_text(json.dumps(actual,indent=2));print(text[-3000:]);print(json.dumps(state))
    control.clean_engine(actual,cid);control.validate_devices(actual['HostConfig']['DeviceRequests'],spec['HostConfig']['DeviceRequests'])
    report=control.read(OUT/'D-full01/report.json');coverage=control.validate_report(report,suite,MODEL)
    for q in report['questions']:assert control.read(OUT/'D-full01'/(q['id'].replace('/','__')+'.json'))==q
    assert {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}==json.loads((OUT/'source-freeze.json').read_text())
    (OUT/'complete.json').write_text(json.dumps({'status':'complete_D_final_SFT905_same_frozen_raw_prompt_suite','source':accepted['source'],'source_step':905,'initial_source_C':accepted['initial_source_C'],'engine_id':cid,'state':state,'coverage':coverage,'report':str(OUT/'D-full01/report.json'),'tokenizer_guard_adapter_receipt':str(OUT/'tokenizer-compatibility.json'),'scoring_logic':'original eval.py score/pair/likelihood/aggregation unchanged; migrated full-vocab guard adapter only','chat_template_applied':False,'source_payload_integrity':'not_checked_explicit_user_waiver','A_C_B_rerun':False},indent=2));print('ACCEPTED',coverage)
