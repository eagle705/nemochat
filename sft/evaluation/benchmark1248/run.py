"""Export final SFT1248 on CPU, then evaluate E only under cooperative locks."""
import sys,json,time,struct,fcntl,subprocess,hashlib,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=ROOT/'sft/evaluation/benchmark1248';OUT=ROOT/'results/midtraining/sft1248-benchmark-gpu-01';MODEL=ROOT/'checkpoints/exports/nemochat_sft1248_hf_bf16_01';OLD=ROOT/'midtraining/evaluation/ablation_gpu'
sys.path.insert(0,str(ROOT/'results/pretrain_followup'))
from engine_helper import request,Connection
s=importlib.util.spec_from_file_location('old_controller',OLD/'controller.py');control=importlib.util.module_from_spec(s);s.loader.exec_module(control)
LABEL={'nemochat.task':'sft1248-benchmark-gpu-01'}
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
def job(phase):
    cpu=phase=='export'
    cache=OUT/(phase+'-runtime');cache.mkdir(exist_ok=False)
    for p in ['home','triton','inductor']:(cache/p).mkdir()
    command=['python','-u',str(HERE/'convert_sft1248.py')] if cpu else ['python','-u',str(HERE/'score_sft1248.py'),'score','--model',str(MODEL),'--suite',str(control.SUITE),'--out',str(OUT/'E-full01')]
    spec={'Image':control.IMAGE,'Cmd':['/usr/bin/timeout','--signal=TERM','--kill-after=30','1800',*command],'Entrypoint':[],'WorkingDir':str(ROOT),'User':'0:0' if cpu else '1001:1001','Env':['CUDA_VISIBLE_DEVICES=' if cpu else 'CUDA_VISIBLE_DEVICES=0','NVIDIA_VISIBLE_DEVICES=void' if cpu else 'NVIDIA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','TOKENIZERS_PARALLELISM=false','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PYTHONDONTWRITEBYTECODE=1','CUBLAS_WORKSPACE_CONFIG=:4096:8',f'HOME={cache}/home',f'HF_HOME={cache}/home/hf',f'TRITON_CACHE_DIR={cache}/triton',f'TORCHINDUCTOR_CACHE_DIR={cache}/inductor','TMPDIR=/nct','TMP=/nct','TEMP=/nct'],'HostConfig':{'Binds':[f'{ROOT}:{ROOT}:ro',f'{OUT}:{OUT}:rw']+([f'{MODEL}:{MODEL}:rw'] if cpu else []),'NetworkMode':'none','ReadonlyRootfs':True,'NanoCpus':4000000000,'Memory':137438953472,'ShmSize':1073741824,'Tmpfs':{'/nct':'rw,exec,nosuid,nodev,size=4g,mode=1777'},'CapDrop':[] if cpu else ['ALL'],'SecurityOpt':['no-new-privileges'],'DeviceRequests':[] if cpu else [{'Driver':'nvidia','Count':0,'DeviceIDs':['0'],'Capabilities':[['gpu']],'Options':{}}]},'Labels':{**LABEL,'nemochat.phase':phase}}
    name='nemochat-sft1248-'+phase+'-01';cid=request('POST','/containers/create?name='+name,spec)['Id'];write(phase+'-engine.create.json',{'id':cid,'name':name,'spec':spec})
    inspection=request('GET',f'/containers/{cid}/json')
    assert inspection['Id']==cid and inspection['Name']=='/'+name and all(inspection['Config']['Labels'].get(k)==v for k,v in spec['Labels'].items())
    try:
        request('POST',f'/containers/{cid}/start');deadline=time.monotonic()+1860
        while True:
            state=request('GET',f'/containers/{cid}/json')['State']
            if not state['Running']:break
            if time.monotonic()>deadline:raise TimeoutError(phase+' bounded deadline')
            time.sleep(2)
    finally:
        live=request('GET',f'/containers/{cid}/json')['State']
        if live['Running']:request('POST',f'/containers/{cid}/stop?t=10')
        actual=request('GET',f'/containers/{cid}/json');state=actual['State']
        conn=Connection('localhost',timeout=120);conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1');raw=conn.getresponse().read();conn.close();parts=[]
        while len(raw)>=8:
            n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
        text=b''.join(parts).decode(errors='replace');(OUT/(phase+'-engine.log')).write_text(text);write(phase+'-engine.final.json',actual);print(text[-4000:],flush=True);print(json.dumps(state),flush=True)
    control.clean_engine(actual,cid)
    assert actual['HostConfig']['ReadonlyRootfs'] and actual['HostConfig']['NetworkMode']=='none'
    assert all(actual['Config']['Labels'].get(k)==v for k,v in spec['Labels'].items())
    if cpu:assert not actual['HostConfig'].get('DeviceRequests') and not actual['HostConfig'].get('Devices')
    else:control.validate_devices(actual['HostConfig']['DeviceRequests'],spec['HostConfig']['DeviceRequests'])
    return cid,actual

def main():
    suite=control.load_suite()
    baseline_paths={a:control.OUT/(a+'-full01/report.json') for a in ['A','C','B']};baseline_paths['D']=ROOT/'results/midtraining/sft905-benchmark-gpu-01/D-full01/report.json'
    baseline_bytes={a:p.read_bytes() for a,p in baseline_paths.items()}
    for arm,p in baseline_paths.items():control.validate_report(json.loads(baseline_bytes[arm]),suite,ROOT/'checkpoints/exports/nemochat_sft905_hf_bf16_01' if arm=='D' else control.MODELS[arm])
    accepted=control.read(ROOT/'results/midtraining/sft-continue1500-02/complete.json');assert accepted['final_step']==1248 and accepted['segment_updates']==343 and accepted['rank_counts']==[343]*8
    files=[OLD/'eval.py',OLD/'controller.py',HERE/'convert_sft1248.py',HERE/'score_sft1248.py',Path(__file__),ROOT/'midtraining/publication/export_kd1000/convert_kd1000.py',ROOT/'results/pretraining/nemochat_baby/hf_transformers_conversion/convert_base.py',control.SUITE]
    freeze={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};write('source-freeze.json',freeze)
    export_id,export_engine=job('export')
    export=control.read(OUT/'export_result.json');assert export['source_step']==1248 and export['source']==accepted['checkpoint'] and export['parameter_numel']==9790967416 and export['source_tensor_records']==619 and export['hf_tensor_records']==447
    locks=[]
    try:
        for path in [ROOT/'logs/gpu_execution.lock',ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock']:
            h=path.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
        gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
        assert not apps.strip() and all(int(r.split(',')[1])==0 and int(r.split(',')[2])==0 for r in gpus.strip().splitlines())
        write('gpu-preflight.json',{'time':time.time(),'gpus':gpus,'apps':apps,'locks':[str(x.name) for x in locks]})
        cid,actual=job('eval')
        report=control.read(OUT/'E-full01/report.json');coverage=control.validate_report(report,suite,MODEL)
        for q in report['questions']:assert control.read(OUT/'E-full01'/(q['id'].replace('/','__')+'.json'))==q
        assert {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}==freeze
        assert all(p.read_bytes()==baseline_bytes[a] for a,p in baseline_paths.items())
        write('evaluation-accepted.json',{'status':'accepted_E_final_SFT1248_same_frozen_raw_prompt_suite','source':accepted['checkpoint'],'source_step':1248,'resume_source_step':905,'segment_updates':343,'export_engine_id':export_id,'engine_id':cid,'state':actual['State'],'coverage':coverage,'report':str(OUT/'E-full01/report.json'),'baseline_reruns':False,'baseline_reports_unchanged':True,'source_payload_integrity':'not_checked_explicit_user_waiver','native_HF_logit_parity':'not_run','chat_template_applied':False})
    finally:
        for h in reversed(locks):fcntl.flock(h,fcntl.LOCK_UN);h.close()
    idle=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    assert not apps.strip() and all(int(r.split(',')[1])==0 and int(r.split(',')[2])==0 for r in idle.strip().splitlines())
    write('gpu-postflight.json',{'time':time.time(),'gpus':idle,'apps':apps,'locks_released':True})
    print('ACCEPTED',coverage,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:write('failure.json',{'type':type(exc).__name__,'error':str(exc)});raise
