#!/usr/bin/env python3
"""Parent-released native SFT runtime05. Preview has no external side effects."""
from __future__ import annotations
import argparse
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT/'sft/archive/runtime05'
OUT = ROOT/'results/midtraining/sft-after-kd-runtime05'
SOURCE = 'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
MANIFEST = ROOT/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json'
MANIFEST_SHA = '78ab81fbbd9c0b13c55392b7d87c9e7035ad75778abad0dd3795d64121125c21'
DEADLINE = datetime.fromisoformat('2026-10-19T12:00:00+09:00').timestamp()
TRAIN_STOP = DEADLINE-1800
UNIT_PREFIX = 'nemochat-sft-sealed03-r05-'
OWNER = 'nemochat.sft.runtime05.owner'
IMAGE = 'nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
CAPACITY = 34135
FILES = {'train-tokens.npy','train-masks.npy','train-dialogues.jsonl','heldout-tokens.npy','heldout-masks.npy','heldout-dialogues.jsonl','source-schema-rejections.jsonl'}

def require(value, message):
    if not value:
        raise PermissionError(message)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    partial.replace(path)

def validate_release(path):
    from readiness import validate_parent_release
    return validate_parent_release(path,sys.modules[__name__])


def source_paths():
    return sorted(str(p.relative_to(ROOT)) for p in HERE.glob('*.py'))+[
        'sft/archive/runtime05/start.sh','sft/archive/runtime05/README.md',
        'sft/archive/runtime05/README.ko.md',
        'sft/configs/sealed03-contract.json',
        'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json',
        'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json',
        SOURCE+'/config.json',SOURCE+'/model.safetensors.index.json',
        'results/pretraining/nemochat_baby/tokenizer_migration/manifest.json',
        'sft/training/native_sft_v1.py','sft/configs/recipe-v1.yaml',
        'sft/training/controls.py','midtraining/training/native_kd.py',
        'midtraining/patches/student_embedding_import.py',
        'results/pretraining/nemochat_baby/training/runner.py','results/pretraining/nemochat_baby/training/train.yaml',
        'results/pretrain_followup/engine_helper.py']

def timing_budget(all_rows, now):
    require(len(all_rows)==8,'all eight native ranks required')
    for rows in all_rows:
        require([x['update'] for x in rows]==list(range(1,17)),'exact 16-call calibration required')
        for x in rows:
            require(x.get('native_optimizer') is True and x.get('finite_positive_gradient') is True,'actual native updates required')
            require(all(math.isfinite(x[k]) and x[k]>0 for k in ('iteration_seconds','grad_norm','lr')) and math.isfinite(x['loss']),'finite positive native calibration required')
    warm=[max(rows[i-1]['iteration_seconds'] for rows in all_rows) for i in range(9,17)]
    mean=statistics.mean(warm)
    # Includes a declared 15% periodic-save/evaluation/data overhead assumption.
    # The absolute native stop remains authoritative when projection is wrong.
    conservative=max(max(warm),mean/0.85)
    remaining=TRAIN_STOP-now-300  # fresh main model import/startup allowance
    cap=min(CAPACITY,math.floor(remaining/conservative))
    require(cap>=2,'insufficient clock for fresh main and finalization reserve')
    return dict(warm_updates=list(range(9,17)),warm_allrank_max_seconds=warm,
        warm_mean_seconds=mean,assumed_overhead_fraction=.15,
        startup_allowance_seconds=300,execution_update_cap=cap,
        native_full_epoch_updates=CAPACITY,conservative_iteration_seconds=conservative,
        training_stop_timestamp=TRAIN_STOP,save_eval_buffer_seconds=1800,
        hard_deadline='2026-10-19T12:00:00+09:00',epochs=1,allow_repetition=False,
        projection_not_guarantee=True)

def stage_config(stage, manifest, now, budget=None):
    import yaml
    require(stage in ('calibration','main'),'unknown stage')
    require(math.ceil(manifest['train_dialogues']/128)==CAPACITY,'one epoch capacity changed')
    require(now<TRAIN_STOP,'no training time before stop')
    c=yaml.safe_load((ROOT/'sft/configs/recipe-v1.yaml').read_text())
    if stage=='main':
        require(budget is not None,'actual accepted calibration budget required')
    seconds=budget['conservative_iteration_seconds'] if stage=='main' else 120
    # Native full sampler/scheduler/eval allocation never shrinks to 16 calls.
    c.update(execution_mode='calibration',max_updates=CAPACITY,max_hours=min((TRAIN_STOP-now)/3600,1.5 if stage=='calibration' else 48),
        execution_update_cap=16 if stage=='calibration' else budget['execution_update_cap'],
        runtime_stage=stage,training_stop_timestamp=TRAIN_STOP,
        conservative_iteration_seconds=seconds,launch_ready=True,
        load_hf_weights=True,calibration_receipt=str(OUT/'calibration-accepted.json') if stage=='main' else None)
    require(1<=c['execution_update_cap']<=CAPACITY,'finite one-epoch cap required')
    return c

def unit(stage):
    require(stage in ('calibration','main'),'unknown stage')
    return UNIT_PREFIX+stage+'.service'

def supervisor_argv(stage,release):
    remaining=max(1,int(DEADLINE-time.time()-120))
    python=str(ROOT/'.venv/bin/python')
    return ['systemd-run','--user','--unit='+unit(stage),'--service-type=exec',
        '--working-directory='+str(ROOT),'--property=Restart=no',
        '--property=KillMode=control-group','--property=TimeoutStopSec=120',
        '--property=RuntimeMaxSec='+str(remaining),
        '--property=ExecStopPost='+python+' '+str(HERE/'controller.py')+' --cleanup '+stage,
        '--property=StandardOutput=append:'+str(OUT/(stage+'-supervisor.log')),
        '--property=StandardError=append:'+str(OUT/(stage+'-supervisor.log')),
        python,'-u',str(HERE/'controller.py'),'--execute',stage,'--release',str(release)]

def preview(stage):
    return dict(status='INERT_PARENT_REVIEW_REQUIRED',execution_implemented=True,gpu_launches=0,input_mode='sealed_scm_rights',
        source_hf=SOURCE,model_only_fresh_optimizer_scheduler_rng_cursor=True,
        hard_deadline='2026-10-19T12:00:00+09:00',training_stop='2026-10-19T11:30:00+09:00',
        native_full_epoch_updates=CAPACITY,calibration_calls=16,warm_updates=list(range(9,17)),
        source_freeze=str(OUT/'source-freeze.json'),parent_release=str(OUT/'parent-release.json'),
        native_wrapper=str(HERE/'native.py'),recipe=str(ROOT/'sft/configs/recipe-v1.yaml'),
        supervisor_argv=supervisor_argv(stage,OUT/'parent-release.json'),
        main_from_calibration_checkpoint=False,auto_restart=False,
        checkpoint_integrity='not_checked_explicit_user_waiver')

def check_owned(actual,cid,segment):
    labels=actual.get('Config',{}).get('Labels',{})
    own={k:v for k,v in labels.items() if k.startswith('nemochat.sft.runtime05.')}
    require(actual.get('Id')==cid and actual.get('Name')=='/nemochat-sft-sealed03-r05-'+segment
        and own=={OWNER:segment} and labels.get('nemochat.sealed03')=='v1','exact runtime05 Engine ownership mismatch')

def process_env(home,threads=4):
    """Set before Python import/spawn; /nct is a container-private exec tmpfs."""
    return ['PYTHONPATH='+str(ROOT/'src')+':'+str(ROOT),'PYTHONDONTWRITEBYTECODE=1',
        'WANDB_MODE=disabled','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
        'HOME='+str(home),'TMPDIR=/nct','TMP=/nct','TEMP=/nct',
        'TORCHINDUCTOR_CACHE_DIR='+str(home/'inductor'),'TRITON_CACHE_DIR='+str(home/'triton'),
        'CUDA_CACHE_PATH='+str(home/'cuda'),'OMP_NUM_THREADS='+str(threads),'NCCL_IB_DISABLE=1']


def container_payload(segment,config,limit,config_only=False):
    require(segment in ('calibration','reload','main','config'),'unknown Engine segment')
    home=OUT/(segment+'-home')
    cmd=['--signal=TERM','--kill-after=60',str(max(1,int(limit))),'python','-m','torch.distributed.run',
         '--nnodes=1','--nproc_per_node='+('1' if config_only else '8'),
         '--master_addr=127.0.0.1','--master_port=29619',str(HERE/'native.py'),
         '--config',str(config),'--manifest',str(MANIFEST),'--output',str(OUT/segment)]
    if config_only:cmd+=['--config-only']
    elif segment=='reload':cmd+=['--reload-checkpoint',str(OUT/'calibration/checkpoint')]
    else:cmd+=['--release',str(OUT/'native-release.json')]
    # The pinned image's editable /opt/Megatron-Bridge is root-readable only.
    # Match the already executed v1 CPU config probe, without changing SDK perms.
    return dict(Image=IMAGE,User='0:0',WorkingDir=str(ROOT),
        Entrypoint=['timeout'],Cmd=cmd,Labels={OWNER:segment,'nemochat.sealed03':'v1'},
        Env=process_env(home)+[
            'NVIDIA_VISIBLE_DEVICES='+('void' if config_only else '0,1,2,3,4,5,6,7'),
            'NVIDIA_DRIVER_CAPABILITIES=compute,utility'],
        HostConfig=dict(Runtime='runc',NetworkMode='none',ReadonlyRootfs=True,Privileged=False,
            CapDrop=['ALL'],CapAdd=['DAC_OVERRIDE'],SecurityOpt=['no-new-privileges:true'],Memory=1500*1024**3,
            ShmSize=32*1024**3,PidsLimit=4096,RestartPolicy={'Name':'no'},
            DeviceRequests=[] if config_only else [dict(Driver='nvidia',Count=0,DeviceIDs=list(map(str,range(8))),Capabilities=[['gpu']],Options={})],
            Binds=[str(ROOT)+':'+str(ROOT)+':ro',str(OUT)+':'+str(OUT)+':rw']+socket_bind(config_only),
            Tmpfs={'/nct':'rw,exec,nosuid,nodev,mode=1777,size=8g'}))

def socket_bind(config_only=False):
    if config_only:return []
    r=read(OUT/'native-release.json');ready=read(r['ready_path'])
    directory=str(Path(ready['socket_path']).parent)
    require(Path(directory).is_absolute() and directory not in ('/','/run','/tmp'), 'private socket directory required')
    return [directory+':'+directory+':ro']


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]

def accept_native(output,state,c,stage):
    if stage=='main':require(time.time()<=DEADLINE,'late main cannot be accepted as on-time completion')
    require(state.get('Running') is False and state.get('ExitCode')==0 and state.get('OOMKilled') is False
        and not state.get('Error'),'clean native Engine exit required')
    output=Path(output);all_rows=[];counts=[]
    for rank in range(8):
        if stage=='reload':
            r=read(output/f'rank-{rank}-reload.json')
            require(r.get('status')=='native_full_state_reload_eval_complete' and r.get('step')==16
                and r.get('scheduler_num_steps')==2048 and r.get('optimizer_updates')==0
                and r.get('optimizer_rng_native_load_required') is True and math.isfinite(r['heldout_ce']),
                'all-rank native full-state operational reload/eval required')
            continue
        rows=jsonl(output/f'rank-{rank}-updates.jsonl');n=len(rows);counts.append(n)
        require(0<n<=c['execution_update_cap'] and [x['update'] for x in rows]==list(range(1,n+1)),
            'contiguous bounded optimizer calls required')
        if stage=='calibration':require(n==16,'calibration must complete exactly 16 optimizer calls')
        for row in rows:
            require(row.get('native_optimizer') is True and row.get('finite_positive_gradient') is True
                and math.isfinite(row['loss']) and all(math.isfinite(row[k]) and row[k]>0 for k in ('lr','grad_norm','iteration_seconds')),
                'all-rank finite positive native optimizer evidence required')
        load=read(output/f'rank-{rank}-load.json')
        require(load.get('source')==SOURCE and load.get('stage_step')==0 and load.get('model_only') is True
            and load.get('optimizer_scheduler_rng_cursor_reset') is True,'original C model-only fresh state required')
        hf=read(output/f'rank-{rank}-hf-import.json')
        dst=read(output/f'rank-{rank}-destination-import.json')
        require(hf.get('source_key_coverage')=='all_index_keys_have_native_mappings' and hf.get('source_tensor_keys')==6289
            and hf.get('native_embedding_transform_calls',0)>0 and hf.get('native_embedding_shape')==[131072,1024],
            'all original C HF keys and actual learned embedding import required')
        require(dst.get('destination_mapping_complete') is True and dst.get('destination_required_tensors',0)>0
            and dst['destination_required_tensors']==dst.get('destination_covered_tensors')
            and dst.get('native_transform_outputs',0)>=dst['destination_required_tensors'],
            'complete actual native destination import required; no random missing keys')
        entry=read(output/f'rank-{rank}-entrypoint.json')
        require(entry.get('native_exit_code')==0 and entry.get('updates')==n,'native entrypoint/update mismatch')
        saves=read(output/f'rank-{rank}-save.json')
        require(n in saves.get('save_steps',[]) and saves.get('save_optim') is True and saves.get('save_rng') is True
            and saves.get('native_synchronous_save_completed') is True,'final native full-state save missing')
        held=jsonl(output/f'rank-{rank}-heldout.jsonl')
        require(any(x.get('update')==n and math.isfinite(x['heldout_ce']) for x in held),'final-update finite native heldout required')
        all_rows.append(rows)
    require(not counts or len(set(counts))==1,'all-rank update disagreement')
    return dict(status='native_full_state_reload_accepted' if stage=='reload' else 'native_updates_full_state_save_accepted',
        stage=stage,updates=counts[0] if counts else 0,all_ranks=8,engine_state=state,
        checkpoint_integrity='not_checked_explicit_user_waiver',all_rows=all_rows)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def engine():
    return load_module('runtime05_engine',ROOT/'results/pretrain_followup/engine_helper.py')

def owned(e,cid,segment):
    actual=e.request('GET','/containers/'+cid+'/json')
    check_owned(actual,cid,segment)
    return actual

def cleanup(stage,e=None):
    unit(stage) # validates scope, never enumerate or terminate unrelated jobs
    e=e or engine();result={}
    segments=['main'] if stage=='main' else ['calibration','reload']
    if stage=='calibration' and (OUT/'calibration-owns-main.json').exists():
        require(read(OUT/'calibration-owns-main.json').get('owner')==unit(stage),'chained cleanup owner mismatch')
        segments.append('main')
    for segment in segments:
        p=OUT/(segment+'-container.json')
        if not p.exists():
            # Recover only the deterministic *owned* name after an exclusive
            # execution claim. Covers Engine CREATE success/client-write races.
            if not (OUT/(stage+'-execution-claim.json')).exists():continue
            name='nemochat-sft-sealed03-r05-'+segment
            try:a=e.request('GET','/containers/'+name+'/json')
            except RuntimeError as exc:
                if ': 404:' in str(exc):continue
                raise
            check_owned(a,a['Id'],segment)
            write(p,dict(id=a['Id'],name=name,recovered_create_identity=True))
        cid=read(p)['id'];a=owned(e,cid,segment)
        if a['State']['Running']:
            e.request('POST','/containers/'+cid+'/stop?t=60',timeout=90)
        a=owned(e,cid,segment)
        require(a['State']['Running'] is False,'owned Engine still running after cleanup')
        result[segment]=a['State']
    write(OUT/(stage+'-cleanup.json'),result)
    return result

STOP_REQUESTED=False

def stop_requested(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True

def service_properties(stage):
    s=subprocess.run(['systemctl','--user','show',unit(stage),'-p','MainPID','-p','ActiveState','-p','ExecMainStatus'],capture_output=True,text=True,check=True)
    return dict(x.split('=',1) for x in s.stdout.splitlines() if '=' in x)

def own_service(stage,release):
    props=service_properties(stage)
    controls=load_module('runtime05_identity_controls',ROOT/'sft/training/controls.py')
    actual=controls.process_identity(os.getpid())
    expected=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'controller.py'),'--execute',stage,'--release',str(release)]
    require(props.get('MainPID')==str(os.getpid()) and actual['uid']==os.getuid()
        and actual['argv']==expected,'exact systemd MainPID/UID/full argv required')
    write(OUT/(stage+'-ownership.json'),dict(unit=unit(stage),process=actual,service=props))

def idle():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    rows=[list(map(int,x.replace(' ','').split(','))) for x in raw.splitlines() if x.strip()]
    require(len(rows)==8 and sorted(x[0] for x in rows)==list(range(8)) and not apps.strip()
        and all(x[1]<=1024 and x[2]==0 for x in rows),'all-eight GPUs must be live idle while shared lock is held')
    return dict(gpus=rows,compute_apps=apps,time=time.time())

def capture_logs(e,cid,segment):
    import struct
    conn=e.Connection('localhost',timeout=60)
    conn.request('GET','/v1.52/containers/'+cid+'/logs?stdout=1&stderr=1')
    response=conn.getresponse();raw=response.read();conn.close()
    require(response.status==200,'Engine log read failed')
    parts=[];offset=0
    while offset+8<=len(raw):
        n=struct.unpack('>I',raw[offset+4:offset+8])[0]
        require(offset+8+n<=len(raw),'Engine log framing truncated')
        parts.append(raw[offset+8:offset+8+n]);offset+=8+n
    (OUT/(segment+'-engine.log')).write_bytes(b''.join(parts))

def run_container(segment,config,limit,config_only=False):
    require(limit>120,'insufficient bounded Engine runtime')
    e=engine()
    output=OUT/segment
    output.mkdir(exist_ok=False)
    home=OUT/(segment+'-home')
    for part in ('tmp','inductor','triton','cuda'):(home/part).mkdir(parents=True,exist_ok=False)
    if not config_only:
        from readiness import validate_native_release
        validate_native_release(read(OUT/'native-release.json'),sys.modules[__name__],host=True)
        require(not STOP_REQUESTED and time.time()<DEADLINE-120,'stop/deadline prevents native release')
        write(OUT/(segment+'-gpu-idle.json'),idle())
    payload=container_payload(segment,config,limit,config_only)
    write(OUT/(segment+'-request.json'),payload)
    name='nemochat-sft-sealed03-r05-'+segment
    cid=e.request('POST','/containers/create?name='+name,payload)['Id']
    write(OUT/(segment+'-container.json'),dict(id=cid,name=name,segment=segment))
    started=time.monotonic()
    try:
        owned(e,cid,segment)
        e.request('POST','/containers/'+cid+'/start')
        a=owned(e,cid,segment)
        write(OUT/(segment+'-start-readback.json'),dict(id=cid,state=a['State']))
        while a['State']['Running']:
            write(OUT/(segment+'-engine-state.json'),dict(id=cid,state=a['State'],time=time.time()))
            if STOP_REQUESTED or time.time()>=DEADLINE-65 or time.monotonic()-started>limit+60:
                raise TimeoutError('owned native Engine stop/deadline/watchdog; never success')
            time.sleep(5)
            a=owned(e,cid,segment)
    finally:
        a=owned(e,cid,segment)
        if a['State']['Running']:
            e.request('POST','/containers/'+cid+'/stop?t=60',timeout=90)
        a=owned(e,cid,segment)
        write(OUT/(segment+'-engine-final.json'),dict(id=cid,state=a['State'],time=time.time()))
        capture_logs(e,cid,segment)
    require(a['State']['Running'] is False and a['State']['ExitCode']==0 and not a['State']['OOMKilled']
        and not a['State'].get('Error'),'native Engine did not exit cleanly')
    return a['State']

def calibration_gate():
    accepted=read(OUT/'calibration-accepted.json')
    require(accepted.get('status')=='calibration_and_native_reload_accepted','actual saved/reloaded calibration missing')
    require(accepted.get('manifest_sha256')==MANIFEST_SHA and accepted.get('source_hf')==SOURCE,'calibration lineage changed')
    require(accepted.get('source_freeze_sha256')==sha(OUT/'source-freeze.json'),'calibration producer source changed')
    c=read(OUT/'calibration-config.json')
    e=engine()
    native=accept_native(OUT/'calibration',owned(e,read(OUT/'calibration-container.json')['id'],'calibration')['State'],c,'calibration')
    accept_native(OUT/'reload',owned(e,read(OUT/'reload-container.json')['id'],'reload')['State'],c,'reload')
    return timing_budget(native['all_rows'],time.time())

def write_config(stage,c):
    import yaml
    write(OUT/(stage+'-config.json'),c)
    path=OUT/(stage+'-config.yaml')
    path.write_text(yaml.safe_dump(c,sort_keys=False))
    return path

def execute(stage,release_path):
    OUT.mkdir(parents=True,exist_ok=True)
    # State is exclusive and permanent: no implicit retry, resume or overwrite.
    claim=OUT/(stage+'-execution-claim.json')
    with claim.open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),stage=stage),f)
    lock=None
    result=None
    try:
        r=validate_release(release_path)
        own_service(stage,release_path)
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop_requested)
        lock=(ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(OUT/(stage+'-gpu-lock.json'),dict(path=lock.name,pid=os.getpid(),time=time.time()))
        require(not STOP_REQUESTED,'stop requested')
        benchmark_props=subprocess.run(['systemctl','--user','show','nemochat-offline2b-ablation-gpu-01.service','-p','MainPID','-p','ActiveState','-p','ExecMainStatus'],capture_output=True,text=True,check=True)
        props=dict(x.split('=',1) for x in benchmark_props.stdout.splitlines() if '=' in x)
        require(props.get('MainPID')=='0' and props.get('ActiveState')=='inactive' and props.get('ExecMainStatus')=='0','accepted benchmark must remain exited; never rerun')
        write(OUT/'native-release.json',dict(status='runtime05_verified_parent_release',manifest_sha256=MANIFEST_SHA,
            source_hf=SOURCE,parent_release_sha256=sha(release_path),source_freeze_sha256=sha(OUT/'source-freeze.json'),
            input_mode='sealed_scm_rights',ready_path=r['ready_path'],ready_sha256=r['ready_sha256'],
            snapshot_id=r['snapshot_id'],holder_identity=r['holder_identity'],producer_source_freeze_sha256=r['producer_source_freeze_sha256']))
        m=read(MANIFEST)
        if stage=='calibration':
            # Reserve full calibration+save+operational reload+minimal main time.
            require(TRAIN_STOP-time.time()>3*3600,'insufficient remaining clock for bounded calibration/reload/main')
            c=stage_config('calibration',m,time.time());p=write_config('calibration',c)
            state=run_container('calibration',p,min(c['max_hours']*3600+900,DEADLINE-time.time()-120))
            native=accept_native(OUT/'calibration',state,c,'calibration')
            write(OUT/'calibration-native-accepted.json',{k:v for k,v in native.items() if k!='all_rows'})
            state=run_container('reload',p,min(1800,TRAIN_STOP-time.time()-600))
            reload_result=accept_native(OUT/'reload',state,c,'reload')
            write(OUT/'calibration-accepted.json',dict(status='calibration_and_native_reload_accepted',
                source_hf=SOURCE,manifest_sha256=MANIFEST_SHA,source_freeze_sha256=sha(OUT/'source-freeze.json'),
                native={k:v for k,v in native.items() if k!='all_rows'},reload=reload_result,time=time.time()))
            if r.get('automatic_main_after_calibration') is not True:return
            # Same durable supervisor owns both jobs. Parent must explicitly bind
            # automatic main release; no learned calibration weights are used.
            write(OUT/'calibration-owns-main.json',dict(owner=unit(stage),time=time.time()))
            with (OUT/'main-execution-claim.json').open('x') as f:json.dump({'owner':unit(stage),'time':time.time()},f)
        validate_release(release_path) # READY/source revocation still blocks main
        budget=calibration_gate()
        write(OUT/'actual-timing-budget.json',budget)
        c=stage_config('main',m,time.time(),budget)
        p=write_config('main',c)
        state=run_container('main',p,DEADLINE-time.time()-90)
        result=accept_native(OUT/'main',state,c,'main')
        result.pop('all_rows')
        write(OUT/'main-accepted.json',dict(result,manifest_sha256=MANIFEST_SHA,source_hf=SOURCE,
            fresh_original_c_not_calibration=True,finished_before_deadline=time.time()<=DEADLINE,time=time.time()))
    except BaseException as exc:
        write(OUT/(stage+'-failure.json'),dict(status='failed_not_accepted',type=type(exc).__name__,message=str(exc),time=time.time()))
        raise
    finally:
        try:cleanup_result=cleanup(stage)
        finally:
            if lock is not None:lock.close()
    if result is not None:
        require(time.time()<=DEADLINE,'cleanup did not finish before hard deadline')
        write(OUT/'complete.json',dict(status='native_fresh_c_sft_saved_and_owned_cleanup_complete',
            main=result,cleanup=cleanup_result,gpu_lock_released=True,source_hf=SOURCE,
            checkpoint_integrity='not_checked_explicit_user_waiver',time=time.time()))

def start(stage,release_path):
    validate_release(release_path)
    require(not (OUT/(stage+'-execution-claim.json')).exists(),'stage previously attempted; explicit new generation required')
    if stage=='main':
        p=service_properties('calibration')
        require(p.get('MainPID')=='0' and p.get('ActiveState')=='inactive' and p.get('ExecMainStatus')=='0','calibration service/cleanup must finish before separate main')
        calibration_gate()
    # Parent starts a uniquely named user service; no explicit User=/Group=.
    subprocess.run(supervisor_argv(stage,release_path),check=True)
    props=service_properties(stage)
    require(int(props.get('MainPID','0'))>0,'new service MainPID not live')
    controls=load_module('runtime05_start_identity',ROOT/'sft/training/controls.py')
    identity=controls.process_identity(int(props['MainPID']))
    expected=[str(ROOT/'.venv/bin/python'),'-u',str(HERE/'controller.py'),'--execute',stage,'--release',str(release_path)]
    require(identity['uid']==os.getuid() and identity['argv']==expected,'started service exact UID/argv mismatch')
    write(OUT/(stage+'-service-started.json'),dict(service=props,identity=identity))
    return dict(status='supervisor_started_not_optimizer_evidence',stage=stage,MainPID=identity['pid'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    for verb in ('preview','start','execute','cleanup'):g.add_argument('--'+verb,choices=['calibration','main'])
    g.add_argument('--run',choices=['main'],help='alias for --start main')
    p.add_argument('--release',type=Path,default=OUT/'parent-release.json')
    args=p.parse_args()
    try:
        if args.preview:result=preview(args.preview)
        elif args.cleanup:result=cleanup(args.cleanup)
        elif args.execute:execute(args.execute,args.release.resolve());return
        else:result=start(args.start or args.run,args.release.resolve())
        print(json.dumps(result,indent=2,allow_nan=False))
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr)
        raise SystemExit(78)

if __name__=='__main__':
    main()
