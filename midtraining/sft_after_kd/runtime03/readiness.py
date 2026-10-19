"""Truthful sealed03 readiness; no compatibility with original buffered READY."""
import json, os, re, stat, subprocess, time
from pathlib import Path
C_MANIFEST_SHA='23c1a94b4c87942585cc7e4a4fc693653c8f3565843a91172e70545a1699dc63'
NORMALIZED=('all7_original_sft_exact_sha','all16_c_exact_sha','fresh_c_reserved_id_absence',
    'sft_masks_bounds_target_counts','sft_alias18_29_audited','tokenizer_provenance',
    'pinned_cpu_same_fd_all7_hash','readonly_sealed_descriptors')

def require(ok,message):
    if not ok:raise PermissionError(message)

def live_holder(ready,c):
    identity=ready['holder_identity'];pid=int(identity['pid'])
    props=subprocess.run(['systemctl','--user','show',ready['holder_unit'],'-p','MainPID','-p','ActiveState'],capture_output=True,text=True,check=True)
    props=dict(x.split('=',1) for x in props.stdout.splitlines() if '=' in x)
    require(props.get('ActiveState')=='active' and props.get('MainPID')==str(pid),'holder service is not exact active MainPID')
    proc=Path('/proc')/str(pid);raw=(proc/'stat').read_text().rsplit(')',1)[1].split()
    ticks=identity.get('start_ticks',identity.get('starttime'))
    require(type(ticks) is int and ('starttime' not in identity or identity['starttime']==ticks),'invalid/conflicting holder start ticks')
    require(int(raw[19])==ticks and proc.stat().st_uid==identity['uid'] and raw[0] not in ('Z','X'),'dead/reused holder identity')
    if 'argv' in identity:require((proc/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')==identity['argv'],'holder argv changed')
    socket=Path(ready['socket_path'])
    require(socket.is_absolute() and stat.S_ISSOCK(socket.stat().st_mode),'holder socket is absent or not UNIX socket')
    return True

def validate_ready(ready,c,host):
    require(ready.get('schema')=='nemochat-sealed-full03-v1' and ready.get('status')=='accepted_sealed_snapshot_full03','RAM-only sealed READY required')
    for key in ('snapshot_id','holder_identity','holder_unit','socket_path','files','audits','provenance','producer_source_freeze_sha256','cpu_consumer_receipt'):
        require(bool(ready.get(key)),'incomplete sealed READY: '+key)
    require(ready['holder_unit']=='nemochat-sft-sealed03-holder.service','wrong holder unit')
    require(ready['sft_manifest_sha256']==c.MANIFEST_SHA and ready['c_manifest_sha256']==C_MANIFEST_SHA,'sealed manifest bindings changed')
    require(c.sha(c.MANIFEST)==c.MANIFEST_SHA,'original SFT manifest metadata drift')
    manifest=c.read(c.MANIFEST)
    require(len(manifest['files'])==7 and set(manifest['files'])==c.FILES,'exact original seven-file inventory required')
    files=ready['files']
    for name,info in manifest['files'].items():
        fid='sft:'+name
        require(fid in files,'original SFT descriptor missing')
        require(files[fid]['sha256']==info['sha256'] and files[fid]['bytes']==info['bytes'],'snapshot differs from immutable expected SFT bytes')
    cmanifest=c.ROOT/'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json'
    require(c.sha(cmanifest)==C_MANIFEST_SHA,'original C manifest metadata drift')
    cm=c.read(cmanifest);expected_c=set()
    for source,splits in cm['sources'].items():
        for split,entry in splits.items():
            for suffix in ('bin','idx'):
                fid=f'c:{source}/{split}/corpus.{suffix}';expected_c.add(fid)
                require(fid in files and files[fid]['sha256']==entry[suffix+'_sha256'] and files[fid]['bytes']==entry[suffix+'_bytes'],'C snapshot descriptor missing/changed: '+fid)
    require(len(expected_c)==16 and {fid for fid in files if fid.startswith('c:')}==expected_c,'all sixteen exact C descriptors required')
    for fid,info in files.items():
        require(info.get('file_id')==fid and info.get('seals')==15 and info.get('access_mode')=='O_RDONLY','descriptor identity/seals/access mismatch')
        require(isinstance(info.get('bytes'),int) and info['bytes']>0 and re.fullmatch('[0-9a-f]{64}',info.get('sha256','')) is not None,'invalid descriptor byte/hash metadata')
    for split in ('train','heldout'):
        n=manifest[split+'_dialogues'];seq=manifest['sequence_length']
        require(seq==16384 and n=={'train':4369280,'heldout':45090}[split],'full03 row count/context changed')
        for name,shape,dtypes in [('tokens',[n,seq+1],('uint32','<u4')),('masks',[n,seq],('uint8','|u1'))]:
            info=files[f'sft:{split}-{name}.npy']
            require(info.get('shape')==shape and info.get('dtype') in dtypes,'sealed array role/header mismatch')
        info=files['rowhash:'+split]
        require(info.get('shape')==[n,32] and info.get('dtype') in ('uint8','|u1') and info['bytes']==n*32,'sealed row catalog mismatch')
    if host:live_holder(ready,c)
    return ready

def validate_parent_release(path,c):
    r=c.read(path)
    require(r.get('schema')=='nemochat-sealed-runtime03-parent-release-v1' and r.get('status')=='parent_accepted_runtime03','new parent runtime03 release required')
    require(r.get('input_mode')=='sealed_scm_rights','original/buffered/direct input forbidden')
    require(r.get('source_hf')==c.SOURCE,'fresh original C-only source required')
    freeze=c.OUT/'source-freeze.json'
    require(r.get('source_freeze_sha256')==c.sha(freeze),'parent must bind exact runtime03 freeze')
    frozen=c.read(freeze)
    require(set(frozen)==set(c.source_paths()),'runtime source allowlist changed')
    for name,digest in frozen.items():require(c.sha(c.ROOT/name)==digest,'reviewed runtime source changed: '+name)
    path=Path(r['ready_path'])
    require(path.is_absolute() and path.name=='READY.sealed.json','exact absolute sealed READY path required')
    require(c.sha(path)==r.get('ready_sha256'),'raw sealed READY changed')
    ready=validate_ready(c.read(path),c,True)
    require(r.get('snapshot_id')==ready['snapshot_id'] and r.get('holder_identity')==ready['holder_identity'],'release snapshot/holder mismatch')
    producer_path=r['producer_source_freeze_path']
    producer=c.read(producer_path)
    if isinstance(producer,dict) and 'files' in producer:
        import hashlib
        records=producer['files']
        require(isinstance(records,dict) and bool(records),'empty producer source inventory')
        for name,digest in records.items():
            require(name.startswith('midtraining/sft_after_kd/sealed_snapshot03/') and c.sha(c.ROOT/name)==digest,'producer source changed: '+name)
        producer_digest=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(producer.get('sha256')==producer_digest,'producer canonical source digest changed')
    else:
        producer_digest=c.sha(producer_path)
    require(r.get('producer_source_freeze_sha256')==ready['producer_source_freeze_sha256']==producer_digest,'producer freeze changed')
    normalized=r['normalization']
    require(normalized.get('ready_sha256')==r['ready_sha256'],'normalization not bound to raw sealed READY')
    for key in NORMALIZED:require(normalized.get(key) is True,'parent CPU proof missing: '+key)
    require(r.get('benchmark_acceptance_sha256')==c.sha(c.ROOT/'results/midtraining/offline2b-ablation-gpu-01/parent-final-acceptance.json'),'all three completed benchmark acceptance changed')
    require(r.get('checkpoint_integrity')=='not_checked_explicit_user_waiver','checkpoint byte waiver must remain explicit')
    proof_path=c.OUT/'actual-sealed-cpu-proof.json'
    require(r.get('actual_cpu_adapter_receipt_sha256')==c.sha(proof_path),'parent must bind real producer-client CPU adapter receipt')
    proof=c.read(proof_path)
    require(proof.get('status')=='ACTUAL_SEALED_CLIENT_CPU_ADAPTER_PASS' and proof.get('ready_sha256')==r['ready_sha256'] and proof.get('snapshot_id')==r['snapshot_id'] and proof.get('holder_identity')==r['holder_identity'] and proof.get('runtime_source_freeze_sha256')==r['source_freeze_sha256'],'actual CPU adapter lineage mismatch')
    ranks=proof.get('ranks',[])
    require(len(ranks)==8 and [x.get('rank') for x in ranks]==list(range(8)) and all(x.get('producer_client')=='REAL_PUBLIC_CLIENT' and x.get('received_fds')==7 and x.get('ready_sha256')==r['ready_sha256'] and x.get('status')=='ACTUAL_SEALED_CLIENT_CPU_ADAPTER_PASS' for x in ranks),'all eight real producer-client CPU adapter ranks required')
    actual=c.engine().request('GET','/containers/'+proof['container_id']+'/json')
    require(actual.get('Id')==proof['container_id'] and actual.get('Name')=='/nemochat-sft-sealed03-actual-cpu' and actual['Config'].get('Image')==c.IMAGE and actual['Config'].get('Labels',{}).get(c.OWNER)=='actual-sealed-cpu' and actual['Config'].get('Labels',{}).get('nemochat.sealed03')=='v1','real CPU adapter exact Engine identity mismatch')
    require(not actual['HostConfig'].get('DeviceRequests') and actual['State'].get('Running') is False and actual['State'].get('ExitCode')==0 and actual['State'].get('OOMKilled') is False,'real CPU adapter clean no-device Engine exit required')
    require(type(r.get('automatic_main_after_calibration')) is bool,'explicit automatic main policy required')
    require(time.time()<c.TRAIN_STOP,'absolute training stop exhausted')
    return r

def validate_native_release(r,c,host=False):
    require(r.get('status')=='runtime03_verified_parent_release' and r.get('input_mode')=='sealed_scm_rights','native RAM-only release missing')
    require(r.get('manifest_sha256')==c.MANIFEST_SHA==c.sha(c.MANIFEST) and r.get('source_hf')==c.SOURCE,'native source/manifest drift')
    require(r.get('source_freeze_sha256')==c.sha(c.OUT/'source-freeze.json'),'native runtime freeze changed')
    for path,digest in c.read(c.OUT/'source-freeze.json').items():require(c.sha(c.ROOT/path)==digest,'native source changed: '+path)
    require(c.sha(r['ready_path'])==r.get('ready_sha256'),'native raw READY changed')
    ready=validate_ready(c.read(r['ready_path']),c,host)
    require(r.get('snapshot_id')==ready['snapshot_id'] and r.get('holder_identity')==ready['holder_identity'] and r.get('producer_source_freeze_sha256')==ready['producer_source_freeze_sha256'],'native snapshot/holder/producer mismatch')
    return ready
