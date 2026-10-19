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
    require(ready['holder_unit']=='nemochat-sft-continue1500-holder.service','wrong holder unit')
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

def validate_native_release(r,c,host=False):
    require(r.get('status') in ('parent_accepted_continue1500','continue1500_verified_parent_release'),'parent continuation release missing')
    require(r.get('input_mode')=='sealed_scm_rights' and r.get('resume_checkpoint')==__import__('contract').CHECKPOINT,'native905 full-state RAM-only release required')
    require(r.get('checkpoint_integrity')=='not_checked_explicit_user_waiver' and r.get('full_state_resume') is True,'full-state checkpoint waiver contract drift')
    require(r.get('manifest_sha256')==c.MANIFEST_SHA,'manifest release changed')
    freeze=c.OUT/'source-freeze.json'
    require(c.sha(freeze)==r.get('source_freeze_sha256'),'reviewed runtime freeze changed')
    frozen=c.read(freeze)
    require(set(frozen)==set(c.source_paths()),'runtime source allowlist changed')
    for path,digest in frozen.items():require(c.sha(c.ROOT/path)==digest,'frozen runtime source changed: '+path)
    ready_path=c.ROOT/'results/midtraining/sft-continue1500-snapshot-02/READY.sealed.json'
    require(r.get('ready_path')==str(ready_path) and c.sha(ready_path)==r.get('ready_sha256'),'fresh actual READY binding required')
    ready=validate_ready(c.read(ready_path),c,host)
    require(len(ready['files'])==25 and r.get('snapshot_id')==ready['snapshot_id'] and r.get('holder_identity')==ready['holder_identity'],'fresh25-file holder/snapshot identity mismatch')
    require(r.get('parent_data_transport_received') is True and r.get('parent_identical_authenticated_data_audits_accepted') is True,'parent fresh transport and authenticated input acceptance required')
    require(r.get('producer_source_freeze_sha256')==ready['producer_source_freeze_sha256'],'producer freeze READY mismatch')
    path=r['producer_source_freeze_path'];producer=c.read(path)
    records=producer['files']
    import hashlib
    digest=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    require(producer.get('sha256')==digest==r['producer_source_freeze_sha256'],'producer canonical freeze changed')
    for path,expected in records.items():
        require(path.startswith(('sft/archive/snapshot/','sft/archive/sealed_snapshot03/')) and c.sha(c.ROOT/path)==expected,'producer source changed: '+path)
    return ready

def validate_parent_release(path,c):
    r=c.read(path)
    require(r.get('schema')=='nemochat-native905-continue1500-parent-release-v1' and r.get('status')=='parent_accepted_continue1500','new parent review required')
    validate_native_release(r,c,True)
    guard=c.OUT/'config-sdk03/actual-post-wrapper-guard.json'
    regression=c.OUT/'config-sdk03/source-guard-regression.json'
    require(r.get('source_guard_receipt_sha256')==c.sha(guard) and r.get('source_guard_regression_sha256')==c.sha(regression),'CPU post-wrapper source proof changed')
    require(c.read(guard).get('unchanged_sdk_guard_invoked') is True and c.read(regression).get('status')=='PINNED_ACTUAL_SOURCE_GUARD_REGRESSIONS_PASS','real unchanged native SDK guard proof required')
    require(time.time()<c.TRAIN_STOP,'absolute training stop exhausted')
    return r
