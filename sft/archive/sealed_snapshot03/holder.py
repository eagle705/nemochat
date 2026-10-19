"""Bounded direct-first sealed RAM capture; no GPU, no source writes."""
import argparse
import array
import concurrent.futures
from datetime import datetime
import fcntl
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import signal
import socket
import struct
import threading
import time
import traceback
import uuid
import numpy as np
try:
    from .client import CHUNK, digest_fd, require, row_digest, request_header, OpenedSnapshotFile, validate_descriptor
except ImportError:
    from client import CHUNK, digest_fd, require, row_digest, request_header, OpenedSnapshotFile, validate_descriptor

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT/'sft/archive/sealed_snapshot03'
OUT = ROOT/'results/midtraining/sft-after-kd-sealed03'
CONTRACT = ROOT/'sft/configs/sealed03-contract.json'


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n'); pending.replace(path)

def stat(path):
    s = os.stat(path)
    return {k: getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def seal_readonly(fd):
    fcntl.fcntl(fd, 1033, 15)
    ro = os.open('/proc/self/fd/'+str(fd), os.O_RDONLY|os.O_CLOEXEC)
    require(fcntl.fcntl(ro,1034) == 15, 'seals')
    return ro

def vote_block(a,b,c):
    require(len(a) == len(b) == len(c), 'vote_size')
    pages = (len(a)+4095)//4096
    if a == b == c:
        return a, {'unanimous':pages,'majority':0,'tie':0,'changed':0}
    selected = bytearray(a); counts = {'unanimous':0,'majority':0,'tie':0,'changed':0}
    for p in range(0,len(a),4096):
        x,y,z = a[p:p+4096], b[p:p+4096], c[p:p+4096]
        if x == y == z: v,k = x,'unanimous'
        elif x == y or x == z: v,k = x,'majority'
        elif y == z: v,k = y,'majority'
        else: v,k = x,'tie'
        counts[k] += 1
        if v != x: selected[p:p+len(v)] = v; counts['changed'] += 1
    return selected, counts

def capture(path, expected, report, progress=None):
    size = expected['bytes']; before = stat(path)
    require(before['st_size'] == size, 'source_size')
    report.update(path=str(path), expected=expected, source_before=before, views=[])
    candidate = os.memfd_create('sealed03-'+Path(path).name, os.MFD_ALLOW_SEALING|os.MFD_CLOEXEC)
    ro = None
    try:
        os.ftruncate(candidate,size)
        h = hashlib.sha256()
        view = {'kind':'initial_O_DIRECT_to_memfd','before':stat(path),'bytes':0}
        report['views'].append(view)
        fd = os.open(path,os.O_RDONLY|os.O_DIRECT|os.O_CLOEXEC)
        try:
            with mmap.mmap(-1,CHUNK) as aligned:
                for offset in range(0,size,CHUNK):
                    wanted = min(CHUNK,size-offset)
                    got = os.preadv(fd,[aligned],offset)
                    require(got == wanted,'direct_short_read')
                    data = aligned[:got]; h.update(data)
                    require(os.pwrite(candidate,data,offset) == wanted,'memfd_short_write')
                    view['bytes'] += got
                    if progress and offset % (1024**3) < CHUNK: progress(offset+got,size)
        finally: os.close(fd)
        view.update(after=stat(path),sha256=h.hexdigest())
        require(before == view['before'] == view['after'],'source_stat_changed')
        initial = digest_fd(candidate,size)
        report['initial_memfd_readback_sha256'] = initial
        require(initial == h.hexdigest(),'initial_memfd_unstable')
        if initial != expected['sha256']:
            hashes = [hashlib.sha256(), hashlib.sha256()]
            views = [{'kind':kind,'before':stat(path),'bytes':0} for kind in ('buffered_observation','fresh_O_DIRECT_observation')]
            report['views'].extend(views)
            counts = {k:0 for k in ('unanimous','majority','tie','changed')}
            buffered = os.open(path,os.O_RDONLY|os.O_CLOEXEC)
            direct = os.open(path,os.O_RDONLY|os.O_DIRECT|os.O_CLOEXEC)
            try:
                with mmap.mmap(-1,CHUNK) as aligned:
                    for offset in range(0,size,CHUNK):
                        wanted = min(CHUNK,size-offset)
                        a = os.pread(candidate,wanted,offset)
                        b = os.pread(buffered,wanted,offset)
                        got = os.preadv(direct,[aligned],offset); c = aligned[:got]
                        require(len(a) == len(b) == got == wanted,'vote_short_read')
                        for h,v,meta in zip(hashes,(b,c),views): h.update(v); meta['bytes'] += len(v)
                        selected, delta = vote_block(a,b,c)
                        for k,v in delta.items(): counts[k] += v
                        if delta['changed']: require(os.pwrite(candidate,selected,offset) == wanted,'vote_write')
                        if progress and offset % (1024**3) < CHUNK: progress(offset+got,size,'vote')
            finally: os.close(buffered); os.close(direct)
            for h,meta in zip(hashes,views):
                meta.update(after=stat(path),sha256=h.hexdigest())
                require(meta['before'] == meta['after'] == before,'source_stat_changed')
            report['page_votes'] = counts
        ro = seal_readonly(candidate)
        report.update(sealed_readback_sha256=digest_fd(ro,size),seals=fcntl.fcntl(ro,1034),source_after=stat(path))
        require(report['source_after'] == before,'source_stat_changed')
        require(report['sealed_readback_sha256'] == expected['sha256'],'whole_sha256')
        report['accepted'] = True
        os.close(candidate); candidate = None
        result = ro; ro = None
        return result
    except BaseException:
        report['accepted'] = False
        report['all_descriptors_closed_on_failure'] = True
        raise
    finally:
        if candidate is not None: os.close(candidate)
        if ro is not None: os.close(ro)

def process_identity(pid=None):
    pid = pid or os.getpid()
    status = Path(f'/proc/{pid}/status').read_text()
    return {'pid':pid,'uid':int(next(x for x in status.splitlines() if x.startswith('Uid:')).split()[1]),'start_ticks':int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'argv':Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0') and [x.decode() for x in Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').split(b'\0')]}

def freeze_source():
    records = {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in sorted(CODE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    return {'files':records,'sha256':sha(json.dumps(records,sort_keys=True,separators=(',',':')).encode())}

def inventory(contract):
    manifests = {}
    for role in ('sft','c'):
        data = (ROOT/contract[role+'_manifest']).read_bytes()
        require(sha(data) == contract[role+'_manifest_sha256'],role+'_manifest_sha256')
        manifests[role] = json.loads(data)
    jobs = []
    sft = manifests['sft']; corpus = manifests['c']
    for name,e in sft['files'].items():
        jobs.append({'file_id':'sft:'+name,'path':str((ROOT/contract['sft_manifest']).parent/name),'expected':e})
    for source,splits in corpus['sources'].items():
        for split,e in splits.items():
            for ext in ('bin','idx'):
                jobs.append({'file_id':f'c:{source}/{split}/corpus.{ext}','path':e['prefix']+'.'+ext,'expected':{'bytes':e[ext+'_bytes'],'sha256':e[ext+'_sha256']}})
    require(len(jobs) == contract['all_original_file_count_including_c'] == 23,'inventory_count')
    require(sum(x['expected']['bytes'] for x in jobs) == contract['all_original_bytes_including_c'],'inventory_bytes')
    for job in jobs: job['source_stat'] = stat(job['path'])
    return manifests, jobs

def authenticate_provenance():
    priorpath = ROOT/'results/midtraining/sft-after-kd-data-acceptance-01/provenance.json'
    data = priorpath.read_bytes(); p = json.loads(data)
    require(p['status'] == 'provenance_authenticated' and p['producer_engine_clean_exit'] and p['producer_source_current_matches_historical_handoff'],'historical_provenance')
    for r in list(p['records'].values()) + list(p['tokenizer_assets'].values()):
        b = Path(r['path']).read_bytes()
        require(len(b) == r['bytes'] and sha(b) == r['sha256'],'provenance_file:'+r['path'])
        if 'historical_expected_sha256' in r: require(sha(b) == r['historical_expected_sha256'],'producer_source')
    base = ROOT/'results/pretraining/nemochat_baby/nemotron_sft/cpu-full-03'
    engine = json.loads((base/'full-preparation.final-state.json').read_text())
    create = json.loads((base/'full-preparation.create.json').read_text())
    wait = json.loads((base/'full-preparation.wait.json').read_text())
    require(engine['Id'] == create['Id'] == p['producer_id'] and engine['State']['ExitCode'] == wait['StatusCode'] == 0 and not engine['State']['Running'] and not engine['State']['OOMKilled'],'producer_exit')
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from nemochat.tokenization.posttraining import compare_tokenizers
    assets = p['tokenizer_assets']
    old = json.loads(Path(assets['old/tokenizer.json']['path']).read_bytes())
    new = json.loads(Path(assets['SFT/tokenizer.json']['path']).read_bytes())
    comparison = compare_tokenizers(old,new)
    require(json.loads(Path(assets['C-corpus/tokenizer.json']['path']).read_bytes()) == new,'corpus_tokenizer')
    require(json.loads(Path(assets['C-export-original/tokenizer.json']['path']).read_bytes()) == old,'export_tokenizer')
    return {'status':'authenticated_historical_chain_rechecked','receipt_sha256':sha(data),'prior_receipt':str(priorpath),'producer_id':engine['Id'],'producer_exit_code':0,'tokenizer_comparison':comparison,'mask_lineage':p['mask_lineage'],'original_counted_audit':p['original_counted_audit'],'checkpoint_integrity':'not_checked_explicit_user_waiver'}

def file_metadata(file_id, expected, fd):
    meta = dict(file_id=file_id,bytes=expected['bytes'],sha256=expected['sha256'],expected_original_bytes=expected['bytes'],expected_original_sha256=expected['sha256'],seals=15,access_mode='O_RDONLY',shape=None,dtype=None)
    if file_id.endswith('.npy'):
        b = io.BytesIO(os.pread(fd,65536,0)); version = np.lib.format.read_magic(b)
        require(version in ((1,0),(2,0)),'npy_version')
        shape,fortran,dtype = (np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0)(b)
        require(not fortran,'fortran')
        meta.update(shape=list(shape),dtype=str(dtype))
    validate_descriptor(fd,meta)
    return meta

def audit_array(obj, kind, expected_shape, expected_targets=None):
    a = obj.array
    require(list(a.shape) == expected_shape and a.dtype == np.dtype('<u4' if kind == 'tokens' else 'u1'),'audit_shape_dtype')
    r = {'authenticated':True,'shape':list(a.shape),'dtype':str(a.dtype),'min':int(a.min()),'max':int(a.max()),'reserved_counts':{str(i):0 for i in range(18,30)},'targets':0,'out_of_bounds':0,'invalid_masks':0}
    flat = a.reshape(-1)
    for start in range(0,flat.size,4_000_000):
        x = flat[start:start+4_000_000]
        if kind == 'tokens':
            r['out_of_bounds'] += int(np.count_nonzero(x >= 131072))
            vals,counts = np.unique(x[(x>=18)&(x<=29)],return_counts=True)
            for v,c in zip(vals,counts): r['reserved_counts'][str(int(v))] += int(c)
        else:
            r['invalid_masks'] += int(np.count_nonzero(x>1)); r['targets'] += int(x.sum(dtype=np.uint64))
    require(not r['out_of_bounds'] and not r['invalid_masks'],'array_semantics')
    if expected_targets is not None: require(r['targets'] == expected_targets,'target_count')
    return r

def audit_index(fd, bin_meta, expected):
    data = os.pread(fd,os.fstat(fd).st_size,0)
    require(data[:9] == b'MMIDIDX\0\0' and struct.unpack_from('<Q',data,9)[0] == 1 and data[17] == 4,'idx_header_dtype')
    ns,nd = struct.unpack_from('<QQ',data,18)
    require(ns == expected['documents'] and nd == ns+1 and len(data) == 34+12*ns+8*nd,'idx_counts_size')
    sizes = np.frombuffer(data,'<i4',count=ns,offset=34)
    ptr = np.frombuffer(data,'<i8',count=ns,offset=34+4*ns)
    docs = np.frombuffer(data,'<i8',count=nd,offset=34+12*ns)
    require(bool(np.all(sizes > 0)) and int(sizes.sum(dtype=np.int64)) == expected['tokens'] and expected['tokens']*4 == bin_meta['bytes'],'idx_bin_link')
    require(bool(np.array_equal(ptr,np.cumsum(np.r_[0,sizes[:-1]],dtype=np.int64)*4)),'idx_pointers')
    require(bool(np.array_equal(docs,np.arange(ns+1,dtype=np.int64))),'idx_document_boundaries')
    return {'authenticated':True,'magic':'MMIDIDX-NUL-NUL','version':1,'dtype':'int32','dtype_code':4,'sequences':ns,'document_boundaries':nd,'tokens':expected['tokens'],'all_pointers_and_document_boundaries_valid':True,'bin_sha256':bin_meta['sha256'],'format_source':'results/pretraining/nemochat_baby/climbmix/corpus_sdk.py'}

def audit_c(fd, expected):
    low,high,bad = 131072,0,0; reserved = {str(i):0 for i in range(18,30)}
    for offset in range(0,expected['bin_bytes'],CHUNK):
        data = os.pread(fd,min(CHUNK,expected['bin_bytes']-offset),offset)
        x = np.frombuffer(data,'<i4'); low=min(low,int(x.min()));high=max(high,int(x.max()))
        bad += int(np.count_nonzero((x<0)|(x>=131072)))
        vals,counts=np.unique(x[(x>=18)&(x<=29)],return_counts=True)
        for v,c in zip(vals,counts):reserved[str(int(v))] += int(c)
    require(not bad and not sum(reserved.values()),'C_bounds_alias_absence')
    return {'authenticated':True,'min':low,'max':high,'out_of_bounds':bad,'reserved_counts':reserved,'all_bin_bytes_scanned':expected['bin_bytes']}

def audit_dialogues(fd, split, manifest):
    r = dict(rows=0,targets=0,nonpadding_tokens=0,per_source={},selected_assistant_messages=0,context_only_assistant_messages=0,structured_tool_calls=0,authenticated=True)
    excluded = {x.removeprefix('nvidia/') for x in manifest['excluded_datasets']}
    with os.fdopen(os.dup(fd),'rb',buffering=CHUNK) as stream:
        stream.seek(0)
        for line in stream:
            row = json.loads(line)
            source = row['source_key']; require(source.split('--',1)[0] not in excluded,'excluded_source')
            r['rows'] += 1; r['targets'] += row['supervised_tokens']; r['nonpadding_tokens'] += row['nonpadding_tokens']
            r['per_source'][source] = r['per_source'].get(source,0)+1
            require(len(row['messages']) == len(row['selected']),'selection_alignment')
            r['selected_assistant_messages'] += sum(row['selected'])
            r['context_only_assistant_messages'] += sum(m['role']=='assistant' and not selected for m,selected in zip(row['messages'],row['selected']))
            r['structured_tool_calls'] += sum(len(m.get('tool_calls') or []) for m in row['messages'])
    require(r['rows'] == manifest[split+'_dialogues'] and r['targets'] == manifest[split+'_loss_tokens'] and r['nonpadding_tokens'] == manifest['nonpadding_tokens'][split],'dialogue_counts')
    for source,count in r['per_source'].items(): require(manifest['per_source'][source][split] == count,'dialogue_source_count')
    return r

def derive_catalog(tokens, masks, label):
    n = len(tokens); size = n*32
    fd = os.memfd_create('sealed03-rowhash-'+label,os.MFD_ALLOW_SEALING|os.MFD_CLOEXEC)
    try:
        os.ftruncate(fd,size); h = hashlib.sha256()
        for start in range(0,n,4096):
            data = b''.join(row_digest(i,tokens[i],masks[i]) for i in range(start,min(n,start+4096)))
            require(os.pwrite(fd,data,start*32) == len(data),'catalog_write'); h.update(data)
        ro = seal_readonly(fd)
        try:
            require(digest_fd(ro,size) == h.hexdigest(),'catalog_readback')
            meta = {'file_id':'rowhash:'+label,'bytes':size,'sha256':h.hexdigest(),'seals':15,'access_mode':'O_RDONLY','shape':[n,32],'dtype':'uint8','derived_from':['sft:'+label+'-tokens.npy','sft:'+label+'-masks.npy'],'row_domain_hex':b'nemochat-sft-row-v1\0'.hex()}
            return ro,meta
        except BaseException: os.close(ro);raise
    finally: os.close(fd)

def base_ready(contract, snapshot_id, freeze):
    directory=Path('/run/user/1001')/('ncs03-'+snapshot_id[:12])
    directory.mkdir(mode=0o700)
    require(stat(directory)['st_size']>=0 and directory.stat().st_uid==1001 and directory.stat().st_mode & 0o777 == 0o700,'socket_dir')
    return {'schema':contract['ready_schema'],'status':contract['ready_status'],'snapshot_id':snapshot_id,'sft_manifest_sha256':contract['sft_manifest_sha256'],'c_manifest_sha256':contract['c_manifest_sha256'],'holder_identity':process_identity(),'holder_unit':contract['holder_unit'],'socket_path':str(directory/'s'),'files':{},'audits':{},'provenance':{},'producer_source_freeze_sha256':freeze,'input_mode':'sealed_scm_rights','original_mmap_inputs_blocked':True,'checkpoint_integrity':'not_checked_explicit_user_waiver','holder_not_after':contract['holder_not_after']}

def fixture():
    try:from .transport import Server,consume
    except ImportError:from transport import Server,consume
    contract=json.loads(CONTRACT.read_bytes())
    ready=base_ready(contract,uuid.uuid4().hex,'fixture-not-production')
    fds={};server=None;report={'schema':'nemochat-sealed03-genuine-fixture-v1','status':'BLOCKED','tests':{}}
    try:
        for split in ('train','heldout'):
            for kind,dtype,width in [('tokens','<u4',4),('masks','u1',3)]:
                a=np.arange(12 if kind=='tokens' else 9,dtype=dtype).reshape(3,width)
                if kind=='masks':a[:]=a%2
                b=io.BytesIO();np.save(b,a,allow_pickle=False);data=b.getvalue()
                fd=os.memfd_create('sealed03-fixture',os.MFD_ALLOW_SEALING);os.write(fd,data)
                ro=seal_readonly(fd);os.close(fd)
                file_id=f'sft:{split}-{kind}.npy';fds[file_id]=ro
                ready['files'][file_id]=file_metadata(file_id,dict(bytes=len(data),sha256=sha(data)),ro)
            for name in ([split+'-dialogues.jsonl'] + (['source-schema-rejections.jsonl'] if split=='train' else [])):
                data=b'{"genuine":"fixture"}\n';fd=os.memfd_create('sealed03-fixture-json',os.MFD_ALLOW_SEALING);os.write(fd,data)
                ro=seal_readonly(fd);os.close(fd);file_id='sft:'+name;fds[file_id]=ro
                ready['files'][file_id]=file_metadata(file_id,dict(bytes=len(data),sha256=sha(data)),ro)
        server=Server(ready,fds);server.start()
        # Reject a same-UID host process despite its valid request/header.
        with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as sock:
            sock.settimeout(20);sock.connect(ready['socket_path']);sock.send(json.dumps(request_header(ready,next(iter(fds)))).encode())
            try:
                denied=json.loads(sock.recv(65536));require('peer_not_owned_pinned_container' in denied['denied'],'host_peer_not_rejected')
            except ConnectionResetError:
                require(any('peer_not_owned_pinned_container' in x['error'] for x in server.errors),'host_reset_without_auth_rejection')
        report['tests']['host_same_uid_rejected']=True
        report['tests']['pinned_init_full_read']=consume(ready,CODE,{},timeout=120)
        report['tests']['pinned_descendant_full_read']=consume(ready,CODE,{},timeout=120,descendant_probe=True)
        wrong=dict(ready,snapshot_id='incorrect-fixture-snapshot')
        report['tests']['wrong_request_identity']=consume(wrong,CODE,{},timeout=120,negative='header_identity')
        report['peer_receipts']=server.receipts
        require(any(x['peer_pid']!=x['container_init_pid'] for x in server.receipts),'descendant_not_exercised')
        require(all(r['cleanup_verified_absent'] for r in report['tests'].values() if isinstance(r,dict)),'fixture_cleanup')
        report['status']='PASS'
    except BaseException as error:
        report['error']=repr(error);report['traceback']=traceback.format_exc();raise
    finally:
        if server:server.close()
        for fd in fds.values():os.close(fd)
        Path(ready['socket_path']).parent.rmdir()
        report['all_fixture_fds_closed']=True
        save(OUT/'FIXTURE.json',report)
    return report

def memory_available():
    return int(next(line for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')).split()[1])*1024

def cgroup_metrics():
    try:
        rel=next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')).split('::',1)[1]
        p=Path('/sys/fs/cgroup')/rel.lstrip('/')
        return {'path':str(p),**{name:(p/name).read_text().strip() for name in ('memory.current','memory.peak','memory.max','memory.high','memory.events','cpu.max','cpu.stat')}}
    except Exception as error:return {'error':repr(error)}

def hold(request_path):
    try:from .transport import Server,consume
    except ImportError:from transport import Server,consume
    request=json.loads(Path(request_path).read_bytes());contract_bytes=CONTRACT.read_bytes();contract=json.loads(contract_bytes)
    fds={};objects={};server=None;ready=None;published=False
    report={'schema':'nemochat-sealed03-capture-evidence-v1','status':'capturing','snapshot_id':request['snapshot_id'],'files':{},'checkpoint_integrity':'not_checked_explicit_user_waiver','checkpoint_payloads_read':False,'GPU_allocated':False,'source_writes':False,'host_repaired':False,'original_mmap_inputs_blocked':True,'holder_identity':process_identity(),'request_path':str(request_path)}
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('absolute_capture_stage_3600_seconds')
    def stop(*_):raise InterruptedError('holder_stop_or_absolute_expiry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(contract['snapshot_capture_timeout_seconds'])
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        require(sha(contract_bytes)==request['contract_sha256'],'contract_changed')
        require(freeze_source()==request['source_freeze'],'producer_source_changed')
        require(json.loads((OUT/'FIXTURE.json').read_bytes())['status']=='PASS','real_fixture_gate')
        require(sha((OUT/'FIXTURE.json').read_bytes())==request['fixture_sha256'],'fixture_changed')
        report['mem_available_before_capture']=memory_available()
        require(report['mem_available_before_capture']>=contract['minimum_mem_available_before_start_bytes'],'MemAvailable_below_1TiB')
        require(datetime.now().astimezone()<datetime.fromisoformat(contract['holder_not_after']),'expired_contract')
        provenance=authenticate_provenance();manifests,jobs=inventory(contract)
        require(jobs==request['inventory'],'source_stat_freeze_changed')
        ready=base_ready(contract,request['snapshot_id'],request['source_freeze']['sha256'])
        ready['provenance']=provenance
        report['provenance']=provenance;report['producer_source_freeze']=request['source_freeze'];report['cgroup_initial']=cgroup_metrics()
        limits=report['cgroup_initial']
        require(limits['path'].endswith('/'+contract['holder_unit']) and limits['memory.max']==str(768*1024**3) and limits['memory.high']==str(640*1024**3),'durable_holder_unit_memory_limits')
        quota,period=(int(x) for x in limits['cpu.max'].split())
        require(quota==8*period,'durable_holder_cpu_quota')
        save(OUT/'holder-state.json',report)
        # Serial capture keeps one candidate; never duplicate a hundreds-GB file.
        for job in sorted(jobs,key=lambda x:x['expected']['bytes']):
            file_id=job['file_id'];evidence={};report['files'][file_id]=evidence
            report['current_file_id']=file_id;save(OUT/'holder-state.json',report)
            def progress(done,total,phase='initial'):
                save(OUT/'progress.json',{'snapshot_id':ready['snapshot_id'],'file_id':file_id,'phase':phase,'bytes':done,'total':total,'elapsed_seconds':time.monotonic()-started,'holder_pid':os.getpid(),'cgroup':cgroup_metrics()})
            try:fd=capture(Path(job['path']),job['expected'],evidence,progress)
            finally:
                save(OUT/('file-'+sha(file_id.encode())[:16]+'.json'),evidence);save(OUT/'holder-state.json',report)
            fds[file_id]=fd;meta=file_metadata(file_id,job['expected'],fd);meta['source_stat']=job['source_stat'];meta['authenticated_sealed_readback_sha256']=evidence['sealed_readback_sha256'];ready['files'][file_id]=meta
            if file_id.endswith('.npy'):
                obj=OpenedSnapshotFile(os.dup(fd),meta);objects[file_id]=obj
                split=Path(job['path']).name.split('-')[0];kind='tokens' if '-tokens.' in file_id else 'masks'
                ready['audits'][file_id]=audit_array(obj,kind,[manifests['sft'][split+'_dialogues'],16385 if kind=='tokens' else 16384],None if kind=='tokens' else manifests['sft'][split+'_loss_tokens'])
            elif file_id.startswith('sft:') and '-dialogues.jsonl' in file_id:
                split=Path(job['path']).name.split('-')[0];ready['audits'][file_id]=audit_dialogues(fd,split,manifests['sft'])
            elif file_id=='sft:source-schema-rejections.jsonl':
                with os.fdopen(os.dup(fd),'rb') as stream:rows=[json.loads(line) for line in stream]
                require(len(rows)==manifests['sft']['source_schema_policy']['rejected_records']==39,'schema_rejections')
                ready['audits'][file_id]={'authenticated':True,'rows':len(rows)}
            evidence['audit']=ready['audits'].get(file_id)
            save(OUT/('file-'+sha(file_id.encode())[:16]+'.json'),evidence)
        for source,splits in manifests['c']['sources'].items():
            for split,expected in splits.items():
                prefix=f'c:{source}/{split}/corpus.'
                ready['audits'][prefix+'idx']=audit_index(fds[prefix+'idx'],ready['files'][prefix+'bin'],expected)
                ready['audits'][prefix+'bin']=audit_c(fds[prefix+'bin'],expected)
        # Every JSONL row was parsed from the same authenticated immutable FD.
        counted=provenance['original_counted_audit']
        for field,expected in [('selected_assistant_messages','all_selected_assistant_messages'),('context_only_assistant_messages','all_context_only_assistant_messages'),('structured_tool_calls','all_structured_tool_calls')]:
            require(sum(ready['audits']['sft:'+split+'-dialogues.jsonl'][field] for split in ('train','heldout'))==counted[expected],'full_dialogue_'+field)
        ready['audits']['SFT_alias_policy']={'classification':'legitimate new official aliases, not blanket zero','semantics':provenance['tokenizer_comparison']['renames'],'counts_by_file':{k:v['reserved_counts'] for k,v in ready['audits'].items() if k.endswith('-tokens.npy')}}
        for split in ('train','heldout'):
            fd,meta=derive_catalog(objects['sft:'+split+'-tokens.npy'].array,objects['sft:'+split+'-masks.npy'].array,split)
            meta['source_sha256']={k:ready['files'][k]['sha256'] for k in meta['derived_from']};fds[meta['file_id']]=fd;ready['files'][meta['file_id']]=meta
        require(len(fds)==len(ready['files'])==25,'final_file_count')
        server=Server(ready,fds);server.start()
        report['cpu_consumer_receipt']={}
        try:consume(ready,CODE,report['cpu_consumer_receipt'],timeout=min(1200,max(1,3600-int(time.monotonic()-started))))
        finally:save(OUT/'CPU-consumer.json',report['cpu_consumer_receipt'])
        result=report['cpu_consumer_receipt']['result']
        require(result['file_count']==25 and sum(k.startswith('sft:') for k in result['files'])==7,'all_seven_consumer_proof')
        require(all(result['files'][key]['stream_sha256']==result['files'][key]['mmap_sha256']==meta['sha256'] for key,meta in ready['files'].items()),'consumer_complete_SHA')
        # Re-read all same sealed holder descriptors after the real container exits.
        for file_id,fd in fds.items():
            validate_descriptor(fd,ready['files'][file_id]);require(digest_fd(fd,ready['files'][file_id]['bytes'])==ready['files'][file_id]['sha256'],'postconsumer_sha256')
        require(freeze_source()==request['source_freeze'],'source_freeze_postconsumer')
        for job in jobs:require(stat(job['path'])==job['source_stat'],'source_stat_postconsumer')
        ready['cpu_consumer_receipt']={'path':str(OUT/'CPU-consumer.json'),'sha256':sha((OUT/'CPU-consumer.json').read_bytes()),'container_id':report['cpu_consumer_receipt']['container_id'],'exit_code':0,'cleanup_verified_absent':True,'verified_original_sft_files':7,'verified_total_files':25,'full_stream_and_mmap_sha256':True}
        ready['captured_original_file_count']=23;ready['file_count']=25;ready['peer_policy']='owned_pinned_docker_init_or_verified_descendant_only';ready['created_at']=datetime.now().astimezone().isoformat()
        require(all(k in ready for k in contract['ready_required_fields']),'ready_required_fields')
        for meta in ready['files'].values():require(all(k in meta for k in contract['file_required_fields']),'file_required_fields')
        save(OUT/'READY.sealed.json',ready);published=True;signal.alarm(0)
        report.update(status='holding_ready',file_count=25,original_file_count=23,capture_elapsed_seconds=time.monotonic()-started,ready_sha256=sha((OUT/'READY.sealed.json').read_bytes()))
        while datetime.now().astimezone()<datetime.fromisoformat(contract['holder_not_after']):
            require(server.thread.is_alive(),'server_thread_dead')
            report.update(cgroup=cgroup_metrics(),peer_receipts=server.receipts,peer_denials=server.errors,heartbeat_at=datetime.now().astimezone().isoformat())
            save(OUT/'holder-state.json',report);time.sleep(10)
    except BaseException as error:
        report.update(status='holder_terminated' if published else 'BLOCKED',error=repr(error),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-started)
        if not published:save(OUT/'BLOCKED.json',report)
    finally:
        signal.alarm(0)
        if server:server.close()
        if ready:Path(ready['socket_path']).parent.rmdir()
        for obj in objects.values():obj.close()
        closed=[]
        for fd in fds.values():
            os.close(fd)
            try:os.fstat(fd)
            except OSError:closed.append(fd)
        report['all_ram_fds_closed_verified']=len(closed)==len(fds);report['cgroup_after_close']=cgroup_metrics()
        if published:report['status']='holder_closed_ready_stale';save(OUT/'HOLDER-CLOSED.json',report)
        else:save(OUT/'BLOCKED.json',report)
        save(OUT/'holder-state.json',report)
    return 0 if published else 1

def main():
    parser=argparse.ArgumentParser(description='CPU-only bounded full03 sealed RAM staging; no GPU release')
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('fixture',help='genuine tiny pinned Docker SCM_RIGHTS fixtures only')
    holder=sub.add_parser('hold',help='systemd-owned bounded capture then RAM FD service');holder.add_argument('--request',required=True)
    args=parser.parse_args()
    if args.command=='fixture':print(json.dumps({'status':fixture()['status'],'receipt':str(OUT/'FIXTURE.json')}));return 0
    return hold(args.request)

if __name__=='__main__':raise SystemExit(main())
