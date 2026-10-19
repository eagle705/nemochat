"""Single authorized CPU recovery generation; old source remains immutable."""
import concurrent.futures
from datetime import datetime
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
import uuid
from sft.archive.sealed_snapshot03 import holder as old
from sft.archive.sealed_snapshot03 import transport
from sft.archive.sealed_snapshot03.client import require, request_header, OpenedSnapshotFile, validate_descriptor, CHUNK, digest_fd, row_digest
stat=old.stat
seal_readonly=old.seal_readonly
ROOT=old.ROOT
CODE=ROOT/'sft/archive/snapshot/recovery02'
OUT=ROOT/'results/midtraining/sft-continue1500-snapshot-02'
PARENT=ROOT/'sft/resume/request.json'
PRIOR=ROOT/'results/midtraining/sft-after-kd-sealed03/READY.sealed.json'
PRIOR_SHA='24498630aee5a3b1ad3f6aa3a2e8dbfcaec5fca98e3acb899d7fbc55678da5fb'
VOTE_KEYS=('unanimous','majority','tie','changed','fastpath_chunks','fastpath_bytes','fallback_chunks','fallback_bytes','chunk_disagreements')


def vote_chunk(a,b,c):
    require(len(a)==len(b)==len(c),'vote_size')
    stats={k:0 for k in VOTE_KEYS}
    stats['chunk_disagreements']=int(not a==b==c)
    if a==b or a==c: selected=a
    elif b==c: selected=b
    else:
        selected,pages=old.vote_block(a,b,c)
        stats.update(pages)
        stats.update(fallback_chunks=1,fallback_bytes=len(a))
        return selected,stats
    stats.update(fastpath_chunks=1,fastpath_bytes=len(a))
    return selected,stats


def derive_catalog_parallel(tokens,masks,label):
    n=len(tokens);require(len(masks)==n,'catalog_rows')
    size=n*32
    fd=os.memfd_create('recovery02-rowhash-'+label,os.MFD_ALLOW_SEALING|os.MFD_CLOEXEC)
    started=time.monotonic()
    try:
        os.ftruncate(fd,size)
        # Exactly eight coarse jobs. Each pwrite is bounded to 128KiB.
        ranges=[(n*k//8,n*(k+1)//8) for k in range(8)]
        def worker(bounds):
            begin,end=bounds;written=0
            for start in range(begin,end,4096):
                stop=min(end,start+4096)
                data=b''.join(row_digest(i,tokens[i],masks[i]) for i in range(start,stop))
                require(len(data)==(stop-start)*32,'catalog_chunk_size')
                require(os.pwrite(fd,data,start*32)==len(data),'catalog_write')
                written+=stop-start
            return begin,end,written
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            completed=list(pool.map(worker,ranges))
        require([(a,b) for a,b,c in completed]==ranges and all(c==b-a for a,b,c in completed) and sum(c for a,b,c in completed)==n,'catalog_exact_coverage')
        # Hash the actual complete produced descriptor, seal, then same-FD readback.
        expected_actual=digest_fd(fd,size)
        ro=seal_readonly(fd)
        try:
            require(digest_fd(ro,size)==expected_actual,'catalog_readback')
            meta={'file_id':'rowhash:'+label,'bytes':size,'sha256':expected_actual,'seals':15,'access_mode':'O_RDONLY','shape':[n,32],'dtype':'uint8','derived_from':['sft:'+label+'-tokens.npy','sft:'+label+'-masks.npy'],'row_domain_hex':b'nemochat-sft-row-v1\0'.hex()}
            evidence={'event':'catalog_complete','split':label,'rows':n,'bytes':size,'sha256':expected_actual,'workers':8,'completed_ranges':completed,'elapsed_seconds':time.monotonic()-started}
            if label in ('train','heldout'):old.save(OUT/('catalog-'+label+'.json'),evidence)
            print(json.dumps(evidence),flush=True)
            return ro,meta
        except BaseException:os.close(ro);raise
    finally:os.close(fd)

def capture(path, expected, report, progress=None):
    started_capture=time.monotonic()
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
            counts = {k:0 for k in VOTE_KEYS}; disagreement = hashlib.sha256()
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
                        selected, delta = vote_chunk(a,b,c)
                        if not a == b == c:
                            disagreement.update(offset.to_bytes(8,'little'))
                            for observation in (a,b,c): disagreement.update(hashlib.sha256(observation).digest())
                        for k,v in delta.items(): counts[k] += v
                        if selected != a: require(os.pwrite(candidate,selected,offset) == wanted,'vote_write')
                        if progress and offset % (1024**3) < CHUNK: progress(offset+got,size,'vote')
            finally: os.close(buffered); os.close(direct)
            for h,meta in zip(hashes,views):
                meta.update(after=stat(path),sha256=h.hexdigest())
                require(meta['before'] == meta['after'] == before,'source_stat_changed')
            report['vote_evidence'] = {'counts':counts,'page_counts_scope':'fallback chunks only; no inferred per-page fastpath counts','chunk_disagreement_evidence_sha256':disagreement.hexdigest(),'chunk_disagreement_domain':'offset LE64 followed by SHA256(a), SHA256(b), SHA256(c) for non-unanimous chunks'}
        ro = seal_readonly(candidate)
        report.update(sealed_readback_sha256=digest_fd(ro,size),seals=fcntl.fcntl(ro,1034),source_after=stat(path))
        require(report['source_after'] == before,'source_stat_changed')
        require(report['sealed_readback_sha256'] == expected['sha256'],'whole_sha256')
        report['accepted'] = True
        report['elapsed_seconds']=time.monotonic()-started_capture
        report['actual_source_view_bytes']=sum(v['bytes'] for v in report['views'])
        print(json.dumps({'event':'captured','path':str(path),'bytes':size,'actual_source_view_bytes':report['actual_source_view_bytes'],'views':len(report['views']),'sealed_readback_sha256':report['sealed_readback_sha256'],'seconds':report['elapsed_seconds']}),flush=True)
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

def inherit_identical(prior, files):
    for key,meta in files.items():
        require(all(meta[k]==prior['files'][key][k] for k in ('bytes','sha256')), 'inheritance_byte_identity:'+key)
    return True

def freeze():
    paths=list(CODE.rglob('*.py'))+[old.CODE/n for n in ('holder.py','client.py','transport.py')]+[CODE.parent/n for n in ('holder.py','client.py','consumer.py')]
    records={str(p.relative_to(ROOT)):old.sha(p.read_bytes()) for p in sorted(paths)}
    return {'files':records,'sha256':old.sha(json.dumps(records,sort_keys=True,separators=(',',':')).encode())}

def hold(request_path):
    request=json.loads(Path(request_path).read_bytes()); parent=json.loads(PARENT.read_bytes())
    fds={}; objects={}; server=None; ready=None;published=False
    report={'status':'capturing','snapshot_id':request['snapshot_id'],'holder_identity':old.process_identity(),'files':{},'GPU_allocated':False,'checkpoint_payloads_read':False,'source_writes':False,'original_mmap_inputs_blocked':True}
    started=time.monotonic()
    def expired(*_):
        report.update(status='holder_terminated' if published else 'BLOCKED',error='capture_3600_seconds_or_absolute_1515_or_stop',kernel_closes_all_process_fds_on_exit=True)
        old.save(OUT/('HOLDER-CLOSED.json' if published else 'BLOCKED.json'),report)
        old.save(OUT/'holder-state.json',report)
        os._exit(1)
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired);signal.signal(signal.SIGINT,expired);signal.alarm(min(3600,max(1,int((datetime.fromisoformat(parent['holder_not_after'])-datetime.now().astimezone()).total_seconds()))))
    try:
        require(old.sha(PARENT.read_bytes())==request['parent_sha256'],'parent_changed')
        require(freeze()==request['source_freeze'],'source_changed')
        require(request['absolute_capture_timeout_seconds']==3600 and request['max_generations']==1,'bounded_single_generation')
        require(old.sha((OUT/'FIXTURE.json').read_bytes())==request['fixture_sha256'] and json.loads((OUT/'FIXTURE.json').read_bytes())['status']=='PASS','genuine_fixture_gate')
        require(datetime.now().astimezone()<datetime.fromisoformat(parent['holder_not_after']),'absolute_holder_expired')
        require(old.sha(PRIOR.read_bytes())==PRIOR_SHA==parent['prior_snapshot_sha256'],'prior_ready_identity')
        prior=json.loads(PRIOR.read_bytes())
        receipt=prior['cpu_consumer_receipt'];require(old.sha(Path(receipt['path']).read_bytes())==receipt['sha256'],'prior_consumer_receipt')
        require(old.memory_available()>=1024**4,'MemAvailable_below_1TiB')
        metrics=old.cgroup_metrics();report['cgroup_initial']=metrics
        require(metrics['path'].endswith('/'+parent['new_holder_unit']),'unit_scope')
        require(metrics['memory.max']==str(768*1024**3) and metrics['memory.high']==str(640*1024**3),'memory_bounds')
        quota,period=map(int,metrics['cpu.max'].split());require(quota==8*period,'cpu_quota')
        contract=json.loads(old.CONTRACT.read_bytes());contract.update(holder_unit=parent['new_holder_unit'],holder_not_after=parent['holder_not_after'])
        manifests,jobs=old.inventory(contract)
        require(jobs==request['inventory'],'source_inventory_changed')
        # Old base creator is not used: fresh distinct private socket generation.
        directory=Path('/run/user/1001')/('ncs05-'+request['snapshot_id'][:12]);directory.mkdir(mode=0o700)
        require(directory.stat().st_uid==1001 and directory.stat().st_mode&0o777==0o700,'socket_directory')
        ready={k:prior[k] for k in ('schema','status','sft_manifest_sha256','c_manifest_sha256','input_mode','original_mmap_inputs_blocked','checkpoint_integrity','peer_policy')}
        ready.update(snapshot_id=request['snapshot_id'],holder_identity=old.process_identity(),holder_unit=parent['new_holder_unit'],holder_not_after=parent['holder_not_after'],socket_path=str(directory/'s'),producer_source_freeze_sha256=request['source_freeze']['sha256'],files={})
        report['producer_source_freeze']=request['source_freeze'];report['mem_available_before_capture']=old.memory_available()
        old.save(OUT/'holder-state.json',report)
        # Disjoint memfds; no duplicate hundreds-GB candidates. Each uses the
        # unchanged bounded direct-first capture, never a source mmap fallback.
        def acquire(job):
            key=job['file_id']; evidence={};report['files'][key]=evidence
            def progress(done,total,phase='initial'):
                old.save(OUT/('progress-'+old.sha(key.encode())[:16]+'.json'),{'file_id':key,'bytes':done,'total':total,'phase':phase,'elapsed_seconds':time.monotonic()-started})
            try:
                fd=capture(Path(job['path']),job['expected'],evidence,progress)
                fds[key]=fd
                meta=old.file_metadata(key,job['expected'],fd)
                meta.update(source_stat=job['source_stat'],authenticated_sealed_readback_sha256=evidence['sealed_readback_sha256'])
                return key,meta
            finally:old.save(OUT/('file-'+old.sha(key.encode())[:16]+'.json'),evidence)
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=3)
        futures=[pool.submit(acquire,j) for j in sorted(jobs,key=lambda j:-j['expected']['bytes'])]
        try:
            for future in concurrent.futures.as_completed(futures):
                key,meta=future.result();ready['files'][key]=meta
                report['completed_files']=len(ready['files']);old.save(OUT/'holder-state.json',report)
        finally:pool.shutdown(wait=True,cancel_futures=True)
        inherit_identical(prior,ready['files'])
        print(json.dumps({'event':'all23_captured','elapsed_seconds':time.monotonic()-started}),flush=True)
        report['phase']='deriving_row_catalogs';old.save(OUT/'holder-state.json',report)
        for split in ('train','heldout'):
            keys=['sft:'+split+'-'+kind+'.npy' for kind in ('tokens','masks')]
            for key in keys:objects[key]=OpenedSnapshotFile(os.dup(fds[key]),ready['files'][key])
            fd,meta=derive_catalog_parallel(objects[keys[0]].array,objects[keys[1]].array,split)
            fds[meta['file_id']]=fd
            meta['source_sha256']={k:ready['files'][k]['sha256'] for k in meta['derived_from']}
            require(meta==prior['files'][meta['file_id']],'catalog_must_match_authenticated_prior')
            ready['files'][meta['file_id']]=meta
        require(len(fds)==len(ready['files'])==25,'all25')
        inherit_identical(prior,ready['files'])
        ready['audits']=prior['audits'];ready['provenance']=prior['provenance']
        ready['audit_inheritance']={'status':'INHERITED_ON_IDENTICAL_BYTES','prior_ready_path':str(PRIOR),'prior_ready_sha256':PRIOR_SHA,'scope':'All prior semantic bounds/masks/targets/dialogue counts/C-alias absence and provenance; no new semantic rescan claimed. All 23 fresh originals have exact whole SHA+size and sealed-FD readback; both catalogs freshly regenerated with exact prior SHA.','prior_cpu_full_stream_mmap_receipt':receipt,'prior_transport_proof_is_not_new_fd_generation':True}
        server=transport.Server(ready,fds);server.start()
        # A valid header from an unowned same-UID host must still be rejected.
        with socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET) as sock:
            sock.settimeout(20);sock.connect(ready['socket_path']);sock.send(json.dumps(request_header(ready,next(iter(fds)))).encode())
            try:require('peer_not_owned_pinned_container' in json.loads(sock.recv(65536))['denied'],'wrong_peer_not_denied')
            except ConnectionResetError:require(any('peer_not_owned_pinned_container' in x['error'] for x in server.errors),'wrong_peer_reset_not_authenticated')
        proof={'host_same_uid_denied':True,'negative':{}}
        transport.consume(dict(ready,snapshot_id='incorrect-continuation-snapshot'),CODE,proof['negative'],timeout=120,negative='header_identity')
        transport.consume(ready,CODE,proof,timeout=180)
        require(proof['result']['file_count']==25 and proof['result']['original_sft_files']==7,'fresh_all25_transport')
        require(proof['result']['sample_rows']['train']==[0,115839,115840,4369279],'resume_cursor_samples')
        proof.update(peer_receipts=server.receipts,peer_denials=server.errors,fresh_full_original_hash=False,content_full_hash_scope='holder same sealed FD readback; historical container full stream+mmap inherited only on identical bytes')
        old.save(OUT/'CPU-consumer.json',proof)
        for key,fd in fds.items():validate_descriptor(fd,ready['files'][key])
        require(freeze()==request['source_freeze'],'source_changed_after_capture')
        for job in jobs:require(old.stat(job['path'])==job['source_stat'],'source_stat_after')
        ready.update(cpu_consumer_receipt={'path':str(OUT/'CPU-consumer.json'),'sha256':old.sha((OUT/'CPU-consumer.json').read_bytes()),'container_id':proof['container_id'],'exit_code':0,'cleanup_verified_absent':True,'verified_original_sft_files':7,'verified_total_files':25,'full_stream_and_mmap_sha256':False,'fresh_transport_all25_headers_seals_readonly_shapes_and_sample_rows':True,'historical_full_stream_mmap_status':'INHERITED_ON_IDENTICAL_BYTES'},captured_original_file_count=23,file_count=25,created_at=datetime.now().astimezone().isoformat())
        old.save(OUT/'READY.sealed.json',ready);published=True;signal.alarm(max(1,int((datetime.fromisoformat(parent['holder_not_after'])-datetime.now().astimezone()).total_seconds())))
        print(json.dumps({'event':'READY','snapshot_id':ready['snapshot_id'],'ready':str(OUT/'READY.sealed.json')}),flush=True)
        report.update(status='holding_ready',file_count=25,capture_elapsed_seconds=time.monotonic()-started,ready_sha256=old.sha((OUT/'READY.sealed.json').read_bytes()))
        while datetime.now().astimezone()<datetime.fromisoformat(parent['holder_not_after']):
            require(server.thread.is_alive(),'server_dead')
            report.update(cgroup=old.cgroup_metrics(),peer_receipts=server.receipts,peer_denials=server.errors,heartbeat_at=datetime.now().astimezone().isoformat())
            old.save(OUT/'holder-state.json',report);time.sleep(10)
    except BaseException as error:
        report.update(status='holder_terminated' if published else 'BLOCKED',error=repr(error),traceback=traceback.format_exc())
    finally:
        signal.alarm(0)
        if server:server.close()
        if ready:Path(ready['socket_path']).parent.rmdir()
        for obj in objects.values():obj.close()
        for fd in fds.values():os.close(fd)
        report['all_ram_fds_closed']=True
        old.save(OUT/('HOLDER-CLOSED.json' if published else 'BLOCKED.json'),report);old.save(OUT/'holder-state.json',report)
    return 0 if published else 1

if __name__=='__main__':raise SystemExit(hold(sys.argv[1]))
