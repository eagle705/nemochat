"""Single diagnostic, three source views maximum; never a training release."""
import os, sys, json, time, hashlib, mmap, fcntl, socket, http.client, urllib.parse, signal, resource, traceback
from pathlib import Path
# Linux UAPI constants from /usr/include/linux/fcntl.h and asm-generic/fcntl.h;
# this host Python build omits the symbolic exports.
for _k,_v in {'F_ADD_SEALS':1033,'F_GET_SEALS':1034,'F_SEAL_SEAL':1,'F_SEAL_SHRINK':2,'F_SEAL_GROW':4,'F_SEAL_WRITE':8}.items():
    if not hasattr(fcntl,_k): setattr(fcntl,_k,_v)
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'results/midtraining/sft-after-kd-ram-probe-01'
IMAGE='nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
EXPECTED='19a28db000dabf1bae97cc85c2abbf6c38ac2837f84b08baf71686724182aaf4'
SIZE=2955198728
CHUNK=4*1024*1024

def vote_page(v):
    a,b,c=v
    if a==b==c: return a,'unanimous'
    if a==b or a==c: return a,'majority'
    if b==c: return b,'majority'
    return a,'tie'

class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect('/var/run/docker.sock')

def api(method,path,payload=None,raw=False):
    c=UnixHTTP('localhost',timeout=60)
    body=None if payload is None else json.dumps(payload).encode()
    c.request(method,'/v1.44'+path,body,{'Content-Type':'application/json'})
    r=c.getresponse(); b=r.read(); status=r.status; c.close()
    if status>=300: raise RuntimeError(f'Docker {method} {path}: {status} {b[:2000]!r}')
    return b if raw else (json.loads(b) if b else None)

def stat(p):
    s=os.stat(p); return {k:getattr(s,k) for k in ['st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns']}

def digest_fd(fd,n):
    h=hashlib.sha256()
    for p in range(0,n,CHUNK): h.update(os.pread(fd,min(CHUNK,n-p),p))
    return h.hexdigest()

def seal(fd):
    os.fchmod(fd,0o400)
    flags=fcntl.F_SEAL_WRITE|fcntl.F_SEAL_GROW|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_SEAL
    fcntl.fcntl(fd,fcntl.F_ADD_SEALS,flags)
    assert fcntl.fcntl(fd,fcntl.F_GET_SEALS)==flags
    denied={}
    for key,fn in [('write',lambda:os.pwrite(fd,b'x',0)),('grow',lambda:os.ftruncate(fd,os.fstat(fd).st_size+1)),('shrink',lambda:os.ftruncate(fd,0))]:
        try: fn(); denied[key]=False
        except OSError as e: denied[key]={'errno':e.errno}
    assert all(denied.values())
    return {'mask':flags,'mutation_denials':denied}

CONSUMER=r'''
import os,json,hashlib,mmap,fcntl,ast,struct
import numpy as np
p='/snapshot.npy'; fd=os.open(p,os.O_RDONLY); n=os.fstat(fd).st_size
r={'uid':os.getuid(),'bytes':n,'seals':fcntl.fcntl(fd,fcntl.F_GET_SEALS)}
h=hashlib.sha256()
while True:
 b=os.read(fd,4194304)
 if not b:break
 h.update(b)
r['read_sha256']=h.hexdigest()
m=mmap.mmap(fd,0,access=mmap.ACCESS_READ); h=hashlib.sha256()
for off in range(0,n,4194304):h.update(m[off:off+4194304])
r['mmap_sha256']=h.hexdigest();r['mmap_readonly']=m.readonly if hasattr(m,'readonly') else True
r['header_hex']=m[:128].hex()
if os.environ['IS_NPY']=='1':
 a=np.load(p,mmap_mode='r',allow_pickle=False)
 r.update(shape=list(a.shape),dtype=str(a.dtype),array_writeable=bool(a.flags.writeable),token_min=int(a.min()),token_max=int(a.max()))
 r['token_bounds_ok']=r['token_min']>=0 and r['token_max']<131072
 r['header_shape_ok']=list(a.shape)==[45090,16385] and str(a.dtype)=='uint32'
try:
 w=os.open(p,os.O_RDWR);r['rw_open_denied']=False;os.close(w)
except OSError as e:r['rw_open_denied']={'errno':e.errno}
r['matches_original_expected']=r['read_sha256']==os.environ['EXPECTED_SHA'] and r['mmap_sha256']==os.environ['EXPECTED_SHA']
print(json.dumps(r),flush=True)
'''

def consumer(fd,expected,npy,label,report):
    path=f'/proc/{os.getpid()}/fd/{fd}'
    payload={'Image':IMAGE,'User':'1001:1001','Entrypoint':['python3','-c',CONSUMER], 'Cmd':[], 'Env':['EXPECTED_SHA='+expected,'IS_NPY='+str(int(npy)),'NVIDIA_VISIBLE_DEVICES=void','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1'], 'Labels':{'nemochat.ram-probe':'01','probe.phase':label},'NetworkDisabled':True,'HostConfig':{'Binds':[path+':/snapshot.npy:ro'],'NetworkMode':'none','CapDrop':['ALL'],'SecurityOpt':['no-new-privileges:true'],'ReadonlyRootfs':True,'Memory':8*1024**3,'MemorySwap':8*1024**3,'NanoCpus':4*10**9,'PidsLimit':64,'DeviceRequests':[],'Devices':[],'AutoRemove':False}}
    cid=None
    try:
        cid=api('POST','/containers/create?name=nemochat-ram-probe-01-'+label,payload)['Id']
        report['container_id']=cid
        check=api('GET','/containers/'+cid+'/json')
        report['inspected_before_start']={'Image':check['Image'],'User':check['Config']['User'],'HostConfig':check['HostConfig'],'Mounts':check['Mounts']}
        hc=check['HostConfig']; assert hc['NetworkMode']=='none' and hc['Memory']==8*1024**3 and hc['NanoCpus']==4*10**9 and not hc.get('DeviceRequests') and hc['CapDrop']==['ALL']
        assert all(not m['RW'] for m in check['Mounts'])
        api('POST','/containers/'+cid+'/start')
        while True:
            state=api('GET','/containers/'+cid+'/json')['State']
            if not state['Running']:break
            time.sleep(.25)
        logs=api('GET','/containers/'+cid+'/logs?stdout=1&stderr=1',raw=True)
        # Docker non-TTY multiplex stream.
        text=b''; p=0
        while p+8<=len(logs):
            length=int.from_bytes(logs[p+4:p+8],'big'); text+=logs[p+8:p+8+length];p+=8+length
        report['logs']=text.decode(errors='replace'); report['state']=state
        if state['ExitCode']!=0:raise RuntimeError('consumer failed: '+report['logs'])
        report['result']=json.loads(report['logs'].strip().splitlines()[-1])
        rr=report['result']; assert rr['uid']==1001 and rr['seals']==15 and rr['read_sha256']==rr['mmap_sha256'] and rr['rw_open_denied']
        return rr
    finally:
        if cid:
            api('DELETE','/containers/'+cid+'?force=1')
            try:api('GET','/containers/'+cid+'/json')
            except RuntimeError as e:
                if ': 404 ' not in str(e):raise
                report['cleanup_verified_absent']=True


def main():
    started=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'PROBE.json').exists():
        raise RuntimeError('Single bounded real-file probe already consumed; no automatic rerun authorized.')
    report={'schema':'nemochat-bounded-ram-probe-v1','verdict':'INVALIDATED','release_allowed':False,'SFT_accepted':False,'expected_sha256':EXPECTED,'expected_bytes':SIZE,'limitations':['Diagnostic memory candidate is not new training data.','Host root cause unknown; no host fix or weight integrity claim.','No golden-producer semantic proof.','Three observational read views are not independent oracles.','Full production requires parent/user review, full immutable SHA coverage of all files, memory/cgroup capacity and lifetime design, reviewed source audits and production consumer validation.'],'controls':{},'source_views':[]}
    fds=[]; candidate=None
    def timeout_handler(*_):raise TimeoutError('600-second total runtime bound')
    signal.signal(signal.SIGALRM,timeout_handler);signal.alarm(590)
    # RLIMIT_AS bounds diagnostic holder virtual memory; single 2.96GB memfd + chunks.
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    try:
        img=api('GET','/images/'+urllib.parse.quote(IMAGE,safe='')+'/json');report['pinned_image']={'reference':IMAGE,'Id':img['Id'],'RepoDigests':img.get('RepoDigests')}
        # Tiny real bytes, independent gold digest, three differently corrupted observed fixture views.
        gold=bytes(range(256))*48; goldhash=hashlib.sha256(gold).hexdigest()
        vv=[]
        for i in range(3):
            v=bytearray(gold);v[i*4096:(i+1)*4096]=bytes([111+i])*4096;vv.append(v)
        tiny=os.memfd_create('ram-probe-tiny',os.MFD_ALLOW_SEALING);fds.append(tiny)
        for p in range(0,len(gold),4096):os.write(tiny,vote_page([bytes(v[p:p+4096]) for v in vv])[0])
        assert digest_fd(tiny,len(gold))==goldhash
        report['controls']['genuine_fixture']={'bytes':len(gold),'gold_sha256':goldhash,'view_sha256':[hashlib.sha256(v).hexdigest() for v in vv],'candidate_exact_match':True,'seals':seal(tiny)}
        report['controls']['docker']={}
        prior=OUT/'tiny-docker-bind-failure.json'
        if prior.exists():
            previous=json.loads(prior.read_text());report['controls']['docker']=previous['controls']['docker'];report['controls']['docker']['error']=previous['error'];report['controls']['docker']['reused_observation_no_retry']=True
        else:
            try: assert consumer(tiny,goldhash,False,'tiny',report['controls']['docker'])['matches_original_expected']
            except Exception as ex: report['controls']['docker']['error']=repr(ex)
        os.close(tiny);fds.remove(tiny)
        manifest=ROOT/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json'
        mb=manifest.read_bytes(); m=json.loads(mb); e=m['files']['heldout-tokens.npy']
        assert e=={'bytes':SIZE,'sha256':EXPECTED}
        source=manifest.parent/'heldout-tokens.npy';report['source']=str(source);report['manifest_sha256']=hashlib.sha256(mb).hexdigest();report['source_before']=stat(source);assert report['source_before']['st_size']==SIZE
        for flags in [os.O_RDONLY,os.O_RDONLY|os.O_DIRECT,os.O_RDONLY|os.O_DIRECT]:fds.append(os.open(source,flags))
        sourcefds=fds[-3:]
        candidate=os.memfd_create('ram-probe-candidate',os.MFD_ALLOW_SEALING);fds.append(candidate);os.ftruncate(candidate,SIZE)
        aligned=[mmap.mmap(-1,CHUNK) for _ in range(2)]
        hashes=[hashlib.sha256() for _ in range(3)]; counts={'pages':0,'unanimous':0,'majority':0,'tie':0,'changed_from_buffered':0}; disagreements=[]; pairdiff=[0,0,0]
        evidence=open(OUT/'disagreements.jsonl','w')
        for off in range(0,SIZE,CHUNK):
            wanted=min(CHUNK,SIZE-off); views=[os.pread(sourcefds[0],wanted,off)]
            for i in range(2):
                nr=os.preadv(sourcefds[i+1],[aligned[i]],off);assert nr==wanted,(off,nr,wanted);views.append(aligned[i][:nr])
            assert all(len(v)==wanted for v in views)
            for h,v in zip(hashes,views):h.update(v)
            chunk=bytearray(views[0])
            for p in range(0,wanted,4096):
                pages=[v[p:p+4096] for v in views]; selected,kind=vote_page(pages);counts['pages']+=1;counts[kind]+=1
                if selected!=pages[0]:counts['changed_from_buffered']+=1;chunk[p:p+len(selected)]=selected
                if kind!='unanimous':
                    for j,(a,b) in enumerate([(0,1),(0,2),(1,2)]):pairdiff[j]+=pages[a]!=pages[b]
                    row={'offset':off+p,'length':len(selected),'vote':kind,'view_sha256':[hashlib.sha256(x).hexdigest() for x in pages]}
                    evidence.write(json.dumps(row)+'\n')
                    if len(disagreements)<8:disagreements.append(row)
            assert os.pwrite(candidate,chunk,off)==wanted
            if off%(256*1024*1024)==0:print(json.dumps({'progress_bytes':off+wanted,'elapsed':time.monotonic()-started}),flush=True)
        evidence.close()
        for a in aligned:a.close()
        report['source_after']=stat(source);report['source_metadata_unchanged']=report['source_before']==report['source_after'];assert report['source_metadata_unchanged']
        report['source_views']=[{'kind':k,'bytes':SIZE,'sha256':h.hexdigest(),'matches_original_expected':h.hexdigest()==EXPECTED} for k,h in zip(['buffered_pread','O_DIRECT_preadv_aligned_1','O_DIRECT_preadv_aligned_2'],hashes)]
        report['source_bytes_read']=SIZE*3;report['page_votes']=counts;report['pair_disagreement_pages']=pairdiff;report['disagreement_examples']=disagreements
        for fd in sourcefds:os.close(fd);fds.remove(fd)
        report['seals']=seal(candidate);report['candidate_sha256']=digest_fd(candidate,SIZE);report['candidate_expected_hash_match']=report['candidate_sha256']==EXPECTED
        report['host_header_hex']=os.pread(candidate,128,0).hex()
        readonly_candidate=os.open(f'/proc/self/fd/{candidate}',os.O_RDONLY);fds.append(readonly_candidate)
        with mmap.mmap(readonly_candidate,0,access=mmap.ACCESS_READ) as mm:
            h=hashlib.sha256()
            for p in range(0,SIZE,CHUNK):h.update(mm[p:p+CHUNK])
            report['host_readonly_mmap_sha256']=h.hexdigest()
        assert report['host_readonly_mmap_sha256']==report['candidate_sha256']
        if report['controls']['docker'].get('error'):
            report['real_consumer']={'attempted':False,'reason':'Tiny sealed memfd bind failed in runc; identical real bind not retried.'}
            report['sealedfd_consumer_readback']=False
            report['verdict']='PARTIAL' if report['candidate_expected_hash_match'] else 'INVALIDATED'
            return
        report['real_consumer']={};rr=consumer(candidate,EXPECTED,True,'real',report['real_consumer'])
        report['sealedfd_consumer_readback']=rr['read_sha256']==report['candidate_sha256'] and rr['header_hex']==report['host_header_hex']
        if report['candidate_expected_hash_match'] and rr['matches_original_expected'] and rr['token_bounds_ok'] and rr['header_shape_ok'] and report['sealedfd_consumer_readback']:report['verdict']='VALIDATED'
        else:report['verdict']='INVALIDATED'
    except BaseException as e:
        report['error']=repr(e);report['traceback']=traceback.format_exc()
        if report.get('candidate_expected_hash_match'):report['verdict']='PARTIAL'
    finally:
        signal.alarm(0)
        for fd in fds:
            try:os.close(fd)
            except OSError:pass
        report['holder_fds_closed']=True;report['elapsed_seconds']=time.monotonic()-started;report['peak_holder_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['ram_design']={'candidate_bytes':SIZE,'maximum_holder_address_space':8*1024**3,'views_streamed_not_retained':True,'consumer_memory_limit':8*1024**3,'same_memfd_pages_shared_not_copied':True}
        (OUT/'PROBE.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'verdict':report['verdict'],'report':str(OUT/'PROBE.json'),'error':report.get('error'),'elapsed_seconds':report['elapsed_seconds']}),flush=True)

if __name__=='__main__':main()
