"""Independent GNU dd/SHA read protocol, complete immutable inventory."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import verify

OUT=verify.OUT

def one(path,expected,label,mode):
    before=verify.identity(path);start=time.monotonic()
    cmd=['dd','if='+str(path),'bs=4M','status=none']
    if mode=='direct':cmd+=['iflag=direct']
    dd=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    sha=subprocess.Popen(['sha256sum'],stdin=dd.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    dd.stdout.close();so,se=sha.communicate();de=dd.stderr.read();dc=dd.wait();after=verify.identity(path)
    digest=so.decode().split()[0] if so else None
    r=dict(path=str(path),mode=mode,expected=expected,observed_sha256=digest,before=before,after=after,dd_exit=dc,sha_exit=sha.returncode,dd_stderr=de.decode(),sha_stderr=se.decode(),command=cmd,elapsed_seconds=time.monotonic()-start)
    r['accepted']=dc==0 and sha.returncode==0 and digest==expected['sha256'] and before==after and before['st_size']==expected['bytes']
    verify.save(OUT/('serial-'+label+'-'+mode+'.json'),r)
    print(json.dumps(r),flush=True)
    return label,mode,r

def main():
    jobs=[]
    m=json.loads(verify.MANIFEST.read_text())
    for name,e in m['files'].items():jobs.append((verify.MANIFEST.parent/name,e,'sft-'+name))
    c=json.loads(verify.CORPUS.read_text())
    for source,splits in c['sources'].items():
      for split,e in splits.items():
       for ext in ('bin','idx'):jobs.append((Path(e['prefix']+'.'+ext),{'bytes':e[ext+'_bytes'],'sha256':e[ext+'_sha256']},'c-'+source+'-'+split+'.'+ext))
    result={'status':'running','protocol':'independent GNU dd bs=4M to GNU sha256sum; no source writes; aligned direct requests including short EOF','required_files':len(jobs),'results':{}}
    verify.save(OUT/'serial-status.json',result)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
      fs=[pool.submit(one,p,e,label,mode) for p,e,label in sorted(jobs,key=lambda j:-j[1]['bytes']) for mode in ('buffered','direct')]
      for f in concurrent.futures.as_completed(fs):
        label,mode,r=f.result();result['results'].setdefault(label,{})[mode]=r;verify.save(OUT/'serial-status.json',result)
    result['status']='complete';result['accepted']=all(r['accepted'] for views in result['results'].values() for r in views.values());verify.save(OUT/'serial-status.json',result)

if __name__=='__main__':main()
