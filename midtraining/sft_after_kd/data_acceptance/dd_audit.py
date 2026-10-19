"""Full C token census and SHA from the same independent direct-dd stream."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import verify

out=verify.OUT

def scan(source,split,expected):
 p=Path(expected['prefix']+'.bin');before=verify.identity(p);h=hashlib.sha256();n=0;counts={str(i):0 for i in range(18,30)};lo=131072;hi=-1;bad=0
 dd=subprocess.Popen(['dd','if='+str(p),'iflag=direct','bs=4M','status=none'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 while True:
  b=dd.stdout.read(8*1024**2)
  if not b:break
  h.update(b);a=np.frombuffer(b,dtype='<i4');n+=int(a.size)
  if a.size:
   lo=min(lo,int(a.min()));hi=max(hi,int(a.max()));bad+=int(np.count_nonzero((a<0)|(a>=131072)))
   vals,cs=np.unique(a[(a>=18)&(a<=29)],return_counts=True)
   for v,c in zip(vals,cs):counts[str(int(v))]+=int(c)
 err=dd.stderr.read().decode();rc=dd.wait();after=verify.identity(p);sha=h.hexdigest()
 r=dict(source=source,split=split,path=str(p),before=before,after=after,expected_sha256=expected['bin_sha256'],observed_sha256=sha,expected_bytes=expected['bin_bytes'],bytes=n*4,tokens_scanned=n,expected_tokens=expected['tokens'],reserved_counts=counts,min_token_id=lo,max_token_id=hi,out_of_bounds=bad,dd_exit=rc,dd_stderr=err,parser_sha_same_immutable_chunks=True)
 r['authenticated_bytes']=rc==0 and before==after and n*4==expected['bin_bytes'] and sha==expected['bin_sha256']
 r['accepted_absence_and_bounds']=r['authenticated_bytes'] and n==expected['tokens'] and not bad and not sum(counts.values())
 verify.save(out/('c-direct-census-'+source+'-'+split+'.json'),r)
 return source+'/'+split,r

if __name__=='__main__':
 c=json.loads(verify.CORPUS.read_text());jobs=[(source,split,e) for source,splits in c['sources'].items() for split,e in splits.items()]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:r=dict(pool.map(lambda a:scan(*a),jobs))
 verify.save(out/'c-direct-census.json',{'status':'passed' if all(v['accepted_absence_and_bounds'] for v in r.values()) else 'blocked','corpus_manifest_sha256':hashlib.sha256(verify.CORPUS.read_bytes()).hexdigest(),'files':r,'old_corpus_absence_not_reused':True})
 print(json.dumps({k:{'authenticated':v['authenticated_bytes'],'reserved':v['reserved_counts'],'out_of_bounds':v['out_of_bounds']} for k,v in r.items()},indent=2))
