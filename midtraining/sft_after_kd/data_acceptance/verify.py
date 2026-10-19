"""Read-only paired buffered/O_DIRECT audit. No checkpoint access or release."""
import concurrent.futures
import hashlib
import io
import json
import mmap
import os
from pathlib import Path
import struct
import time
import traceback
import numpy as np

ROOT=Path('/raid/yoon/nemochat')
OUT=ROOT/'results/midtraining/sft-after-kd-data-acceptance-01'
MANIFEST=ROOT/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json'
CORPUS=ROOT/'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json'

def identity(p):
    s=p.stat()
    return {k:getattr(s,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.pending');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(p)

def check_file(path,expected,kind,out,chunk=8*1024**2):
    path=Path(path);before=identity(path);size=before['st_size'];t=time.monotonic()
    hb=hashlib.sha256();hd=hashlib.sha256();pos=0;diffblocks=0;first=None
    audit={'kind':kind,'elements':0,'reserved_counts':{str(i):0 for i in range(18,30)},'out_of_bounds':0,'min':None,'max':None,'ones':0,'invalid_mask_values':0,'rows':0,'metadata_loss_tokens':0,'nonpadding_tokens':0,'per_source':{},'selected_assistant_messages':0,'context_only_assistant_messages':0,'structured_tool_calls':0}
    tail=b'';offset=0;dtype=None;shape=None;parse_error=None
    fd=os.open(path,os.O_RDONLY|os.O_DIRECT)
    buf=mmap.mmap(-1,chunk)
    # O_DIRECT's final request remains fully aligned; short read at EOF is
    # explicit, never reopened through a buffered fallback.
    try:
      with path.open('rb',buffering=0) as buffered:
       while pos<size:
        want=min(chunk,size-pos)
        got=os.preadv(fd,[buf],pos)
        if got!=want:raise OSError(f'direct short read {got} != {want} at {pos}')
        direct=bytes(buf[:got]);ordinary=buffered.read(want)
        if len(ordinary)!=want:raise OSError('buffered short read')
        hb.update(ordinary);hd.update(direct)
        if ordinary!=direct:
            diffblocks+=1
            if first is None:
                left=np.frombuffer(ordinary,dtype='u1');right=np.frombuffer(direct,dtype='u1');different=np.flatnonzero(left!=right)
                i=int(different[0]);start=i//4096*4096;end=min(start+4096,want)
                tag=hashlib.sha256(str(path).encode()).hexdigest()[:16]
                bp=Path(out)/(tag+'-buffered-first-page.bin');dp=Path(out)/(tag+'-direct-first-page.bin')
                bp.write_bytes(ordinary[start:end]);dp.write_bytes(direct[start:end])
                first={'offset':pos+i,'page_offset':pos+start,'different_bytes_in_first_chunk':int(different.size),'buffered_snapshot':str(bp),'direct_snapshot':str(dp),'buffered_snapshot_sha256':hashlib.sha256(ordinary[start:end]).hexdigest(),'direct_snapshot_sha256':hashlib.sha256(direct[start:end]).hexdigest()}
        if parse_error is None:
          try:
            data=direct
            if pos==0 and path.suffix=='.npy':
                f=io.BytesIO(data);v=np.lib.format.read_magic(f)
                if v==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
                elif v==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
                else:raise ValueError('unsupported npy version')
                if fortran:raise ValueError('Fortran order forbidden')
                offset=f.tell();data=data[offset:];audit.update(shape=list(shape),dtype=str(dtype),header_bytes=offset)
                if dtype!=np.dtype('uint32' if kind=='tokens' else 'uint8'):raise ValueError('unexpected dtype')
            elif kind=='corpus':dtype=np.dtype('<i4')
            if kind in ('tokens','corpus','masks'):
                a=np.frombuffer(data,dtype=dtype)
                audit['elements']+=int(a.size)
                if a.size:
                    amin=int(a.min());amax=int(a.max());audit['min']=amin if audit['min'] is None else min(amin,audit['min']);audit['max']=amax if audit['max'] is None else max(amax,audit['max'])
                    if kind=='masks':
                        audit['ones']+=int(a.sum(dtype=np.uint64));audit['invalid_mask_values']+=int(np.count_nonzero(a>1))
                    else:
                        audit['out_of_bounds']+=int(np.count_nonzero((a<0)|(a>=131072)))
                        reserved=a[(a>=18)&(a<=29)]
                        if reserved.size:
                            vals,counts=np.unique(reserved,return_counts=True)
                            for v,c in zip(vals,counts):audit['reserved_counts'][str(int(v))]+=int(c)
            elif kind=='dialogues':
                lines=(tail+data).split(b'\n');tail=lines.pop()
                for line in lines:
                    r=json.loads(line);audit['rows']+=1;audit['metadata_loss_tokens']+=r['supervised_tokens'];audit['nonpadding_tokens']+=r['nonpadding_tokens']
                    source=r['source_key'];audit['per_source'][source]=audit['per_source'].get(source,0)+1
                    audit['selected_assistant_messages']+=sum(r['selected'])
                    audit['context_only_assistant_messages']+=sum(msg['role']=='assistant' and not flag for msg,flag in zip(r['messages'],r['selected']))
                    audit['structured_tool_calls']+=sum(len(msg.get('tool_calls') or []) for msg in r['messages'])
            elif kind=='jsonl':
                lines=(tail+data).split(b'\n');tail=lines.pop()
                for line in lines:json.loads(line);audit['rows']+=1
          except Exception as e:parse_error=repr(e)
        pos+=want
        if pos%(1024**3)<chunk:
            print(json.dumps({'progress':str(path),'bytes':pos,'total':size,'seconds':round(time.monotonic()-t,1)}),flush=True)
    finally:
        buf.close();os.close(fd)
    after=identity(path)
    if tail and kind in ('dialogues','jsonl') and parse_error is None:parse_error='unterminated JSONL'
    if shape is not None and audit['elements']!=int(np.prod(shape,dtype=np.int64)):parse_error='array size mismatch'
    hashes={'buffered':hb.hexdigest(),'direct':hd.hexdigest()}
    good=size==expected['bytes'] and all(h==expected['sha256'] for h in hashes.values()) and before==after and diffblocks==0
    semantic=parse_error is None and not audit['out_of_bounds'] and not audit['invalid_mask_values'] and (kind!='corpus' or not sum(audit['reserved_counts'].values()))
    result=dict(path=str(path),expected=expected,observed_sha256=hashes,bytes_read_per_view=pos,before=before,after=after,bytes_accepted=good,semantic_accepted=semantic,audit=audit,parse_error=parse_error,differing_chunks=diffblocks,first_difference=first,elapsed_seconds=time.monotonic()-t,parser_view='same immutable direct chunks used for direct SHA',buffered_fallback=False)
    print(json.dumps({'completed':str(path),'bytes_accepted':good,'semantic_accepted':semantic,'hashes':hashes}),flush=True)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mbytes=MANIFEST.read_bytes();m=json.loads(mbytes);cbytes=CORPUS.read_bytes();c=json.loads(cbytes)
    assert hashlib.sha256(mbytes).hexdigest()=='78ab81fbbd9c0b13c55392b7d87c9e7035ad75778abad0dd3795d64121125c21'
    assert hashlib.sha256(cbytes).hexdigest()=='23c1a94b4c87942585cc7e4a4fc693653c8f3565843a91172e70545a1699dc63'
    jobs=[]
    for name,e in m['files'].items():
        kind='tokens' if 'tokens.npy' in name else 'masks' if 'masks.npy' in name else 'dialogues' if 'dialogues.jsonl' in name else 'jsonl'
        jobs.append((MANIFEST.parent/name,e,kind,'sft-'+name))
    for source,splits in c['sources'].items():
        for split,e in splits.items():
            for ext in ('bin','idx'):
                jobs.append((Path(e['prefix']+'.'+ext),{'bytes':e[ext+'_bytes'],'sha256':e[ext+'_sha256']},'corpus' if ext=='bin' else 'index','c-'+source+'-'+split+'.'+ext))
    summary=dict(status='running',manifest=str(MANIFEST),manifest_sha256=hashlib.sha256(mbytes).hexdigest(),corpus_manifest=str(CORPUS),corpus_manifest_sha256=hashlib.sha256(cbytes).hexdigest(),required_files=len(jobs),results={},checkpoint_payloads_read=False,host_repaired=False)
    save(OUT/'scan-status.json',summary)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(check_file,p,e,k,OUT):(label,p) for p,e,k,label in jobs}
        for f in concurrent.futures.as_completed(futures):
            label,p=futures[f]
            try:r=f.result()
            except Exception:r={'path':str(p),'bytes_accepted':False,'semantic_accepted':False,'exception':traceback.format_exc()}
            save(OUT/(label+'.json'),r);summary['results'][label]=r;save(OUT/'scan-status.json',summary)
    summary['status']='scans_complete';summary['all_bytes_accepted']=all(r['bytes_accepted'] for r in summary['results'].values());summary['all_semantics_accepted']=all(r['semantic_accepted'] for r in summary['results'].values())
    summary['metadata_unchanged']=MANIFEST.read_bytes()==mbytes and CORPUS.read_bytes()==cbytes
    save(OUT/'scan-status.json',summary)

if __name__=='__main__':main()
