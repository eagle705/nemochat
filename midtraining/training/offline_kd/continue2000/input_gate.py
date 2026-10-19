"""Bounded independent SAME-document direct assembly; never repair/rebase inputs.

Native GPT sample ownership formula was executed in sampler-proof.json. Only a
schema-valid digest disagreement is a data-class rejection. Structural errors,
unsupported datasets/configs, model and collectives remain fatal.
"""
import hashlib
import mmap
import os
from pathlib import Path
from .broker import CacheRejected
from midtraining.training.offline_kd.prefix.filtered.transport import dio_alignment, identity
KEYS=('tokens','labels','loss_mask','position_ids')
class SourceDigestMismatch(CacheRejected):pass

def sample_digest(sample):
    return hashlib.sha256(b''.join(sample[k].cpu().contiguous().numpy().tobytes() for k in KEYS)).hexdigest()
def validate_schema(sample):
    import torch
    if set(sample)-set(KEYS) not in (set(),{'dataset_id'}) or not set(KEYS)<=set(sample):raise ValueError('unsupported native sample schema')
    for k in KEYS:
        a=sample[k]
        if a.device.type!='cpu' or tuple(a.shape)!=(4096,) or a.dtype!=(torch.float32 if k=='loss_mask' else torch.int64):
            raise ValueError('native input shape/dtype/device invalid')
    if (not torch.equal(sample['position_ids'],torch.arange(4096))
        or not bool((sample['loss_mask']==1).all())
        or any(int(sample[k].min())<0 or int(sample[k].max())>=131072 for k in ('tokens','labels'))):
        raise ValueError('native input IDs/mask/positions schema invalid')
def compare_digest(sample,expected,index,**extra):
    actual=sample_digest(sample)
    if actual!=expected:raise SourceDigestMismatch(dict(error='SourceDigestMismatch',source_index=int(index),
        expected_sha256=expected,observed_sha256=actual,**extra))

def direct_region(path,offset,size):
    path=Path(path)
    if path.resolve()!=path or path.is_symlink() or offset<0 or not 0<size<=1024*1024:
        raise ValueError('invalid bounded direct input region')
    before=identity(path.stat())
    if offset+size>before[2]:raise ValueError('source pointer beyond file')
    fd=os.open(path,os.O_RDONLY|os.O_DIRECT|os.O_NOFOLLOW)
    try:
        if identity(os.fstat(fd))!=before:raise RuntimeError('source stat changed')
        _,alignment=dio_alignment(fd)
        start=offset//4096*4096;length=((offset+size-start+4095)//4096)*4096
        with mmap.mmap(-1,length) as buf:
            got=os.preadv(fd,[buf],start)
            if got<offset-start+size:raise OSError('short independent source direct read')
            raw=bytes(buf[offset-start:offset-start+size])
        if identity(os.fstat(fd))!=before or identity(path.stat())!=before:raise RuntimeError('source identity changed')
        return raw
    finally:os.close(fd)

def verify_array(array,start,count=1):
    # Native indices are mmap-backed .npy arrays. Verify the exact accessed bytes
    # independently; invalid/changed source ownership is NOT digest quarantine.
    import numpy as np
    if not isinstance(array,np.memmap) or not array.filename:raise ValueError('native index is not pinned mmap')
    stride=array.dtype.itemsize*(array.shape[1] if array.ndim==2 else 1)
    raw=direct_region(array.filename,int(array.offset)+int(start)*stride,int(count)*stride)
    if raw!=array[start:start+count].tobytes():raise ValueError('source index mmap/direct disagreement: structural fatal')

def direct_text(source,index):
    import numpy as np
    if not 0<=index<len(source):raise ValueError('source index outside original full dataset')
    verify_array(source.dataset_index,index);verify_array(source.dataset_sample_index,index)
    did=int(source.dataset_index[index]);local=int(source.dataset_sample_index[index])
    if not 0<=did<len(source.datasets):raise ValueError('invalid blended dataset ID')
    g=source.datasets[did]
    if not 0<=local<len(g):raise ValueError('invalid native local sample ID')
    cfg=g.config
    if (cfg.sequence_length!=4096 or not cfg.add_extra_token_to_sequence or cfg.reset_position_ids
        or cfg.reset_attention_mask or cfg.eod_mask_loss or cfg.create_attention_mask):raise ValueError('unsupported source semantics')
    # Native __getitem__ loads lazy index maps before this routine is called.
    verify_array(g.shuffle_index,local);sh=int(g.shuffle_index[local])
    if not 0<=sh<len(g.sample_index)-1:raise ValueError('invalid native shuffled ID')
    verify_array(g.sample_index,sh,2)
    beg=g.sample_index[sh];end=g.sample_index[sh+1];first,offset=map(int,beg);last,endoff=map(int,end)
    if not 0<=first<=last<len(g.document_index) or last-first>4097:raise ValueError('invalid document boundary')
    verify_array(g.document_index,first,last-first+1)
    idx=g.dataset.index;dtype=np.dtype(idx.dtype);parts=[];docs=[]
    if dtype!=np.dtype('int32'):raise ValueError('unexpected original source bin dtype')
    base=np.frombuffer(idx.bin_buffer,dtype=np.uint8)
    for d in range(first,last+1):
        doc=int(g.document_index[d])
        if not 0<=doc<len(g.dataset):raise ValueError('invalid raw document ID')
        length=int(g.dataset.sequence_lengths[doc]);pointer=int(idx.sequence_pointers[doc])
        for name in ('sequence_lengths','sequence_pointers'):
            a=getattr(idx,name);off=int(a.ctypes.data-base.ctypes.data)+doc*a.dtype.itemsize
            if direct_region(g.dataset_path+'.idx',off,a.dtype.itemsize)!=a[doc:doc+1].tobytes():
                raise ValueError('raw idx metadata disagreement: structural fatal')
        lo=offset if d==first else 0;hi=endoff+1 if d==last else length
        if not 0<=lo<hi<=length:raise ValueError('invalid native raw document offsets')
        parts.append(np.frombuffer(direct_region(g.dataset_path+'.bin',pointer+lo*dtype.itemsize,(hi-lo)*dtype.itemsize),dtype=dtype).astype(np.int64))
        docs.append(doc)
    text=np.concatenate(parts)
    if len(text)!=4097 or int(text.min())<0 or int(text.max())>=131072:raise ValueError('independent source token schema invalid')
    return text,dict(dataset_id=did,local_sample=local,shuffled_sample=sh,documents=docs,dataset_path=g.dataset_path)

def verified_sample(source,index,expected=None,given=None):
    import torch
    # Clone now; accepted sample buffers must not alias mmap/native mask caches.
    sample={k:(v.clone() if hasattr(v,'clone') else v.copy()) for k,v in (source[index] if given is None else given).items()}
    validate_schema(sample)
    text,ownership=direct_text(source,index)
    if 'dataset_id' in sample and int(sample['dataset_id'])!=ownership['dataset_id']:raise ValueError('native batch dataset_id ownership mismatch')
    independent=dict(sample,tokens=torch.from_numpy(text[:-1].copy()),labels=torch.from_numpy(text[1:].copy()))
    compare_digest(sample,sample_digest(independent),index,oracle='same_docs_O_DIRECT',ownership=ownership)
    if expected is not None:compare_digest(sample,expected['sha256'],index,oracle='historical_teacher',ownership=ownership)
    return sample

class VerifiedSource:
    def __init__(self,source,expected):self.source=source;self.expected=expected
    def __len__(self):return len(self.source)
    def __getitem__(self,index):
        physical,offset=divmod(index,128);micro=offset//8
        return verified_sample(self.source,index,self.expected[physical*16+micro])
