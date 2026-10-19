"""Fail-closed sealed FD dataset and exact pre-forward row authentication.

No original-data loader, fallback, quarantine, resampling or optimizer exists here.
The public producer client is imported only when a snapshot is actually opened.
"""
from __future__ import annotations
import fcntl, hashlib, io, json, math, os
from pathlib import Path
import numpy as np

DOMAIN=b'nemochat-sft-row-v1\0'
IDS=['sft:train-tokens.npy','sft:train-masks.npy','sft:heldout-tokens.npy',
     'sft:heldout-masks.npy','sft:heldout-dialogues.jsonl','rowhash:train','rowhash:heldout']

def check(ok,message):
    if not ok:raise PermissionError(message)

def row_digest(row_index,token_row,mask_row):
    return hashlib.sha256(DOMAIN+int(row_index).to_bytes(8,'little')+
        np.ascontiguousarray(token_row,dtype='<u4').tobytes()+np.ascontiguousarray(mask_row,dtype='u1').tobytes()).digest()

def scalar(value):
    if isinstance(value,(list,tuple)):
        check(len(value)==1,'microbatch size must be one');value=value[0]
    if hasattr(value,'numel'):
        check(value.numel()==1,'scalar identity required');value=value.item()
    return value

def verify_item(item,expected_digest,expected_row,split,snapshot_id,sequence_length=16384):
    import torch
    check(scalar(item['snapshot_id'])==snapshot_id and scalar(item['split'])==split,'wrong snapshot/role')
    check(int(scalar(item['row_index']))==expected_row,'selected row ID mismatch')
    # Clone first, then authenticate those exact tensors, never a second source read.
    result={k:item[k].detach().clone() for k in ('tokens','labels','loss_mask','position_ids')}
    for key,t in result.items():
        check(t.device.type=='cpu' and tuple(t.shape) in ((sequence_length,),(1,sequence_length)), 'wrong actual input shape/device: '+key)
    tokens,labels,mask,pos=[result[k].reshape(-1) for k in ('tokens','labels','loss_mask','position_ids')]
    check(tokens.dtype==labels.dtype==pos.dtype==torch.int64,'native tokens/labels/positions require int64')
    check(mask.dtype==torch.float32,'native mask requires float32')
    check(bool(((tokens>=0)&(tokens<131072)).all()) and bool(((labels>=0)&(labels<131072)).all()),'token outside C vocabulary')
    check(torch.equal(tokens[1:],labels[:-1]),'actual label shift/overlap mismatch')
    check(torch.equal(pos,torch.arange(sequence_length)),'position IDs differ from native arange')
    check(bool(((mask==0)|(mask==1)).all()),'mask must be exactly 0/1')
    full=torch.cat((tokens,labels[-1:])).numpy().astype('<u4')
    if expected_row<0:
        check(expected_digest is None and bool((tokens==11).all()) and bool((labels==11).all()) and not bool(mask.any()),'invalid native zero-loss tail')
    else:
        check(row_digest(expected_row,full,mask.numpy())==bytes(expected_digest),'actual native input row SHA256 mismatch')
    return result

def collective_read(read,verify,reduce_success):
    error=None;result=None
    try:result=verify(read())
    except Exception as exc:error=exc
    okay=reduce_success(error is None)
    if not okay:raise RuntimeError('all-rank fatal input authentication/read failure; no optimizer/LR advance') from error
    return result

class Snapshot:
    def __init__(self,ready,manifest):
        from midtraining.sft_after_kd.sealed_snapshot03.client import open_snapshot
        self.ready=ready;self.manifest=manifest;self.opened=open_snapshot(ready,IDS)
        check(set(self.opened)==set(IDS),'missing received descriptor')
        for fid,obj in self.opened.items():
            info=ready['files'][fid]
            check(fcntl.fcntl(obj.fd,fcntl.F_GET_SEALS)==15,'FD not fully sealed')
            check(fcntl.fcntl(obj.fd,fcntl.F_GETFL)&os.O_ACCMODE==os.O_RDONLY,'FD not read only')
            check(os.fstat(obj.fd).st_size==info['bytes'],'FD size mismatch')
        for split in ('train','heldout'):
            n=manifest[split+'_dialogues'];seq=manifest['sequence_length']
            for name,shape,dtype in [('tokens',(n,seq+1),np.dtype('<u4')),('masks',(n,seq),np.dtype('u1'))]:
                a=self.opened[f'sft:{split}-{name}.npy'].array
                check(a.shape==shape and a.dtype==dtype and not a.flags.writeable,'sealed array header/type/shape mismatch')
            a=self.opened['rowhash:'+split].array
            check(a.shape==(n,32) and a.dtype==np.uint8 and not a.flags.writeable,'row catalog header/type mismatch')
        self.datasets={}
    def heldout_groups(self):
        from collections import defaultdict
        groups=defaultdict(list);fd=self.opened['sft:heldout-dialogues.jsonl'].fd
        # Independent seek offset; never close or seek producer-owned descriptor.
        size=os.fstat(fd).st_size;offset=0;pending=b'';index=0
        while offset<size:
            block=os.pread(fd,min(1024*1024,size-offset),offset)
            check(bool(block),'sealed metadata short read');offset+=len(block);pending+=block
            lines=pending.split(b'\n');pending=lines.pop()
            for line in lines:
                if line:groups[json.loads(line)['source_key']].append(index);index+=1
        if pending:groups[json.loads(pending)['source_key']].append(index);index+=1
        check(index==self.manifest['heldout_dialogues'],'sealed heldout metadata count mismatch')
        return groups
    def verify(self,item,expected_split=None):
        split=scalar(item['split']);check(split in self.datasets,'unregistered split')
        if expected_split is not None:check(split==expected_split,'wrong train/validation role at native forward')
        dataset=self.datasets[split];index=int(scalar(item['source_index']))
        row=dataset.expected_row(index)
        digest=None if row<0 else self.opened['rowhash:'+split].array[row].tobytes()
        return verify_item(item,digest,row,split,self.ready['snapshot_id'],dataset.sequence_length)
    def close(self):
        # Do not call until native datasets and all views are no longer used.
        for obj in self.opened.values():obj.close()

def rows_class(base,snapshot):
    class SealedRows(base):
        def __init__(self,data,split,samples,batch=128,*,epochs=1,seed=42,window=None,sequence_length=4096):
            check(sequence_length==16384 and epochs==1,'sealed03 requires unique one epoch seq16384')
            self.snapshot=snapshot # strong holder reference outlives numpy views
            self.tokens=snapshot.opened[f'sft:{split}-tokens.npy'].array
            self.masks=snapshot.opened[f'sft:{split}-masks.npy'].array
            self.n=len(self.tokens);self.samples=samples;self.split=split;self.batch=batch
            self.epoch_samples=math.ceil(self.n/batch)*batch;self.sequence_length=sequence_length
            check(samples>=1 and (split!='train' or samples<=self.epoch_samples),'native sample budget exceeded')
            self.orders=[np.random.default_rng(seed).permutation(self.n)] if split=='train' else [np.arange(min(self.n,window or self.n))]
            self.fixed_window=split!='train' and window is not None
            if self.fixed_window:
                from collections import deque
                groups=snapshot.heldout_groups()
                queues=[deque(np.random.default_rng(seed).permutation(groups[key]).tolist()) for key in sorted(groups)]
                selected=[]
                while len(selected)<min(window,self.n):
                    for queue in queues:
                        if queue and len(selected)<min(window,self.n):selected.append(queue.popleft())
                self.orders=[np.asarray(selected)]
            self.order=self.orders[0];self.window_samples=math.ceil(len(self.order)/batch)*batch
            snapshot.datasets[split]=self
        def expected_row(self,index):
            check(0<=index<self.samples,'native source cursor outside allocation')
            epoch,offset=divmod(index,self.epoch_samples) if self.split=='train' else (0,index%(self.window_samples if self.fixed_window else len(self.order)))
            return -1 if (self.split=='train' and offset>=self.n) or (self.fixed_window and offset>=len(self.order)) else int(self.orders[epoch][offset])
        def __getitem__(self,index):
            import torch
            row=self.expected_row(index)
            if row<0:
                tok=torch.full((self.sequence_length+1,),11,dtype=torch.long);mask=torch.zeros(self.sequence_length)
            else:
                tok=torch.tensor(self.tokens[row].astype(np.int64));mask=torch.tensor(self.masks[row],dtype=torch.float32)
            return dict(tokens=tok[:-1],labels=tok[1:],loss_mask=mask,position_ids=torch.arange(self.sequence_length),row_index=row,source_index=index,split=self.split,snapshot_id=snapshot.ready['snapshot_id'])
    return SealedRows

def guarded_forward(forward,snapshot,dist,torch):
    def wrapped(iterator,model):
        def reduce_success(ok):
            backend=dist.get_backend()
            device=torch.cuda.current_device() if backend=='nccl' else 'cpu'
            flag=torch.tensor(int(ok),device=device,dtype=torch.int32)
            dist.all_reduce(flag,op=dist.ReduceOp.MIN)
            return bool(flag.item())
        def verify_and_stage(item):
            check(hasattr(model,'training'),'native model train/eval role unavailable')
            item=snapshot.verify(item,'train' if model.training else 'heldout')
            # Move verified clones once before the success collective. Frozen
            # native forward .to(same device) keeps exactly these tensor objects.
            if dist.get_backend()=='nccl':item={k:t.to(torch.cuda.current_device()) for k,t in item.items()}
            return item
        item=collective_read(lambda:next(iterator),verify_and_stage,reduce_success)
        return forward(iter([item]),model)
    return wrapped
