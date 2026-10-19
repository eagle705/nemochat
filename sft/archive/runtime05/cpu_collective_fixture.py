"""Test-local SCM_RIGHTS public API fixture, never imported by production.

Run with torchrun --nproc_per_node=8 on CPU. Exercises production Snapshot,
SealedRows, exact cloned native-forward inputs and all-rank MIN rejection.
Does not exercise producer server authorization or real full03 bytes.
"""
import array, fcntl, hashlib, importlib.util, io, json, mmap, os, socket, sys, threading, types
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
sys.path.insert(0,str(Path(__file__).resolve().parent))
import sealed_input as s

class Opened:
    def __init__(self,fd,info):
        self.fd=fd;self.mapping=mmap.mmap(fd,0,access=mmap.ACCESS_READ)
        if info['format']=='npy':
            stream=io.BytesIO(self.mapping[:256]);version=np.lib.format.read_magic(stream)
            shape,fortran,dtype=np.lib.format._read_array_header(stream,version)
            self.array=np.ndarray(shape,dtype=dtype,buffer=self.mapping,offset=stream.tell(),order='F' if fortran else 'C')
        elif info['format']=='catalog':self.array=np.ndarray(tuple(info['shape']),dtype='u1',buffer=self.mapping)
        else:self.array=None
    def close(self):
        self.array=None;self.mapping.close();os.close(self.fd)

def fixture_client(ready,file_ids):
    conn=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET);conn.connect(ready['socket_path'])
    conn.send(json.dumps(file_ids).encode());result={}
    for fid in file_ids:
        data,anc,flags,addr=conn.recvmsg(4096,socket.CMSG_SPACE(4))
        if flags&(socket.MSG_TRUNC|socket.MSG_CTRUNC):raise RuntimeError('truncated fixture SCM frame')
        info=json.loads(data);fds=array.array('i');fds.frombytes(anc[0][2]);assert len(fds)==1
        assert info['file_id']==fid and fcntl.fcntl(fds[0],fcntl.F_GET_SEALS)==15
        result[fid]=Opened(fds[0],info)
    conn.close();return result

def make_fixture(directory):
    n=17;seq=16384;files={};fds={}
    for split in ('train','heldout'):
        tokens=(np.arange(n*(seq+1),dtype='<u4')%131072).reshape(n,seq+1)
        masks=np.zeros((n,seq),dtype='u1');masks[:,7::3]=1
        catalog=np.stack([np.frombuffer(s.row_digest(i,tokens[i],masks[i]),dtype='u1') for i in range(n)])
        for fid,value,fmt in [(f'sft:{split}-tokens.npy',tokens,'npy'),(f'sft:{split}-masks.npy',masks,'npy'),('rowhash:'+split,catalog,'catalog')]:
            buffer=io.BytesIO()
            if fmt=='npy':np.save(buffer,value);raw=buffer.getvalue()
            else:raw=value.tobytes()
            fd=os.memfd_create(fid,os.MFD_ALLOW_SEALING);os.write(fd,raw);fcntl.fcntl(fd,fcntl.F_ADD_SEALS,15)
            ro=os.open('/proc/self/fd/'+str(fd),os.O_RDONLY);os.close(fd);fds[fid]=ro
            files[fid]=dict(file_id=fid,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),shape=list(value.shape),dtype=str(value.dtype),seals=15,access_mode='O_RDONLY',format=fmt)
    raw=b''.join(json.dumps({'source_key':'group-'+str(i%3)}).encode()+b'\n' for i in range(n))
    fid='sft:heldout-dialogues.jsonl';fd=os.memfd_create(fid,os.MFD_ALLOW_SEALING);os.write(fd,raw);fcntl.fcntl(fd,fcntl.F_ADD_SEALS,15)
    fds[fid]=os.open('/proc/self/fd/'+str(fd),os.O_RDONLY);os.close(fd)
    files[fid]=dict(file_id=fid,bytes=len(raw),format='jsonl')
    sock=socket.socket(socket.AF_UNIX,socket.SOCK_SEQPACKET);path=str(directory/'fixture.sock');sock.bind(path);sock.listen(16)
    errors=[]
    def server():
        try:
            for _ in range(8):
                conn,_=sock.accept();ids=json.loads(conn.recv(4096))
                for fid in ids:conn.sendmsg([json.dumps(files[fid]).encode()],[(socket.SOL_SOCKET,socket.SCM_RIGHTS,array.array('i',[fds[fid]]))])
                conn.close()
        except BaseException as exc:errors.append(repr(exc))
    thread=threading.Thread(target=server);thread.start()
    return dict(snapshot_id='CPU-FIXTURE-NOT-PRODUCTION',socket_path=path,files=files),dict(train_dialogues=n,heldout_dialogues=n,sequence_length=seq),thread,errors

def main():
    torch.set_num_threads(1);dist.init_process_group('gloo');rank=dist.get_rank()
    assert dist.get_world_size()==8
    directory=Path(sys.argv[1]);directory.mkdir(parents=True,exist_ok=True)
    payload=[None];thread=None
    if rank==0:
        ready,manifest,thread,errors=make_fixture(directory);payload[0]=(ready,manifest)
    dist.broadcast_object_list(payload,src=0);ready,manifest=payload[0]
    # Exact test-local stub for the declared but independently-owned public API.
    module=types.ModuleType('sft.archive.sealed_snapshot03.client');module.open_snapshot=fixture_client
    sys.modules[module.__name__]=module
    snapshot=s.Snapshot(ready,manifest)
    path=Path(__file__).resolve().parents[3]/'sft/training/native_sft_v1.py'
    spec=importlib.util.spec_from_file_location('cpu_frozen_rows',path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    rows=s.rows_class(base.PreparedRows,snapshot)
    dataset=rows('/MUST_NOT_BE_OPENED','train',128,batch=128,sequence_length=16384)
    heldout=rows('/MUST_NOT_BE_OPENED','heldout',128,batch=8,window=16,sequence_length=16384)
    assert np.array_equal(dataset.order,np.random.default_rng(42).permutation(17))
    assert dataset.expected_row(17)==-1 and dataset.expected_row(127)==-1
    calls=[];last=[]
    def native_forward(iterator,model):
        item=next(iterator);calls.append(1);last.append(item)
        assert item['tokens'].shape==(16384,) and item['labels'].dtype==torch.int64
        return 'NATIVE_FORWARD_FIXTURE'
    forward=s.guarded_forward(native_forward,snapshot,dist,torch)
    assert forward(iter([dataset[rank]]),types.SimpleNamespace(training=True))=='NATIVE_FORWARD_FIXTURE'
    assert forward(iter([heldout[rank]]),types.SimpleNamespace(training=False))=='NATIVE_FORWARD_FIXTURE'
    assert forward(iter([dataset[127]]),types.SimpleNamespace(training=True))=='NATIVE_FORWARD_FIXTURE'
    rejected=[]
    for fault in ('row','tokens','labels','mask','position','dtype','role','snapshot','read'):
        item=dataset[rank]
        if rank==7:
            if fault=='row':item['row_index']+=1
            elif fault=='tokens':item['tokens'][0]+=1
            elif fault=='labels':item['labels'][-1]+=1
            elif fault=='mask':item['loss_mask'][1]=.5
            elif fault=='position':item['position_ids'][0]+=1
            elif fault=='dtype':item['tokens']=item['tokens'].float()
            elif fault=='role':item['split']='heldout'
            elif fault=='snapshot':item['snapshot_id']='wrong'
        class Reader:
            def __next__(self):
                if fault=='read' and rank==7:raise OSError('fixture native loader error')
                return item
        try:forward(Reader(),types.SimpleNamespace(training=True))
        except RuntimeError:rejected.append(fault)
        else:raise AssertionError('all ranks must reject '+fault)
    assert len(calls)==3 and len(rejected)==9
    dist.barrier()
    (directory/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,status='CPU_FIXTURE_SEALED_FD_AND_ALLRANK_GUARD_PASS',received_fds=7,good_forward_fixture_calls=3,rejected_faults=rejected,no_model=True,no_optimizer=True,producer_client='TEST_LOCAL_PUBLIC_API_STUB',native_sampler_order_retained=True)))
    if thread:thread.join(30);assert not thread.is_alive() and not errors
    dist.destroy_process_group()

if __name__=='__main__':main()
