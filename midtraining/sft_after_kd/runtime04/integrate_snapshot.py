#!/usr/bin/env python3
"""Parent-run actual snapshot adapter integration, CPU only, no waiting/polling.

Uses real producer client, never test-local fixture. This command does NOT start
training or authorize a release; its output is a parent-review CPU receipt.
"""
import argparse,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import controller as c
from readiness import validate_ready

def worker(args):
    import torch,torch.distributed as dist
    import sealed_input as s
    from types import SimpleNamespace
    torch.set_num_threads(1);dist.init_process_group('gloo');rank=dist.get_rank()
    c.require(dist.get_world_size()==8,'all eight CPU ranks required')
    c.require(c.sha(args.ready)==args.ready_sha256,'raw ready changed before actual client integration')
    ready=validate_ready(c.read(args.ready),c,False);manifest=c.read(c.MANIFEST)
    snapshot=s.Snapshot(ready,manifest)
    spec=__import__('importlib').util.spec_from_file_location('sealed03_actual_native_rows',c.ROOT/'midtraining/sft_after_kd/native_sft_v1.py')
    base=__import__('importlib').util.module_from_spec(spec);spec.loader.exec_module(base)
    Rows=s.rows_class(base.PreparedRows,snapshot)
    train=Rows(None,'train',c.CAPACITY*128,128,sequence_length=16384)
    heldout=Rows(None,'heldout',128,8,window=128,sequence_length=16384)
    seen=[]
    def forward(iterator,model):
        item=next(iterator);seen.append({'token_shape':list(item['tokens'].shape),'loss_targets':int(item['loss_mask'].sum())})
        return True
    guarded=s.guarded_forward(forward,snapshot,dist,torch)
    for index in (rank,rank+120,rank+2040):
        guarded(iter([train[index]]),SimpleNamespace(training=True))
    guarded(iter([heldout[rank]]),SimpleNamespace(training=False))
    dist.barrier()
    c.write(args.output/f'rank-{rank}.json',dict(status='ACTUAL_SEALED_CLIENT_CPU_ADAPTER_PASS',rank=rank,
        snapshot_id=ready['snapshot_id'],ready_sha256=args.ready_sha256,received_fds=len(snapshot.opened),
        producer_client='REAL_PUBLIC_CLIENT',verified_rows=seen,no_gpu=True,no_model=True,no_optimizer=True))
    dist.destroy_process_group()

def parent(args):
    ready=validate_ready(c.read(args.ready),c,True);digest=c.sha(args.ready)
    output=c.OUT/'actual-sealed-cpu';output.mkdir(parents=True,exist_ok=False)
    payload=c.container_payload('config',c.OUT/'unused',300,config_only=True)
    home=c.OUT/'actual-sealed-cpu-home'
    for part in ('tmp','inductor','triton','cuda'):(home/part).mkdir(parents=True,exist_ok=False)
    payload['Env']=[x for x in payload['Env'] if not x.startswith(('HOME=','TMPDIR=','TORCHINDUCTOR_CACHE_DIR=','TRITON_CACHE_DIR=','CUDA_CACHE_PATH=','OMP_NUM_THREADS='))]
    payload['Env']+=['HOME='+str(home),'TMPDIR='+str(home/'tmp'),'OMP_NUM_THREADS=1']
    payload['Labels']={c.OWNER:'actual-sealed-cpu','nemochat.sealed03':'v1'}
    directory=str(Path(ready['socket_path']).parent)
    payload['HostConfig']['Binds'].append(directory+':'+directory+':ro')
    payload['HostConfig']['Memory']=64*1024**3
    payload['Cmd']=['--signal=TERM','--kill-after=30','300','python','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=8','--master_addr=127.0.0.1','--master_port=29639',str(HERE/'integrate_snapshot.py'),'--worker','--ready',str(args.ready),'--ready-sha256',digest,'--output',str(output)]
    e=c.engine();name='nemochat-sft-sealed03-r04-actual-cpu'
    c.write(c.OUT/'actual-cpu-request.json',payload)
    cid=e.request('POST','/containers/create?name='+name,payload)['Id'];c.write(c.OUT/'actual-cpu-container.json',{'id':cid})
    def exact():
        a=e.request('GET','/containers/'+cid+'/json')
        c.require(a['Id']==cid and a['Name']=='/'+name and a['Config']['Labels'].get(c.OWNER)=='actual-sealed-cpu' and a['Config']['Labels'].get('nemochat.sealed03')=='v1' and not a['HostConfig'].get('DeviceRequests'),'actual CPU owned Engine mismatch')
        return a
    exact();e.request('POST','/containers/'+cid+'/start');start=time.monotonic()
    try:
        while exact()['State']['Running']:
            if time.monotonic()-start>340:raise TimeoutError('actual CPU adapter integration timed out')
            time.sleep(2)
    finally:
        a=exact()
        if a['State']['Running']:e.request('POST','/containers/'+cid+'/stop?t=30')
        a=exact();c.write(c.OUT/'actual-cpu-final.json',{'id':cid,'state':a['State']});c.capture_logs(e,cid,'actual-cpu')
    c.require(a['State']['ExitCode']==0 and not a['State']['OOMKilled'],'actual client CPU integration failed')
    ranks=[c.read(output/f'rank-{rank}.json') for rank in range(8)]
    c.require(all(row['rank']==rank and row['ready_sha256']==digest for rank,row in enumerate(ranks)),'actual CPU rank identity mismatch')
    receipt=dict(status='ACTUAL_SEALED_CLIENT_CPU_ADAPTER_PASS',ready_sha256=digest,snapshot_id=ready['snapshot_id'],holder_identity=ready['holder_identity'],runtime_source_freeze_sha256=c.sha(c.OUT/'source-freeze.json'),container_id=cid,ranks=ranks,gpu_launches=0)
    c.write(c.OUT/'actual-sealed-cpu-proof.json',receipt);print(json.dumps(receipt,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ready',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--ready-sha256');p.add_argument('--output',type=Path)
    a=p.parse_args();a.ready=a.ready.resolve()
    if a.worker:worker(a)
    else:parent(a)

if __name__=='__main__':main()
