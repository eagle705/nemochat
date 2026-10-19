#!/usr/bin/env python3
"""CPU-only fixture of pinned multiprocessing and MCore storage APIs."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import tempfile


def close_queue(queue):
    # SyncManager Queue is a proxy, not multiprocessing.Queue. In a spawned
    # child this pinned AutoProxy has no _close finalizer; close its real IPC
    # connection instead. The parent still owns manager shutdown.
    close=getattr(queue,'_close',None)
    if close is not None:close()
    else:queue._tls.connection.close()


def queue_child(queue):
    import tempfile
    item=queue.get(timeout=20)
    queue.put(dict(item=item,tmpdir=os.environ['TMPDIR'],cached_tempdir=tempfile.gettempdir(),pid=os.getpid()))
    close_queue(queue)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inspect',action='store_true');args=parser.parse_args()
    import torch
    import torch.multiprocessing as mp
    from megatron.core import dist_checkpointing as dc
    from megatron.core.dist_checkpointing.mapping import ShardedTensor
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy,TorchDistLoadShardedStrategy
    assert os.environ['TMPDIR']==os.environ['TMP']==os.environ['TEMP']=='/nct'
    assert tempfile.gettempdir()=='/nct' and not torch.cuda.is_available()
    if args.inspect:
        for obj in (dc.save,dc.load,TorchDistSaveShardedStrategy,TorchDistLoadShardedStrategy,ShardedTensor.from_rank_offsets):
            (args.output/(obj.__name__+'.txt')).write_text(inspect.getsource(obj))
        return
    import ctypes
    import errno
    import multiprocessing.managers
    from unittest.mock import patch
    import torch.distributed as dist
    paths=[]
    original_start=multiprocessing.managers.SyncManager.start
    def observed_start(manager,*a,**kw):
        result=original_start(manager,*a,**kw)
        address=manager._address
        assert isinstance(address,str) and address.startswith('/nct/') and len(os.fsencode(address))<108
        paths.append(dict(path=address,bytes=len(os.fsencode(address)),manager_pid=manager._process.pid))
        return result
    with patch.object(multiprocessing.managers.SyncManager,'start',observed_start):
        manager=mp.Manager();queue=manager.Queue()
        queue.put('fixture-roundtrip')
        child=mp.get_context('spawn').Process(target=queue_child,args=(queue,));child.start();child.join(45)
        assert child.exitcode==0
        roundtrip=queue.get(timeout=20)
        assert roundtrip['item']=='fixture-roundtrip' and roundtrip['tmpdir']==roundtrip['cached_tempdir']=='/nct'
        close_queue(queue);manager.shutdown()
        assert not Path(paths[0]['path']).exists()
        # Exact symptom negative control in a fresh Python process before tempfile caching.
        long=Path('/nct')/('long-task-private-'+'x'*95);long.mkdir()
        negative=subprocess.run(['python','-c','import torch.multiprocessing as m; m.Manager().Queue()'],env=dict(os.environ,TMPDIR=str(long),TMP=str(long),TEMP=str(long)),capture_output=True,text=True,timeout=60)
        (args.output/'long-manager-negative.log').write_text(negative.stdout+negative.stderr)
        assert negative.returncode!=0 and 'AF_UNIX path too long' in negative.stderr
        # Actual executable and compiled shared-library lifecycle, not a mount-string assertion.
        fixture=Path(tempfile.mkdtemp(prefix='exec-'))
        source=fixture/'probe.c';source.write_text('int value(void){return 37;}\nint main(void){return value()==37?0:1;}\n')
        subprocess.run(['cc',str(source),'-o',str(fixture/'probe')],check=True)
        subprocess.run([str(fixture/'probe')],check=True)
        subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(fixture/'probe.so')],check=True)
        assert ctypes.CDLL(str(fixture/'probe.so')).value()==37
        try:Path('/runtime05-root-must-be-readonly').write_text('forbidden')
        except OSError as exc:assert exc.errno==errno.EROFS
        else:raise AssertionError('container root is writable')
        mount=[line for line in Path('/proc/self/mountinfo').read_text().splitlines() if line.split()[4]=='/nct']
        assert len(mount)==1 and ' - tmpfs ' in mount[0] and 'noexec' not in mount[0]
        status=Path('/proc/self/status').read_text();assert 'NoNewPrivs:\t1' in status
        manager_receipt=dict(status='PINNED_TORCH_MANAGER_QUEUE_PASS',paths=paths,roundtrip=roundtrip,queue_proxy_closed=True,manager_shutdown=True,spawn_inherits_short_tmpdir=True,long_path_negative='AF_UNIX path too long',executable_and_shared_object_loaded=True,readonly_root=True,mountinfo=mount,torch_version=torch.__version__)
        (args.output/'manager-proof.json').write_text(json.dumps(manager_receipt,indent=2)+'\n')
        # Real native MCore torch_dist synchronous strategy, CPU fixture only.
        rendezvous=Path('/nct')/'gloo-rendezvous'
        dist.init_process_group('gloo',init_method='file://'+str(rendezvous),rank=0,world_size=1)
        try:
            tensor=torch.arange(24,dtype=torch.float32).reshape(4,6)
            checkpoint=args.output/'tiny-checkpoint';checkpoint.mkdir()
            save_strategy=TorchDistSaveShardedStrategy(backend='torch_dist',version=1)
            save={'fixture':ShardedTensor.from_rank_offsets('fixture',tensor),'fixture_metadata':{'step':16,'source':'CPU_fixture_not_model'}}
            before=len(paths)
            dc.save(save,str(checkpoint),sharded_strategy=save_strategy,validate_access_integrity=True,async_sharded_save=False,verify_integrity=False)
            assert len(paths)>before,'native save did not exercise its real SyncManager'
            target={'fixture':ShardedTensor.from_rank_offsets('fixture',torch.full_like(tensor,-1))}
            loaded=dc.load(target,str(checkpoint),sharded_strategy=TorchDistLoadShardedStrategy(),validate_access_integrity=True,verify_integrity=False)
            assert torch.equal(loaded['fixture'],tensor)
            assert loaded['fixture_metadata']=={'step':16,'source':'CPU_fixture_not_model'}
            assert (checkpoint/'metadata.json').is_file()
            proof=dict(status='PINNED_CPU_MANAGER_AND_NATIVE_TORCH_DIST_STORAGE_PASS',manager=manager_receipt,all_manager_paths=paths,native_save_manager_count=len(paths)-before,native_strategy=type(save_strategy).__name__,native_backend=save_strategy.backend,tensor_shape=list(tensor.shape),tensor_dtype=str(tensor.dtype),tensor_elements=tensor.numel(),loaded_exact_tensor_equal=True,loaded_common_state=True,verify_integrity=False,no_payload_sha_crc=True,no_model_weights=True,no_optimizer=True,gpu_requests=0,gloo_world_size=1,full_model_save_reload_proven=False)
            (args.output/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
            print(json.dumps(proof,indent=2))
        finally:dist.destroy_process_group()

if __name__=='__main__':main()
