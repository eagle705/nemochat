#!/usr/bin/env python3
"""Summarize the preserved genuine no-device native-save blocker; no retry."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import controller as c


def main():
    directory=c.OUT/'storage-cpu-03'
    log=(c.OUT/'storage-cpu-03-engine.log').read_text()
    required=['dc.save(save,str(checkpoint)','sharded_strategy.save(sharded_state_dict, checkpoint_dir)',
        'async_request.execute_sync()','filesystem_async.py", line 248, in preload_tensors',
        'torch.cuda.synchronize()','RuntimeError: No CUDA GPUs are available']
    c.require(all(text in log for text in required),'not the reviewed pinned native CPU CUDA blocker')
    old=c.read(directory/'source-freeze.executed-cpu03.json')
    executed=dict()
    for name in ('cpu_storage.py','controller.py'):
        path=str((c.HERE/name).relative_to(c.ROOT));digest=c.sha(c.HERE/name)
        c.require(old[path]==digest,'executed CPU storage implementation changed: '+name);executed[path]=digest
    manager=c.read(directory/'manager-proof.json')
    c.require(manager['status']=='PINNED_TORCH_MANAGER_QUEUE_PASS' and manager['queue_proxy_closed'] and manager['manager_shutdown'] and manager['executable_and_shared_object_loaded'],'actual Manager/exec evidence missing')
    c.require(all(x['bytes']==len(x['path'].encode()) and x['bytes']<108 and x['path'].startswith('/nct/') for x in manager['paths']),'actual manager path budget missing')
    cid=c.read(directory/'container.json')['id'];a=c.engine().request('GET','/containers/'+cid+'/json')
    c.require(a['Id']==cid and a['Name']=='/nemochat-sft-sealed03-r05-storage-cpu-03' and a['State']['Running'] is False and a['State']['ExitCode']==1 and not a['State']['OOMKilled'] and not a['HostConfig'].get('DeviceRequests'),'exact failed no-device CPU Engine required')
    c.require(dict(x.split('=',1) for x in a['Config']['Env'])['TMPDIR']=='/nct' and 'exec' in a['HostConfig']['Tmpfs']['/nct'].split(','),'short executable task-private temp absent')
    proof=dict(status='PINNED_CPU_MANAGER_PASS_NATIVE_STORAGE_BLOCKED_CUDA',source_freeze_sha256=c.sha(c.OUT/'source-freeze.json'),execution_source_freeze_sha256=c.sha(directory/'source-freeze.executed-cpu03.json'),executed_source_verified_unchanged=executed,container_id=cid,attempt='03',
        manager=manager,all_manager_paths=manager['paths'],native_save_manager_count=None,
        native_backend='torch_dist',native_strategy='TorchDistSaveShardedStrategy',native_save_attempted=True,native_save_completed=False,native_load_executed=False,loaded_exact_tensor_equal=False,
        blocker='Pinned MCore filesystem_async.py:248 preload_tensors unconditionally calls torch.cuda.synchronize(); RuntimeError: No CUDA GPUs are available',
        blocker_stack_log=str(c.OUT/'storage-cpu-03-engine.log'),blocker_log_sha256=c.sha(c.OUT/'storage-cpu-03-engine.log'),engine_state=a['State'],
        no_cuda_mock_or_sdk_patch=True,verify_integrity=False,no_payload_sha_crc=True,no_model_weights=True,no_optimizer=True,gpu_requests=0,full_model_save_reload_proven=False)
    c.write(c.OUT/'storage-cpu-proof.json',proof);print(json.dumps(proof,indent=2))

if __name__=='__main__':main()
