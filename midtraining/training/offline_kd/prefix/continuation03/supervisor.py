"""Durable, exact-identity continuation03 GPU supervisor (repair or student)."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[5]
OUT = REPO/'results/midtraining/offline2b-prefixtrain-01/continuation-03'
SOURCE = REPO/'results/midtraining/offline2b-production1920-01'
sys.path.insert(0, str(REPO/'midtraining/training/offline_kd'))
from launch import IMAGE, freeze
LABEL = 'nemochat.prefix.continuation03.owner'


def publish(name, value):
    path = OUT/(name+'.json'); temporary = path.with_suffix('.json.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)
    print(name, json.dumps(value), flush=True)


def engine():
    spec = importlib.util.spec_from_file_location('e', REPO/'results/pretrain_followup/engine_helper.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def owned(e, cid, stage):
    actual = e.request('GET', f'/containers/{cid}/json')
    if actual['Id'] != cid or actual['Config']['Labels'].get(LABEL) != stage:
        raise ValueError('wrong exact container ownership')
    return actual


def logs(e, cid, stage):
    connection = e.Connection('localhost', timeout=60)
    connection.request('GET', f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1')
    raw = connection.getresponse().read(); connection.close(); parts = []
    while len(raw) >= 8:
        n = struct.unpack('>I', raw[4:8])[0]; parts.append(raw[8:8+n]); raw = raw[8+n:]
    (OUT/f'{stage}-engine.log').write_bytes(b''.join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['repair', 'student'])
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args(); stage = args.stage
    e = engine()
    if args.cleanup:
        receipt = OUT/f'{stage}-engine-created.json'
        if receipt.exists():
            cid = json.loads(receipt.read_text())['id']; actual = owned(e, cid, stage)
            if actual['State']['Running']: e.request('POST', f'/containers/{cid}/stop?t=20')
            publish(f'{stage}-cleanup', dict(id=cid, state=owned(e, cid, stage)['State']))
        return
    expected = json.loads((OUT/f'{stage}-source-freeze.json').read_text())
    actual = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    if expected != dict(prefix=actual, original=freeze()): raise ValueError('source freeze changed')
    unit = f'nemochat-offline2b-prefixtrain-01-continuation03-{stage}'+('02' if stage == 'repair' else '')+'.service'
    pid = int(subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'], text=True))
    if pid != os.getpid() or Path(f'/proc/{pid}').stat().st_uid != os.getuid(): raise ValueError('wrong unit owner')
    publish(f'{stage}-ownership', dict(unit=unit, MainPID=pid, uid=os.getuid(), argv=sys.argv, observed=time.time()))
    lock = (REPO/'results/pretraining/nemochat_baby/training/gpu-all8.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    rows = [list(map(int, line.replace(' ', '').split(','))) for line in gpus.splitlines()]
    if len(rows) != 8 or apps.strip() or any(mem > 1024 or util > 0 for _, mem, util in rows): raise ValueError('all8 not idle')
    publish(f'{stage}-gpu-preflight', dict(gpus=rows, apps=apps, lock=str(lock.name)))
    if stage == 'repair':
        from repair_contract import repair_plan
        request = json.loads((OUT/'repair-request.json').read_text())
        repair_plan(request['iterations'])
        if not request.get('parent_approved'): raise ValueError('parent sparse repair approval required')
    else:
        accepted = json.loads((OUT/'cache-acceptance.json').read_text())
        if accepted['status'] != 'all_closed_shards_codec_validated': raise ValueError('parent cache gate required')
        if accepted['shards'] != 9144 or accepted['coverage'] != [0,1142]: raise ValueError('full completed prefix required')
    scratch = Path(tempfile.mkdtemp(prefix=f'offline1920-prefix03-{stage}-', dir='/home/yoon/.hermes/cache/scratch'))
    old = Path(json.loads((OUT.parent/'scratch.json').read_text())['path'])
    shutil.copytree(old/'sdk', scratch/'sdk'); shutil.copytree(old/'staged', scratch/'staged')
    # NEVER copy generated compiler caches with embedded old absolute paths.
    for name in ('inductor', 'triton', 'home'): (scratch/name).mkdir()
    publish(f'{stage}-scratch', dict(path=str(scratch), sdk_source=str(old), installed_sdk_unchanged=True))
    if not (OUT/'dataset-cache').exists() and (SOURCE/'dataset-cache').exists():
        shutil.copytree(SOURCE/'dataset-cache', OUT/'dataset-cache')
    entry = HERE/('repair_native.py' if stage == 'repair' else 'native.py')
    cmd = ['-m', 'torch.distributed.run', '--nproc_per_node=8', '--nnodes=1', '--node_rank=0',
        '--master_addr=127.0.0.1', '--master_port=29686', str(entry)] + ([] if stage == 'repair' else ['student'])
    payload = dict(Image=IMAGE, User=f'{os.getuid()}:{os.getgid()}', WorkingDir='/repo', Entrypoint=['python'], Cmd=cmd,
        Labels={LABEL:stage}, Env=[f'PYTHONPATH={scratch}/staged/src:{scratch}/sdk:{HERE}:{HERE.parent}:/repo/midtraining/patches/offline_kd_pr/gpu_smoke:/repo/src',
        'PYTHONDONTWRITEBYTECODE=1', f'TMPDIR={scratch}', f'HOME={scratch}/home', 'USER=yoon', 'LOGNAME=yoon',
        'WANDB_MODE=disabled', 'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'TOKENIZERS_PARALLELISM=false',
        'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'NVIDIA_DRIVER_CAPABILITIES=compute,utility',
        'OMP_NUM_THREADS=4', 'MKL_NUM_THREADS=4', 'TORCHINDUCTOR_COMPILE_THREADS=1',
        f'TORCHINDUCTOR_CACHE_DIR={scratch}/inductor', f'TRITON_CACHE_DIR={scratch}/triton',
        'PYTORCH_ALLOC_CONF=expandable_segments:True', 'NCCL_IB_DISABLE=1'],
        HostConfig=dict(Runtime='runc', NetworkMode='none', ReadonlyRootfs=True, CapDrop=['ALL'], SecurityOpt=['no-new-privileges'],
        ShmSize=32*1024**3, Memory=1500*1024**3, NanoCpus=64*10**9, PidsLimit=4096,
        DeviceRequests=[dict(Driver='nvidia', DeviceIDs=[str(i) for i in range(8)], Capabilities=[['gpu']])],
        Binds=[f'{OUT.parent}:/initial:ro', f'{REPO}:/repo:ro', f'{REPO}:{REPO}:ro', f'{OUT}:/output:rw',
            f'{OUT}:{OUT}:ro', f'{scratch}:{scratch}:rw']))
    generation = stage+('02' if stage == 'repair' else '')
    cid = e.request('POST', f'/containers/create?name=nemochat-offline1920-prefix-continuation03-{generation}', payload)['Id']
    actual = owned(e, cid, stage)
    publish(f'{stage}-engine-created', dict(id=cid, image=IMAGE, argv=actual['Config']['Cmd'], mounts=actual['Mounts']))
    e.request('POST', f'/containers/{cid}/start'); deadline = time.monotonic()+12*3600
    try:
        while True:
            os.utime(scratch, None); actual = owned(e, cid, stage); logs(e, cid, stage)
            publish(f'{stage}-phase', dict(id=cid, state=actual['State'], observed=time.time()))
            if not actual['State']['Running']: break
            if time.monotonic() > deadline: raise TimeoutError('12h watchdog')
            time.sleep(10)
    finally:
        if owned(e,cid,stage)['State']['Running']: e.request('POST', f'/containers/{cid}/stop?t=20')
        logs(e,cid,stage); publish(f'{stage}-engine-final', dict(id=cid,state=owned(e,cid,stage)['State']))
    if actual['State']['ExitCode'] != 0: raise RuntimeError('native stage failed')
    ranks = [json.loads((OUT/f'{"repair" if stage == "repair" else "student"}-rank{r}-complete.json').read_text()) for r in range(8)]
    if stage == 'repair':
        if any(r['iterations'] != sorted(set(request['iterations'])) or r['optimizer_calls'] != 0 for r in ranks):
            raise ValueError('incomplete sparse teacher repair')
    elif any(r['completed'] != 1143 or r['skipped'] or r['nonfinite'] for r in ranks): raise ValueError('incomplete native prefix')
    publish(f'{stage}-complete', dict(ranks=ranks, full_schedule_unchanged=True, old_teacher_pipeline_restarted=False))


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        publish('failure-'+(sys.argv[1] if len(sys.argv)>1 else 'unknown'), dict(type=type(exc).__name__,message=str(exc)))
        raise
