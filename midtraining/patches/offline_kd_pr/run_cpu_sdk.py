"""Bounded, no-network/no-GPU pinned SDK pytest runner; no SDK edits.

Run from repository root with host Python. Uses existing local Docker Engine.
The only writable bind is a fresh caller-owned TMPDIR scratch directory.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time

IMAGE = 'nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
REPO = Path(__file__).resolve().parents[3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--test', nargs='+', default=['tests/test_offline_kd_pr_bridge.py'])
    p.add_argument('--apply-patch', action='store_true', help='apply only to scratch SDK copy')
    args = p.parse_args()
    if not args.label.replace('-', '').isalnum():
        p.error('label must be alphanumeric/hyphen')
    spec = importlib.util.spec_from_file_location('engine_helper', REPO / 'results/pretrain_followup/engine_helper.py')
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    import tempfile
    scratch = Path(tempfile.mkdtemp(prefix='bridge-kd-', dir=os.environ['TMPDIR']))
    if os.getuid() != 1001 or os.getgid() != 985:
        raise RuntimeError('expected authorized host UID1001/GID985')
    # Image /opt source ancestors are root-only; stage exact read-only archive
    # bytes from a stopped same-image container, never chmod/edit the SDK.
    import io
    import tarfile
    from urllib.parse import quote
    source_container = '2f1750f9d121488f45a55d1ae66b8390b6ed758e2125e66747c010fe97bc2e92'
    for source, target, namespace in [('/opt/Megatron-Bridge/3rdparty/Megatron-LM/megatron', 'megatron', ''),
                           ('/opt/Megatron-Bridge/src/megatron/bridge', 'bridge', 'megatron')]:
        conn = engine.Connection('localhost', timeout=90)
        conn.request('GET', '/v1.52/containers/' + source_container + '/archive?path=' + quote(source))
        response = conn.getresponse()
        data = response.read()
        conn.close()
        if response.status != 200:
            raise RuntimeError('stopped SDK archive read failed: ' + str(response.status))
        root = scratch / 'sdk' / namespace
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive:
                relative = Path(member.name)
                if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != target:
                    raise ValueError('unexpected SDK archive member')
                dest = root / relative
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(archive.extractfile(member).read())
                elif member.issym():
                    continue  # no external source links accepted
    # Real host driver shared library only; no device nodes/GPU runtime.
    # This satisfies TE's eager dynamic link, not CUDA execution capability.
    import shutil
    (scratch / 'driver').mkdir()
    shutil.copyfile('/usr/lib/x86_64-linux-gnu/libcuda.so.1', scratch / 'driver/libcuda.so.1')
    if args.apply_patch:
        import subprocess
        staged = scratch / 'staged'
        target = staged / 'src/megatron/bridge'
        target.parent.mkdir(parents=True)
        shutil.copytree(scratch/'sdk/megatron/bridge', target)
        patch = Path(__file__).with_name('megatron-bridge-0.6.1-cached-kd.patch')
        for check in (True, False):
            command = ['git','apply'] + (['--check'] if check else []) + [str(patch)]
            subprocess.run(command, cwd=staged, check=True, capture_output=True)
    payload = dict(Image=IMAGE, User='1001:985', WorkingDir='/repo',
                   Entrypoint=['python'], Cmd=['-m','pytest','-q','-p','no:cacheprovider'] + args.test,
                   Env=['NATIVE_KD_SDK_TESTS=1','CUDA_VISIBLE_DEVICES=', 'NVIDIA_VISIBLE_DEVICES=void',
                        'LD_LIBRARY_PATH=/scratch/driver:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib',
                        'NATIVE_KD_PATCH_APPLIED='+str(int(args.apply_patch)),
                        'PYTHONPATH=/scratch/staged/src:/scratch/sdk', 'PYTHONDONTWRITEBYTECODE=1','TMPDIR=/scratch','HOME=/scratch',
                        'USER=yoon','LOGNAME=yoon','TORCHINDUCTOR_CACHE_DIR=/scratch/inductor',
                        'WANDB_MODE=disabled','HF_HUB_OFFLINE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1'],
                   Labels={'nemochat.offline-kd.owner': args.label},
                   HostConfig=dict(Runtime='runc', NetworkMode='none', ReadonlyRootfs=True,
                                   CapDrop=['ALL'], SecurityOpt=['no-new-privileges'],
                                   Memory=8*1024**3, NanoCpus=2*10**9, PidsLimit=256,
                                   DeviceRequests=[], Binds=[f'{REPO}:/repo:ro',f'{scratch}:/scratch:rw']))
    c = engine.request('POST','/containers/create?name=nemochat-offline-kd-'+args.label,payload)
    cid=c['Id']
    inspected=engine.request('GET',f'/containers/{cid}/json')
    assert inspected['Config']['User']=='1001:985'
    assert inspected['HostConfig']['NetworkMode']=='none'
    assert inspected['HostConfig']['Runtime']=='runc'
    assert not inspected['HostConfig']['DeviceRequests']
    assert all(not m['RW'] or m['Source']==str(scratch) for m in inspected['Mounts'])
    engine.request('POST',f'/containers/{cid}/start')
    deadline=time.monotonic()+180
    while True:
        final=engine.request('GET',f'/containers/{cid}/json')
        if not final['State']['Running']:
            break
        if time.monotonic()>deadline:
            engine.request('POST',f'/containers/{cid}/stop?t=5')
            raise TimeoutError(cid)
        time.sleep(1)
    conn=engine.Connection('localhost',timeout=30)
    conn.request('GET',f'/v1.52/containers/{cid}/logs?stdout=1&stderr=1')
    response=conn.getresponse(); raw=response.read();conn.close()
    import struct
    parts=[]
    while len(raw)>=8:
        n=struct.unpack('>I',raw[4:8])[0];parts.append(raw[8:8+n]);raw=raw[8+n:]
    log=b''.join(parts).decode(errors='replace')
    evidence=Path(__file__).with_name('evidence')
    (evidence/f'{args.label}.txt').write_text(log)
    receipt={'id':cid,'image':IMAGE,'scope':{'user':final['Config']['User'],
             'runtime':final['HostConfig']['Runtime'],'network':final['HostConfig']['NetworkMode'],
             'devices':final['HostConfig']['DeviceRequests'],'mounts':final['Mounts']},
             'state':final['State'],'scratch':str(scratch)}
    (evidence/f'{args.label}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(log);print(json.dumps(receipt,indent=2))
    raise SystemExit(final['State']['ExitCode'])


if __name__=='__main__':
    main()
