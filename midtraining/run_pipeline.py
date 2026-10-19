#!/usr/bin/env python3
"""Parent-owned verified download -> text view -> native pilot -> native reload.

Preview is inert. Execution requires a parent-frozen source allowlist and a
supervised service. Never resumes/retries a failed native writer implicitly.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/midtraining/super35-kd-01'
CONTROL = BASE / 'pipeline-01'
PILOT_SEGMENT = 'pilot-01'
RELOAD_SEGMENT = 'reload-01'
DIAGNOSTICS = False
RAW = ROOT / 'checkpoints/teachers/nemotron35-super-base-bf16-4ba52e8f'
REVISION = '4ba52e8ffa79e9693936c1a819ac89f551ceeef6'
TEXT = ROOT / 'checkpoints/teachers/nemotron35-super-base' / (REVISION + '-text')
CHECKPOINT = ROOT / 'checkpoints/midtraining/super35-kd-pilot-01'
DOWNLOAD = BASE / 'teacher-download/receipt.json'
IMAGE = 'nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
SOURCES = ['midtraining/run_pipeline.py', 'midtraining/native_kd.py', 'midtraining/pilot.yaml',
    'midtraining/download_teacher.py', 'midtraining/patches/teacher_text_view.py',
    'midtraining/patches/tokenizer_alias_gate.py', 'midtraining/patches/distillation_provider_shape_guard.py',
    'midtraining/patches/kd_finite_probe.py', 'midtraining/patches/student_embedding_import.py',
    'results/pretrain_followup/engine_helper.py',
    'results/pretraining/nemochat_baby/training/runner.py',
    'src/nemochat/tokenization/posttraining.py']


def configure_generation(generation, diagnostics=False):
    if (not isinstance(generation, str) or len(generation) != 2 or generation == '00'
            or any(character not in '0123456789' for character in generation)):
        raise ValueError('generation must be exactly two ASCII digits 01..99')
    global CONTROL, PILOT_SEGMENT, RELOAD_SEGMENT, DIAGNOSTICS
    CONTROL = BASE / ('pipeline-' + generation)
    PILOT_SEGMENT = 'pilot-' + generation
    RELOAD_SEGMENT = 'reload-' + generation
    DIAGNOSTICS = bool(diagnostics)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_gate():
    expected = read(CONTROL / 'reviewed-sources.json')
    if set(expected) != set(SOURCES):
        raise ValueError('parent source allowlist scope mismatch')
    for name, digest in expected.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('source changed after parent review: ' + name)
    return dict(status='parent_reviewed_source_hashes_match', count=len(expected))


def download_gate(path=DOWNLOAD):
    receipt = read(path)
    if receipt.get('status') != 'complete':
        raise ValueError('complete verified download required')
    if receipt.get('repo_id') != 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16' or receipt.get('revision') != REVISION:
        raise ValueError('teacher identity mismatch')
    spec = importlib.util.spec_from_file_location('midtraining_download_manifest', ROOT / 'midtraining/download_teacher.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    expected = module.build_manifest(read(BASE / 'teacher-hub-metadata.json'))
    files = receipt.get('files', [])
    if (receipt.get('output') != str(RAW) or receipt.get('complete_files') != len(expected)
            or len(files) != len(expected) or receipt.get('expected_files') != len(expected)
            or receipt.get('verified_bytes') != sum(x['size'] for x in expected)
            or receipt.get('expected_bytes') != sum(x['size'] for x in expected)
            or len({x['path'] for x in files}) != len(files)):
        raise ValueError('complete download inventory mismatch')
    registered = {entry['path']: entry for entry in files}
    for entry in expected:
        actual = registered.get(entry['path'], {})
        if (actual.get('verified') is not True or actual.get('size') != entry['size']
                or (entry['sha256'] and actual.get('sha256') != entry['sha256'])
                or (RAW / entry['path']).stat().st_size != entry['size']):
            raise ValueError('download proof/file mismatch')
    return dict(status='verified_teacher_download_accepted', files=len(files),
                bytes=receipt['verified_bytes'], revision=REVISION)


def accept_native(output, engine_state, reload_only=False):
    output = Path(output)
    if engine_state.get('Running') is not False or engine_state.get('ExitCode') != 0 or engine_state.get('OOMKilled') is not False:
        raise ValueError('clean exact Engine exit required')
    learning = []; calls = []; all_evals = []
    for rank in range(8):
        complete = read(output / f'rank-{rank}-complete.json')
        status = 'reload_eval_complete' if reload_only else 'pilot_complete'
        if complete.get('status') != status or complete.get('finite_heldout_evaluations', 0) < 1:
            raise ValueError('all-rank native completion/evaluation missing')
        if reload_only:
            if complete.get('completed_updates') != 0 or complete.get('save_steps'):
                raise ValueError('reload must not train or save')
            all_evals.append(complete['finite_heldout_evaluations'])
            continue
        if complete.get('completed_updates') != 4 or 4 not in complete.get('save_steps', []):
            raise ValueError('all-rank four-update save evidence required')
        rows = [json.loads(line) for line in (output / f'rank-{rank}-updates.jsonl').read_text().splitlines()]
        if [x.get('update') for x in rows] != [1, 2, 3, 4]:
            raise ValueError('all-rank native update sequence mismatch')
        for row in rows:
            metrics = row.get('metrics', {})
            if (set(metrics) != {'LM Loss', 'KD Loss', 'Total Loss', 'Grad Norm'}
                    or not all(math.isfinite(value) for value in metrics.values())
                    or metrics['Grad Norm'] <= 0 or not math.isfinite(row['lr_used']) or row['lr_used'] < 0
                    or row.get('teacher_frozen_no_grads') is not True or row.get('native_optimizer') is not True):
                raise ValueError('finite native learning evidence missing')
        learning.append(sum(row['lr_used'] > 0 for row in rows)); calls.append(len(rows))
    if reload_only:
        return dict(status='native_student_only_reload_eval_accepted', all_ranks=8, optimizer_updates=0)
    return dict(status='native_pilot_save_accepted_reload_not_yet_accepted', all_ranks=8,
                native_optimizer_calls=min(calls), positive_lr_learning_updates=min(learning),
                zero_lr_native_optimizer_calls=min(calls) - min(learning),
                processed_positions=4 * 8 * 4096, checkpoint_integrity='not_checked_explicit_user_waiver')


def container_payload(segment, reload_only=False):
    relative = f'results/midtraining/super35-kd-01/{segment}'
    command = ['--signal=TERM', '--kill-after=120', '6900', 'torchrun', '--nnodes=1',
        '--nproc_per_node=8', '--master_addr=127.0.0.1', '--master_port=29867',
        'midtraining/native_kd.py', '--config', 'midtraining/pilot.yaml', '--output', relative]
    if reload_only:
        command += ['--reload-checkpoint', str(CHECKPOINT.relative_to(ROOT))]
    elif DIAGNOSTICS:
        command += ['--diagnostics']
    return dict(Image=IMAGE, Entrypoint=['timeout'], Cmd=command, WorkingDir='/workspace', User='0:0',
        Env=['PYTHONPATH=/workspace/src', 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
             'TRANSFORMERS_OFFLINE=1', 'WANDB_MODE=disabled', 'OMP_NUM_THREADS=2',
             'HOME=/workspace/results/midtraining/super35-kd-01/runtime-home',
             'CUDA_CACHE_PATH=/workspace/results/midtraining/super35-kd-01/runtime-home/cuda-cache'],
        Labels={'nemochat.stage': 'midtraining', 'nemochat.experiment': 'super35-kd-01', 'nemochat.segment': segment},
        HostConfig=dict(NetworkMode='none', Privileged=False, CapDrop=['ALL'], CapAdd=['DAC_OVERRIDE'],
            SecurityOpt=['no-new-privileges:true'], Binds=[str(ROOT)+':/workspace:ro',
                str(ROOT/'results/midtraining')+':/workspace/results/midtraining:rw',
                str(ROOT/'checkpoints/midtraining')+':/workspace/checkpoints/midtraining:rw'],
            Memory=824633720832, ShmSize=17179869184, RestartPolicy={'Name': 'no'},
            DeviceRequests=[dict(Driver='nvidia', Count=8, Capabilities=[['gpu']])] ))


def engine_module():
    spec = importlib.util.spec_from_file_location('midtraining_engine', ROOT/'results/pretrain_followup/engine_helper.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def run_container(segment, reload_only=False):
    source_gate()
    engine = engine_module(); name = 'nemochat-midtraining-base35-' + segment
    payload = container_payload(segment, reload_only)
    write(CONTROL / (segment+'-payload.json'), payload)
    cid = engine.request('POST', '/containers/create?name='+name, payload)['Id']
    write(CONTROL / (segment+'-container.json'), {'id':cid, 'name':name})
    try:
        engine.request('POST', '/containers/'+cid+'/start')
        engine.request('POST', '/containers/'+cid+'/wait?condition=not-running', timeout=7200)
    except BaseException:
        state = engine.request('GET', '/containers/'+cid+'/json')['State']
        if state['Running']:
            engine.request('POST', '/containers/'+cid+'/stop?t=120', timeout=150)
        raise
    finally:
        state = engine.request('GET', '/containers/'+cid+'/json')['State']
        write(CONTROL / (segment+'-engine-state.json'), {'id':cid, 'state':state})
        connection = engine.Connection('localhost', timeout=60)
        connection.request('GET', '/v1.52/containers/'+cid+'/logs?stdout=true&stderr=true&timestamps=true')
        raw = connection.getresponse().read(); connection.close(); position=0; chunks=[]
        while position + 8 <= len(raw):
            size = struct.unpack('>BBBBI', raw[position:position+8])[-1]; position += 8
            chunks.append(raw[position:position+size].decode(errors='replace')); position += size
        (CONTROL / (segment+'-container.log')).write_text(''.join(chunks))
    result = accept_native(BASE / segment, state, reload_only)
    write(CONTROL / (segment+'-accepted.json'), result)
    return result


def phase(name, **extra):
    value = dict(phase=name, checked_at=datetime.datetime.now().astimezone().isoformat(),
                 production_run=False, sft_automatic_release=False, **extra)
    write(CONTROL / 'state.json', value)
    print(json.dumps(value), flush=True)


def execute():
    if os.environ.get('NEMOCHAT_MIDTRAINING_SUPERVISED') != '01':
        raise RuntimeError('parent-owned systemd execution scope required')
    source_gate()
    (CONTROL / 'execution-claim').open('x').close()
    phase('WAITING_VERIFIED_TEACHER_DOWNLOAD', native_optimizer_updates=0)
    deadline = time.monotonic()+43200
    while not DOWNLOAD.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('teacher download deadline exceeded')
        progress_path = BASE / 'teacher-download/progress.json'
        if progress_path.exists() and read(progress_path).get('status') == 'failed':
            raise RuntimeError('teacher download failed; no GPU launch')
        active = subprocess.run(['systemctl','--user','show','nemochat-midtraining-base35-download-01.service',
                                '-p','ActiveState','--value'], capture_output=True,text=True,check=True).stdout.strip()
        if active not in ('active','activating'):
            raise RuntimeError('teacher download service ended without complete verified receipt')
        time.sleep(20)
    receipt = download_gate(); write(CONTROL/'download-accepted.json', receipt)
    source_gate(); phase('TEXT_BACKBONE_CONVERSION', native_optimizer_updates=0)
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'midtraining/patches/teacher_text_view.py'),
                    '--source',str(RAW),'--output',str(TEXT)],check=True,cwd=ROOT,timeout=7200)
    converted = read(TEXT/'conversion-completed.json')
    if converted.get('status') != 'text_backbone_materialized_not_gpu_validated' or converted.get('language_tensors') != 41643:
        raise ValueError('exact Base backbone conversion inventory mismatch')
    (TEXT/'CONFIG_ONLY_NO_WEIGHTS.json').unlink(missing_ok=True)
    (ROOT/'checkpoints/midtraining').mkdir(parents=True,exist_ok=True)
    (BASE/'runtime-home').mkdir(exist_ok=True)
    lock_path = ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if pids:
            raise RuntimeError('GPUs occupied; foreign workload not stopped')
        recovery = subprocess.check_output(['nvidia-smi','--query-gpu=index,gpu_recovery_action',
                                            '--format=csv,noheader'],text=True)
        (CONTROL/'gpu-recovery-before-native.csv').write_text(recovery)
        if len(recovery.strip().splitlines()) != 8 or any(line.rsplit(',',1)[-1].strip() != 'None'
                                                       for line in recovery.strip().splitlines()):
            raise RuntimeError('GPU recovery action required; no automatic reset/reboot authorized')
        health = subprocess.check_output(['nvidia-smi','-q','-d','PAGE_RETIREMENT,ROW_REMAPPER'],text=True)
        (CONTROL/'gpu-health-before-native.txt').write_text(health)
        if CHECKPOINT.exists():
            raise FileExistsError('fresh pilot checkpoint destination already exists')
        phase('NATIVE_PILOT_STARTING', native_optimizer_updates=0)
        pilot = run_container(PILOT_SEGMENT)
        phase('NATIVE_PILOT_SAVED_RELOAD_PENDING', pilot=pilot)
        reload = run_container(RELOAD_SEGMENT, reload_only=True)
        phase('PILOT_AND_STUDENT_RELOAD_ACCEPTED', pilot=pilot, reload=reload,
              quality_improvement_established=False, long_run_executed=False)


def cleanup_owned(engine=None):
    engine = engine or engine_module()
    results = {}
    for segment in (PILOT_SEGMENT, RELOAD_SEGMENT):
        path = CONTROL / (segment+'-container.json')
        if not path.is_file():
            continue
        identity = read(path); cid = identity['id']
        actual = engine.request('GET', '/containers/'+cid+'/json')
        labels = actual.get('Config', {}).get('Labels', {})
        if (actual.get('Id') != cid or actual.get('Name') != '/nemochat-midtraining-base35-'+segment
                or labels.get('nemochat.stage') != 'midtraining'
                or labels.get('nemochat.experiment') != 'super35-kd-01'
                or labels.get('nemochat.segment') != segment):
            raise RuntimeError('owned container identity mismatch; refuse cleanup')
        if actual['State']['Running']:
            engine.request('POST', '/containers/'+cid+'/stop?t=120', timeout=150)
        results[segment] = engine.request('GET', '/containers/'+cid+'/json')['State']
    write(CONTROL/'cleanup-owned.json', results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', default='01')
    parser.add_argument('--diagnostics', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--cleanup-owned',action='store_true')
    args = parser.parse_args()
    configure_generation(args.generation, diagnostics=args.diagnostics)
    if args.cleanup_owned:
        print(json.dumps(cleanup_owned(), indent=2))
        return
    if not args.execute:
        print(json.dumps(dict(status='inert_preview', source_paths=SOURCES,
            sequence=['verified_teacher_download','text_backbone_conversion','native_four_update_pilot','student_only_reload'],
            checkpoint=str(CHECKPOINT), sft_automatic_release=False),indent=2))
        return
    try:
        execute()
    except BaseException as exc:
        phase('BLOCKED_OR_FAILED', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    main()
