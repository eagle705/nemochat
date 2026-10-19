#!/usr/bin/env python3
"""Parent-reviewed bounded launcher. Default/--preview/--help are inert.

No acquisition, service creation, remote registration or implicit retry/resume.
Use a separate parent-owned systemd service with ExecStopPost --cleanup-owned.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/'results/pretraining/nemochat_baby/training/gpu-all8.lock'
SOURCES = ['midtraining/longrun_native.py', 'midtraining/launch_longrun.py',
    'midtraining/calibration.yaml', 'midtraining/longrun24h.yaml', 'midtraining/longrun24h.mix67.yaml',
    'midtraining/native_kd.py', 'midtraining/run_pipeline.py',
    'midtraining/patches/native_dataset_observer.py',
    'midtraining/patches/student_embedding_import.py',
    'midtraining/patches/distillation_provider_shape_guard.py',
    'midtraining/patches/tokenizer_alias_gate.py',
    'results/pretraining/nemochat_baby/training/runner.py',
    'src/nemochat/tokenization/posttraining.py',
    'results/pretrain_followup/engine_helper.py',
    'results/bridge_wandb/wandb_preflight.py']


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native():
    return load_module('longrun_native_launcher', 'midtraining/longrun_native.py')


def pipeline():
    return load_module('longrun_accepted_pipeline', 'midtraining/run_pipeline.py')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    partial.replace(path)


def source_gate(path):
    expected = read(path)
    if set(expected) != set(SOURCES):
        raise ValueError('exact parent-reviewed source allowlist required')
    for name, digest in expected.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('source changed after parent review: '+name)


def validate_name(run):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,62}', run):
        raise ValueError('fresh filesystem run ID must be 3..63 lowercase ASCII letters/digits/hyphens')


def container_payload(c, run, segment, consumed_config, config_only=False):
    validate_name(run)
    relative = f'results/midtraining/longrun/{run}/{segment}'
    limit = min(c['bounds']['watchdog_seconds']-1800,
                c['train']['exit_duration_in_mins']*60+c['bounds']['save_grace_seconds'])
    if segment == 'reload':
        limit = 1800
    if config_only:
        limit = 600
    command = ['--signal=TERM', '--kill-after=120', str(limit), 'torchrun', '--nnodes=1',
        '--nproc_per_node=1' if config_only else '--nproc_per_node=8',
        '--master_addr=127.0.0.1', '--master_port=29879',
        'midtraining/longrun_native.py', '--config', str(consumed_config), '--output', relative,
        '--config-only' if config_only else '--execute']
    if segment == 'reload':
        command += ['--reload-checkpoint', str(ROOT/c['checkpoint_save'])]
    online = c['tracking']['mode'] == 'online' and segment == 'train' and not config_only
    binds = [str(ROOT)+':'+str(ROOT)+':ro',
        str(ROOT/'results/midtraining/longrun')+':'+str(ROOT/'results/midtraining/longrun')+':rw',
        str(ROOT/'checkpoints/midtraining')+':'+str(ROOT/'checkpoints/midtraining')+':rw']
    if online:
        binds.append(str(ROOT/'.env')+':/run/nemochat-wandb.env:ro')
    return dict(Image=pipeline().IMAGE, Entrypoint=['timeout'], Cmd=command,
        WorkingDir=str(ROOT), User='0:0',
        Env=['PYTHONPATH='+str(ROOT/'src'), 'PYTHONDONTWRITEBYTECODE=1', 'HF_HUB_OFFLINE=1',
             'TRANSFORMERS_OFFLINE=1', 'WANDB_MODE=online' if online else 'WANDB_MODE=disabled',
             'OMP_NUM_THREADS=2', 'HOME='+str(ROOT/'results/midtraining/longrun'/run/'runtime-home'),
             'CUDA_CACHE_PATH='+str(ROOT/'results/midtraining/longrun'/run/'runtime-home/cuda-cache')],
        Labels={'nemochat.stage':'midtraining-longrun', 'nemochat.run':run, 'nemochat.segment':segment},
        HostConfig=dict(NetworkMode='bridge' if online else 'none', Privileged=False,
            CapDrop=['ALL'], CapAdd=['DAC_OVERRIDE'], SecurityOpt=['no-new-privileges:true'],
            Binds=binds, Memory=824633720832, ShmSize=17179869184, RestartPolicy={'Name':'no'},
            DeviceRequests=[dict(Driver='nvidia', DeviceIDs=[str(i) for i in range(8)], Capabilities=[['gpu']])]))


def check_owned(actual, cid, run, segment):
    labels = actual.get('Config', {}).get('Labels', {})
    # Docker merges immutable NVIDIA image labels into explicitly supplied labels.
    # Require the exact task ownership namespace, while retaining image metadata.
    owned_labels = {key: value for key, value in labels.items() if key.startswith('nemochat.')}
    if (actual.get('Id') != cid or actual.get('Name') != '/nemochat-midtraining-'+run+'-'+segment
            or owned_labels != {'nemochat.stage':'midtraining-longrun', 'nemochat.run':run, 'nemochat.segment':segment}):
        raise ValueError('exact owned Engine identity mismatch; refuse cleanup')


def cleanup_owned(run, engine=None):
    validate_name(run)
    engine = engine or pipeline().engine_module()
    control = ROOT/'results/midtraining/longrun'/run/'controller'
    result = {}
    for segment in ('config', 'train', 'reload'):
        path = control/(segment+'-container.json')
        if not path.is_file():
            continue
        identity = read(path)
        cid = identity['id']
        actual = engine.request('GET', '/containers/'+cid+'/json')
        check_owned(actual, cid, run, segment)
        if actual['State']['Running']:
            engine.request('POST', '/containers/'+cid+'/stop?t=120', timeout=150)
        state = engine.request('GET', '/containers/'+cid+'/json')['State']
        if state['Running']:
            raise RuntimeError('owned Engine still running after stop')
        result[segment] = state
    write(control/'cleanup-owned.json', result)
    return result


def accept_native(output, engine_state, stage, reload_only=False):
    """Never accept a run URL, config or Engine create as training evidence."""
    if (engine_state.get('Running') is not False or engine_state.get('ExitCode') != 0
            or engine_state.get('OOMKilled') is not False):
        raise ValueError('clean exact Engine exit required')
    output = Path(output)
    m = native()
    completions, calibrations, hashes = [], [], []
    for rank in range(8):
        try:
            complete = read(output/f'rank-{rank}-complete.json')
            heldout = read(output/f'rank-{rank}-heldout.json')['evaluations']
            preflight = read(output/f'rank-{rank}-preflight.json')
        except (FileNotFoundError, KeyError) as exc:
            raise ValueError('all-rank native completion/heldout/preflight missing') from exc
        if not heldout or not all(math.isfinite(row['lm_loss']) for row in heldout):
            raise ValueError('all-rank finite heldout required')
        hashes.append(preflight['manifest_sha256'])
        if reload_only:
            if (complete.get('status') != 'reload_eval_complete' or complete.get('teacher_free') is not True
                    or complete.get('completed_updates') != 0 or complete.get('save_steps')
                    or any(row.get('teacher_present') is not False for row in heldout)):
                raise ValueError('all-rank teacher-free reload evaluation required')
        else:
            rows = [json.loads(line) for line in (output/f'rank-{rank}-updates.jsonl').read_text().splitlines()]
            count = complete['completed_updates']
            if [row['update'] for row in rows] != list(range(1,count+1)) or not count:
                raise ValueError('all-rank contiguous actual optimizer updates required')
            if count not in complete['save_steps'] or not any(x['update'] == count for x in heldout):
                raise ValueError('final-update heldout and native save required')
            save = read(output/f'rank-{rank}-save.json')
            start = read(output/f'rank-{rank}-start.json')
            data = read(output/f'rank-{rank}-dataset-built.json')
            if (save.get('save_optim') is not True or save.get('save_rng') is not True
                    or save.get('native_synchronous_save_completed') is not True
                    or start.get('teacher_gradients_absent') is not True
                    or data.get('native_train_epochs') != 1 or data.get('native_blend') is not True
                    or set(data.get('source_samples', {})) != set(m.KEYS)
                    or any(x <= 0 for x in data['source_samples'].values())):
                raise ValueError('full-state save, frozen teacher and new-source nonrepeat coverage required')
            for row in rows:
                metrics = row.get('metrics', {})
                if (set(metrics) != {'LM Loss','KD Loss','Total Loss','Grad Norm'}
                        or not all(math.isfinite(x) for x in metrics.values()) or metrics['Grad Norm'] <= 0
                        or not math.isfinite(row['lr_used']) or row['lr_used'] < 0
                        or row.get('native_optimizer') is not True
                        or row.get('teacher_frozen_no_grads') is not True):
                    raise ValueError('actual finite native learning receipts required')
            if stage == 'calibration':
                calibrations.append(m.calibration_summary(rows))
                if complete['status'] != 'calibration_complete_reload_pending':
                    raise ValueError('normal calibration completion required')
            elif stage == 'duration_control':
                if complete['status'] != 'duration_control_complete_reload_pending':
                    raise ValueError('actual one-minute duration control required')
            elif complete['status'] not in ('duration_complete_reload_pending', 'iteration_budget_complete_short_of_wall_target'):
                raise ValueError('production native completion status invalid')
            elif (complete['status'] == 'duration_complete_reload_pending'
                  and complete.get('native_elapsed_seconds',0) < 86400):
                raise ValueError('24h duration claim contradicts actual native elapsed seconds')
            elif complete['status'] == 'iteration_budget_complete_short_of_wall_target':
                consumed = m.load_config(output/'consumed-stage-config.yaml')
                if count != consumed['train']['train_iters']:
                    raise ValueError('short iteration completion disagrees with actual frozen horizon')
        completions.append(complete)
    if len(set(hashes)) != 1 or len({c['completed_updates'] for c in completions}) != 1:
        raise ValueError('all-rank manifest/update cursor disagreement')
    result = dict(status='native_reload_accepted' if reload_only else 'native_save_accepted_reload_pending',
        all_ranks=8, stage=stage, completed_updates=min(c['completed_updates'] for c in completions),
        manifest_sha256=hashes[0], training_24h_accepted=False, remote_tracking_verified=False,
        native_elapsed_seconds=min(c['native_elapsed_seconds'] for c in completions))
    if calibrations:
        if len({x['mean_step_seconds'] for x in calibrations}) != 1:
            raise ValueError('calibration requires all-rank-max wall timing agreement')
        result['calibration'] = dict(calibrations[0], peak_memory_bytes=max(x['peak_memory_bytes'] for x in calibrations))
    result['all_rank_statuses'] = [c['status'] for c in completions]
    return result


def run_container(c, run, segment, consumed, reviewed, config_only=False):
    source_gate(reviewed)
    engine = pipeline().engine_module()
    control = ROOT/'results/midtraining/longrun'/run/'controller'
    payload = container_payload(c, run, segment, consumed, config_only)
    # Payload is deliberately credential-free. Never persist Engine inspect Config.Env.
    write(control/(segment+'-payload.json'), payload)
    name = 'nemochat-midtraining-'+run+'-'+segment
    cid = engine.request('POST', '/containers/create?name='+name, payload)['Id']
    write(control/(segment+'-container.json'), dict(id=cid, name=name))
    started = time.monotonic()
    cap = int(payload['Cmd'][2])+130 # GNU timeout kill grace plus Engine readback
    try:
        actual = engine.request('GET', '/containers/'+cid+'/json')
        check_owned(actual, cid, run, segment)
        engine.request('POST', '/containers/'+cid+'/start')
        while True:
            state = engine.request('GET', '/containers/'+cid+'/json')['State']
            write(control/(segment+'-engine-state.json'), dict(id=cid, state=state))
            if not state['Running']:
                break
            if time.monotonic()-started > cap:
                raise TimeoutError('outer watchdog expired; this is failed training, never completion')
            time.sleep(10)
    except BaseException:
        cleanup_owned(run, engine)
        raise
    state = engine.request('GET', '/containers/'+cid+'/json')['State']
    write(control/(segment+'-engine-state.json'), dict(id=cid, state=state))
    if config_only:
        if state['Running'] or state['ExitCode'] != 0 or state['OOMKilled']:
            raise ValueError('native config-only Engine exit failed')
        receipt = read(ROOT/'results/midtraining/longrun'/run/'config/rank-0-config-only.json')
        if receipt != dict(status='NATIVE_CONFIG_CONSTRUCTED', optimizer_updates=0, weights_loaded=False):
            raise ValueError('actual native config construction receipt required')
        return receipt
    accepted = accept_native(ROOT/'results/midtraining/longrun'/run/segment, state,
                             c['stage'], reload_only=segment == 'reload')
    write(control/(segment+'-accepted.json'), accepted)
    return accepted


def validate_calibration_compatibility(accepted, manifest_hash, weights, compatibility):
    """Parent attestation: recipe expansion OR explicitly changed-mix projection.

    The latter is not a benchmark of the new blend. Bind the old recipe to its
    actual measured seed manifest rather than inferring it from absent metadata.
    """
    m = native()
    weights = m.recognized_weights(weights)
    old_hash = accepted['manifest_sha256']
    if weights == m.WEIGHTS:
        if old_hash == manifest_hash:
            return None
        expected = dict(status='parent_accepted_same_recipe_expansion',
            calibration_manifest_sha256=old_hash, production_manifest_sha256=manifest_hash,
            same_provider_tokenizer_sequence_batch_topology=True, heldout_exclusions_preserved=True)
    else:
        measured_seed = '52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308'
        if old_hash != measured_seed or old_hash == manifest_hash:
            raise ValueError('calibration compatibility requires the actual old-profile seed and new manifest')
        expected = dict(status='parent_accepted_approved_mix_change_profile_projection',
            calibration_manifest_sha256=old_hash, production_manifest_sha256=manifest_hash,
            old_weights=list(m.WEIGHTS), new_weights=list(m.MIX67_WEIGHTS),
            same_provider_tokenizer_sequence_batch_topology=True,
            heldout_exclusions_preserved=True, user_approved_mix_change=True,
            projection_only=True, fresh_new_mix_benchmark=False)
    if compatibility != expected or any(type(compatibility[k]) is not bool
            for k, v in expected.items() if type(v) is bool):
        raise ValueError('calibration compatibility receipt mismatch')
    return compatibility


def execute(args):
    import yaml
    if os.environ.get('NEMOCHAT_LONGRUN_SUPERVISED') != '1':
        raise ValueError('parent-owned independent supervised service required')
    source_gate(args.reviewed_sources)
    m = native()
    c = m.load_config(args.config)
    if args.duration_control:
        if c['stage'] != 'calibration':
            raise ValueError('duration control derives from calibration config only')
        c['stage'] = 'duration_control'
        c['name'] = 'nemochat baby midtraining super35 duration control'
        c['train']['exit_duration_in_mins'] = 1
    if args.manifest:
        c['dataset']['manifest'] = str(args.manifest.resolve().relative_to(ROOT))
    if not c['dataset']['manifest']:
        raise ValueError('producer success manifest required')
    validate_name(args.run_id)
    manifest, manifest_hash = m.load_prepared_manifest(ROOT/c['dataset']['manifest'])
    weights = m.dataset_weights(c, manifest)
    audit = m.audit_reserved_ids(manifest)
    audit_folder = ROOT/'results/midtraining/longrun-corpus-audits'/args.run_id
    audit_folder.mkdir(parents=True, exist_ok=False)
    write(audit_folder/'reserved-id-audit.json', dict(audit, manifest_sha256=manifest_hash))
    m.corpus_tokenizer_gate(manifest, audit)
    calibration = None
    if c['stage'] == 'production':
        if args.calibration is None or args.duration_control_receipt is None:
            raise ValueError('accepted calibration plus saved/reloaded duration control required')
        accepted = read(args.calibration)
        control = read(args.duration_control_receipt)
        if (accepted.get('status') != 'calibration_save_reload_accepted' or accepted.get('all_ranks') != 8
                or control.get('status') != 'duration_control_save_reload_accepted' or control.get('all_ranks') != 8):
            raise ValueError('actual all-rank calibration and duration lifecycle gates missing')
        calibration = accepted['calibration']
        compatibility = validate_calibration_compatibility(accepted, manifest_hash, weights,
            read(args.calibration_compatibility) if args.calibration_compatibility else None)
    c = m.freeze_config(c, calibration)
    if c['stage'] == 'production' and compatibility is not None:
        c['projection']['calibration_compatibility'] = compatibility
    m.capacity_gate(manifest, c['train']['train_iters'], 128, 4096)
    validate_name(args.run_id)
    run = args.run_id
    c['checkpoint_save'] = 'checkpoints/midtraining/'+run
    if c['tracking']['mode'] == 'online':
        # Validate key syntax privately without ever writing its contents or an exception expression.
        env = load_module('longrun_private_wandb_parser', 'results/bridge_wandb/wandb_preflight.py').read_wandb_env(ROOT/'.env')
        if not env.get('WANDB_API_KEY'):
            raise ValueError('online authorization exists but local W&B key is missing')
        c['tracking']['run_id'] = run
    if (ROOT/c['checkpoint_save']).exists():
        raise FileExistsError('existing checkpoints preserved; use a fresh explicit run ID')
    folder = ROOT/'results/midtraining/longrun'/run
    if folder.exists():
        raise FileExistsError('existing run preserved; never implicitly resume/retry')
    # Measured comparable full native optimizer/RNG checkpoint ~137GB. Three
    # retained plus in-flight checkpoint and 16GB cache/log headroom.
    if shutil.disk_usage(ROOT).free < 137_000_000_000*4 + 16_000_000_000:
        raise ValueError('insufficient disk for native full-state retention and in-flight save')
    with LOCK.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
        recovery = subprocess.check_output(['nvidia-smi','--query-gpu=index,gpu_recovery_action','--format=csv,noheader'], text=True)
        if pids or len(recovery.strip().splitlines()) != 8 or any(line.rsplit(',',1)[-1].strip() != 'None' for line in recovery.strip().splitlines()):
            raise ValueError('all-eight GPUs must be idle and have no recovery action; never stop foreign jobs')
        folder.mkdir(parents=True, exist_ok=False)
        (folder/'runtime-home').mkdir()
        consumed = folder/'consumed-config.yaml'
        consumed.write_text(yaml.safe_dump(c, sort_keys=False))
        write(folder/'controller/frozen-inputs.json', dict(manifest_sha256=manifest_hash,
            config_sha256=hashlib.sha256(consumed.read_bytes()).hexdigest(),
            source_review_sha256=hashlib.sha256(args.reviewed_sources.read_bytes()).hexdigest(),
            fresh_model_only_warmstart=True, optimizer_rng_reset=True, previous_control_preserved=True,
            start_time=datetime.datetime.now().astimezone().isoformat()))
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(InterruptedError('supervisor stop')))
        if args.config_only:
            return run_container(c, run, 'config', str(consumed.relative_to(ROOT)), args.reviewed_sources, True)
        trained = run_container(c, run, 'train', str(consumed.relative_to(ROOT)), args.reviewed_sources)
        reloaded = run_container(c, run, 'reload', str(consumed.relative_to(ROOT)), args.reviewed_sources)
        result = dict(trained, status=c['stage']+'_save_reload_accepted', student_only_reload=reloaded,
            training_24h_accepted=c['stage'] == 'production' and all(
                x == 'duration_complete_reload_pending' for x in trained['all_rank_statuses']),
            remote_tracking_verified=False)
        write(folder/'controller/accepted.json', result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'midtraining/calibration.yaml')
    parser.add_argument('--run-id', default='super35-calibration-01')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--reviewed-sources', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--duration-control-receipt', type=Path)
    parser.add_argument('--calibration-compatibility', type=Path)
    parser.add_argument('--duration-control', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--config-only', action='store_true')
    mode.add_argument('--cleanup-owned', action='store_true')
    args = parser.parse_args(argv)
    if args.cleanup_owned:
        print(json.dumps(cleanup_owned(args.run_id), indent=2))
        return
    if not args.execute and not args.config_only:
        c = native().load_config(args.config)
        print(json.dumps(dict(status='inert_preview', config=c, source_paths=SOURCES,
            native_training_executed=False, remote_tracking_run_created=False,
            blockers=['producer success manifest', 'new corpus tokenizer policy/counts',
                      'parent source review', 'native calibration', 'sufficient per-source production capacity',
                      'all-rank save/reload duration control']), indent=2))
        return
    if args.reviewed_sources is None:
        parser.error('--reviewed-sources required')
    try:
        result = execute(args)
        print(json.dumps(result, indent=2))
    except BaseException:
        # No tracking finish call or success receipt on launcher failure.
        raise


if __name__ == '__main__':
    main()
