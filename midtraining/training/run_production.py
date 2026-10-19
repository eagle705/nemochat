#!/usr/bin/env python3
"""Durable production CPU -> native main handoff; default/preview/help are inert."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
UNIT = 'nemochat-midtraining-production-02.service'
RUN = 'super35-longrun-01'
BASE = 'results/midtraining/longrun/'
CALIBRATION = BASE+'super35-newdata-calibration-01/controller/accepted.json'
DURATION = BASE+'super35-newdata-duration-01/controller/accepted.json'
NATIVE_REVIEW = 'results/midtraining/super35-kd-01/longrun-source-review-02.json'
CPU_IDENTITY = 'results/midtraining/super35-kd-01/corpus/production-prepare-01-engine.json'
MANIFEST = 'datasets/midtraining/super35-kd-01/production-24h-v1/data/manifest.json'
SEED = 'datasets/midtraining/super35-kd-01/pilot-10m-v1/manifest.json'
SEED_SHA = '52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308'
OUTPUT = BASE+'super35-production-02/controller'
MAIN_SECONDS = 93600
SUPERVISOR_SECONDS = 140400


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def approved_corpus():
    path = ROOT/CPU_IDENTITY
    if digest(path) != 'b03519f8409217bfe01094d904f4def120fa9a82f0f861aa573bca929845ee76':
        raise ValueError('approved CPU identity/payload receipt changed')
    return read(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_gate(path, names):
    expected = read(path)
    if set(expected) != set(names):
        raise ValueError('exact startup source allowlist required')
    for name, sha in expected.items():
        if digest(ROOT/name) != sha:
            raise ValueError('source changed after parent review: '+name)


def startup_sources():
    launcher = load('production_launcher', 'midtraining/training/launch_longrun.py')
    producer = load('production_source_list', 'midtraining/data/production_corpus.py')
    return sorted(set(launcher.SOURCES) | set(producer.REVIEW_FILES) | {'midtraining/training/run_production.py'})


def check_supervisor(info, pid, uid, actual_argv, expected_argv):
    # A user manager inherits its own credentials when User= is unset. Explicit
    # User= can trigger forbidden supplementary-group changes (216/GROUP).
    # supervisor_gate separately checks the actual /proc process owner UID.
    if (info.get('MainPID') != str(pid) or info.get('User') not in ('', str(uid))
            or info.get('ActiveState') not in ('active', 'activating')
            or actual_argv != expected_argv):
        raise ValueError('exact independent user supervisor identity required')


def supervisor_gate():
    if os.environ.get('NEMOCHAT_PRODUCTION_SUPERVISED') != '1':
        raise ValueError('independent production supervisor required')
    fields = subprocess.check_output(['systemctl', '--user', 'show', UNIT,
        '--property=MainPID,User,ActiveState,SubState'], text=True)
    info = dict(line.split('=', 1) for line in fields.splitlines() if '=' in line)
    proc = Path('/proc')/str(os.getpid())
    actual = proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    # Parent launch uses this exact interpreter, with buffering controlled by environment.
    expected = [sys.executable, str(ROOT/'midtraining/training/run_production.py'), *sys.argv[1:]]
    if proc.stat().st_uid != os.getuid():
        raise ValueError('supervisor process UID mismatch')
    check_supervisor(info, os.getpid(), os.getuid(), actual, expected)



def compare_recipe(old, new):
    """Compare all provider/precision/optimizer/KD/batch fields, not a self-attestation."""
    import copy
    def normalize(value):
        c = copy.deepcopy(value)
        for key in ('name', 'stage', 'checkpoint_save', 'logger', 'tracking', 'bounds', 'projection'):
            c.pop(key, None)
        c['dataset'].pop('manifest', None)
        for key in ('train_iters', 'exit_duration_in_mins'):
            c['train'].pop(key, None)
        for key in ('lr_decay_iters', 'lr_warmup_iters', 'lr_wsd_decay_iters'):
            c['scheduler'].pop(key, None)
        c['validation'].pop('eval_interval', None)
        return c
    if normalize(old) != normalize(new):
        raise ValueError('calibration/production recipe mismatch')


def preserve_heldout(old, new, old_path, new_path):
    reservations = []
    for manifest, path in ((old, old_path), (new, new_path)):
        path = Path(path).with_name('heldout-reservations.jsonl')
        if digest(path) != manifest['heldout_reservations_sha256']:
            raise ValueError('heldout reservation SHA mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        reservations.append({(r['source'], r['content_hash'], r['group_id']) for r in rows})
    if not reservations[0] or not reservations[0] <= reservations[1]:
        raise ValueError('whole prior heldout reservation not preserved')
    for key, source in old['sources'].items():
        for field in ('bin_sha256', 'idx_sha256', 'bin_bytes', 'idx_bytes', 'tokens'):
            if source['validation'][field] != new['sources'][key]['validation'][field]:
                raise ValueError('frozen seed validation changed: '+key)
    return len(reservations[0])


def validate_data(native):
    seed_path, production_path = ROOT/SEED, ROOT/MANIFEST
    if digest(seed_path) != SEED_SHA:
        raise ValueError('immutable old seed manifest changed')
    seed, seed_sha = native.load_prepared_manifest(seed_path)
    manifest, manifest_sha = native.load_prepared_manifest(production_path)
    if (seed_sha != SEED_SHA or manifest.get('seed_manifest_sha256') != seed_sha
            or manifest.get('frozen_seed_heldout_reuse') is not True):
        raise ValueError('production seed lineage mismatch')
    if seed['tokenizer'] != manifest['tokenizer']:
        raise ValueError('complete tokenizer identity changed')
    accepted, duration = read(ROOT/CALIBRATION), read(ROOT/DURATION)
    if (accepted.get('status') != 'calibration_save_reload_accepted' or accepted.get('all_ranks') != 8
            or accepted.get('manifest_sha256') != seed_sha
            or duration.get('status') != 'duration_control_save_reload_accepted' or duration.get('all_ranks') != 8):
        raise ValueError('accepted all-eight calibration/duration lifecycle required')
    calibration_folder = (ROOT/CALIBRATION).parent.parent
    old_config = calibration_folder/'consumed-config.yaml'
    if digest(old_config) != read(calibration_folder/'controller/frozen-inputs.json')['config_sha256']:
        raise ValueError('accepted calibration consumed config changed')
    old, new = native.load_config(old_config), native.load_config(ROOT/'midtraining/configs/longrun24h.yaml')
    compare_recipe(old, new)
    frozen = native.freeze_config(new, accepted['calibration'])
    if frozen['train']['train_iters'] != 3200:
        raise ValueError('parent-approved measured 3200-update horizon changed')
    # Cheap per-source capacity first: a blocked corpus never reaches any GPU launcher.
    capacity = native.capacity_gate(manifest, 3200, 128, 4096)
    reserved = preserve_heldout(seed, manifest, seed_path, production_path)
    audit = native.audit_reserved_ids(manifest)  # all eight pairs: bin SHA/size + idx SHA/size
    native.corpus_tokenizer_gate(manifest, audit)
    compatibility = dict(status='parent_accepted_same_recipe_expansion',
        calibration_manifest_sha256=seed_sha, production_manifest_sha256=manifest_sha,
        same_provider_tokenizer_sequence_batch_topology=True, heldout_exclusions_preserved=True)
    evidence = dict(manifest_sha256=manifest_sha, prior_reserved_documents=reserved,
        capacity=capacity, reserved_id_audit=audit, original_base_model_only_reset=True,
        checkpoint_integrity='not_checked_explicit_user_waiver',
        projection_only_not_fresh_manifest_calibration=True, sustained_24h_proven=False)
    return compatibility, evidence


def check_corpus(actual, frozen):
    labels = actual.get('Config', {}).get('Labels', {}) or {}
    config = actual.get('Config', {})
    expected = frozen['payload']
    if (actual.get('Id') != frozen['id'] or actual.get('Name') != '/'+frozen['name']
            or {k:v for k,v in labels.items() if k.startswith('nemochat.')} != expected['Labels']
            or any(config.get(k) != expected[k] for k in ('Image', 'Cmd'))):
        raise ValueError('exact CPU Engine identity mismatch')


def wait_corpus(engine, frozen, timeout):
    import math
    if not math.isfinite(timeout) or timeout <= 0 or timeout > 43500:
        raise ValueError('finite positive CPU wait timeout <=43500 required')
    cid = frozen['id']
    target = '/containers/'+cid
    check_corpus(engine.request('GET', target+'/json'), frozen)
    waited = engine.request('POST', target+'/wait?condition=not-running', timeout=timeout)
    actual = engine.request('GET', target+'/json')
    check_corpus(actual, frozen)
    state = actual['State']
    if (waited.get('StatusCode') != 0 or waited.get('Error')
            or state.get('Running') is not False or state.get('ExitCode') != 0
            or state.get('OOMKilled') is not False):
        raise ValueError('clean exact CPU exit required; GPU release blocked')
    return state


def write(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix+'.partial')
    with partial.open('w') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def phase(folder, name, **fields):
    row = dict(phase=name, time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               run_id=RUN, **fields)
    with (folder/'phases.jsonl').open('a') as stream:
        stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    write(folder/'state.json', row)


def corpus_wait_seconds():
    started = datetime.datetime.fromisoformat('2026-10-17T11:31:59+00:00')
    now = datetime.datetime.now(datetime.timezone.utc)
    return max(1, min(43500, 43500-(now-started).total_seconds()))


def input_paths():
    calibration_folder = str(Path(CALIBRATION).parent.parent)
    return [CPU_IDENTITY, CALIBRATION, DURATION, NATIVE_REVIEW,
        'results/midtraining/super35-kd-01/corpus/production-reviewed-sources-01.json',
        calibration_folder+'/consumed-config.yaml', calibration_folder+'/controller/frozen-inputs.json']


def main_command(compatibility):
    return [sys.executable, str(ROOT/'midtraining/training/launch_longrun.py'),
        '--config', str(ROOT/'midtraining/configs/longrun24h.yaml'), '--run-id', RUN,
        '--manifest', str(ROOT/MANIFEST), '--calibration', str(ROOT/CALIBRATION),
        '--duration-control-receipt', str(ROOT/DURATION),
        '--calibration-compatibility', str(compatibility),
        '--reviewed-sources', str(ROOT/NATIVE_REVIEW), '--execute']


def execute(reviewed, engine=None):
    supervisor_gate()
    deadline = time.monotonic()+SUPERVISOR_SECONDS-60
    folder = ROOT/OUTPUT
    folder.mkdir(parents=True, exist_ok=False)
    try:
        phase(folder, 'WAITING_CORPUS', native_training_executed=False)
        sources = startup_sources()
        review_sha = digest(reviewed)
        source_gate(reviewed, sources)
        fingerprints = {name: digest(ROOT/name) for name in input_paths()}
        write(folder/'frozen-inputs.json', dict(files=fingerprints, startup_review_sha256=review_sha))
        frozen = approved_corpus()
        engine = engine or load('production_engine', 'results/pretrain_followup/engine_helper.py')
        state = wait_corpus(engine, frozen, corpus_wait_seconds())
        write(folder/'cpu-final.json', dict(id=frozen['id'], name=frozen['name'], State=state))
        source_gate(reviewed, sources)
        phase(folder, 'VALIDATING_DATA', native_training_executed=False)
        compatibility, evidence = validate_data(load('production_native_gates', 'midtraining/training/longrun_native.py'))
        if deadline-time.monotonic() < MAIN_SECONDS:
            raise TimeoutError('full 93600-second native main budget unavailable; release blocked')
        source_gate(reviewed, sources)
        if (digest(reviewed) != review_sha
                or fingerprints != {name: digest(ROOT/name) for name in input_paths()}):
            raise ValueError('frozen startup review/prerequisite receipt changed before release')
        for path in (ROOT/(BASE+RUN), ROOT/('checkpoints/midtraining/'+RUN),
                     ROOT/('results/midtraining/longrun-corpus-audits/'+RUN)):
            if path.exists():
                raise FileExistsError('existing native main preserved; no implicit resume')
        write(folder/'data-accepted.json', evidence)
        receipt = folder/'calibration-compatibility.json'
        write(receipt, compatibility)
        command = main_command(receipt)
        write(folder/'main-command.json', dict(argv=command, timeout_seconds=MAIN_SECONDS))
        source_gate(reviewed, sources)
        phase(folder, 'NATIVE_MAIN_CONTROLLER_RUNNING', optimizer_updates_verified=False)
        env = dict(os.environ, NEMOCHAT_LONGRUN_SUPERVISED='1')
        subprocess.run(command, cwd=ROOT, env=env, timeout=MAIN_SECONDS, check=True)
        result = read(ROOT/(BASE+RUN+'/controller/accepted.json'))
        if (result.get('status') != 'production_save_reload_accepted' or result.get('all_ranks') != 8
                or result.get('manifest_sha256') != compatibility['production_manifest_sha256']
                or type(result.get('training_24h_accepted')) is not bool):
            raise ValueError('native main save/reload acceptance missing or mismatched')
        write(folder/'native-result.json', result)
        phase(folder, 'FINISHED', native_status=result['status'],
              training_24h_accepted=result['training_24h_accepted'], remote_tracking_verified=False)
        return result
    except BaseException as exc:
        # Exception text can contain provider environments/credentials. Persist only its type.
        phase(folder, 'FAILED', error_type=type(exc).__name__, native_acceptance=False)
        raise


def cleanup_owned(reviewed):
    source_gate(reviewed, startup_sources())
    record = ROOT/OUTPUT/'main-command.json'
    if not record.exists():
        return dict(status='no_owned_main_release')
    expected = main_command(ROOT/OUTPUT/'calibration-compatibility.json')
    if read(record) != dict(argv=expected, timeout_seconds=MAIN_SECONDS):
        raise ValueError('exact native main release identity mismatch; refuse cleanup')
    # Existing cleanup verifies each recorded ID/name/task-label namespace before stop.
    return load('production_exact_cleanup', 'midtraining/training/launch_longrun.py').cleanup_owned(RUN)


def interrupted(signum, frame):
    raise InterruptedError('production supervisor stopped')


def supervisor_command(reviewed):
    """Return, never run, the exact parent-owned service command."""
    import shlex
    python = str(ROOT/'.venv/bin/python')
    script = str(ROOT/'midtraining/training/run_production.py')
    reviewed = str(Path(reviewed).resolve())
    log = str(ROOT/'results/midtraining/longrun/super35-production-02-supervisor.log')
    stop = shlex.join([python, script, '--cleanup-owned', '--reviewed-sources', reviewed])
    return ['systemd-run', '--user', '--unit='+UNIT,
        '--property=Type=exec',
        '--property=WorkingDirectory='+str(ROOT), '--property=RuntimeMaxSec=140400',
        '--property=KillMode=control-group', '--property=TimeoutStopSec=300',
        '--property=Restart=no', '--property=ExecStopPost='+stop,
        '--property=StandardOutput=append:'+log, '--property=StandardError=append:'+log,
        '--setenv=NEMOCHAT_PRODUCTION_SUPERVISED=1', '--setenv=PYTHONUNBUFFERED=1',
        '--setenv=PYTHONDONTWRITEBYTECODE=1', python, script, '--execute', '--reviewed-sources', reviewed]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviewed-sources', type=Path,
                        help='parent-created exact startup union SHA256 JSON (not native 14-file review)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--cleanup-owned', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute and not args.cleanup_owned:
        print(json.dumps(dict(status='inert_preview', unit=UNIT, run_id=RUN,
            source_paths=startup_sources(), output=OUTPUT, manifest=MANIFEST,
            cpu_identity=CPU_IDENTITY, native_review=NATIVE_REVIEW,
            supervisor_seconds=SUPERVISOR_SECONDS, native_main_timeout_seconds=MAIN_SECONDS,
            main_argv=main_command(ROOT/OUTPUT/'calibration-compatibility.json'),
            supervisor_argv=supervisor_command(args.reviewed_sources or ROOT/'results/midtraining/super35-kd-01/production-controller-source-review-02.json'),
            native_training_executed=False, remote_tracking_run_created=False), indent=2))
        return 0
    if args.reviewed_sources is None:
        parser.error('--reviewed-sources required for execute/cleanup')
    if args.cleanup_owned:
        print(json.dumps(cleanup_owned(args.reviewed_sources), indent=2))
        return 0
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    try:
        result = execute(args.reviewed_sources)
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as exc:
        # Durable state carries the sanitized failure class; do not echo external exception payloads.
        print('production controller failed: '+type(exc).__name__, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

