#!/usr/bin/env python3
"""Parent-owned full-corpus -> native GB128 -> 48h pipeline; preview is inert."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('results/pretraining/nemochat_baby/nemotron_sft')
RAW = Path('datasets/nemotron_sft/nemochat_baby/raw-full-01')
# Accepted SDK reference is independent of the recovery generation.
SDK_GUARD = BASE / 'runs/nemotron-gb128-calibration-02/rank-0-source-guard.json'
# 48h native + 60m grace + 240s native supervisor + <=2h launcher preflight.
MIN_MAIN_REMAINING_SECONDS = 52 * 3600
PYTHON = ROOT / '.venv/bin/python'
LAUNCHER = 'scripts/posttraining/launch_nemotron_sft.py'


def execution_contract(generation):
    """Only generation 03 changes the accepted data/native execution policy."""
    modern = generation == '03'
    return dict(sequence_length=16384 if modern else 4096,
                excluded_datasets=['nvidia/Nemotron-SFT-Long-Context-Transduction-v1'] if modern else [],
                production_template=Path('configs/posttraining/nemochat_baby_nemotron_sft_16k_48h.yaml'
                                         if modern else 'configs/posttraining/nemochat_baby_nemotron_sft_48h.yaml'),
                calibration_template=BASE / ('seq16k_protocol/calibration.yaml' if modern else 'longrun_protocol/calibration.yaml'),
                calibration_max_hours=1 if modern else 0.25)


def reviewed_input_paths():
    contract = execution_contract(GENERATION)
    return [Path('scripts/posttraining/nemotron_48h_pipeline.py'), Path(LAUNCHER),
            Path('scripts/posttraining/native_nemotron_sft.py'), contract['production_template'],
            contract['calibration_template'], CPU / 'FULL_HANDOFF.json', SDK_GUARD] + (
                [BASE / 'cpu-full-01/acquisition-completed.json',
                 Path('datasets/nemotron_sft/nemochat_baby/prepared-pilot-01/data/heldout-dialogues.jsonl')]
                if GENERATION == '03' else [])


def reviewed_start_paths():
    return reviewed_input_paths() + [Path('tests/test_nemotron_48h_pipeline.py'),
                                     Path('tests/test_nemotron_48h_16k_pipeline.py'),
                                     Path('tests/test_nemotron_process_identity_paths.py')]


def verify_reviewed_start(*, root=ROOT):
    """Never populate or refresh checksums implicitly: parent freezes final inputs."""
    entries = []
    for line in (Path(root) / OUT / 'reviewed-inputs.sha256').read_text().splitlines():
        digest, separator, path = line.partition('  ')
        if (not separator or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('malformed reviewed source allowlist')
        entries.append((path, digest))
    expected = [str(path) for path in reviewed_start_paths()]
    if len(entries) != len(expected) or {path for path, digest in entries} != set(expected):
        raise ValueError('reviewed source allowlist incomplete/duplicated/unapproved')
    controller = Controller(root=root)
    for path, digest in entries:
        if controller.stable_hash(Path(root) / path) != digest:
            raise ValueError('parent-reviewed source changed; explicit re-review/refresh required')
    return dict(status='source_checksums_verified_only_not_execution_authorization',
                generation=GENERATION, checked_files=len(entries), gpu_launches=0)


def configure_generation(generation):
    """Select one process-local namespace before any CLI preview/run/cleanup.

    Generation 01 retains its original paths. Recovery never renames, reads or
    overwrites that controller's state; only the immutable raw inputs are reused.
    """
    if (not isinstance(generation, str) or len(generation) != 2
            or any(c not in '0123456789' for c in generation) or generation == '00'):
        raise ValueError('generation must be exactly two ASCII digits from 01 through 99')
    global GENERATION, CPU, OUT, MANIFEST, READY, PREP_UNIT, PREP_SCRIPT
    global CAL_NAME, MAIN_NAME, MAIN_ID, UNIT, CAL_CONFIG, MAIN_CONFIG
    GENERATION = generation
    CPU = BASE / ('cpu-full-' + generation)
    OUT = BASE / ('longrun_protocol/pipeline' + ('' if generation == '01' else '-' + generation))
    MANIFEST = Path('datasets/nemotron_sft/nemochat_baby') / ('prepared-full-' + generation) / 'data/manifest.json'
    READY = CPU / 'completion-ready.json'
    PREP_UNIT = 'nemochat-nemotron-full-prepare-' + generation + '.service'
    PREP_SCRIPT = str(ROOT / CPU / 'run_full_preparation.py')
    CAL_NAME = 'nemotron-full-gb128-calibration-' + generation
    MAIN_NAME = 'nemotron-main-48h-' + generation
    MAIN_ID = 'nemochat-baby-nemotron-sft-main-48h-' + generation
    UNIT = 'nemochat-nemotron-48h-pipeline-' + generation + '.service'
    CAL_CONFIG = OUT / ('calibration-full-' + generation + '.yaml')
    MAIN_CONFIG = OUT / ('production-48h-' + generation + '.yaml')


configure_generation('01')


def read_json(path):
    return json.loads(Path(path).read_text())


def service_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '-p', 'ActiveState',
                             '-p', 'LoadState', '-p', 'Result', '-p', 'ExecMainStatus'],
                            capture_output=True, text=True, timeout=30)
    props = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode and props.get('LoadState') != 'not-found':
        raise RuntimeError('could not inspect exact owned user service')
    return props


def supervisor_payload():
    return ['systemd-run', '--user', '--unit=' + UNIT, '--service-type=exec',
            '--working-directory=' + str(ROOT), '--property=RuntimeMaxSec=345600',
            '--property=TimeoutStopSec=180', '--property=KillMode=control-group',
            '--property=NoNewPrivileges=yes', '--setenv=PYTHONPATH=src',
            '--property=ExecStopPost=' + shlex.join([str(PYTHON), str(ROOT / 'scripts/posttraining/nemotron_48h_pipeline.py'), '--generation', GENERATION, '--cleanup-owned']),
            '--setenv=NEMOTRON_48H_PIPELINE_SUPERVISED=' + GENERATION,
            '--property=StandardOutput=append:' + str(ROOT / OUT / 'controller.log'),
            '--property=StandardError=append:' + str(ROOT / OUT / 'controller.log'),
            str(PYTHON), str(ROOT / 'scripts/posttraining/nemotron_48h_pipeline.py'), '--generation', GENERATION, '--execute']


def parent_start_script():
    verify = [str(PYTHON), str(ROOT / 'scripts/posttraining/nemotron_48h_pipeline.py'),
              '--generation', GENERATION, '--verify-reviewed-inputs']
    return ('#!/usr/bin/env bash\nset -euo pipefail\n'
            '# Parent review only. Never auto-refresh missing/stale source checksums.\n'
            'if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--verify-only" ) ]]; then exit 64; fi\n'
            'cd -- ' + shlex.quote(str(ROOT)) + '\n' + shlex.join(verify) + '\n'
            'if [[ "${1:-}" == "--verify-only" ]]; then exit 0; fi\n'
            'exec ' + shlex.join(supervisor_payload()) + '\n')


def preview():
    service = service_state(PREP_UNIT)
    identity = None
    if int(service.get('MainPID', 0)):
        try:
            identity = process_identity(int(service['MainPID']), PREP_SCRIPT)
        except FileNotFoundError:
            pass  # exited between systemctl and /proc; execution reconciles final proof
    return dict(status='DATA_PROOF_AVAILABLE_NOT_ACCEPTED' if READY.is_file() else 'WAITING_DATA',
                generation=GENERATION, gpu_launches=0,
                sequence_length=execution_contract(GENERATION)['sequence_length'],
                excluded_datasets=execution_contract(GENERATION)['excluded_datasets'],
                required_atomic_ready=str(READY), manifest=str(MANIFEST),
                preparation_unit=PREP_UNIT, preparation_service=service, preparation_identity=identity,
                calibration_name=CAL_NAME, main_name=MAIN_NAME, main_run_id=MAIN_ID,
                stage_order=['atomic_full_data', 'full_native_GB128_calibration', 'freeze_48h', 'main_supervisor', 'native_final_acceptance'],
                tracking_destination='nvidia/nemochat', calibration_command=calibration_command(),
                main_command=main_command(),
                execution_state=str(OUT / 'controller-state.json'), supervisor_command=supervisor_payload(),
                main_config_target=str(MAIN_CONFIG), production_started=False,
                checkpoint_integrity='not_checked_explicit_user_waiver',
                checkpoint_bytes_read=False)


def derive_horizon(rows, mean, *, overhead=0.10):
    if type(rows) is not int or rows < 50000 or not math.isfinite(mean) or mean <= 0:
        raise ValueError('eligible unique rows and finite measured timing required')
    if not math.isfinite(overhead) or not 0 <= overhead < 1:
        raise ValueError('explicit overhead fraction required')
    horizon = math.ceil(48 * 3600 * (1 - overhead) / mean)
    per_epoch = math.ceil(rows / 128)
    epochs = math.ceil(horizon / per_epoch)
    if epochs > 3 or horizon > 100000:
        raise ValueError('capacity insufficient within <=3 approved epochs; no automatic repeat expansion')
    return dict(max_updates=horizon, max_hours=48, epochs=epochs, authorized_max_epochs=3,
                allow_repetition=epochs > 1, available_updates=epochs * per_epoch,
                additional_epochs_above_one=epochs - 1, estimated_iteration_seconds=mean,
                assumed_overhead_fraction=overhead,
                exposure_disclosure='unique-first; zero-loss padded epoch tails; up to explicit epochs',
                projected_hours=horizon * mean / (1 - overhead) / 3600)


def validate_reused_acquisition(acquisition, archived):
    if acquisition != archived:
        raise ValueError('immutable reused seven-source acquisition receipt mismatch')


def validate_cpu_origin(state, immutable, created=None):
    host = immutable.get('HostConfig', {})
    cid = created.get('Id') if GENERATION == '03' and isinstance(created, dict) else state.get('container_id')
    if (not cid or immutable.get('Id') != cid
            or (GENERATION == '03' and host.get('Devices'))
            or host.get('DeviceRequests') or host.get('NetworkMode') != 'none'
            or host.get('NanoCpus') != 24000000000
            or immutable.get('Config', {}).get('Cmd') != ['-u', str(CPU / 'full_workload.py')]):
        raise ValueError('exact full-preparation CPU Engine origin mismatch')


def validate_data_contract(state, ready, handoff, acquisition, engine, manifest, audit,
                           manifest_sha, audit_sha):
    """Reject stale/mixed proof chains; status alone never authorizes GPU work."""
    target = str(MANIFEST)
    marker = str(READY)
    handoff_policy = handoff.get('source_contract', {}) if GENERATION == '03' else handoff
    content_bytes = handoff_policy.get('source_content_bytes') if GENERATION == '03' else handoff.get('expected_content_bytes')
    if (handoff.get('full_manifest_target') != target or handoff.get('completion_ready_target') != marker
            or content_bytes != 100853847138):
        raise ValueError('full handoff scope mismatch')
    state_valid = (state.get('status') == 'passed_CPU_full_preparation_and_audit'
                   and state.get('unit') == PREP_UNIT and state.get('output') == str(MANIFEST.parent)
                   and state.get('reused_acquisition_receipt') == str(BASE / 'cpu-full-01/acquisition-completed.json')) if GENERATION == '03' else (
                       state.get('status') == 'passed_CPU_full_preparation_and_counted_sample_audit'
                       and state.get('manifest') == target)
    if (not state_valid or ready.get('status') != 'passed'
            or ready.get('manifest') != target or ready.get('manifest_sha256') != manifest_sha
            or ready.get('audit_sha256') != audit_sha or ready.get('source_content_bytes') != 100853847138):
        raise ValueError('atomic completion-ready / preparation proof mismatch')
    if (engine.get('Running') is not False or engine.get('Pid') != 0 or engine.get('ExitCode') != 0
            or engine.get('OOMKilled') is not False or ready.get('engine_state') != engine):
        raise ValueError('clean immutable CPU Engine proof required')
    sources = acquisition.get('completed_sources', [])
    # Generation03 freezes the immutable reused acquisition receipt separately;
    # its data-worker handoff carries policy, not a second copied source registry.
    expected = {s['key']: s for s in (sources if GENERATION == '03' else handoff['sources'])}
    if (acquisition.get('status') != 'passed' or acquisition.get('completed_files') != 7
            or acquisition.get('completed_expected_bytes') != 100853847138 or len(sources) != 7
            or len(expected) != 7 or sum(s['expected_bytes'] for s in sources) != 100853847138
            or {s['key'] for s in sources} != set(expected)):
        raise ValueError('all seven full sources required')
    registered = {s['key']: s for s in manifest.get('sources', [])}
    for source in sources:
        frozen = expected[source['key']]
        if (not source.get('full_file_digest_verified') or source['retained_bytes'] != source['expected_bytes']
                or any(source.get(k) != frozen.get(k) for k in
                       ('dataset', 'path', 'revision', 'expected_bytes', 'expected_sha256'))
                or any(registered.get(source['key'], {}).get(k) != source.get(k) for k in
                       ('dataset', 'path', 'revision', 'expected_bytes', 'expected_sha256'))):
            raise ValueError('full source digest/identity drift')
    if (manifest.get('status') != 'passed' or manifest.get('train_dialogues', 0) < 50000
            or manifest.get('prior_reserved_prompt_groups') != 14
            or manifest.get('train_heldout_overlap') != 0 or manifest.get('kmmlu_exact_question_matches') != 0
            or manifest.get('kmmlu_reserved_questions') != 34871 or manifest.get('training_repetition') is not False
            or manifest.get('tokenizer_revision') != 'fc067b34f6b514014d2f8bddba855d2e4bfdfba6'
            or audit.get('status') != 'passed'
            or audit.get('all_conversations_counted') != manifest['train_dialogues'] + manifest['heldout_dialogues']
            or audit.get('all_loss_tokens_counted') != manifest['train_loss_tokens'] + manifest['heldout_loss_tokens']):
        raise ValueError('full data eligibility / counted audit mismatch')
    if GENERATION == '03':
        contract = execution_contract(GENERATION)
        excluded = contract['excluded_datasets']
        active = sorted(s['key'] for s in sources if s['dataset'] not in excluded)
        if len(active) != 6 or len(registered) != len(manifest.get('sources', [])):
            raise ValueError('16k active source registry mismatch')
        if handoff_policy.get('reused_acquisition_receipt') != str(BASE / 'cpu-full-01/acquisition-completed.json'):
            raise ValueError('16k immutable acquisition receipt reference mismatch')
        for proof in (handoff_policy, ready, manifest):
            keys = proof.get('active_training_source_keys')
            if (proof.get('sequence_length') != 16384 or proof.get('excluded_datasets') != excluded
                    or not isinstance(keys, list) or sorted(keys) != active):
                raise ValueError('16k sequence/exclusion/active source policy mismatch')
        if manifest.get('prior_heldout_reservation_file') != 'datasets/nemotron_sft/nemochat_baby/prepared-pilot-01/data/heldout-dialogues.jsonl':
            raise ValueError('16k complete original heldout blocklist path required')
        presence = manifest.get('reservation_presence', {})
        present = presence.get('physically_present')
        absent = presence.get('not_present_in_selected_corpus')
        inactive = presence.get('inactive_due_excluded_dataset')
        if (any(type(n) is not int or n < 0 for n in (present, absent, inactive))
                or present + absent != 14 or inactive > absent
                or ready.get('reservation_presence') != presence):
            raise ValueError('all 14 prior reservations must remain blocked; inactive groups are not heldout')
        for split in ('train', 'heldout'):
            for kind, width in (('tokens', 16385), ('masks', 16384)):
                shape = [manifest[split + '_dialogues'], width]
                key = split + '_' + kind
                if any(proof.get('array_shapes', {}).get(key) != shape for proof in (manifest, ready)):
                    raise ValueError('16k prepared array shape mismatch')
        for source in sources:
            stats = manifest.get('per_source', {}).get(source['key'], {})
            if source['dataset'] in excluded:
                if (any(stats.get(k, 0) != 0 for k in ('train', 'heldout', 'supervised_tokens', 'accepted'))
                        or stats.get('excluded') is not True
                        or stats.get('exclusion_reason') != 'deferred_long_context_stage'):
                    raise ValueError('excluded dataset has targets/heldout or missing deferred reason')
            elif stats.get('train', 0) <= 0 or stats.get('supervised_tokens', 0) <= 0:
                raise ValueError('active source must contribute real training targets')


def process_identity(pid, expected_arg):
    """Bind the exact owned argv token, UID and Linux starttime before pidfd open."""
    proc = Path('/proc') / str(pid)
    if pid <= 0 or proc.stat().st_uid != os.getuid():
        raise ValueError('owned process UID/PID mismatch')
    argv = (proc / 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    argv = [part.decode() for part in argv]
    matches = expected_arg in argv
    if not matches and Path(expected_arg).is_absolute():
        # Services may execute a repository-relative script from their actual CWD.
        cwd = (proc / 'cwd').resolve()
        target = Path(expected_arg).resolve()
        matches = any((cwd / arg).resolve() == target for arg in argv
                      if arg and not arg.startswith('-'))
    if not matches:
        raise ValueError('owned process argv mismatch')
    start_ticks = int((proc / 'stat').read_text().rsplit(')', 1)[1].split()[19])
    return dict(pid=pid, uid=os.getuid(), expected_arg=expected_arg, start_ticks=start_ticks)


def wait_pidfd(identity, *, deadline):
    """One kernel-blocking exit event, no child-artifact/status polling."""
    fd = os.pidfd_open(identity['pid'])
    try:
        try:
            actual = process_identity(identity['pid'], identity['expected_arg'])
        except ValueError:
            # Exited unreaped processes lose cmdline but retain UID/starttime until reaping.
            proc = Path('/proc') / str(identity['pid'])
            stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            if (stat[0] != 'Z' or int(stat[19]) != identity['start_ticks']
                    or proc.stat().st_uid != identity['uid']):
                raise ValueError('PID reuse / process identity changed') from None
            actual = identity
        if actual != identity:
            raise ValueError('PID reuse / process identity changed')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('controller deadline exceeded')
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        events = poller.poll(max(1, int(remaining * 1000)))
        if not events:
            raise TimeoutError('controller deadline exceeded')
        if not events[0][1] & select.POLLIN:
            raise RuntimeError('unexpected pidfd event')
        return dict(event='pidfd_exit', **identity)
    finally:
        os.close(fd)


def contingent_stages(actions, phase):
    """Pure ordered release; callbacks must raise on any incomplete acceptance."""
    for name, before, after in [
        ('data', 'WAITING_DATA', 'full_data_accepted'),
        ('calibration', 'full_native_calibration', 'full_native_calibration_accepted'),
        ('freeze', 'freeze_measured_48h', 'production_config_frozen'),
        ('launch', 'starting_main_supervisor', 'main_supervisor_started'),
        ('finish', 'waiting_main_pidfd', 'completed'),
    ]:
        phase(before)
        actions[name]()
        phase(after)


def validate_execution_shape(config):
    if GENERATION == '03' and (
            config.get('sequence_length') != 16384
            or config.get('excluded_datasets') != execution_contract('03')['excluded_datasets']
            or config.get('global_batch_size') != 128
            or (config.get('context_parallel_size'), config.get('expert_model_parallel_size')) not in ((1, 8), (2, 4))):
        raise ValueError('16k GB128 exclusion/CP/EP execution shape mismatch')


def validate_calibration_shape(receipt, config):
    if GENERATION == '03' and any(receipt.get(key) != config.get(key) for key in (
            'sequence_length', 'excluded_datasets', 'global_batch_size',
            'context_parallel_size', 'expert_model_parallel_size')):
        raise ValueError('16k calibration sequence/source policy/topology binding mismatch')


def calibration_budget(config, plan):
    validate_execution_shape(config)
    limit = execution_contract(GENERATION)['calibration_max_hours']
    hours = config.get('max_hours', 0)
    updates = config.get('max_updates')
    valid_updates = type(updates) is int and (15 <= updates <= 20 if GENERATION == '03' else updates == 20)
    if (not valid_updates or plan.get('updates') != updates
            or not isinstance(hours, (float, int)) or not math.isfinite(hours) or not 0 < hours <= limit):
        raise ValueError('bounded GB128 native calibration contract mismatch')
    if GENERATION != '03':
        return 1500
    grace = config.get('watchdog_grace_minutes', 0)
    if (config.get('execution_mode') != 'calibration' or config.get('require_final_validation') is not True
            or type(config.get('validation_max_dialogues')) is not int
            or not 128 <= config['validation_max_dialogues'] <= 256
            or not isinstance(grace, (int, float)) or not math.isfinite(grace) or not 1 <= grace <= 60):
        raise ValueError('16k bounded calibration/final-heldout watchdog contract mismatch')
    return math.ceil(hours * 3600 + grace * 60 + 7200 + 240)


def calibration_spec():
    import yaml
    config = yaml.safe_load((ROOT / execution_contract(GENERATION)['calibration_template']).read_text())
    if GENERATION != '03':
        config['name'] = 'nemochat baby Nemotron full GB128 calibration'
    return config, calibration_command()


def calibration_command():
    return [str(PYTHON), LAUNCHER, '--name', CAL_NAME, '--manifest', str(MANIFEST),
            '--config', str(CAL_CONFIG), '--execute']


def main_command():
    return [str(PYTHON), LAUNCHER, '--name', MAIN_NAME, '--manifest', str(MANIFEST),
            '--config', str(MAIN_CONFIG), '--execute', '--supervise', '--online', '--run-id', MAIN_ID]


def production_spec(manifest, calibration, receipt_path, *, calibration_plan=None):
    import yaml
    config = yaml.safe_load((ROOT / execution_contract(GENERATION)['production_template']).read_text())
    validate_execution_shape(config)
    validate_calibration_shape(calibration_plan if calibration_plan is not None else calibration, config)
    if GENERATION == '03' and calibration_plan is not None:
        for key in ('sequence_length', 'excluded_datasets', 'global_batch_size',
                    'context_parallel_size', 'expert_model_parallel_size'):
            if key in calibration and calibration[key] != config.get(key):
                raise ValueError('16k native calibration metadata contradicts accepted producer plan')
    config.update(derive_horizon(manifest['train_dialogues'], calibration['mean_iteration_seconds'], overhead=0.10))
    config['calibration_receipt'] = str(receipt_path)
    native = Controller().load_module('scripts/posttraining/native_nemotron_sft.py', 'pipeline_plan_native')
    return config, native.stage_plan(config, manifest)


class Controller:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.deadline = time.monotonic() + 96 * 3600
        self.evidence = {}

    def run(self):
        if os.environ.get('NEMOTRON_48H_PIPELINE_SUPERVISED') != GENERATION:
            raise ValueError('execution requires exact parent-owned supervisor')
        state_path = self.root / OUT / 'controller-state.json'
        if state_path.exists():
            raise FileExistsError('refusing existing controller state; parent review/new version required')
        service = service_state(UNIT)
        if int(service.get('MainPID', 0)) != os.getpid():
            raise ValueError('execution requires exact parent-owned supervisor MainPID')
        process_identity(os.getpid(), str(self.root / 'scripts/posttraining/nemotron_48h_pipeline.py'))
        self.assert_fresh(CAL_NAME)
        self.assert_fresh(MAIN_NAME)
        (self.root / OUT).mkdir(parents=True, exist_ok=True)
        with (self.root / OUT / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with state_path.open('x') as handle:
                json.dump(dict(status='created', owner_uid=os.getuid(), unit=UNIT), handle)
            self.owned_runs = []
            try:
                self.frozen_inputs = {str(path): self.stable_hash(self.root / path)
                                      for path in reviewed_input_paths()}
                contingent_stages(dict(data=self.data_gate, calibration=self.calibrate, freeze=self.freeze,
                                       launch=self.launch_main, finish=self.finish_main), self.phase)
            except BaseException as exc:
                # Never include arbitrary exception text (SDK/network errors may contain credentials).
                self.evidence['failure_type'] = type(exc).__name__
                self.phase('failed', enforce_deadline=False)
                raise RuntimeError('contingent pipeline failed; inspect exact phase and owned stage logs') from None

    def phase(self, name, *, enforce_deadline=True):
        if enforce_deadline and time.monotonic() >= self.deadline:
            raise TimeoutError('controller 96-hour deadline exceeded')
        state = dict(status=name, unit=UNIT, owner_uid=os.getuid(), controller_pid=os.getpid(),
                     elapsed_seconds=96 * 3600 - (self.deadline - time.monotonic()),
                     owned_runs=getattr(self, 'owned_runs', []),
                     input_hashes=getattr(self, 'frozen_inputs', {}), evidence=self.evidence,
                     checkpoint_integrity='not_checked_explicit_user_waiver', checkpoint_bytes_read=False)
        target = self.root / OUT / 'controller-state.json'
        pending = target.with_suffix('.pending')
        pending.write_text(json.dumps(state, indent=2) + '\n')
        pending.replace(target)
        with (self.root / OUT / 'controller-events.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(status=name, elapsed_seconds=state['elapsed_seconds'])) + '\n')
        print(json.dumps(dict(status=name, elapsed_seconds=state['elapsed_seconds'])), flush=True)

    def verify_frozen_inputs(self):
        for path, expected in getattr(self, 'frozen_inputs', {}).items():
            if self.stable_hash(self.root / path) != expected:
                raise ValueError('controller/runner/template/SDK baseline changed during contingent wait')

    def data_gate(self):
        service = service_state(PREP_UNIT)
        self.evidence['preparation'] = dict(
            unit=PREP_UNIT, proof_root=str(CPU), required_atomic_ready=str(READY),
            service={key: service[key] for key in
                     ('MainPID', 'LoadState', 'ActiveState', 'Result', 'ExecMainStatus') if key in service})
        pid = int(service.get('MainPID', 0))
        if pid:
            try:
                identity = process_identity(pid, PREP_SCRIPT)
                self.evidence['data_wait'] = wait_pidfd(identity, deadline=self.deadline)
            except (ProcessLookupError, FileNotFoundError):
                # Exit races/collected services must reconcile the SAME atomic proof; never restart.
                self.evidence['data_wait'] = dict(event='process_already_exited_reconcile_proof')
        # Exact unit outcome is diagnostic only, never a replacement for readiness.
        if pid:
            final_service = service_state(PREP_UNIT)
            self.evidence['preparation']['service'] = {key: final_service[key] for key in
                ('MainPID', 'LoadState', 'ActiveState', 'Result', 'ExecMainStatus') if key in final_service}
        try:
            ready = read_json(self.root / READY)  # mandatory atomic last publication
        except FileNotFoundError:
            self.evidence['data_failure'] = 'missing_atomic_completion_ready'
            raise
        state = read_json(self.root / CPU / 'preparation-controller-state.json')
        handoff = read_json(self.root / CPU / 'FULL_HANDOFF.json')
        acquisition_path = BASE / 'cpu-full-01/acquisition-completed.json' if GENERATION == '03' else CPU / 'acquisition-completed.json'
        acquisition = read_json(self.root / acquisition_path)
        created = None
        if GENERATION == '03':
            self.verify_frozen_inputs()
            if state.get('reused_acquisition_receipt') != str(acquisition_path):
                raise ValueError('immutable archived acquisition reference mismatch')
            validate_reused_acquisition(acquisition, read_json(self.root / acquisition_path))
            created = read_json(self.root / CPU / 'full-preparation.create.json')
        immutable = read_json(self.root / CPU / 'full-preparation.final-state.json')
        validate_cpu_origin(state, immutable, created)
        manifest = read_json(self.root / MANIFEST)
        audit_path = self.root / MANIFEST.parent / 'counted-source-mask-audit.json'
        audit = read_json(audit_path)
        manifest_sha = self.stable_hash(self.root / MANIFEST)
        validate_data_contract(state, ready, handoff, acquisition, immutable['State'], manifest,
                               audit, manifest_sha, self.stable_hash(audit_path))
        # Dataset-only raw rehash is allowed; never touch checkpoint payload bytes.
        for source in acquisition['completed_sources']:
            raw = self.raw_source_path(source['raw_file'])
            if raw.stat().st_size != source['expected_bytes']:
                raise ValueError('actual full raw dataset file size/path mismatch')
            if self.stable_hash(raw) != source['expected_sha256']:
                raise ValueError('actual full raw dataset SHA mismatch')
        native = self.load_module('scripts/posttraining/native_nemotron_sft.py', 'pipeline_native')
        verification = native.verify_inputs(self.root / MANIFEST, self.root / manifest['tokenizer_manifest'], hash_data=True)
        if self.stable_hash(self.root / MANIFEST) != manifest_sha:
            raise ValueError('full manifest changed during native verification')
        self.manifest = manifest
        self.manifest_sha = manifest_sha
        self.evidence['data'] = dict(manifest_sha256=manifest_sha, native_input_verification=verification,
                                     completion_ready_sha256=self.stable_hash(self.root / READY))

    def raw_source_path(self, raw_file):
        if not isinstance(raw_file, str) or not raw_file or raw_file in ('.', '..') or Path(raw_file).name != raw_file:
            raise ValueError('actual full raw dataset file path mismatch')
        return self.root / RAW / raw_file

    def require_remaining(self, seconds):
        if self.deadline - time.monotonic() < seconds:
            raise ValueError('insufficient controller remaining time for complete downstream stage')

    def calibrate(self):
        import yaml
        if not self.evidence.get('data'):
            raise ValueError('accepted full data proof required before calibration')
        if self.stable_hash(self.root / MANIFEST) != self.manifest_sha:
            raise ValueError('accepted full data manifest changed')
        self.verify_frozen_inputs()
        config, command = calibration_spec()
        native = self.load_module('scripts/posttraining/native_nemotron_sft.py', 'pipeline_native')
        self.cal_plan = native.stage_plan(config, self.manifest)
        timeout = calibration_budget(config, self.cal_plan)
        self.require_remaining(MIN_MAIN_REMAINING_SECONDS + (timeout if GENERATION == '03' else 0))
        self.assert_fresh(CAL_NAME)
        with (self.root / CAL_CONFIG).open('x') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        self.owned_runs.append(CAL_NAME)
        self.phase('full_native_calibration_launch_intent')
        self.run_command(command, 'calibration-launch', max_seconds=timeout)
        self.evidence['calibration_native'] = self.accept_native(CAL_NAME, self.cal_plan)
        path = self.root / BASE / 'runs' / CAL_NAME / 'calibration.json'
        receipt = read_json(path)
        if (receipt.get('manifest_sha256') != self.manifest_sha or receipt.get('status') != 'passed'
                or receipt.get('warm_updates', 0) < 10 or receipt.get('cold_updates_excluded') != 5
                or receipt.get('global_batch_size') != 128 or receipt.get('ranks') != list(range(8))
                or receipt.get('source_checkpoint') != native.SOURCE
                or receipt.get('verification') != str(BASE / 'runs' / CAL_NAME / 'verification.json')):
            raise ValueError('full-specific native calibration receipt/lineage mismatch')
        rows = [json.loads(line) for line in (path.parent / 'rank-0-updates.jsonl').open()][5:]
        timings = [row['iteration_seconds'] for row in rows]
        if (len(timings) < 10 or len(timings) != receipt['warm_updates']
                or not all(math.isfinite(t) and t > 0 for t in timings)
                or not math.isclose(sum(timings) / len(timings), receipt['mean_iteration_seconds'], rel_tol=1e-12)):
            raise ValueError('measured native warm calibration window mismatch')
        self.calibration = receipt
        self.calibration_path = path.relative_to(self.root)
        self.evidence['calibration'] = dict(receipt=str(self.calibration_path), sha256=self.stable_hash(path),
                                             mean_iteration_seconds=receipt['mean_iteration_seconds'])
        if GENERATION == '03':
            self.evidence['calibration'].update(
                accepted_native_run=CAL_NAME, manifest_sha256=self.manifest_sha,
                execution_shape={key: self.cal_plan[key] for key in (
                    'sequence_length', 'excluded_datasets', 'global_batch_size',
                    'context_parallel_size', 'expert_model_parallel_size')},
                shape_origin='accepted native producer config/manifest/script provenance, not added native receipt fields')

    def freeze(self):
        import yaml
        if not self.evidence.get('calibration'):
            raise ValueError('accepted full-native calibration required before freezing production')
        self.verify_frozen_inputs()
        receipt = self.root / self.calibration_path
        if (self.stable_hash(receipt) != self.evidence['calibration']['sha256']
                or self.stable_hash(self.root / MANIFEST) != self.manifest_sha):
            raise ValueError('calibration/full manifest binding changed')
        config, self.main_plan = production_spec(self.manifest, self.calibration, self.calibration_path,
                                                 calibration_plan=self.cal_plan if GENERATION == '03' else None)
        launcher = self.load_module(LAUNCHER, 'pipeline_calibration_validator')
        launcher.validate_calibration(self.main_plan, self.root / MANIFEST)
        with (self.root / MAIN_CONFIG).open('x') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        self.evidence['production'] = dict(config=str(MAIN_CONFIG), sha256=self.stable_hash(self.root / MAIN_CONFIG),
                                            stage_plan=self.main_plan)
        self.run_command([str(PYTHON), LAUNCHER, '--name', MAIN_NAME, '--manifest', str(MANIFEST),
                          '--config', str(MAIN_CONFIG), '--preview'], 'main-preview', max_seconds=7200)

    def launch_main(self):
        if not self.evidence.get('production'):
            raise ValueError('accepted frozen 48h config required before main launch')
        if (self.stable_hash(self.root / MAIN_CONFIG) != self.evidence['production']['sha256']
                or self.stable_hash(self.root / MANIFEST) != self.manifest_sha
                or self.stable_hash(self.root / self.calibration_path) != self.evidence['calibration']['sha256']):
            raise ValueError('frozen main configuration/calibration/data binding drift')
        self.verify_frozen_inputs()
        if self.deadline - time.monotonic() < MIN_MAIN_REMAINING_SECONDS:
            raise ValueError('insufficient 96h controller remaining time for complete 48h stage')
        self.assert_fresh(MAIN_NAME)
        self.evidence['tracking_preflight'] = self.tracking_preflight()
        self.owned_runs.append(MAIN_NAME)
        self.phase('main_supervisor_launch_intent')
        self.run_command(main_command(), 'main-supervisor-launch', max_seconds=7200)
        self.evidence['main_supervisor'] = service_state('nemotron-sft-' + MAIN_NAME + '.service')
        # This is supervisor creation only. Native updates/completion are verified AFTER pidfd exit.

    def finish_main(self):
        if 'main_supervisor' not in self.evidence:
            raise ValueError('main supervisor has not been started by this controller')
        service = service_state('nemotron-sft-' + MAIN_NAME + '.service')
        pid = int(service.get('MainPID', 0))
        if pid:
            try:
                identity = process_identity(pid, str(self.root / LAUNCHER))
                named = process_identity(pid, MAIN_NAME)
                if named['start_ticks'] != identity['start_ticks']:
                    raise ValueError('main supervisor PID reuse')
                self.evidence['main_wait'] = wait_pidfd(identity, deadline=self.deadline)
            except (FileNotFoundError, ProcessLookupError):
                self.evidence['main_wait'] = dict(event='already_exited_reconcile_native_artifacts')
        # A missing/garbage-collected unit is NOT success and is NEVER restarted.
        self.evidence['main_final'] = self.accept_native(MAIN_NAME, self.main_plan)
        self.evidence['tracking_final'] = self.tracking_preflight(final=True)

    def tracking_preflight(self, *, final=False):
        import wandb
        secure = self.load_module('results/bridge_wandb/wandb_preflight.py', 'pipeline_secure_tracking')
        # Runtime-only allowlisted parsing. Never serialize the options/key or exception text.
        options = secure.read_wandb_env(self.root / '.env')
        key = options.get('WANDB_API_KEY') or os.environ.get('WANDB_API_KEY')
        if not key:
            raise ValueError('runtime-only approved tracking credential missing')
        try:
            api = wandb.Api(api_key=key, timeout=45)
            if not final:
                proof = read_json(self.root / 'results/bridge_wandb/latest_preflight.json')
                if (proof.get('entity'), proof.get('project'), proof.get('mode')) != ('nvidia', 'nemochat', 'online'):
                    raise ValueError('approved tracking destination proof mismatch')
                remote = api.run('nvidia/nemochat/' + proof['run_id'])
                if (remote.id != proof['run_id'] or remote.state != 'finished'
                        or remote.summary.get('preflight_ok') is not True
                        or remote.summary.get('training_updates') != 0):
                    raise ValueError('exact prior approved tracking preflight readback failed')
                # Public API list filter avoids treating authentication/transport errors as NotFound.
                existing = list(api.runs('nvidia/nemochat', filters={'name': MAIN_ID}, per_page=1))
                if existing:
                    raise ValueError('stable production tracking ID already exists')
                return dict(entity='nvidia', project='nemochat', preflight_run_id=remote.id,
                            remote_summary_verified=True, fresh_main_run_id=MAIN_ID)
            remote = api.run('nvidia/nemochat/' + MAIN_ID)
            expected_name = self.main_plan['name'] if GENERATION == '03' else 'nemochat baby Nemotron SFT 48h'
            if remote.id != MAIN_ID or remote.name != expected_name or remote.state != 'finished':
                raise ValueError('exact final production tracking identity/state mismatch')
            return dict(entity='nvidia', project='nemochat', run_id=remote.id, display_name=remote.name,
                        remote_state=remote.state, url=remote.url)
        except Exception:
            raise RuntimeError('approved W&B readback failed; details suppressed to protect credentials') from None

    def assert_fresh(self, name):
        if (self.root / BASE / 'runs' / name).exists():
            raise FileExistsError('refusing existing native run: ' + name)
        service = service_state('nemotron-sft-' + name + '.service')
        if service.get('LoadState') not in ('not-found', None) or int(service.get('MainPID', 0)):
            raise FileExistsError('refusing existing native supervisor: ' + name)

    def run_command(self, command, label, *, max_seconds):
        remaining = min(max_seconds, self.deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError('controller deadline exceeded')
        env = dict(os.environ, PYTHONPATH='src')
        with (self.root / OUT / (label + '.log')).open('x') as handle:
            subprocess.run(command, cwd=self.root, env=env, stdout=handle, stderr=subprocess.STDOUT,
                           timeout=remaining, check=True)

    def accept_native(self, name, plan):
        launcher = self.load_module(LAUNCHER, 'pipeline_native_launcher')
        output = self.root / BASE / 'runs' / name
        verification = launcher.verify_run(output, plan, self.manifest)
        if (plan['source_checkpoint'] != 'results/pretraining/nemochat_baby/sft/runs/sft-main-01/checkpoint/iter_0000391'
                or not verification.get('final_validation_accepted')):
            raise ValueError('native SFT391/final-heldout contract drift')
        if read_json(output / 'runner-provenance.json') != launcher.runner_provenance(
                self.root / (CAL_CONFIG if name == CAL_NAME else MAIN_CONFIG), self.root / MANIFEST):
            raise ValueError('native producer config/manifest/script hash mismatch')
        accepted_sdk_guard = read_json(self.root / SDK_GUARD)
        if read_json(output / 'rank-0-source-guard.json') != accepted_sdk_guard:
            raise ValueError('native pinned SDK decorated callable source guard mismatch')
        for rank in range(8):
            load = read_json(output / f'rank-{rank}-load.json')
            if (load.get('optimizer_scheduler_rng_cursor_reset') is not True
                    or load.get('checkpoint_integrity') != 'not_checked_explicit_user_waiver'
                    or load.get('checkpoint_bytes_read') is not False):
                raise ValueError('native full reset / checkpoint waiver mismatch')
        engine = self.load_module('results/pretrain_followup/engine_helper.py', 'pipeline_engine')
        cid = read_json(output / 'container-id.json')['id']
        live = engine.request('GET', '/containers/' + cid + '/json')
        labels = live['Config']['Labels']
        saved = read_json(output / 'container-state.json')
        if (labels.get('nemochat.validation.name') != name
                or labels.get('nemochat.validation.session') != 'nemochat-baby-nemotron-sft'
                or saved['id'] != cid or live['Id'] != cid or live['State']['Running']
                or live['State']['Pid'] or live['State']['ExitCode'] != 0 or live['State'].get('OOMKilled')
                or saved.get('watchdog_exceeded') is not False):
            raise ValueError('fresh exact-owned Engine exit gate failed')
        return dict(verification=verification, engine_id=cid, fresh_engine_state=live['State'])

    def stable_hash(self, path):
        # Only datasets, configurations, scripts and receipts; checkpoint probes are forbidden.
        path = Path(path)
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root.resolve()):
            raise ValueError('repository-local data/config/receipt hash only')
        if any(part in ('checkpoint', 'checkpoints', 'iter_0000391') for part in resolved.parts):
            raise ValueError('checkpoint byte forensics forbidden by explicit waiver')
        before = path.stat()
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
                if time.monotonic() >= self.deadline:
                    raise TimeoutError('controller deadline exceeded')
                digest.update(block)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('dataset/receipt changed during hash verification')
        return digest.hexdigest()

    def load_module(self, path, name):
        spec = importlib.util.spec_from_file_location(name, self.root / path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
        return result


def cleanup_owned(*, root=ROOT):
    root = Path(root)
    path = root / OUT / 'controller-state.json'
    if not path.exists():
        return dict(status='no_owned_controller_state')
    state = read_json(path)
    if state.get('owner_uid') != os.getuid() or state.get('unit') != UNIT:
        raise ValueError('refusing cleanup of unowned controller state')
    if state.get('status') == 'completed':
        return dict(status='completed_no_cleanup_required')
    owned = state.get('owned_runs', [])
    if not isinstance(owned, list) or len(set(owned)) != len(owned) or set(owned) - {CAL_NAME, MAIN_NAME}:
        raise ValueError('refusing unrecognized cleanup scope')
    controller = Controller(root=root)
    launcher = controller.load_module(LAUNCHER, 'pipeline_cleanup_launcher') if owned else None
    receipts = {}
    for name in reversed(owned):
        if name == MAIN_NAME:
            # Stop only the exact child service. Its own ExecStopPost verifies exact Engine labels.
            service = service_state('nemotron-sft-' + MAIN_NAME + '.service')
            if int(service.get('MainPID', 0)):
                process_identity(int(service['MainPID']), str(root / LAUNCHER))
                process_identity(int(service['MainPID']), MAIN_NAME)
                subprocess.run(['systemctl', '--user', 'stop', 'nemotron-sft-' + MAIN_NAME + '.service'],
                               check=True, timeout=150)
        receipts[name] = launcher.cleanup_owned(name)
    result = dict(status='failure_or_cancel_cleanup', owned=receipts)
    with (root / OUT / ('cleanup-' + str(time.time_ns()) + '.json')).open('x') as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', default='01', help='isolated recovery namespace: exactly 01 through 99 (default: 01)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-reviewed-inputs', action='store_true', help='inert strict parent-frozen startup checksum verification')
    mode.add_argument('--preview', action='store_true', help='inert default: no GPU work or credential access')
    mode.add_argument('--execute', action='store_true', help='parent-reviewed authorization of contingent future gates')
    mode.add_argument('--cleanup-owned', action='store_true', help='supervisor exact-owned failure/cancel cleanup only')
    args = parser.parse_args()
    try:
        configure_generation(args.generation)
    except ValueError as exc:
        parser.error(str(exc))
    os.chdir(ROOT)
    if args.verify_reviewed_inputs:
        print(json.dumps(verify_reviewed_start()))
    elif args.cleanup_owned:
        print(json.dumps(cleanup_owned()))
    elif args.execute:
        Controller().run()
    else:
        print(json.dumps(preview(), indent=2))


if __name__ == '__main__':
    main()
