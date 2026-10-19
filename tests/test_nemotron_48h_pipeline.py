"""CPU-only controls: handwritten inputs are NOT native acceptance evidence."""
import copy
import importlib.util
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/posttraining/nemotron_48h_pipeline.py'


def module():
    assert SCRIPT.exists(), 'parent-owned pipeline not implemented'
    spec = importlib.util.spec_from_file_location('pipeline_control', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generation02_namespaces_are_explicit_and_default01_is_preserved():
    p = module()
    assert hasattr(p, 'configure_generation'), 'explicit generation configuration missing'
    p.configure_generation('02')
    assert p.GENERATION == '02'
    assert p.CPU == p.BASE / 'cpu-full-02'
    assert p.OUT == p.BASE / 'longrun_protocol/pipeline-02'
    assert str(p.MANIFEST) == 'datasets/nemotron_sft/nemochat_baby/prepared-full-02/data/manifest.json'
    assert p.READY == p.CPU / 'completion-ready.json'
    assert p.PREP_UNIT == 'nemochat-nemotron-full-prepare-02.service'
    assert p.PREP_SCRIPT == str(p.ROOT / p.CPU / 'run_full_preparation.py')
    assert p.UNIT == 'nemochat-nemotron-48h-pipeline-02.service'
    assert p.CAL_NAME == 'nemotron-full-gb128-calibration-02'
    assert p.MAIN_NAME == 'nemotron-main-48h-02'
    assert p.MAIN_ID == 'nemochat-baby-nemotron-sft-main-48h-02'
    assert p.CAL_CONFIG == p.OUT / 'calibration-full-02.yaml'
    assert p.MAIN_CONFIG == p.OUT / 'production-48h-02.yaml'
    assert p.RAW == Path('datasets/nemotron_sft/nemochat_baby/raw-full-01')
    assert p.SDK_GUARD == p.BASE / 'runs/nemotron-gb128-calibration-02/rank-0-source-guard.json'
    default = module()  # Never share mutated module globals between controls.
    assert default.GENERATION == '01'
    assert default.OUT == default.BASE / 'longrun_protocol/pipeline'
    assert default.CAL_NAME.endswith('-01') and default.MAIN_ID.endswith('-01')


@pytest.mark.parametrize('token', ['00', '0', '2', '002', '-1', '../02', '02/', ' 02', '02\n', '０２', 2, None])
def test_generation_rejects_unsafe_tokens_without_mutating_namespace(token):
    p = module()
    assert hasattr(p, 'configure_generation'), 'generation validation missing'
    before = (p.CPU, p.OUT, p.MANIFEST, p.UNIT)
    with pytest.raises(ValueError, match='generation'):
        p.configure_generation(token)
    assert (p.CPU, p.OUT, p.MANIFEST, p.UNIT) == before


def test_generation02_cli_preview_forwards_supervisor_and_cleanup_generation():
    result = subprocess.run([sys.executable, str(SCRIPT), '--generation', '02', '--preview'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    preview = json.loads(result.stdout)
    assert preview['generation'] == '02' and preview['gpu_launches'] == 0
    assert preview['production_started'] is False
    assert 'cpu-full-02/completion-ready.json' in preview['required_atomic_ready']
    assert 'prepared-full-02/data/manifest.json' in preview['manifest']
    assert 'pipeline-02/controller-state.json' in preview['execution_state']
    assert preview['preparation_unit'] == 'nemochat-nemotron-full-prepare-02.service'
    assert preview['calibration_name'] == 'nemotron-full-gb128-calibration-02'
    assert preview['main_name'] == 'nemotron-main-48h-02'
    assert preview['main_run_id'] == 'nemochat-baby-nemotron-sft-main-48h-02'
    cmd = preview['supervisor_command']
    assert '--unit=nemochat-nemotron-48h-pipeline-02.service' in cmd
    assert '--setenv=NEMOTRON_48H_PIPELINE_SUPERVISED=02' in cmd
    assert cmd[-3:] == ['--generation', '02', '--execute']
    stop = shlex.split(next(x.split('=', 2)[2] for x in cmd if x.startswith('--property=ExecStopPost=')))
    assert stop[-3:] == ['--generation', '02', '--cleanup-owned']
    assert 'prepared-full-02/data/manifest.json' in ' '.join(preview['calibration_command'])
    assert 'pipeline-02/production-48h-02.yaml' in ' '.join(preview['main_command'])


def test_generation02_requires_matching_environment_and_new_unit_mainpid(tmp_path, monkeypatch):
    p = module()
    p.configure_generation('02')
    c = p.Controller(root=tmp_path)
    monkeypatch.setenv('NEMOTRON_48H_PIPELINE_SUPERVISED', '01')
    with pytest.raises(ValueError, match='exact parent-owned supervisor'):
        c.run()
    monkeypatch.setenv('NEMOTRON_48H_PIPELINE_SUPERVISED', '02')
    units = []
    def wrong_mainpid(unit):
        units.append(unit)
        return {'MainPID': '0', 'LoadState': 'not-found'}
    monkeypatch.setattr(p, 'service_state', wrong_mainpid)
    with pytest.raises(ValueError, match='MainPID'):
        c.run()
    assert units == ['nemochat-nemotron-48h-pipeline-02.service']
    assert not (tmp_path / p.OUT).exists()


def test_generation02_cleanup_never_reads_old_state_or_accepts_old_owned_runs(tmp_path):
    p = module()
    old = tmp_path / p.OUT / 'controller-state.json'
    old.parent.mkdir(parents=True)
    original = json.dumps(dict(owner_uid=os.getuid(), unit=p.UNIT, status='failed', owned_runs=[p.MAIN_NAME]))
    old.write_text(original)
    p.configure_generation('02')
    assert p.cleanup_owned(root=tmp_path) == {'status': 'no_owned_controller_state'}
    state = tmp_path / p.OUT / 'controller-state.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps(dict(owner_uid=os.getuid(), unit=p.UNIT, status='failed',
                                    owned_runs=['nemotron-main-48h-01'])))
    with pytest.raises(ValueError, match='scope'):
        p.cleanup_owned(root=tmp_path)
    assert old.read_text() == original


def test_generation02_rejects_old_proofs_accepts_only_new_targets_and_origin():
    p = module()
    old_handoff = json.loads((ROOT / p.CPU / 'FULL_HANDOFF.json').read_text())
    sources = [dict(s, full_file_digest_verified=True, retained_bytes=s['expected_bytes']) for s in old_handoff['sources']]
    manifest = dict(status='passed', train_dialogues=2000000, heldout_dialogues=1000,
                    train_loss_tokens=5000000, heldout_loss_tokens=10000, prior_reserved_prompt_groups=14,
                    train_heldout_overlap=0, kmmlu_exact_question_matches=0, kmmlu_reserved_questions=34871,
                    training_repetition=False, tokenizer_revision='fc067b34f6b514014d2f8bddba855d2e4bfdfba6', sources=sources)
    state = dict(status='passed_CPU_full_preparation_and_counted_sample_audit', manifest=old_handoff['full_manifest_target'])
    engine = dict(Running=False, Pid=0, ExitCode=0, OOMKilled=False)
    ready = dict(status='passed', manifest=old_handoff['full_manifest_target'], manifest_sha256='Handwritten-sha',
                 audit_sha256='Handwritten-audit', source_content_bytes=100853847138, engine_state=engine)
    acquisition = dict(status='passed', completed_files=7, completed_expected_bytes=100853847138, completed_sources=sources)
    audit = dict(status='passed', all_conversations_counted=2001000, all_loss_tokens_counted=5010000)
    args = [state, ready, old_handoff, acquisition, engine, manifest, audit, 'Handwritten-sha', 'Handwritten-audit']
    old_cpu = p.CPU
    p.configure_generation('02')
    with pytest.raises(ValueError, match='handoff scope'):
        p.validate_data_contract(*args)
    new = copy.deepcopy(args)
    new[0]['manifest'] = new[1]['manifest'] = new[2]['full_manifest_target'] = str(p.MANIFEST)
    new[2]['completion_ready_target'] = str(p.READY)
    p.validate_data_contract(*new)
    for index, key, value in [(0, 'manifest', str(module().MANIFEST)), (1, 'manifest', str(module().MANIFEST)),
                               (2, 'completion_ready_target', str(module().READY))]:
        bad = copy.deepcopy(new)
        bad[index][key] = value
        with pytest.raises(ValueError):
            p.validate_data_contract(*bad)
    origin = dict(Id='Handwritten-id', Config={'Cmd': ['-u', str(old_cpu / 'full_workload.py')]},
                  HostConfig={'DeviceRequests': [], 'NetworkMode': 'none', 'NanoCpus': 24000000000})
    with pytest.raises(ValueError, match='CPU Engine origin'):
        p.validate_cpu_origin({'container_id': 'Handwritten-id'}, origin)
    origin['Config']['Cmd'] = ['-u', str(p.CPU / 'full_workload.py')]
    p.validate_cpu_origin({'container_id': 'Handwritten-id'}, origin)


def test_generation02_raw_reuse_is_explicit_and_rejects_traversal(tmp_path):
    p = module()
    p.configure_generation('02')
    assert hasattr(p.Controller, 'raw_source_path'), 'immutable raw input path guard missing'
    c = p.Controller(root=tmp_path)
    assert c.raw_source_path('Handwritten-source.jsonl') == tmp_path / p.RAW / 'Handwritten-source.jsonl'
    for token in ['../raw-full-02/source.jsonl', '/etc/passwd', 'nested/file.jsonl', '', '.', '..']:
        with pytest.raises(ValueError, match='raw dataset'):
            c.raw_source_path(token)


def test_generation02_missing_readiness_records_exact_preparation_failure_without_gpu_release(tmp_path, monkeypatch):
    p = module()
    p.configure_generation('02')
    monkeypatch.setenv('NEMOTRON_48H_PIPELINE_SUPERVISED', '02')
    monkeypatch.setattr(p, 'process_identity', lambda *args: {'fixture': 'Handwritten'})
    def service(unit):
        if unit == p.UNIT:
            return {'MainPID': str(os.getpid())}
        assert unit == p.PREP_UNIT
        return dict(MainPID='0', LoadState='loaded', ActiveState='failed', Result='exit-code', ExecMainStatus='1')
    monkeypatch.setattr(p, 'service_state', service)
    c = p.Controller(root=tmp_path)
    monkeypatch.setattr(c, 'assert_fresh', lambda name: None)
    monkeypatch.setattr(c, 'stable_hash', lambda path: 'Handwritten-source-only-not-GPU-evidence')
    with pytest.raises(RuntimeError, match='contingent pipeline failed'):
        c.run()
    state = json.loads((tmp_path / p.OUT / 'controller-state.json').read_text())
    assert state['status'] == 'failed' and state['owned_runs'] == []
    assert state['evidence']['failure_type'] == 'FileNotFoundError'
    prep = state['evidence']['preparation']
    assert prep['unit'] == p.PREP_UNIT
    assert prep['proof_root'] == str(p.CPU)
    assert prep['required_atomic_ready'] == str(p.READY)
    assert prep['service']['ExecMainStatus'] == '1'
    assert state['evidence']['data_failure'] == 'missing_atomic_completion_ready'
    assert not (tmp_path / module().OUT).exists()


def test_generation02_failed_source_freeze_is_durable_and_never_starts_stages(tmp_path, monkeypatch):
    p = module()
    p.configure_generation('02')
    monkeypatch.setenv('NEMOTRON_48H_PIPELINE_SUPERVISED', '02')
    monkeypatch.setattr(p, 'service_state', lambda unit: {'MainPID': str(os.getpid())})
    monkeypatch.setattr(p, 'process_identity', lambda *args: {'fixture': 'Handwritten'})
    c = p.Controller(root=tmp_path)
    monkeypatch.setattr(c, 'assert_fresh', lambda name: None)
    with pytest.raises(RuntimeError, match='contingent pipeline failed'):
        c.run()
    state = json.loads((tmp_path / p.OUT / 'controller-state.json').read_text())
    assert state['status'] == 'failed' and state['owned_runs'] == []
    assert state['evidence']['failure_type'] == 'FileNotFoundError'


def test_generation02_parent_start_artifact_has_real_source_hashes_and_inert_verification(tmp_path):
    p = module()
    p.configure_generation('02')
    artifact = ROOT / p.OUT / 'start-parent-reviewed.sh'
    assert artifact.is_file(), 'fresh parent-reviewed start artifact missing'
    checked = subprocess.run(['bash', '-n', str(artifact)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    verified = subprocess.run(['bash', str(artifact), '--verify-only'], cwd=ROOT, capture_output=True, text=True)
    # Historical review bytes are immutable. Current 16k source changes must make
    # the cancelled generation02 launcher fail closed, never refresh its receipt.
    import hashlib
    source_list = (ROOT / p.OUT / 'reviewed-inputs.sha256').read_text()
    drift = []
    for line in source_list.splitlines():
        digest, path = line.split('  ', 1)
        target = ROOT / path
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            drift.append(path)
    if drift:
        assert verified.returncode != 0
        assert all(path + ': FAILED' in verified.stdout for path in drift)
        return  # Successful shell forwarding is exercised by generation03 CPU fixtures.
    assert verified.returncode == 0, verified.stdout + verified.stderr
    # Exercise exact shell argv through an inert CPU substitute, not systemd.
    fake = tmp_path / 'systemd-run'
    fake.write_text('#!' + sys.executable + '\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o700)
    launched = subprocess.run(['bash', str(artifact)], cwd=ROOT,
                              env=dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH']),
                              capture_output=True, text=True)
    assert launched.returncode == 0, launched.stderr
    argv = json.loads(launched.stdout.splitlines()[-1])
    assert argv == p.supervisor_payload()[1:]
    source_list = (ROOT / p.OUT / 'reviewed-inputs.sha256').read_text()
    assert 'scripts/posttraining/nemotron_48h_pipeline.py' in source_list
    assert str(p.SDK_GUARD) in source_list
    assert 'cpu-full-01' not in source_list
    assert 'checkpoint/' not in source_list


def test_real_pidfd_waits_for_owned_short_lived_process():
    p = module()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(0.2)', 'Handwritten-pidfd-control'])
    try:
        identity = p.process_identity(child.pid, 'Handwritten-pidfd-control')
        result = p.wait_pidfd(identity, deadline=time.monotonic() + 5)
        assert result['event'] == 'pidfd_exit'
        assert result['start_ticks'] == identity['start_ticks']
        assert child.wait(timeout=2) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.parametrize('rows,epochs', [(2000000, 1), (900000, 2), (600000, 3)])
def test_handwritten_horizon_uses_minimum_epochs_without_shortening(rows, epochs):
    p = module()
    assert hasattr(p, 'derive_horizon'), 'measured horizon planner missing'
    plan = p.derive_horizon(rows, 12.352091671, overhead=0.10)
    assert plan['epochs'] == epochs
    assert plan['allow_repetition'] is (epochs > 1)
    assert plan['max_updates'] * 12.352091671 / 0.9 >= 48 * 3600
    assert plan['available_updates'] >= plan['max_updates']
    assert plan['additional_epochs_above_one'] == epochs - 1
    with pytest.raises(ValueError, match='capacity'):
        p.derive_horizon(82557, 12.352091671, overhead=0.10)


def test_handwritten_atomic_readiness_contract_rejects_stale_or_incomplete_evidence():
    p = module()
    assert hasattr(p, 'validate_data_contract'), 'atomic full-data gate missing'
    handoff = json.loads((ROOT / 'results/pretraining/nemochat_baby/nemotron_sft/cpu-full-01/FULL_HANDOFF.json').read_text())
    sources = [dict(s, full_file_digest_verified=True, retained_bytes=s['expected_bytes']) for s in handoff['sources']]
    m = dict(status='passed', train_dialogues=2000000, heldout_dialogues=1000,
             train_loss_tokens=5000000, heldout_loss_tokens=10000, prior_reserved_prompt_groups=14,
             train_heldout_overlap=0, kmmlu_exact_question_matches=0, kmmlu_reserved_questions=34871,
             training_repetition=False, tokenizer_revision='fc067b34f6b514014d2f8bddba855d2e4bfdfba6',
             sources=sources)
    state = dict(status='passed_CPU_full_preparation_and_counted_sample_audit', manifest=handoff['full_manifest_target'])
    ready = dict(status='passed', manifest=handoff['full_manifest_target'], manifest_sha256='Handwritten-sha',
                 audit_sha256='Handwritten-audit', source_content_bytes=100853847138)
    engine = dict(Running=False, Pid=0, ExitCode=0, OOMKilled=False)
    ready['engine_state'] = engine
    acq = dict(status='passed', completed_files=7, completed_expected_bytes=100853847138, completed_sources=sources)
    audit = dict(status='passed', all_conversations_counted=2001000, all_loss_tokens_counted=5010000)
    args = [state, ready, handoff, acq, engine, m, audit, 'Handwritten-sha', 'Handwritten-audit']
    p.validate_data_contract(*args)
    for index, key, value in [(1, 'manifest_sha256', 'old'), (0, 'status', 'waiting_full_acquisition'),
                               (4, 'ExitCode', 1), (5, 'prior_reserved_prompt_groups', 13),
                               (6, 'all_conversations_counted', 100), (3, 'completed_files', 6),
                               (5, 'train_heldout_overlap', 1)]:
        bad = copy.deepcopy(args)
        bad[index][key] = value
        with pytest.raises(ValueError):
            p.validate_data_contract(*bad)


def test_default_cli_preview_is_inert_and_emits_fresh_future_supervisor():
    target = ROOT / module().OUT / 'controller-state.json'
    before = target.read_bytes() if target.exists() else None
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().startswith('{'), 'inert CLI preview not implemented'
    preview = json.loads(result.stdout)
    assert preview['status'] in {'WAITING_DATA', 'DATA_PROOF_AVAILABLE_NOT_ACCEPTED'}
    assert preview['gpu_launches'] == 0
    assert preview['required_atomic_ready'].endswith('completion-ready.json')
    cmd = preview['supervisor_command']
    assert 'systemd-run' == cmd[0]
    assert '--property=RuntimeMaxSec=345600' in cmd
    assert '--execute' in cmd
    assert '--setenv=PYTHONPATH=src' in cmd
    assert preview['calibration_name'] == 'nemotron-full-gb128-calibration-01'
    assert preview['main_run_id'] == 'nemochat-baby-nemotron-sft-main-48h-01'
    assert preview['stage_order'] == ['atomic_full_data', 'full_native_GB128_calibration', 'freeze_48h', 'main_supervisor', 'native_final_acceptance']
    assert preview['tracking_destination'] == 'nvidia/nemochat'
    assert preview['calibration_command'][-1] == '--execute'
    assert '--supervise' in preview['main_command']
    if preview['preparation_identity'] is not None:
        assert preview['preparation_identity']['uid'] == os.getuid()
    assert ROOT / preview['execution_state'] == target
    assert (target.read_bytes() if target.exists() else None) == before


@pytest.mark.parametrize('blocked', [None, 'data', 'calibration', 'freeze', 'launch', 'finish'])
def test_handwritten_contingency_releases_only_ordered_accepted_stages(blocked):
    p = module()
    assert hasattr(p, 'contingent_stages'), 'ordered contingent release missing'
    calls, phases = [], []

    def step(name):
        def handwritten_control():
            calls.append(name)
            if name == blocked:
                raise ValueError('Handwritten blocked gate')
            return {'fixture': 'Handwritten; not acceptance evidence'}
        return handwritten_control

    actions = {name: step(name) for name in ('data', 'calibration', 'freeze', 'launch', 'finish')}
    if blocked:
        with pytest.raises(ValueError, match='Handwritten'):
            p.contingent_stages(actions, phases.append)
        assert calls == list(actions)[:list(actions).index(blocked) + 1]
        assert 'completed' not in phases
    else:
        p.contingent_stages(actions, phases.append)
        assert calls == list(actions)
        assert phases[-1] == 'completed'
        assert phases.index('main_supervisor_started') < phases.index('completed')


def test_handwritten_manifest_alone_never_passes_atomic_data_gate(tmp_path, monkeypatch):
    p = module()
    assert hasattr(p, 'Controller'), 'production gate controller missing'
    monkeypatch.setattr(p, 'service_state', lambda unit: {'MainPID': '0', 'LoadState': 'not-found'})
    manifest = tmp_path / p.MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"status":"passed","fixture":"Handwritten"}')
    control = p.Controller(root=tmp_path)
    with pytest.raises((ValueError, FileNotFoundError), match='completion-ready'):
        control.data_gate()
    assert not (tmp_path / p.OUT / 'controller-state.json').exists()


def test_calibration_spec_is_fresh_full_specific_and_data_gate_is_mandatory(tmp_path):
    p = module()
    assert hasattr(p, 'calibration_spec'), 'full native calibration specification missing'
    config, cmd = p.calibration_spec()
    assert config['source_checkpoint'].endswith('sft-main-01/checkpoint/iter_0000391')
    assert config['max_updates'] == 20 and config['max_hours'] == 0.25
    assert config['global_batch_size'] == 128
    assert config['require_final_validation'] and config['validation_max_dialogues'] <= 1000
    assert '--execute' in cmd and '--supervise' not in cmd
    assert cmd[cmd.index('--manifest') + 1] == str(p.MANIFEST)
    assert cmd[cmd.index('--name') + 1] == 'nemotron-full-gb128-calibration-01'
    assert hasattr(p.Controller, 'calibrate'), 'native calibration execution missing'
    with pytest.raises(ValueError, match='full data'):
        p.Controller(root=tmp_path).calibrate()


def test_handwritten_production_spec_freezes_measured_unique_first_config_without_writes(tmp_path):
    p = module()
    assert hasattr(p, 'production_spec'), '48h measured production config missing'
    m = json.loads((ROOT / 'datasets/nemotron_sft/nemochat_baby/prepared-main-01/data/manifest.json').read_text())
    m['train_dialogues'] = 2000000
    m['heldout_dialogues'] = 1000
    config, plan = p.production_spec(m, {'mean_iteration_seconds': 12.352091671}, 'Handwritten-only.json')
    assert config['epochs'] == 1 and config['allow_repetition'] is False
    assert config['calibration_receipt'] == 'Handwritten-only.json'
    assert config['name'] == 'nemochat baby Nemotron SFT 48h'
    assert config['source_checkpoint'].endswith('sft-main-01/checkpoint/iter_0000391')
    assert plan['updates'] == config['max_updates']
    assert plan['available_updates'] >= plan['updates']
    assert config['watchdog_grace_minutes'] == 60
    assert getattr(p, 'MIN_MAIN_REMAINING_SECONDS', 0) >= 52 * 3600, 'main requires room for launcher preflight plus full watchdog'
    assert hasattr(p.Controller, 'freeze'), 'real freeze gate missing'
    with pytest.raises(ValueError, match='calibration'):
        p.Controller(root=tmp_path).freeze()
    assert not (ROOT / p.MAIN_CONFIG).exists()


def test_main_command_is_supervised_distinct_online_and_requires_frozen_gate(tmp_path):
    p = module()
    assert hasattr(p, 'main_command'), 'main native supervisor command missing'
    cmd = p.main_command()
    assert {'--execute', '--supervise', '--online'} <= set(cmd)
    assert cmd[cmd.index('--run-id') + 1] == p.MAIN_ID
    assert cmd[cmd.index('--manifest') + 1] == str(p.MANIFEST)
    assert 'prepared-main-01' not in ' '.join(cmd)
    assert hasattr(p.Controller, 'launch_main'), 'gated main launch missing'
    with pytest.raises(ValueError, match='frozen'):
        p.Controller(root=tmp_path).launch_main()
    assert hasattr(p.Controller, 'finish_main'), 'main lifecycle reconciliation missing'
    with pytest.raises(ValueError, match='supervisor'):
        p.Controller(root=tmp_path).finish_main()


def test_execute_requires_exact_parent_supervisor_before_any_writes():
    target = ROOT / module().OUT / 'controller-state.json'
    before = target.read_bytes() if target.exists() else None
    result = subprocess.run([sys.executable, str(SCRIPT), '--execute'], cwd=ROOT,
                            env={k: v for k, v in os.environ.items() if k != 'NEMOTRON_48H_PIPELINE_SUPERVISED'},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'exact parent-owned supervisor' in result.stderr, 'execution authorization guard missing'
    assert (target.read_bytes() if target.exists() else None) == before


def test_existing_controller_state_refuses_reexecution_before_any_launch(tmp_path, monkeypatch):
    p = module()
    assert hasattr(p.Controller, 'run'), 'durable controller execution missing'
    target = tmp_path / p.OUT / 'controller-state.json'
    target.parent.mkdir(parents=True)
    target.write_text('{"status":"completed","fixture":"Handwritten"}')
    monkeypatch.setenv('NEMOTRON_48H_PIPELINE_SUPERVISED', '01')
    with pytest.raises(FileExistsError, match='existing controller'):
        p.Controller(root=tmp_path).run()
    assert json.loads(target.read_text())['status'] == 'completed'


def test_cleanup_without_owned_state_is_inert(tmp_path):
    p = module()
    assert hasattr(p, 'cleanup_owned'), 'controller exact-owned failure cleanup missing'
    assert p.cleanup_owned(root=tmp_path) == {'status': 'no_owned_controller_state'}


def test_real_pidfd_exit_race_accepts_same_zombie_identity():
    p = module()
    child = subprocess.Popen([sys.executable, '-c', "import sys; print('READY', flush=True); sys.stdin.readline()",
                              'Handwritten-exit-race'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'READY'
        identity = p.process_identity(child.pid, 'Handwritten-exit-race')
        child.stdin.write('EXIT\n')
        child.stdin.flush()
        # Wait for exit without reaping: deterministically exercise a real zombie.
        import os
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        assert p.wait_pidfd(identity, deadline=time.monotonic() + 2)['event'] == 'pidfd_exit'
        assert child.wait() == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_real_pidfd_rejects_stale_starttime_and_times_out_without_killing():
    p = module()
    child = subprocess.Popen([sys.executable, '-c', "import time; print('READY', flush=True); time.sleep(5)",
                              'Handwritten-timeout-control'], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'READY'
        identity = p.process_identity(child.pid, 'Handwritten-timeout-control')
        with pytest.raises(ValueError, match='PID reuse'):
            p.wait_pidfd(dict(identity, start_ticks=identity['start_ticks'] + 1), deadline=time.monotonic() + 1)
        with pytest.raises(TimeoutError):
            p.wait_pidfd(identity, deadline=time.monotonic() + 0.02)
        assert child.poll() is None
        with pytest.raises(ValueError, match='argv'):
            p.process_identity(child.pid, 'not-the-owned-command')
    finally:
        child.kill()
        child.wait()


def test_handwritten_hashing_rejects_checkpoints_even_with_symlinks(tmp_path):
    p = module()
    c = p.Controller(root=tmp_path)
    data = tmp_path / 'datasets/tiny.json'
    data.parent.mkdir()
    data.write_text('Handwritten-data-only')
    assert c.stable_hash(data) == __import__('hashlib').sha256(data.read_bytes()).hexdigest()
    cp = tmp_path / 'checkpoints/forbidden'
    cp.parent.mkdir()
    cp.write_text('Handwritten-not-a-model')
    alias = tmp_path / 'datasets/alias'
    alias.symlink_to(cp)
    with pytest.raises(ValueError, match='forensics'):
        c.stable_hash(cp)
    with pytest.raises(ValueError, match='forensics'):
        c.stable_hash(alias)
    with pytest.raises(ValueError, match='repository-local'):
        c.stable_hash('/proc/self/cmdline')


def test_handwritten_owned_cpu_command_is_exercised_and_logs_are_never_overwritten(tmp_path):
    p = module()
    c = p.Controller(root=tmp_path)
    (tmp_path / p.OUT).mkdir(parents=True)
    c.run_command([sys.executable, '-c', 'print("Handwritten CPU command; no GPU")'], 'finite-control', max_seconds=5)
    target = tmp_path / p.OUT / 'finite-control.log'
    assert target.read_text().strip() == 'Handwritten CPU command; no GPU'
    with pytest.raises(FileExistsError):
        c.run_command([sys.executable, '-c', 'raise SystemExit(1)'], 'finite-control', max_seconds=5)


def test_handwritten_freshness_refuses_existing_run_and_scope_cleanup_is_narrow(tmp_path, monkeypatch):
    p = module()
    c = p.Controller(root=tmp_path)
    monkeypatch.setattr(p, 'service_state', lambda unit: {'MainPID': '0', 'LoadState': 'not-found'})
    c.assert_fresh(p.MAIN_NAME)
    run = tmp_path / p.BASE / 'runs' / p.MAIN_NAME
    run.mkdir(parents=True)
    with pytest.raises(FileExistsError, match='existing native run'):
        c.assert_fresh(p.MAIN_NAME)
    state = tmp_path / p.OUT / 'controller-state.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps(dict(owner_uid=os.getuid(), unit=p.UNIT, owned_runs=['unrelated-run'])))
    with pytest.raises(ValueError, match='scope'):
        p.cleanup_owned(root=tmp_path)
    state.write_text(json.dumps(dict(owner_uid=os.getuid(), unit=p.UNIT, status='completed')))
    assert p.cleanup_owned(root=tmp_path)['status'] == 'completed_no_cleanup_required'


def test_handwritten_full_cpu_engine_origin_requires_exact_owned_no_gpu_workload():
    p = module()
    assert hasattr(p, 'validate_cpu_origin'), 'exact CPU origin binding missing'
    state = {'container_id': 'Handwritten-container-id'}
    engine = dict(Id='Handwritten-container-id',
                  Config={'Cmd': ['-u', str(p.CPU / 'full_workload.py')]},
                  HostConfig={'DeviceRequests': [], 'NetworkMode': 'none', 'NanoCpus': 24000000000})
    p.validate_cpu_origin(state, engine)
    for mutation in ('wrong-id', 'gpu', 'different-workload'):
        bad = copy.deepcopy(engine)
        if mutation == 'wrong-id':
            bad['Id'] = 'Handwritten-other-container'
        elif mutation == 'gpu':
            bad['HostConfig']['DeviceRequests'] = [{'Driver': 'nvidia'}]
        else:
            bad['Config']['Cmd'] = ['unrelated.py']
        with pytest.raises(ValueError, match='CPU Engine origin'):
            p.validate_cpu_origin(state, bad)
