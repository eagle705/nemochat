"""CPU contract controls only; never evidence of native 16k acceptance."""
import copy
import json
from pathlib import Path

import pytest
from test_nemotron_48h_pipeline import module, ROOT

LONG = 'nvidia/Nemotron-SFT-Long-Context-Transduction-v1'


def test_generation03_execution_contract_is_explicit_not_legacy():
    p = module()
    assert hasattr(p, 'execution_contract'), 'generation-specific execution contract missing'
    c = p.execution_contract('03')
    assert c['sequence_length'] == 16384
    assert c['excluded_datasets'] == [LONG]
    assert c['production_template'] == Path('configs/posttraining/nemochat_baby_nemotron_sft_16k_48h.yaml')
    assert c['calibration_template'] == p.BASE / 'seq16k_protocol/calibration.yaml'
    assert c['calibration_max_hours'] == 1
    assert p.execution_contract('02')['sequence_length'] == 4096
    assert p.execution_contract('01')['excluded_datasets'] == []
    p.configure_generation('03')
    paths = p.reviewed_input_paths()
    assert c['production_template'] in paths and c['calibration_template'] in paths
    assert Path('configs/posttraining/nemochat_baby_nemotron_sft_48h.yaml') not in paths
    assert p.SDK_GUARD in paths
    assert p.BASE / 'cpu-full-01/acquisition-completed.json' in paths
    assert Path('datasets/nemotron_sft/nemochat_baby/prepared-pilot-01/data/heldout-dialogues.jsonl') in paths
    assert p.OUT == p.BASE / 'longrun_protocol/pipeline-03'
    assert p.CAL_NAME == 'nemotron-full-gb128-calibration-03'
    assert p.UNIT == 'nemochat-nemotron-48h-pipeline-03.service'


def data_fixture(p):
    handoff = json.loads((ROOT / p.BASE / 'cpu-full-01/FULL_HANDOFF.json').read_text())
    handoff.update(full_manifest_target=str(p.MANIFEST), completion_ready_target=str(p.READY))
    sources = [dict(s, full_file_digest_verified=True, retained_bytes=s['expected_bytes']) for s in handoff['sources']]
    active = sorted(s['key'] for s in sources if s['dataset'] != LONG)
    policy = dict(sequence_length=16384, excluded_datasets=[LONG], active_training_source_keys=active)
    handoff['source_contract'] = dict(policy, source_content_bytes=100853847138,
        active_training_source_bytes=99631965701,
        reused_acquisition_receipt=str(p.BASE / 'cpu-full-01/acquisition-completed.json'))
    del handoff['sources'], handoff['expected_content_bytes']
    manifest = dict(status='passed', train_dialogues=2000000, heldout_dialogues=1000,
        train_loss_tokens=5000000, heldout_loss_tokens=10000, prior_reserved_prompt_groups=14,
        reservation_presence=dict(physically_present=13, not_present_in_selected_corpus=1, inactive_due_excluded_dataset=1), train_heldout_overlap=0, kmmlu_exact_question_matches=0,
        kmmlu_reserved_questions=34871, training_repetition=False,
        prior_heldout_reservation_file='datasets/nemotron_sft/nemochat_baby/prepared-pilot-01/data/heldout-dialogues.jsonl',
        tokenizer_revision='fc067b34f6b514014d2f8bddba855d2e4bfdfba6', sources=sources, **policy)
    manifest['per_source'] = {s['key']: dict(train=1, heldout=1, supervised_tokens=10) if s['dataset'] != LONG
        else dict(train=0, heldout=0, supervised_tokens=0, excluded=True, exclusion_reason='deferred_long_context_stage') for s in sources}
    manifest['array_shapes'] = {split + '_' + kind: [manifest[split + '_dialogues'], width]
        for split in ('train', 'heldout') for kind, width in [('tokens', 16385), ('masks', 16384)]}
    state = dict(status='passed_CPU_full_preparation_and_audit', unit=p.PREP_UNIT,
        output=str(p.MANIFEST.parent), reused_acquisition_receipt=str(p.BASE / 'cpu-full-01/acquisition-completed.json'))
    engine = dict(Running=False, Pid=0, ExitCode=0, OOMKilled=False)
    ready = dict(status='passed', manifest=str(p.MANIFEST), manifest_sha256='Handwritten-sha',
        audit_sha256='Handwritten-audit', source_content_bytes=100853847138, engine_state=engine, **policy)
    acquisition = dict(status='passed', completed_files=7, completed_expected_bytes=100853847138, completed_sources=sources)
    ready.update(array_shapes=manifest['array_shapes'], reservation_presence=manifest['reservation_presence'])
    audit = dict(status='passed', all_conversations_counted=2001000, all_loss_tokens_counted=5010000)
    return [state, ready, handoff, acquisition, engine, manifest, audit, 'Handwritten-sha', 'Handwritten-audit']


def test_generation03_rejects_long_training_despite_valid_seven_file_archive():
    p = module()
    p.configure_generation('03')
    args = data_fixture(p)
    p.validate_data_contract(*args)
    for index, key, value in [(5, 'sequence_length', 4096), (5, 'excluded_datasets', []),
        (1, 'excluded_datasets', []), (2, 'source_contract', dict(args[2]['source_contract'], sequence_length=4096)),
        (5, 'reservation_presence', dict(physically_present=14, not_present_in_selected_corpus=1, inactive_due_excluded_dataset=1)),
        (1, 'array_shapes', {}), (5, 'array_shapes', {}), (5, 'prior_heldout_reservation_file', None),
        (5, 'active_training_source_keys', [])]:
        bad = copy.deepcopy(args)
        bad[index][key] = value
        with pytest.raises(ValueError):
            p.validate_data_contract(*bad)
    long_key = next(s['key'] for s in args[5]['sources'] if s['dataset'] == LONG)
    for field, value in [('train', 1), ('heldout', 1), ('supervised_tokens', 1), ('exclusion_reason', ''), ('excluded', False)]:
        bad = copy.deepcopy(args)
        bad[5]['per_source'][long_key][field] = value
        with pytest.raises(ValueError, match='excluded'):
            p.validate_data_contract(*bad)


def test_generation03_reads_only_16k_templates_and_preserves_display_name(tmp_path, monkeypatch):
    import yaml
    p = module()
    p.configure_generation('03')
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    c = p.execution_contract('03')
    target = tmp_path / c['calibration_template']
    target.parent.mkdir(parents=True)
    target.write_text(yaml.safe_dump(dict(name='actual 16k calibration', sequence_length=16384)))
    config, command = p.calibration_spec()
    assert config['sequence_length'] == 16384
    assert config['name'] == 'actual 16k calibration'
    assert command[command.index('--manifest') + 1] == str(p.MANIFEST)
    target.unlink()
    legacy = tmp_path / p.execution_contract('02')['calibration_template']
    legacy.parent.mkdir(parents=True)
    legacy.write_text(yaml.safe_dump(dict(sequence_length=4096)))
    with pytest.raises(FileNotFoundError):
        p.calibration_spec()


def test_generation03_calibration_shape_time_and_future_main_reservation():
    p = module()
    p.configure_generation('03')
    assert hasattr(p, 'validate_execution_shape'), 'strict 16k native shape gate missing'
    assert hasattr(p, 'calibration_budget'), 'bounded 16k calibration budget missing'
    config = dict(sequence_length=16384, excluded_datasets=[LONG], global_batch_size=128,
        context_parallel_size=1, expert_model_parallel_size=8, max_hours=.75,
        max_updates=20, watchdog_grace_minutes=30, execution_mode='calibration',
        require_final_validation=True, validation_max_dialogues=256)
    p.validate_execution_shape(config)
    p.calibration_budget(dict(config, max_updates=15), dict(config, updates=15))
    seconds = p.calibration_budget(config, dict(config, updates=20))
    assert seconds >= .75 * 3600 + 30 * 60 + 7200
    for key, value in [('max_hours', 1.01), ('max_hours', 0), ('max_updates', 19),
        ('validation_max_dialogues', 1000), ('require_final_validation', False),
        ('sequence_length', 4096), ('excluded_datasets', []), ('global_batch_size', 64),
        ('context_parallel_size', 3), ('expert_model_parallel_size', 4)]:
        bad = dict(config, **{key: value})
        with pytest.raises(ValueError):
            p.calibration_budget(bad, dict(bad, updates=20))
    # CP2 is merely a compatible static shape, not native topology acceptance.
    p.validate_execution_shape(dict(config, context_parallel_size=2, expert_model_parallel_size=4))
    with pytest.raises(ValueError):
        p.calibration_budget(config, dict(config, updates=19))
    assert hasattr(p.Controller, 'require_remaining'), 'stage reservation gate missing'
    controller = p.Controller()
    import time
    controller.deadline = time.monotonic() + p.MIN_MAIN_REMAINING_SECONDS + seconds - 10
    with pytest.raises(ValueError, match='remaining'):
        controller.require_remaining(p.MIN_MAIN_REMAINING_SECONDS + seconds)


def test_generation03_calibrate_uses_new_budget_before_any_run_write(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    p = module()
    p.configure_generation('03')
    config = dict(sequence_length=16384, excluded_datasets=[LONG], global_batch_size=128,
        context_parallel_size=1, expert_model_parallel_size=8, max_hours=.75,
        max_updates=20, watchdog_grace_minutes=30, execution_mode='calibration',
        require_final_validation=True, validation_max_dialogues=256)
    monkeypatch.setattr(p, 'calibration_spec', lambda: (config, ['Handwritten-no-GPU']))
    c = p.Controller(root=tmp_path)
    c.evidence['data'] = {'fixture': 'Handwritten'}
    c.manifest_sha = 'Handwritten-sha'
    c.manifest = {}
    c.owned_runs = []
    monkeypatch.setattr(c, 'stable_hash', lambda path: 'Handwritten-sha')
    monkeypatch.setattr(c, 'verify_frozen_inputs', lambda: None)
    monkeypatch.setattr(c, 'load_module', lambda *a: SimpleNamespace(stage_plan=lambda *a: dict(config, updates=20)))
    monkeypatch.setattr(c, 'assert_fresh', lambda *a: None)
    monkeypatch.setattr(c, 'phase', lambda *a: None)
    c.deadline = time.monotonic() + p.MIN_MAIN_REMAINING_SECONDS + 100
    with pytest.raises(ValueError, match='remaining'):
        c.calibrate()
    assert not (tmp_path / p.CAL_CONFIG).exists()
    c.deadline = time.monotonic() + 96 * 3600
    (tmp_path / p.OUT).mkdir(parents=True)
    invoked = []
    def stop_before_native_accept(command, label, *, max_seconds):
        invoked.append((command, max_seconds))
        raise RuntimeError('Handwritten stop; no actual GPU invocation')
    monkeypatch.setattr(c, 'run_command', stop_before_native_accept)
    with pytest.raises(RuntimeError, match='Handwritten stop'):
        c.calibrate()
    assert invoked == [(['Handwritten-no-GPU'], p.calibration_budget(config, dict(config, updates=20)))]


def test_generation03_production_binds_measured_16k_topology_not_4k(tmp_path, monkeypatch):
    import yaml
    from types import SimpleNamespace
    p = module()
    p.configure_generation('03')
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    target = tmp_path / p.execution_contract('03')['production_template']
    target.parent.mkdir(parents=True)
    config = dict(name='native chosen 16k display', sequence_length=16384, excluded_datasets=[LONG],
        global_batch_size=128, context_parallel_size=1, expert_model_parallel_size=8)
    target.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(p.Controller, 'load_module', lambda *a: SimpleNamespace(stage_plan=lambda cfg, m: dict(cfg)))
    receipt = dict(config, status='passed', manifest_sha256='Handwritten-full-sha', mean_iteration_seconds=50)
    manifest = dict(train_dialogues=2000000, sequence_length=16384, excluded_datasets=[LONG])
    frozen, plan = p.production_spec(manifest, receipt, 'Handwritten-receipt.json')
    assert frozen['name'] == config['name']
    assert frozen['max_hours'] == 48 and frozen['sequence_length'] == 16384
    assert frozen['max_updates'] * 50 / .9 >= 48 * 3600
    # Existing native receipt schema omits shape; bind the already accepted
    # producer plan (whose exact config/manifest hashes accept_native verified).
    plain = dict(status='passed', manifest_sha256='Handwritten-full-sha', mean_iteration_seconds=50)
    p.production_spec(manifest, plain, 'Handwritten-receipt.json', calibration_plan=config)
    with pytest.raises(ValueError, match='calibration'):
        p.production_spec(manifest, plain, 'Handwritten-receipt.json', calibration_plan=dict(config, sequence_length=4096))
    for key, value in [('sequence_length', 4096), ('excluded_datasets', []),
        ('context_parallel_size', 2), ('expert_model_parallel_size', 4), ('global_batch_size', 64)]:
        with pytest.raises(ValueError, match='calibration'):
            p.production_spec(manifest, dict(receipt, **{key: value}), 'Handwritten-receipt.json')


def test_generation03_durable_source_freeze_uses_only_approved_templates(tmp_path, monkeypatch):
    import os
    p = module()
    p.configure_generation('03')
    monkeypatch.setenv('NEMOTRON_48H_PIPELINE_SUPERVISED', '03')
    monkeypatch.setattr(p, 'service_state', lambda unit: {'MainPID': str(os.getpid())} if unit == p.UNIT else {'MainPID': '0'})
    monkeypatch.setattr(p, 'process_identity', lambda *a: {'fixture': 'Handwritten'})
    c = p.Controller(root=tmp_path)
    monkeypatch.setattr(c, 'assert_fresh', lambda *a: None)
    seen = []
    def record(path):
        seen.append(path.relative_to(tmp_path))
        return 'Handwritten-source-only'
    monkeypatch.setattr(c, 'stable_hash', record)
    with pytest.raises(RuntimeError, match='contingent pipeline failed'):
        c.run()
    assert seen == p.reviewed_input_paths()
    state = json.loads((tmp_path / p.OUT / 'controller-state.json').read_text())
    assert state['status'] == 'failed' and state['owned_runs'] == []
    assert state['evidence']['data_failure'] == 'missing_atomic_completion_ready'


def test_generation03_final_tracking_requires_frozen_template_display_name(monkeypatch):
    import sys
    from types import SimpleNamespace
    p = module()
    p.configure_generation('03')
    remote = SimpleNamespace(id=p.MAIN_ID, name='nemochat baby Nemotron SFT 16k 48h', state='finished', url='Handwritten-no-network')
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(Api=lambda **kw: SimpleNamespace(run=lambda target: remote)))
    c = p.Controller()
    c.main_plan = dict(name=remote.name)
    monkeypatch.setattr(c, 'load_module', lambda *a: SimpleNamespace(read_wandb_env=lambda path: {'WANDB_API_KEY': 'Handwritten-not-a-secret'}))
    assert c.tracking_preflight(final=True)['display_name'] == c.main_plan['name']
    remote.name = 'nemochat baby Nemotron SFT 48h'
    with pytest.raises(RuntimeError, match='readback failed'):
        c.tracking_preflight(final=True)


def test_generation03_preview_is_inert_without_native_templates(tmp_path, monkeypatch):
    p = module()
    p.configure_generation('03')
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'service_state', lambda unit: {'MainPID': '0', 'LoadState': 'not-found'})
    preview = p.preview()
    assert preview['generation'] == '03' and preview['gpu_launches'] == 0
    assert preview['sequence_length'] == 16384 and preview['excluded_datasets'] == [LONG]
    assert preview['calibration_command'][-1] == '--execute'
    assert preview['status'] == 'WAITING_DATA'
    assert not (tmp_path / p.OUT / 'controller-state.json').exists()


def test_generation03_reviewed_source_guard_requires_complete_exact_current_allowlist(tmp_path):
    import hashlib
    p = module()
    p.configure_generation('03')
    assert hasattr(p, 'verify_reviewed_start'), 'parent-reviewed source guard missing'
    assert hasattr(p, 'reviewed_start_paths'), 'exact startup input allowlist missing'
    with pytest.raises(FileNotFoundError):
        p.verify_reviewed_start(root=tmp_path)
    paths = p.reviewed_start_paths()
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('Handwritten CPU source guard control ' + str(path))
    checksum = tmp_path / p.OUT / 'reviewed-inputs.sha256'
    checksum.parent.mkdir(parents=True, exist_ok=True)
    lines = [hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() + '  ' + str(path) + '\n' for path in paths]
    checksum.write_text(''.join(lines))
    assert p.verify_reviewed_start(root=tmp_path)['checked_files'] == len(paths)
    checksum.write_text(''.join(lines[:-1]))
    with pytest.raises(ValueError, match='allowlist'):
        p.verify_reviewed_start(root=tmp_path)
    checksum.write_text(''.join(lines + lines[:1]))
    with pytest.raises(ValueError, match='allowlist'):
        p.verify_reviewed_start(root=tmp_path)
    checksum.write_text(''.join(lines))
    (tmp_path / paths[0]).write_text('changed source; no auto-refresh')
    with pytest.raises(ValueError, match='changed'):
        p.verify_reviewed_start(root=tmp_path)


def test_generation03_parent_start_script_real_guard_and_inert_systemd_substitute(tmp_path, monkeypatch):
    import hashlib
    import os
    import subprocess
    import sys
    p = module()
    p.configure_generation('03')
    assert hasattr(p, 'parent_start_script'), 'generation03 reviewed start script generator missing'
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'PYTHON', Path(sys.executable))
    paths = p.reviewed_start_paths()
    source = ROOT / 'scripts/posttraining/nemotron_48h_pipeline.py'
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text() if path == Path('scripts/posttraining/nemotron_48h_pipeline.py') else 'Handwritten CPU fixture')
    script = tmp_path / p.OUT / 'start-parent-reviewed.sh'
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(p.parent_start_script())
    fake = tmp_path / 'systemd-run'
    fake.write_text('#!' + sys.executable + '\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o700)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'])
    before = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert before.returncode != 0 and not before.stdout.strip().endswith('--execute')
    checksum = tmp_path / p.OUT / 'reviewed-inputs.sha256'
    checksum.write_text(''.join(hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() + '  ' + str(path) + '\n' for path in paths))
    verified = subprocess.run(['bash', str(script), '--verify-only'], env=env, capture_output=True, text=True)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    launched = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert launched.returncode == 0, launched.stdout + launched.stderr
    assert json.loads(launched.stdout.splitlines()[-1]) == p.supervisor_payload()[1:]
    assert not (tmp_path / p.OUT / 'controller-state.json').exists()


def test_generation03_reused_archive_is_exact_not_a_new_self_asserted_registry():
    p = module()
    p.configure_generation('03')
    assert hasattr(p, 'validate_reused_acquisition'), 'immutable archived acquisition comparison missing'
    archive = data_fixture(p)[3]
    p.validate_reused_acquisition(archive, copy.deepcopy(archive))
    changed = copy.deepcopy(archive)
    changed['completed_sources'][0]['expected_sha256'] = 'Handwritten-changed'
    with pytest.raises(ValueError, match='immutable'):
        p.validate_reused_acquisition(changed, archive)


def test_generation03_reads_actual_data_worker_paths_and_created_engine_identity(tmp_path, monkeypatch):
    p = module()
    p.configure_generation('03')
    args = data_fixture(p)
    state, ready, handoff, archive, engine, manifest, audit = args[:7]
    immutable = dict(Id='Handwritten-engine', State=engine,
        Config={'Cmd': ['-u', str(p.CPU / 'full_workload.py')]},
        HostConfig={'DeviceRequests': [], 'Devices': [], 'NetworkMode': 'none', 'NanoCpus': 24000000000})
    fixtures = {p.READY: ready, p.CPU / 'preparation-controller-state.json': state,
        p.CPU / 'FULL_HANDOFF.json': handoff, p.BASE / 'cpu-full-01/acquisition-completed.json': archive,
        p.CPU / 'full-preparation.final-state.json': immutable, p.CPU / 'full-preparation.create.json': {'Id': 'Handwritten-engine'},
        p.MANIFEST: manifest, p.MANIFEST.parent / 'counted-source-mask-audit.json': audit}
    for path, value in fixtures.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))
    monkeypatch.setattr(p, 'service_state', lambda *a: {'MainPID': '0'})
    c = p.Controller(root=tmp_path)
    monkeypatch.setattr(c, 'verify_frozen_inputs', lambda: None)
    def stop_before_data_hash(path):
        raise RuntimeError('Handwritten manifest hash boundary; no GPU')
    monkeypatch.setattr(c, 'stable_hash', stop_before_data_hash)
    with pytest.raises(RuntimeError, match='manifest hash boundary'):
        c.data_gate()
    (tmp_path / p.CPU / 'full-preparation.create.json').write_text(json.dumps({'Id': 'wrong-engine'}))
    with pytest.raises(ValueError, match='CPU Engine origin'):
        c.data_gate()


def test_generation03_records_verified_calibration_shape_without_fabricating_native_receipt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    p = module()
    p.configure_generation('03')
    cfg = dict(sequence_length=16384, excluded_datasets=[LONG], global_batch_size=128,
        context_parallel_size=1, expert_model_parallel_size=8, max_hours=.75,
        max_updates=15, watchdog_grace_minutes=15, execution_mode='calibration',
        require_final_validation=True, validation_max_dialogues=128)
    source = 'results/pretraining/nemochat_baby/sft/runs/sft-main-01/checkpoint/iter_0000391'
    c = p.Controller(root=tmp_path)
    c.evidence['data'] = {'fixture': 'Handwritten'}
    c.manifest, c.manifest_sha, c.owned_runs = {}, 'Handwritten-manifest', []
    monkeypatch.setattr(p, 'calibration_spec', lambda: (cfg, ['Handwritten-no-GPU']))
    monkeypatch.setattr(c, 'load_module', lambda *a: SimpleNamespace(SOURCE=source, stage_plan=lambda *a: dict(cfg, updates=15)))
    monkeypatch.setattr(c, 'stable_hash', lambda *a: 'Handwritten-manifest')
    monkeypatch.setattr(c, 'verify_frozen_inputs', lambda: None)
    monkeypatch.setattr(c, 'assert_fresh', lambda *a: None)
    monkeypatch.setattr(c, 'phase', lambda *a: None)
    monkeypatch.setattr(c, 'run_command', lambda *a, **kw: None)
    monkeypatch.setattr(c, 'accept_native', lambda *a: {'fixture': 'Handwritten accepted native substitute, NOT GPU evidence'})
    run = tmp_path / p.BASE / 'runs' / p.CAL_NAME
    run.mkdir(parents=True)
    (tmp_path / p.OUT).mkdir(parents=True)
    receipt = dict(status='passed', manifest_sha256=c.manifest_sha, warm_updates=10,
        cold_updates_excluded=5, global_batch_size=128, ranks=list(range(8)), source_checkpoint=source,
        verification=str(p.BASE / 'runs' / p.CAL_NAME / 'verification.json'), mean_iteration_seconds=50)
    (run / 'calibration.json').write_text(json.dumps(receipt))
    (run / 'rank-0-updates.jsonl').write_text(''.join(json.dumps(dict(iteration_seconds=50)) + '\n' for _ in range(15)))
    c.calibrate()
    assert c.calibration == receipt  # Native receipt bytes/fields are never forged.
    evidence = c.evidence['calibration']
    assert evidence['accepted_native_run'] == p.CAL_NAME
    assert evidence['manifest_sha256'] == c.manifest_sha
    assert evidence['execution_shape'] == {key: cfg[key] for key in ('sequence_length', 'excluded_datasets',
        'global_batch_size', 'context_parallel_size', 'expert_model_parallel_size')}
