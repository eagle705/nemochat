"""CPU synthetic controls only; no GPU/service/production Engine access."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def controller():
    path = ROOT / 'midtraining/run_production.py'
    assert path.exists(), 'contingent production controller is missing'
    spec = importlib.util.spec_from_file_location('production_controller_test', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_source_freeze_is_exact_separate_and_detects_changes(tmp_path):
    m = controller()
    m.ROOT = tmp_path
    (tmp_path/'a').write_text('reviewed')
    review = tmp_path/'review.json'
    digest = __import__('hashlib').sha256(b'reviewed').hexdigest()
    review.write_text(json.dumps({'a': digest}))
    m.source_gate(review, ['a'])
    with pytest.raises(ValueError, match='allowlist'):
        m.source_gate(review, ['a', 'b'])
    (tmp_path/'a').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        m.source_gate(review, ['a'])


def test_supervisor_checks_uid_pid_and_full_argv():
    m = controller()
    argv = ['/python', '/repo/midtraining/run_production.py', '--execute']
    info = {'MainPID': str(os.getpid()), 'ActiveState': 'activating',
            'SubState': 'start', 'User': str(os.getuid())}
    m.check_supervisor(info, os.getpid(), os.getuid(), argv, argv)
    for key, value in [('MainPID', '1'), ('User', '0'), ('ActiveState', 'failed')]:
        bad = dict(info, **{key: value})
        with pytest.raises(ValueError, match='supervisor'):
            m.check_supervisor(bad, os.getpid(), os.getuid(), argv, argv)
    with pytest.raises(ValueError, match='supervisor'):
        m.check_supervisor(info, os.getpid(), os.getuid(), argv + ['foreign'], argv)


def test_user_manager_default_credentials_keep_exact_pid_and_argv_guards():
    m = controller()
    argv = ['/python', '/repo/midtraining/run_production.py', '--execute']
    info = {'MainPID': str(os.getpid()), 'ActiveState': 'active', 'User': ''}
    m.check_supervisor(info, os.getpid(), os.getuid(), argv, argv)
    with pytest.raises(ValueError, match='supervisor'):
        m.check_supervisor(dict(info, MainPID='1'), os.getpid(), os.getuid(), argv, argv)
    with pytest.raises(ValueError, match='supervisor'):
        m.check_supervisor(info, os.getpid(), os.getuid(), argv+['foreign'], argv)


def test_same_recipe_comparison_allows_only_intentional_differences():
    m = controller()
    native = m.load('test_native_recipe', 'midtraining/longrun_native.py')
    old = native.load_config(ROOT/'midtraining/calibration.yaml')
    new = native.load_config(ROOT/'midtraining/longrun24h.yaml')
    m.compare_recipe(old, new)
    for section in ('optimizer', 'ddp', 'parallel', 'kd', 'rng', 'checkpoint'):
        bad = copy.deepcopy(new)
        bad[section]['synthetic_change'] = True
        with pytest.raises(ValueError, match='recipe'):
            m.compare_recipe(old, bad)
    for field in ('teacher_revision', 'student_hf', 'sequence_length'):
        bad = copy.deepcopy(new)
        bad[field] = 'foreign'
        with pytest.raises(ValueError, match='recipe'):
            m.compare_recipe(old, bad)
    bad = copy.deepcopy(new)
    bad['train']['micro_batch_size'] = 2
    with pytest.raises(ValueError, match='recipe'):
        m.compare_recipe(old, bad)


def test_preserve_whole_heldout_and_validation_digests(tmp_path):
    m = controller()
    old_path, new_path = tmp_path/'old', tmp_path/'new'
    old_path.mkdir(); new_path.mkdir()
    row = {'source': 'synthetic', 'content_hash': 'a'*64, 'group_id': 'b'*64}
    for path in (old_path, new_path):
        (path/'heldout-reservations.jsonl').write_text(json.dumps(row)+'\n')
    sha = m.digest(old_path/'heldout-reservations.jsonl')
    entry = dict(bin_sha256='c'*64, idx_sha256='d'*64, bin_bytes=4, idx_bytes=8, tokens=1)
    old = dict(heldout_reservations_sha256=sha, sources={'synthetic': {'validation': entry}})
    new = copy.deepcopy(old)
    assert m.preserve_heldout(old, new, old_path/'manifest.json', new_path/'manifest.json') == 1
    bad = copy.deepcopy(new)
    bad['sources']['synthetic']['validation']['bin_sha256'] = 'e'*64
    with pytest.raises(ValueError, match='validation'):
        m.preserve_heldout(old, bad, old_path/'manifest.json', new_path/'manifest.json')
    (new_path/'heldout-reservations.jsonl').write_text('')
    new['heldout_reservations_sha256'] = m.digest(new_path/'heldout-reservations.jsonl')
    with pytest.raises(ValueError, match='reservation'):
        m.preserve_heldout(old, new, old_path/'manifest.json', new_path/'manifest.json')


def test_data_validation_calls_native_gates_and_derives_exact_compatibility(tmp_path, monkeypatch):
    m = controller()
    n = m.load('test_real_native_capacity', 'midtraining/longrun_native.py')
    calls = []
    seed = {'tokenizer': {'synthetic': True}, 'sources': {}}
    production = copy.deepcopy(seed)
    production.update(seed_manifest_sha256=m.SEED_SHA, frozen_seed_heldout_reuse=True)
    prod_path, seed_path = tmp_path/'prod.json', tmp_path/'seed.json'
    seed_path.write_text('synthetic seed')
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'SEED', 'seed.json')
    monkeypatch.setattr(m, 'MANIFEST', 'prod.json')
    monkeypatch.setattr(m, 'SEED_SHA', m.digest(seed_path))
    production['seed_manifest_sha256'] = m.SEED_SHA
    accepted = dict(status='calibration_save_reload_accepted', all_ranks=8,
                    manifest_sha256=m.SEED_SHA, calibration={'synthetic': True})
    duration = dict(status='duration_control_save_reload_accepted', all_ranks=8)
    for relative, value in [(m.CALIBRATION, accepted), (m.DURATION, duration)]:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    old = n.load_config(ROOT/'midtraining/calibration.yaml')
    new = n.load_config(ROOT/'midtraining/longrun24h.yaml')
    cfg = tmp_path/(m.BASE+'super35-newdata-calibration-01/consumed-config.yaml')
    cfg.write_text('synthetic config')
    frozen = cfg.parent/'controller/frozen-inputs.json'
    frozen.write_text(json.dumps({'config_sha256': m.digest(cfg)}))
    monkeypatch.setattr(m, 'preserve_heldout', lambda *a: 7858)
    class Native:
        def load_prepared_manifest(self, path):
            calls.append(('load', path))
            return (seed, m.SEED_SHA) if path == seed_path else (production, 'newsha')
        def load_config(self, path):
            return old if path == cfg else new
        def freeze_config(self, config, calibration):
            return dict(config, train=dict(config['train'], train_iters=3200))
        def capacity_gate(self, manifest, updates, batch, sequence):
            calls.append(('capacity', updates, batch, sequence))
            # Use actual native per-source capacity, synthetic token metadata.
            synthetic = {'sources': {k: {'train': {'tokens': 2_000_000_000},
                                       'validation': {'tokens': 4097}} for k in n.KEYS}}
            return n.capacity_gate(synthetic, updates, batch, sequence)
        def audit_reserved_ids(self, manifest):
            calls.append(('audit',))
            return {'actual_prepared_bin_scan': True}
        def corpus_tokenizer_gate(self, manifest, audit):
            calls.append(('tokenizer',))
    compatibility, evidence = m.validate_data(Native())
    assert compatibility == dict(status='parent_accepted_same_recipe_expansion',
        calibration_manifest_sha256=m.SEED_SHA, production_manifest_sha256='newsha',
        same_provider_tokenizer_sequence_batch_topology=True, heldout_exclusions_preserved=True)
    assert ('capacity', 3200, 128, 4096) in calls
    assert ('audit',) in calls and ('tokenizer',) in calls
    assert evidence['prior_reserved_documents'] == 7858
    assert evidence['capacity']['processed_positions'] == 1677721600
    production['tokenizer']['synthetic'] = False
    with pytest.raises(ValueError, match='tokenizer'):
        m.validate_data(Native())
    production['tokenizer']['synthetic'] = True
    class Insufficient(Native):
        def capacity_gate(self, manifest, updates, batch, sequence):
            synthetic = {'sources': {k: {'train': {'tokens': 1},
                                       'validation': {'tokens': 4097}} for k in n.KEYS}}
            return n.capacity_gate(synthetic, updates, batch, sequence)
        def audit_reserved_ids(self, manifest):
            pytest.fail('insufficient capacity must block before full audit or native launch')
    with pytest.raises(ValueError, match='capacity insufficient'):
        m.validate_data(Insufficient())


@pytest.mark.parametrize('mutation', ['Id', 'Name', 'Image', 'Cmd', 'owned-label', 'oom', 'exit', 'wait-error'])
def test_cpu_rejection_never_succeeds(mutation):
    m = controller(); frozen, actual = identity()
    if mutation in ('Id', 'Name'):
        actual[mutation] = 'foreign'
    elif mutation in ('Image', 'Cmd'):
        actual['Config'][mutation] = 'foreign'
    elif mutation == 'owned-label':
        actual['Config']['Labels']['nemochat.stage'] = 'foreign'
    class Engine:
        def request(self, method, path, **kwargs):
            if method == 'POST':
                actual['State']['Running'] = False
                if mutation == 'oom':
                    actual['State']['OOMKilled'] = True
                if mutation == 'exit':
                    actual['State']['ExitCode'] = 1
                return dict(StatusCode=0, **({'Error': 'synthetic wait failure'} if mutation == 'wait-error' else {}))
            return copy.deepcopy(actual)
    with pytest.raises(ValueError):
        m.wait_corpus(Engine(), frozen, 10)


def test_end_to_end_synthetic_cpu_exit_to_real_child_and_durable_finish(tmp_path, monkeypatch):
    m = controller()
    frozen, actual = identity()
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    p = tmp_path/m.CPU_IDENTITY
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(frozen))
    review = tmp_path/'startup.json'; review.write_text('{}')
    events = []
    monkeypatch.setattr(m, 'supervisor_gate', lambda: events.append('supervised'))
    monkeypatch.setattr(m, 'startup_sources', lambda: [])
    monkeypatch.setattr(m, 'source_gate', lambda *a: events.append('source'))
    monkeypatch.setattr(m, 'validate_data', lambda n: (dict(production_manifest_sha256='synthetic'),
                                                     dict(manifest_sha256='synthetic')))
    monkeypatch.setattr(m, 'corpus_wait_seconds', lambda: 60, raising=False)
    monkeypatch.setattr(m, 'approved_corpus', lambda: frozen)
    monkeypatch.setattr(m, 'input_paths', lambda: [m.CPU_IDENTITY], raising=False)
    monkeypatch.setattr(m, 'load', lambda *a: None)
    accepted = tmp_path/(m.BASE+m.RUN+'/controller/accepted.json')
    script = tmp_path/'synthetic_native_child.py'
    script.write_text('import os,json,pathlib\n'
        "assert os.environ['NEMOCHAT_LONGRUN_SUPERVISED']=='1'\n"
        f'p=pathlib.Path({str(accepted)!r})\np.parent.mkdir(parents=True)\n'
        "p.write_text(json.dumps(dict(status='production_save_reload_accepted',all_ranks=8,"
        "manifest_sha256='synthetic',training_24h_accepted=False,completed_updates=3200)))\n")
    monkeypatch.setattr(m, 'main_command', lambda *a: [__import__('sys').executable, str(script)], raising=False)
    class Engine:
        def request(self, method, path, **kwargs):
            if method == 'POST':
                events.append('CPU exited')
                actual['State']['Running'] = False
                return {'StatusCode': 0}
            return copy.deepcopy(actual)
    result = m.execute(review, Engine())
    state = json.loads((tmp_path/m.OUTPUT/'state.json').read_text())
    assert result['training_24h_accepted'] is False
    assert state['phase'] == 'FINISHED'
    phases = [json.loads(x)['phase'] for x in (tmp_path/m.OUTPUT/'phases.jsonl').read_text().splitlines()]
    assert phases == ['WAITING_CORPUS', 'VALIDATING_DATA', 'NATIVE_MAIN_CONTROLLER_RUNNING', 'FINISHED']
    assert events.index('source') < events.index('CPU exited')
    with pytest.raises(FileExistsError):
        m.execute(review, Engine())


def test_cpu_failure_is_durable_and_never_launches_gpu(tmp_path, monkeypatch):
    m = controller(); frozen, actual = identity()
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    p = tmp_path/m.CPU_IDENTITY; p.parent.mkdir(parents=True); p.write_text(json.dumps(frozen))
    monkeypatch.setattr(m, 'supervisor_gate', lambda: None)
    monkeypatch.setattr(m, 'startup_sources', lambda: [])
    monkeypatch.setattr(m, 'source_gate', lambda *a: None)
    monkeypatch.setattr(m, 'corpus_wait_seconds', lambda: 60, raising=False)
    monkeypatch.setattr(m, 'approved_corpus', lambda: frozen)
    monkeypatch.setattr(m, 'input_paths', lambda: [m.CPU_IDENTITY], raising=False)
    monkeypatch.setattr(m, 'main_command', lambda *a: pytest.fail('GPU launch on CPU failure'), raising=False)
    class Engine:
        def request(self, method, path, **kwargs):
            if method == 'POST':
                actual['State'].update(Running=False, ExitCode=137, OOMKilled=True)
                return {'StatusCode': 137}
            return copy.deepcopy(actual)
    (tmp_path/'review.json').write_text('{}')
    with pytest.raises(ValueError, match='CPU exit'):
        m.execute(tmp_path/'review.json', Engine())
    assert json.loads((tmp_path/m.OUTPUT/'state.json').read_text())['phase'] == 'FAILED'
    assert not (tmp_path/m.OUTPUT/'calibration-compatibility.json').exists()


def test_cli_preview_and_help_are_inert_and_forward_exact_command(tmp_path):
    import subprocess
    import sys
    m = controller()
    for args in ([], ['--preview'], ['--help']):
        result = subprocess.run([sys.executable, str(ROOT/'midtraining/run_production.py'), *args],
                                cwd=tmp_path, capture_output=True, text=True, check=True)
        assert result.stdout
    preview = json.loads(subprocess.check_output([sys.executable,
        str(ROOT/'midtraining/run_production.py'), '--preview'], cwd=tmp_path, text=True))
    assert preview['status'] == 'inert_preview'
    assert preview['unit'] == m.UNIT
    command = m.main_command(tmp_path/'compatibility.json')
    assert command[-1] == '--execute'
    assert command[command.index('--run-id')+1] == 'super35-longrun-01'
    assert command[command.index('--reviewed-sources')+1] == str(ROOT/m.NATIVE_REVIEW)
    assert '--calibration-compatibility' in command
    assert not list(tmp_path.iterdir())


def test_cleanup_refuses_without_owned_release_record(tmp_path, monkeypatch):
    m = controller()
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    monkeypatch.setattr(m, 'startup_sources', lambda: [])
    review = tmp_path/'review.json'; review.write_text('{}')
    monkeypatch.setattr(m, 'load', lambda *a: pytest.fail('must not import cleanup without release'))
    assert m.cleanup_owned(review) == {'status': 'no_owned_main_release'}


def test_signal_handler_interrupts_without_touching_cpu():
    m = controller()
    with pytest.raises(InterruptedError):
        m.interrupted(15, None)


def test_source_review_rejects_duplicate_json_keys(tmp_path):
    m = controller(); m.ROOT = tmp_path
    (tmp_path/'a').write_text('reviewed')
    sha = m.digest(tmp_path/'a')
    review = tmp_path/'review.json'
    review.write_text('{"a": "'+sha+'", "a": "'+sha+'"}')
    with pytest.raises(ValueError, match='duplicate'):
        m.source_gate(review, ['a'])


def test_cpu_reference_is_pinned_to_actual_approved_payload():
    m = controller()
    frozen = m.approved_corpus()
    assert frozen['id'] == '9e0861343af8efa265c622623ff78714109132a1cd48cc59c4a2731ec456bab5'
    assert frozen['name'] == 'nemochat-midtraining-production-prepare-01'


def test_wait_requires_positive_finite_bound():
    m = controller(); frozen, actual = identity()
    class Engine:
        def request(self, *a, **kw):
            pytest.fail('invalid timeout must not access Engine')
    for timeout in (0, -1, float('inf'), float('nan')):
        with pytest.raises(ValueError, match='timeout'):
            m.wait_corpus(Engine(), frozen, timeout)


@pytest.mark.parametrize('block', ['capacity', 'source', 'receipt', 'review', 'deadline'])
def test_release_gate_failure_is_durable_and_never_launches(tmp_path, monkeypatch, block):
    m = controller(); frozen, actual = identity()
    m.ROOT = tmp_path
    p = tmp_path/m.CPU_IDENTITY; p.parent.mkdir(parents=True); p.write_text(json.dumps(frozen))
    review = tmp_path/'review.json'; review.write_text('{}')
    monkeypatch.setattr(m, 'supervisor_gate', lambda: None)
    monkeypatch.setattr(m, 'startup_sources', lambda: [])
    monkeypatch.setattr(m, 'approved_corpus', lambda: frozen)
    monkeypatch.setattr(m, 'input_paths', lambda: [m.CPU_IDENTITY])
    monkeypatch.setattr(m, 'load', lambda *a: None)
    monkeypatch.setattr(m, 'corpus_wait_seconds', lambda: 60)
    monkeypatch.setattr(m, 'main_command', lambda *a: pytest.fail('blocked release launched native'))
    def source_gate(*args):
        if block == 'source' and actual['State']['Running'] is False:
            raise ValueError('synthetic changed source')
    monkeypatch.setattr(m, 'source_gate', source_gate)
    def validate(*args):
        if block == 'capacity':
            raise ValueError('synthetic capacity insufficient')
        if block == 'receipt':
            p.write_text('changed receipt')
        if block == 'review':
            review.write_text('{"silently_refreshed": true}')
        return {}, {}
    monkeypatch.setattr(m, 'validate_data', validate)
    if block == 'deadline':
        times = iter([0, m.SUPERVISOR_SECONDS-m.MAIN_SECONDS+1])
        monkeypatch.setattr(m.time, 'monotonic', lambda: next(times))
    class Engine:
        def request(self, method, path, **kwargs):
            if method == 'POST':
                actual['State']['Running'] = False
                return {'StatusCode': 0}
            return copy.deepcopy(actual)
    with pytest.raises((ValueError, TimeoutError)):
        m.execute(review, Engine())
    assert m.read(tmp_path/m.OUTPUT/'state.json')['phase'] == 'FAILED'
    assert not (tmp_path/m.OUTPUT/'main-command.json').exists()


def test_supervisor_argv_has_exact_bounds_cleanup_and_env():
    m = controller()
    command = m.supervisor_command(ROOT/'startup-review.json')
    assert command[:2] == ['systemd-run', '--user']
    assert '--unit='+m.UNIT in command
    for value in ('RuntimeMaxSec=140400', 'KillMode=control-group'):
        assert '--property='+value in command
    assert not any(value.startswith('--property=User=') for value in command)
    stop = next(x for x in command if x.startswith('--property=ExecStopPost='))
    assert '--cleanup-owned' in stop and 'startup-review.json' in stop
    assert '--setenv=NEMOCHAT_PRODUCTION_SUPERVISED=1' in command
    assert command[-4:] == [str(ROOT/'midtraining/run_production.py'), '--execute',
                           '--reviewed-sources', str(ROOT/'startup-review.json')]


def test_supervisor_gate_rejects_foreign_interpreter(tmp_path, monkeypatch):
    import sys
    m = controller()
    proc = tmp_path/str(os.getpid()); proc.mkdir()
    script = str(ROOT/'midtraining/run_production.py')
    (proc/'cmdline').write_bytes(('foreign-python\0'+script+'\0--execute\0').encode())
    real_path = Path
    monkeypatch.setattr(m, 'Path', lambda *a: real_path(tmp_path) if a == ('/proc',) else real_path(*a))
    monkeypatch.setattr(sys, 'argv', [script, '--execute'])
    monkeypatch.setenv('NEMOCHAT_PRODUCTION_SUPERVISED', '1')
    monkeypatch.setattr(m.subprocess, 'check_output', lambda *a, **kw:
        f'MainPID={os.getpid()}\nUser={os.getuid()}\nActiveState=active\nSubState=running\n')
    with pytest.raises(ValueError, match='supervisor'):
        m.supervisor_gate()


def identity():
    payload = dict(Image='synthetic-cpu-image', Cmd=['synthetic-cpu-producer'],
                   Labels={'nemochat.stage': 'synthetic-cpu'})
    frozen = dict(id='a'*64, name='synthetic-cpu-producer', payload=payload)
    actual = dict(Id=frozen['id'], Name='/'+frozen['name'], Config=copy.deepcopy(payload),
                  State=dict(Running=True, ExitCode=0, OOMKilled=False))
    actual['Config']['Labels']['com.nvidia.inherited'] = 'allowed'
    return frozen, actual


def test_exact_cpu_wait_blocks_once_and_checks_identity_after():
    m = controller()
    frozen, actual = identity()
    calls = []
    class Engine:
        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))
            if method == 'POST':
                actual['State']['Running'] = False
                return {'StatusCode': 0}
            return copy.deepcopy(actual)
    result = m.wait_corpus(Engine(), frozen, timeout=100)
    assert result['ExitCode'] == 0
    assert [x[:2] for x in calls] == [
        ('GET', '/containers/'+frozen['id']+'/json'),
        ('POST', '/containers/'+frozen['id']+'/wait?condition=not-running'),
        ('GET', '/containers/'+frozen['id']+'/json')]
    assert calls[1][2]['timeout'] == 100
    for field in ('Image', 'Cmd'):
        wrong = copy.deepcopy(actual)
        wrong['Config'][field] = 'foreign'
        with pytest.raises(ValueError, match='identity'):
            m.check_corpus(wrong, frozen)
    wrong = copy.deepcopy(actual)
    wrong['Config']['Labels']['nemochat.foreign'] = 'no'
    with pytest.raises(ValueError, match='identity'):
        m.check_corpus(wrong, frozen)
