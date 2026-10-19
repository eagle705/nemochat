"""Approved 1.920B production profile; CPU contracts, not GPU results."""
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location('offline1920_' + name, ROOT / 'midtraining/training/offline_kd' / (name + '.py'))
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def test_approved_unique1920_profile_preserves_gate_and_old_budget():
    launch = load('launch')
    p = launch.plan('production1920')
    assert p['train_iters'] == 3662
    assert p['positions'] == 1919942656
    assert p['train_samples'] == 468736
    assert p['warmup_iters'] == 37 and p['decay_iters'] == 366
    assert p['weights'] == [.67, .10, .03, .20]
    assert p['num_microbatches'] == 16 and p['world_size'] == 8
    assert launch.plan('gate')['train_iters'] == 4
    assert launch.plan('production')['train_iters'] == 3815


def test_approved_profile_cli_still_requires_parent_review():
    import subprocess
    output = ROOT / 'results/midtraining/offline2b-never-created-approved1920'
    assert not output.exists()
    completed = subprocess.run([sys.executable, str(ROOT / 'midtraining/training/offline_kd/launch.py'),
        '--mode', 'production1920', '--output', str(output), '--start'],
        capture_output=True, text=True)
    assert completed.returncode != 0
    assert 'parent production review receipt required' in completed.stderr
    assert not output.exists()


def test_approved_profile_has_only_teacher_and_student_stages():
    supervisor = load('supervisor')
    assert supervisor.stages('production1920') == ('teacher', 'student')


def test_approved_profile_reads_only_new_unique_manifest(tmp_path, monkeypatch):
    supervisor = load('supervisor')
    captured = []
    class Native:
        def load_prepared_manifest(self, path):
            captured.append(path)
            raise RuntimeError('captured manifest before IO')
    monkeypatch.setattr(supervisor, 'helper', lambda *a: Native())
    with pytest.raises(RuntimeError, match='captured manifest'):
        supervisor.prepare_manifest('production1920', tmp_path, load('launch').plan('production1920'))
    assert captured == [ROOT / 'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json']


def test_budget_only_gate_compatibility_keeps_native_model_loss_sources_pinned():
    import copy
    import json
    launch = load('launch')
    gate = json.loads((ROOT / 'results/midtraining/offline2b-gate-02/report.json').read_text())
    compatibility = {
        'status': 'parent_accepted_budget_only_adjustment',
        'gate_source_sha256': gate['source_sha256'],
        'old_production_calls': 3815,
        'new_production_calls': 3662,
        'original_base_reset': True,
        'same_teacher_tokenizer_precision_sequence_batch_topology_objective': True,
        'user_approved_unique_budget': True,
        'gate_execution_reused_not_reexecuted': True,
    }
    review = {'source_sha256': launch.freeze(), 'budget_compatibility': compatibility}
    assert hasattr(launch, 'validate_gate_review'), 'budget-only parent compatibility validation is missing'
    launch.validate_gate_review(review, gate, 'production1920')
    broken = copy.deepcopy(review)
    broken['source_sha256']['midtraining/training/offline_kd/native.py'] = '0' * 64
    with pytest.raises(ValueError, match='native|source'):
        launch.validate_gate_review(broken, gate, 'production1920')
    broken = copy.deepcopy(review)
    broken['budget_compatibility']['user_approved_unique_budget'] = False
    with pytest.raises(ValueError, match='compatibility'):
        launch.validate_gate_review(broken, gate, 'production1920')
    with pytest.raises(ValueError, match='compatibility|source'):
        launch.validate_gate_review(review, gate, 'production')


def test_live_owned_scratch_heartbeat_prevents_idle_expiry(tmp_path):
    import os
    supervisor = load('supervisor')
    scratch = tmp_path / 'owned'
    scratch.mkdir()
    os.utime(scratch, (1, 1))
    assert hasattr(supervisor, 'keep_scratch_alive'), 'long-run scratch heartbeat is missing'
    supervisor.keep_scratch_alive(scratch)
    assert scratch.stat().st_mtime > 1
    assert list(scratch.iterdir()) == []


def test_new_production_requires_parent_corpus_digest_and_exact_clean_producer():
    import copy
    supervisor = load('supervisor')
    manifest = ROOT / 'datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json'
    proof = dict(status='parent_verified_native_corpus', manifest_path=str(manifest),
        manifest_sha256='a' * 64, producer_id='owned-id', producer_name='/owned-corpus',
        task_labels={'nemochat.offline1920.owner': 'owned-corpus'})
    actual = dict(Id='owned-id', Name='/owned-corpus',
        Config=dict(Image=supervisor.IMAGE, Labels=proof['task_labels'] | {'com.nvidia.image': 'inherited'}),
        HostConfig=dict(DeviceRequests=[], NetworkMode='none'),
        State=dict(Running=False, ExitCode=0, OOMKilled=False))
    class Engine:
        def request(self, method, path):
            assert method == 'GET' and path == '/containers/owned-id/json'
            return actual
    assert hasattr(supervisor, 'verify_corpus_approval'), 'exact producer acceptance gate is missing'
    supervisor.verify_corpus_approval({'corpus_approval': proof}, manifest, 'a' * 64, Engine())
    with pytest.raises(ValueError, match='corpus|manifest'):
        supervisor.verify_corpus_approval({'corpus_approval': proof}, manifest, 'b' * 64, Engine())
    actual['State']['ExitCode'] = 1
    with pytest.raises(ValueError, match='producer'):
        supervisor.verify_corpus_approval({'corpus_approval': proof}, manifest, 'a' * 64, Engine())
    actual['State']['ExitCode'] = 0
    actual['Config']['Labels']['nemochat.offline1920.owner'] = 'foreign'
    with pytest.raises(ValueError, match='producer'):
        supervisor.verify_corpus_approval({'corpus_approval': proof}, manifest, 'a' * 64, Engine())
