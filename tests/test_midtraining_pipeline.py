"""Controller gates use exact receipts, never a launch-success inference."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    path = ROOT / 'midtraining/training/run_pipeline.py'
    assert path.is_file(), 'parent-owned midtraining controller missing'
    spec = importlib.util.spec_from_file_location('midtraining_pipeline', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_download_gate_rejects_partial_and_wrong_teacher(tmp_path):
    module = load()
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps({'status': 'running'}))
    with pytest.raises(ValueError, match='complete'):
        module.download_gate(path)
    path.write_text(json.dumps({'status': 'complete', 'repo_id': 'wrong', 'revision': 'wrong'}))
    with pytest.raises(ValueError, match='identity'):
        module.download_gate(path)


def test_native_acceptance_requires_every_rank_and_clean_engine(tmp_path):
    module = load()
    with pytest.raises((FileNotFoundError, ValueError)):
        module.accept_native(tmp_path, {'Running': False, 'ExitCode': 0, 'OOMKilled': False})
    for rank in range(8):
        (tmp_path / f'rank-{rank}-complete.json').write_text(json.dumps({
            'status': 'pilot_complete', 'completed_updates': 4, 'save_steps': [4], 'finite_heldout_evaluations': 1}))
        rows = [dict(update=i, lr_used=0. if i == 1 else 1e-5,
                     metrics={'LM Loss': 2., 'KD Loss': .3, 'Total Loss': 3., 'Grad Norm': .5},
                     teacher_frozen_no_grads=True, native_optimizer=True) for i in range(1, 5)]
        (tmp_path / f'rank-{rank}-updates.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    result = module.accept_native(tmp_path, {'Running': False, 'ExitCode': 0, 'OOMKilled': False})
    assert result['all_ranks'] == 8
    assert result['native_optimizer_calls'] == 4
    assert result['positive_lr_learning_updates'] == 3
    with pytest.raises(ValueError, match='Engine'):
        module.accept_native(tmp_path, {'Running': False, 'ExitCode': 1, 'OOMKilled': False})
    (tmp_path / 'rank-7-complete.json').unlink()
    with pytest.raises(FileNotFoundError):
        module.accept_native(tmp_path, {'Running': False, 'ExitCode': 0, 'OOMKilled': False})


def test_cleanup_refuses_foreign_container_without_stopping_it(tmp_path, monkeypatch):
    module = load()
    assert hasattr(module, 'cleanup_owned'), 'exact-owned native cleanup is missing'
    monkeypatch.setattr(module, 'CONTROL', tmp_path)
    (tmp_path / 'pilot-01-container.json').write_text(json.dumps({'id': 'owned-id'}))
    calls = []
    class Engine:
        def request(self, method, path, **kwargs):
            calls.append((method, path))
            return {'Id': 'owned-id', 'Name': '/foreign-run', 'Config': {'Labels': {}}, 'State': {'Running': True}}
    with pytest.raises(RuntimeError, match='identity'):
        module.cleanup_owned(Engine())
    assert all(method == 'GET' for method, path in calls)


def test_new_generation_isolated_from_failed_proof_and_opt_in_diagnostics():
    module = load()
    assert hasattr(module, 'configure_generation'), 'fresh diagnostic generation is not supported'
    module.configure_generation('02', diagnostics=True)
    assert module.CONTROL.name == 'pipeline-02'
    assert module.PILOT_SEGMENT == 'pilot-02'
    assert module.RELOAD_SEGMENT == 'reload-02'
    assert '--diagnostics' in module.container_payload('pilot-02')['Cmd']
    assert '--diagnostics' not in module.container_payload('reload-02', reload_only=True)['Cmd']
    with pytest.raises(ValueError):
        module.configure_generation('../')


def test_payload_uses_native_eight_ranks_and_isolated_mounts():
    module = load()
    payload = module.container_payload('pilot-01')
    assert payload['HostConfig']['NetworkMode'] == 'none'
    assert payload['HostConfig']['DeviceRequests'][0]['Count'] == 8
    assert '--nproc_per_node=8' in payload['Cmd']
    assert 'midtraining/training/native_kd.py' in payload['Cmd']
    assert any(x.endswith(':/workspace:ro') for x in payload['HostConfig']['Binds'])
    assert not payload['HostConfig'].get('Privileged', False)
