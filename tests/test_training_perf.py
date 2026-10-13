"""Real tiny CPU updates; CUDA tests are explicit hardware-dependent gates."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from nemochat.training_perf import (DenseCausalTransformer, compare_results, run_benchmark,
                                    validate_config, workload_estimates)


def smoke(checkpoint=False, profile=False):
    return {"schema_version": 1, "experiment": {"name": "cpu-control", "seed": 42},
            "workload": {"device": "cpu", "dtype": "float32", "batch_size": 2,
                         "sequence_length": 16, "hidden_size": 32, "ffn_size": 64,
                         "layers": 1, "vocab_size": 128, "learning_rate": .001,
                         "activation_checkpointing": checkpoint},
            "measurement": {"warmup_steps": 1, "measured_steps": 3, "repeats": 2,
                            "profile": profile},
            "resources": {"max_seconds": 60, "max_gpu_count": 0}}


@pytest.fixture(scope="module")
def actual():
    a = run_benchmark(smoke())
    b = run_benchmark(smoke(checkpoint=True))
    assert a['status'] == b['status'] == 'success'
    return a, b


def test_cpu_actual_updates_and_repeat_state(actual):
    a, b = actual
    for result in (a, b):
        assert result['evidence_scope'] == 'CPU-model-executed'
        first, second = result['repeats']
        assert first['losses'] == second['losses']
        assert first['gradient_norms'] == second['gradient_norms']
        assert first['initial_parameter_sha256'] == second['initial_parameter_sha256']
        assert first['final_parameter_sha256'] == second['final_parameter_sha256']
        assert first['initial_parameter_sha256'] != first['final_parameter_sha256']
        for row in result['repeats']:
            assert row['completed_updates'] == 3
            assert row['processed_tokens'] == row['loss_tokens'] == 96
            assert row['memory']['peak_allocated_bytes'] is None
            assert row['memory']['peak_reserved_bytes'] is None
            assert row['gradients_finite'] and row['parameters_changed']
            assert math.isfinite(row['tokens_per_second'])
    report = compare_results(a, b)
    assert report['decision'] == 'experimental-only'
    assert report['memory_tradeoff']['status'] == 'unavailable-on-CPU'


def test_rng_threads_restored():
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    result = run_benchmark(smoke(checkpoint=True))
    assert result['status'] == 'success'
    assert torch.equal(state, torch.get_rng_state())
    assert threads == torch.get_num_threads()


@pytest.mark.parametrize('section,key,value', [
    ('workload', 'learning_rate', float('nan')), ('workload', 'learning_rate', float('inf')),
    ('workload', 'learning_rate', 0), ('workload', 'batch_size', True),
    ('workload', 'sequence_length', 0), ('workload', 'vocab_size', 1),
    ('workload', 'dtype', 'float16'), ('workload', 'activation_checkpointing', 1),
    ('measurement', 'profile', 'false'), ('measurement', 'warmup_steps', -1),
    ('measurement', 'measured_steps', 0), ('measurement', 'repeats', 1.5),
    ('resources', 'max_seconds', float('inf')), ('resources', 'max_gpu_count', 2),
    ('experiment', 'seed', -1), ('experiment', 'name', ''),
])
def test_malformed(section, key, value):
    c = smoke(); c[section][key] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_unknown_missing_schema():
    for c in (None, [], {}, dict(smoke(), unknown=1)):
        with pytest.raises(ValueError):
            validate_config(c)
    c = smoke(); del c['workload']['layers']
    with pytest.raises(ValueError):
        validate_config(c)
    c = smoke(); c['schema_version'] = True
    with pytest.raises(ValueError):
        validate_config(c)
    c = smoke(); c['workload']['device'] = 'cuda'
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize('mutation', ['seed', 'measurement', 'environment', 'scope', 'counter',
                                     'nan', 'correctness', 'workload', 'gradient', 'memory'])
def test_incompatible_comparison(actual, mutation):
    a, b = deepcopy(actual)
    if mutation == 'seed': b['config']['experiment']['seed'] += 1
    elif mutation == 'measurement': b['config']['measurement']['warmup_steps'] += 1
    elif mutation == 'environment': b['environment']['torch'] = 'different'
    elif mutation == 'scope': b['evidence_scope'] = 'GPU-smoke-executed'
    elif mutation == 'counter': b['repeats'][0]['processed_tokens'] = 0
    elif mutation == 'nan': b['repeats'][0]['elapsed_seconds'] = float('nan')
    elif mutation == 'correctness': b['correctness']['status'] = 'not-established'
    elif mutation == 'workload': b['config']['workload']['hidden_size'] *= 2
    elif mutation == 'gradient': b['repeats'][0]['gradient_norms'][0] = float('inf')
    elif mutation == 'memory': b['repeats'][0]['memory']['peak_allocated_bytes'] = 123
    with pytest.raises(ValueError):
        compare_results(a, b)


def test_no_overwrite_profile_and_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(smoke(profile=True)))
    output = tmp_path / 'measurement'
    command = [sys.executable, str(root/'scripts/benchmark_training.py'), '--config', str(config), '--output', str(output)]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads((output/'result.json').read_text())
    assert result['profile']['status'] == 'captured-separate-from-timing'
    trace = json.loads((output/'trace.json').read_text())
    labels = {event.get('name') for event in trace['traceEvents']}
    assert {'data', 'forward', 'backward', 'optimizer'} <= labels
    before = (output/'result.json').read_bytes()
    assert subprocess.run(command, capture_output=True).returncode == 1
    assert (output/'result.json').read_bytes() == before
    compare = [sys.executable, str(root/'scripts/compare_training_runs.py'), '--baseline', str(output/'result.json'),
               '--candidate', str(output/'result.json'), '--output', str(tmp_path/'compare.json')]
    assert subprocess.run(compare, capture_output=True).returncode == 0
    assert subprocess.run(compare, capture_output=True).returncode == 1


def test_nonfinite_and_oom_machine_status(monkeypatch):
    from nemochat import training_perf as perf
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    def bad(*args, **kwargs):
        raise torch.OutOfMemoryError('test fixture OOM')
    monkeypatch.setattr(perf, '_step', bad)
    result = run_benchmark(smoke())
    assert result['status'] == 'failed' and result['failure_kind'] == 'out-of-memory'
    assert torch.equal(state, torch.get_rng_state()) and threads == torch.get_num_threads()
    def nan(*args, **kwargs):
        return torch.tensor(float('nan')), torch.tensor(True), torch.tensor(1.)
    monkeypatch.setattr(perf, '_step', nan)
    result = run_benchmark(smoke())
    assert result['status'] == 'failed' and result['failure_kind'] == 'nonfinite-or-no-update'


def test_blocked_cuda(tmp_path):
    if torch.cuda.is_available():
        pytest.skip('absence-path test requires CPU-only host')
    c = smoke(); c['workload']['device'] = 'cuda'; c['resources']['max_gpu_count'] = 1
    config = tmp_path/'cuda.json'; config.write_text(json.dumps(c))
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'blocked'
    p = subprocess.run([sys.executable, str(root/'scripts/benchmark_training.py'), '--config', str(config),
                        '--output', str(output)], capture_output=True)
    assert p.returncode == 3
    result = json.loads((output/'result.json').read_text())
    assert result['status'] == 'blocked' and result['evidence_scope'] is None
    assert result['repeats'] == []


def test_estimates():
    estimate = workload_estimates(smoke()['workload'])
    qkv = estimate['operators'][0]
    assert qkv['forward_flops_per_instance'] == 2*32*96*32
    assert qkv['minimum_bytes_per_instance'] == 4*(32*32+32*96+32*96)
    assert 'not measured DRAM' in estimate['byte_model']
    assert estimate['forward_backward_projection_flops_per_update'] == 3*estimate['forward_projection_flops_per_update']


def test_checkpoint_math_actual():
    w = smoke()['workload']
    a = DenseCausalTransformer(w)
    w = dict(w, activation_checkpointing=True)
    b = DenseCausalTransformer(w); b.load_state_dict(a.state_dict())
    tokens = torch.arange(32).reshape(2, 16) % 128
    ya, yb = a(tokens), b(tokens)
    torch.testing.assert_close(ya, yb, rtol=0, atol=0)
    ya.square().mean().backward(); yb.square().mean().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)


def test_cpu_bfloat16_rejected_by_shared_contract():
    c = smoke(); c['workload']['dtype'] = 'bfloat16'
    with pytest.raises(ValueError, match='CPU bfloat16'):
        run_benchmark(c)


def test_budget_failure_machine_status(monkeypatch):
    from nemochat import training_perf as perf
    def exhausted(*args, **kwargs):
        raise perf.BenchmarkTimeout('fixture cooperative deadline')
    monkeypatch.setattr(perf, '_step', exhausted)
    result = run_benchmark(smoke())
    assert result['status'] == 'failed'
    assert result['failure_kind'] == 'time-budget-exceeded'


@pytest.mark.parametrize('field,value', [('correctness', None), ('profile', None), ('repeats', [None, None])])
def test_malformed_artifact(actual, field, value):
    a, b = deepcopy(actual)
    b[field] = value
    with pytest.raises(ValueError):
        compare_results(a, b)


@pytest.mark.parametrize('kind,exit_code', [
    ('out-of-memory', 3), ('time-budget-exceeded', 4),
    ('nonfinite-or-no-update', 1), ('execution-error', 1),
])
def test_direct_cli_typed_failure_exit(tmp_path, monkeypatch, capsys, kind, exit_code):
    import runpy
    from nemochat import training_perf as perf
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(smoke()))
    monkeypatch.setattr(perf, 'run_benchmark', lambda *args: {
        'status': 'failed', 'failure_kind': kind, 'evidence_scope': 'CPU-model-executed'})
    monkeypatch.setattr(sys, 'argv', ['benchmark_training.py', '--config', str(config),
                                    '--output', str(tmp_path / 'measurement')])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(root / 'scripts/benchmark_training.py'), run_name='__main__')
    assert exc.value.code == exit_code
    captured = capsys.readouterr()
    assert json.loads(captured.out)['failure_kind'] == kind
    assert captured.err == ''


@pytest.mark.parametrize('script', ['benchmark_training.py', 'compare_training_runs.py'])
@pytest.mark.parametrize('payload', [
    b'{"x":1,"x":2}', b'{"x":{"y":1,"y":2}}', b'{"x":NaN}',
    b'{"x":Infinity}', b'{"x":-Infinity}', b'{"x":1e999}', b'{', b'\xff',
    b'[' * 2000 + b'0' + b']' * 2000, b'{"x":' + b'9' * 5000 + b'}',
])
def test_direct_cli_strict_json_rejection(tmp_path, monkeypatch, capsys, script, payload):
    import runpy
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'bad.json'
    source.write_bytes(payload)
    output = tmp_path / 'output'
    arguments = ['--config', str(source)] if script == 'benchmark_training.py' else [
        '--baseline', str(source), '--candidate', str(source)]
    monkeypatch.setattr(sys, 'argv', [script, *arguments, '--output', str(output)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(root / 'scripts' / script), run_name='__main__')
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert 'rejected:' in captured.err and 'Traceback' not in captured.err
    assert captured.out == '' and not output.exists()


def test_comparison_huge_measurement_integer_clean_rejection(actual, tmp_path, monkeypatch, capsys):
    import runpy
    a, b = deepcopy(actual)
    b['repeats'][0]['elapsed_seconds'] = 10**400
    baseline, candidate = tmp_path / 'baseline.json', tmp_path / 'candidate.json'
    baseline.write_text(json.dumps(a)); candidate.write_text(json.dumps(b))
    output = tmp_path / 'comparison.json'
    monkeypatch.setattr(sys, 'argv', ['compare_training_runs.py', '--baseline', str(baseline),
                                    '--candidate', str(candidate), '--output', str(output)])
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(root / 'scripts/compare_training_runs.py'), run_name='__main__')
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert 'comparison rejected:' in captured.err and 'Traceback' not in captured.err
    assert not output.exists()


@pytest.mark.parametrize('script,limit', [
    ('benchmark_training.py', 32768), ('compare_training_runs.py', 8 * 1024 * 1024),
])
def test_direct_cli_json_size_limit(tmp_path, monkeypatch, capsys, script, limit):
    import runpy
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'large.json'
    source.write_bytes(b' ' * (limit + 1))
    output = tmp_path / 'output'
    arguments = ['--config', str(source)] if script == 'benchmark_training.py' else [
        '--baseline', str(source), '--candidate', str(source)]
    monkeypatch.setattr(sys, 'argv', [script, *arguments, '--output', str(output)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(root / 'scripts' / script), run_name='__main__')
    assert exc.value.code == 1
    assert f'exceeds {limit} bytes' in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA full-update timing/allocator/checkpoint gate requires authorized GPU')
def test_cuda_actual():
    c = smoke(checkpoint=True); c['workload']['device'] = 'cuda'; c['resources']['max_gpu_count'] = 1
    result = run_benchmark(c)
    assert result['status'] == 'success'
    assert result['evidence_scope'] == 'GPU-smoke-executed'
    assert result['repeats'][0]['memory']['peak_allocated_bytes'] > 0
