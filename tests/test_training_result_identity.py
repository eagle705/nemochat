"""Comparison identity validation; descriptors are user-controlled, not authentication."""
from copy import deepcopy
from pathlib import Path

import pytest

from nemochat.experiments import load_config
from nemochat.training_perf import compare_results, run_benchmark


ENV_FIELDS = (
    'python', 'torch', 'cuda_runtime', 'cudnn', 'platform', 'machine', 'processor',
    'device', 'cpu_threads', 'deterministic_algorithms', 'matmul_precision',
    'cuda_matmul_allow_tf32', 'cudnn_allow_tf32', 'implementation_sha256', 'gpu',
)
PROTOCOL_FIELDS = (
    'version', 'optimizer', 'model', 'timing', 'repeat_state', 'tokens', 'dtype',
    'cpu_memory', 'cuda_memory', 'budget',
)
GPU_FIELDS = (
    'name', 'index', 'compute_capability', 'total_memory_bytes',
    'multiprocessor_count', 'driver_version',
)


@pytest.fixture(scope='module')
def actual_cpu():
    template = Path(__file__).resolve().parents[1] / 'configs/experiments/dense_cpu.yaml'
    config = load_config(template).to_dict()
    config['workload'].update(batch_size=1, sequence_length=2, hidden_size=4,
                              ffn_size=8, layers=1, vocab_size=8)
    config['measurement'].update(warmup_steps=0, measured_steps=1, repeats=1, profile=False)
    result = run_benchmark(config)
    assert result['status'] == 'success', result
    return result


def reject_symmetric(result):
    with pytest.raises(ValueError):
        compare_results(deepcopy(result), deepcopy(result))


def test_actual_tiny_cpu_accepted(actual_cpu):
    assert set(actual_cpu['environment']) == set(ENV_FIELDS)
    assert set(actual_cpu['protocol']) == set(PROTOCOL_FIELDS)
    assert compare_results(actual_cpu, deepcopy(actual_cpu))['status'] == 'success'


def test_reviewer_stripped_identity(actual_cpu):
    result = deepcopy(actual_cpu)
    result['environment'] = {k: result['environment'][k]
                             for k in ('device', 'cpu_threads', 'python', 'torch', 'gpu')}
    result['protocol'] = {'version': 1}
    reject_symmetric(result)


@pytest.mark.parametrize('field', ENV_FIELDS)
def test_symmetric_missing_environment_field(actual_cpu, field):
    result = deepcopy(actual_cpu)
    del result['environment'][field]
    reject_symmetric(result)


@pytest.mark.parametrize('field', PROTOCOL_FIELDS)
def test_symmetric_missing_protocol_field(actual_cpu, field):
    result = deepcopy(actual_cpu)
    del result['protocol'][field]
    reject_symmetric(result)


@pytest.mark.parametrize('field,value', [
    ('implementation_sha256', None), ('implementation_sha256', 'f' * 63),
    ('implementation_sha256', 'g' * 64), ('implementation_sha256', 'A' * 64),
    ('python', ''), ('torch', 1), ('platform', ''), ('machine', None), ('processor', 0),
    ('cuda_runtime', 12), ('cuda_runtime', ''), ('cudnn', True), ('cudnn', 0),
    ('device', 'cuda:0'), ('cpu_threads', True), ('cpu_threads', 0), ('cpu_threads', 3),
    ('deterministic_algorithms', 1), ('matmul_precision', 'unknown'),
    ('cuda_matmul_allow_tf32', 0), ('cudnn_allow_tf32', 'false'), ('gpu', {}),
])
def test_invalid_environment_types_and_cpu_consistency(actual_cpu, field, value):
    result = deepcopy(actual_cpu)
    result['environment'][field] = value
    reject_symmetric(result)


@pytest.mark.parametrize('field', PROTOCOL_FIELDS)
@pytest.mark.parametrize('value', [None, True, 'unknown protocol'])
def test_unknown_protocol_values(actual_cpu, field, value):
    result = deepcopy(actual_cpu)
    result['protocol'][field] = value
    reject_symmetric(result)


def test_nullable_cpu_runtime_empty_processor_and_extensions(actual_cpu):
    result = deepcopy(actual_cpu)
    result['environment'].update(cuda_runtime=None, cudnn=None, processor='', extra='annotation')
    result['protocol']['extra'] = 'annotation'
    assert compare_results(result, deepcopy(result))['status'] == 'success'
    # CPU execution can also use a CUDA-enabled torch build without claiming GPU execution.
    result['environment'].update(cuda_runtime='12.8', cudnn=91002)
    assert compare_results(result, deepcopy(result))['status'] == 'success'


@pytest.fixture
def synthetic_gpu(actual_cpu):
    """Schema-only mutation, never evidence that CUDA executed."""
    result = deepcopy(actual_cpu)
    result['config']['workload']['device'] = 'cuda'
    result['config']['resources']['max_gpu_count'] = 1
    result['evidence_scope'] = 'GPU-smoke-executed'
    result['environment'].update(device='cuda:0', cuda_runtime='12.8', cudnn=None, gpu={
        'name': 'synthetic-schema-fixture', 'index': 0, 'compute_capability': [9, 0],
        'total_memory_bytes': 1024, 'multiprocessor_count': 1,
        'driver_version': 'not queried (no subprocess or environment capture)',
    })
    for row in result['repeats']:
        row['memory'] = {'status': 'CUDA-allocator-measured',
                         'peak_allocated_bytes': 1, 'peak_reserved_bytes': 2}
    return result


def test_synthetic_gpu_schema_accepted_without_authentication(synthetic_gpu):
    assert compare_results(synthetic_gpu, deepcopy(synthetic_gpu))['status'] == 'success'


@pytest.mark.parametrize('field', GPU_FIELDS)
def test_symmetric_missing_gpu_field(synthetic_gpu, field):
    del synthetic_gpu['environment']['gpu'][field]
    reject_symmetric(synthetic_gpu)


@pytest.mark.parametrize('field,value', [
    ('name', ''), ('index', True), ('index', 1), ('compute_capability', [9]),
    ('compute_capability', [True, 0]), ('compute_capability', [0, 0]),
    ('compute_capability', [9, -1]), ('compute_capability', '9.0'),
    ('total_memory_bytes', True), ('total_memory_bytes', 0),
    ('multiprocessor_count', 0), ('driver_version', None),
])
def test_invalid_gpu_schema(synthetic_gpu, field, value):
    synthetic_gpu['environment']['gpu'][field] = value
    reject_symmetric(synthetic_gpu)


@pytest.mark.parametrize('field,value', [('gpu', None), ('gpu', {}), ('cuda_runtime', None)])
def test_gpu_runtime_consistency(synthetic_gpu, field, value):
    synthetic_gpu['environment'][field] = value
    reject_symmetric(synthetic_gpu)
