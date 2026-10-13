"""YAML and direct JSON entry points must enforce the same experiment contract."""
from copy import deepcopy
from pathlib import Path
import pytest
from nemochat.experiments import load_config, validate_config as yaml_validate
from nemochat.training_perf import validate_config as benchmark_validate


@pytest.mark.parametrize('section,key,value', [
    ('workload', 'dtype', 'bfloat16'),
    ('workload', 'batch_size', 100000),
    ('workload', 'vocab_size', 1),
    ('workload', 'hidden_size', 3),
    ('resources', 'max_gpu_count', 1),
    ('resources', 'max_seconds', 3601),
    ('experiment', 'name', 'bad name'),
    ('experiment', 'seed', 2**32),
    ('measurement', 'measured_steps', 1001),
    ('workload', 'learning_rate', 1.01),
])
def test_invalid_contract_is_rejected_by_both_entry_points(section, key, value):
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / 'configs/experiments/dense_cpu.yaml').to_dict()
    config[section][key] = value
    for validator in (yaml_validate, benchmark_validate):
        with pytest.raises(ValueError):
            validator(deepcopy(config))


def test_valid_config_is_equal_through_both_entry_points():
    root = Path(__file__).resolve().parents[1]
    for name in ('dense_cpu.yaml', 'dense_cuda.yaml'):
        c = load_config(root / 'configs/experiments' / name).to_dict()
        assert yaml_validate(deepcopy(c)) == benchmark_validate(deepcopy(c))
