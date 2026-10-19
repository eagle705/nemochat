"""The run-bound opt-in registry must never blanket-skip portable tests."""
import importlib.util
from pathlib import Path


def policy():
    path = Path(__file__).with_name('conftest.py')
    assert path.is_file(), 'explicit historical collection policy is missing'
    spec = importlib.util.spec_from_file_location('historical_policy_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_matching_is_exact_except_parameter_suffix():
    module = policy()
    contracts = {'tests/test_example.py::TestA::test_saved': 'requires saved receipt'}
    assert module.historical_reason('tests/test_example.py::TestA::test_saved[x]', contracts)
    assert module.historical_reason('tests/test_example.py::TestA::test_saved', contracts)
    assert module.historical_reason('tests/test_example.py::TestA::test_saved_other', contracts) is None
    assert module.historical_reason('tests/test_example.py::TestB::test_saved', contracts) is None
    assert module.historical_reason('tests/test_other.py::TestA::test_saved', contracts) is None


PORTABLE_GUARDS = (
    'tests/test_offline1920_directio.py::test_activate_requires_opt_in',
    'tests/test_offline1920_directio.py::test_actual_statx_alignment_fail_closed',
    'tests/test_offline1920_directio.py::test_gate_native_dataset_is_available',
)


def test_known_portable_guards_are_not_historical():
    import json
    contracts = json.loads(Path(__file__).with_name('historical_contracts.json').read_text())['tests']
    assert not set(PORTABLE_GUARDS).intersection(contracts)


def test_historical_registry_ids_resolve_to_collected_tests():
    import json
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q', 'tests'],
        cwd=root, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                         TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled'),
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    collected = {line.split('[', 1)[0] for line in result.stdout.splitlines()
                 if line.startswith('tests/') and '::' in line}
    contracts = json.loads((root / 'tests/historical_contracts.json').read_text())['tests']
    assert set(contracts) <= collected, sorted(set(contracts) - collected)
