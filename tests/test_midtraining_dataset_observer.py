"""The native registry observation adapter stays in the user-requested patch directory."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def test_registry_adapter_is_isolated_and_receives_native_audit_helpers(monkeypatch):
    spec = importlib.util.spec_from_file_location('observer_parent', ROOT/'midtraining/longrun_native.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    restored = object()
    def install(*args, **kwargs):
        calls.append((args, kwargs))
        return restored
    def load(name, path):
        assert path == 'midtraining/patches/native_dataset_observer.py'
        return SimpleNamespace(install=install)
    monkeypatch.setattr(module, 'helper', load)
    config, publish = {}, lambda *args: None
    assert module.install_native_dataset_audit(config, publish, True) is restored
    args, kwargs = calls[0]
    assert args == (config, publish, True)
    assert kwargs['no_repeat_gate'] is module.no_repeat_gate
    assert kwargs['evaluation_budget_with_startup'] is module.evaluation_budget_with_startup
    assert kwargs['keys'] == module.KEYS
    assert kwargs['weights'] == module.WEIGHTS


def test_owned_engine_allows_inherited_image_labels_but_not_foreign_ownership():
    import copy
    import pytest
    spec = importlib.util.spec_from_file_location('ownership_parent', ROOT/'midtraining/launch_longrun.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels = {'nemochat.stage': 'midtraining-longrun', 'nemochat.run': 'fresh-01',
              'nemochat.segment': 'config', 'com.nvidia.build.id': '443194436'}
    actual = {'Id': 'exact-id', 'Name': '/nemochat-midtraining-fresh-01-config',
              'Config': {'Labels': labels}}
    module.check_owned(actual, 'exact-id', 'fresh-01', 'config')
    for key in ('nemochat.stage', 'nemochat.run', 'nemochat.segment'):
        foreign = copy.deepcopy(actual)
        foreign['Config']['Labels'][key] = 'foreign'
        with pytest.raises(ValueError, match='identity'):
            module.check_owned(foreign, 'exact-id', 'fresh-01', 'config')
    for key in ('Id', 'Name'):
        foreign = copy.deepcopy(actual)
        foreign[key] = 'foreign'
        with pytest.raises(ValueError, match='identity'):
            module.check_owned(foreign, 'exact-id', 'fresh-01', 'config')
