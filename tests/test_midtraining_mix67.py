"""CPU-only approved blend controls; not native execution evidence."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from test_midtraining_longrun import module, prepared, ROOT

NEW = (.67, .1, .03, .2)
OLD_PROFILE_SHA = '52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308'


def test_versioned_config_freezes_approved_budget_with_honest_profile_label():
    m = module()
    c = m.load_config(ROOT/'midtraining/configs/longrun24h.mix67.yaml')
    assert c['name'] == c['logger']['wandb_exp_name'] == 'nemochat baby midtraining super35 24h'
    assert m.dataset_weights(c) == NEW
    assert c['train']['train_iters'] == 3200
    # Measured old-profile mean; fixture is not a fresh new-mix benchmark.
    stats = dict(warm_updates=list(range(9, 17)), profiled=False,
                 global_batch_size=128, sequence_length=4096,
                 mean_step_seconds=24.29540627438837)
    frozen = m.freeze_config(c, stats)
    assert frozen['train']['train_iters'] == 3200
    assert frozen['scheduler']['lr_warmup_iters'] == 32
    assert frozen['scheduler']['lr_wsd_decay_iters'] == 320
    assert frozen['projection']['timing_recipe_weights'] == list(m.WEIGHTS)
    assert frozen['projection']['production_weights'] == list(NEW)
    assert frozen['projection']['fresh_new_mix_benchmark'] is False
    assert frozen['projection']['actual_new_main_warm_steps_pending'] is True
    launcher = module('launch_longrun')
    assert 'midtraining/configs/longrun24h.mix67.yaml' in launcher.SOURCES
    assert 'midtraining/configs/longrun24h.yaml' in launcher.SOURCES


def test_execute_checks_projection_before_any_training_side_effect(monkeypatch, tmp_path):
    from types import SimpleNamespace
    launcher = module('launch_longrun')
    m = module()
    cpath = ROOT/'midtraining/configs/longrun24h.mix67.yaml'
    manifest = prepared(tmp_path/'pairs', tokens=2_000_000_000)['result']
    manifest['weights'] = dict(zip(m.KEYS, NEW))
    target = 'b'*64
    accepted = dict(status='calibration_save_reload_accepted', all_ranks=8,
        manifest_sha256=OLD_PROFILE_SHA, calibration=dict(warm_updates=list(range(9, 17)),
        profiled=False, global_batch_size=128, sequence_length=4096,
        mean_step_seconds=24.29540627438837))
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps(accepted))
    control = tmp_path/'control.json'
    control.write_text(json.dumps(dict(status='duration_control_save_reload_accepted', all_ranks=8)))
    compat = tmp_path/'compat.json'
    compat.write_text(json.dumps(compatibility(m, target)))
    monkeypatch.setenv('NEMOCHAT_LONGRUN_SUPERVISED', '1')
    monkeypatch.setattr(launcher, 'ROOT', tmp_path)
    monkeypatch.setattr(launcher, 'source_gate', lambda *a: None)
    monkeypatch.setattr(launcher, 'native', lambda: m)
    monkeypatch.setattr(m, 'load_prepared_manifest', lambda *a: (manifest, target))
    monkeypatch.setattr(m, 'audit_reserved_ids', lambda *a: {})
    monkeypatch.setattr(m, 'corpus_tokenizer_gate', lambda *a: None)
    calls = []
    class StopBeforeExternalAction(Exception): pass
    def capacity(value, updates, batch, sequence):
        calls.append((value, updates, batch, sequence))
        raise StopBeforeExternalAction
    monkeypatch.setattr(m, 'capacity_gate', capacity)
    args = SimpleNamespace(config=cpath, reviewed_sources=tmp_path/'review.json',
        duration_control=False, manifest=tmp_path/'manifest.json', run_id='test-mix67-01',
        calibration=calibration, duration_control_receipt=control, calibration_compatibility=compat)
    with pytest.raises(StopBeforeExternalAction):
        launcher.execute(args)
    assert calls == [(manifest, 3200, 128, 4096)]
    args.run_id = 'test-mix67-02'
    compat.write_text(json.dumps(dict(compatibility(m), fresh_new_mix_benchmark=True)))
    with pytest.raises(ValueError, match='compatibility'):
        launcher.execute(args)
    assert len(calls) == 1


def compatibility(m, target='b'*64):
    return dict(status='parent_accepted_approved_mix_change_profile_projection',
        calibration_manifest_sha256=OLD_PROFILE_SHA, production_manifest_sha256=target,
        old_weights=list(m.WEIGHTS), new_weights=list(NEW),
        same_provider_tokenizer_sequence_batch_topology=True,
        heldout_exclusions_preserved=True, user_approved_mix_change=True,
        projection_only=True, fresh_new_mix_benchmark=False)


def test_approved_projection_compatibility_is_exact_not_same_recipe():
    launcher = module('launch_longrun')
    m = module()
    accepted = dict(manifest_sha256=OLD_PROFILE_SHA)
    expected = compatibility(m)
    assert launcher.validate_calibration_compatibility(accepted, 'b'*64, NEW, expected) == expected
    for key, bad in [('status', 'parent_accepted_same_recipe_expansion'),
                     ('old_weights', list(NEW)), ('new_weights', list(m.WEIGHTS)),
                     ('production_manifest_sha256', 'c'*64),
                     ('same_provider_tokenizer_sequence_batch_topology', False),
                     ('heldout_exclusions_preserved', False), ('projection_only', False),
                     ('fresh_new_mix_benchmark', True), ('user_approved_mix_change', False)]:
        value = dict(expected, **{key: bad})
        with pytest.raises(ValueError, match='compatibility'):
            launcher.validate_calibration_compatibility(accepted, 'b'*64, NEW, value)
    with pytest.raises(ValueError, match='compatibility'):
        launcher.validate_calibration_compatibility(accepted, 'b'*64, NEW, None)
    with pytest.raises(ValueError, match='compatibility'):
        launcher.validate_calibration_compatibility(dict(manifest_sha256='c'*64), 'b'*64, NEW, expected)
    legacy = dict(status='parent_accepted_same_recipe_expansion',
        calibration_manifest_sha256=OLD_PROFILE_SHA, production_manifest_sha256='b'*64,
        same_provider_tokenizer_sequence_batch_topology=True, heldout_exclusions_preserved=True)
    assert launcher.validate_calibration_compatibility(accepted, 'b'*64, m.WEIGHTS, legacy) == legacy
    assert launcher.validate_calibration_compatibility(accepted, OLD_PROFILE_SHA, m.WEIGHTS, None) is None
    with pytest.raises(ValueError, match='compatibility'):
        launcher.validate_calibration_compatibility(accepted, OLD_PROFILE_SHA, NEW, expected)


def test_actual_observer_receipt_keeps_new_weights_and_one_epoch(monkeypatch):
    import sys
    import numpy as np
    from types import ModuleType, SimpleNamespace
    m = module()
    c = m.load_config(ROOT/'midtraining/configs/longrun24h.mix67.yaml')
    class GPT: pass
    config = ModuleType('megatron.bridge.training.config')
    config.GPTDatasetConfig = GPT
    data = ModuleType('megatron.bridge.data')
    # Synthetic native-index contract fixture, not a measured builder result.
    index = np.array([0]*86 + [1]*13 + [2]*4 + [3]*25)
    train = SimpleNamespace(dataset_index=index, datasets=[
        SimpleNamespace(indices=[0, 1], document_index=[1, 0]) for _ in range(4)])
    original = lambda samples, cfg: (train, None, None)
    data.utils = SimpleNamespace(_REGISTRY={GPT: original})
    monkeypatch.setitem(sys.modules, 'megatron.bridge.training.config', config)
    monkeypatch.setitem(sys.modules, 'megatron.bridge.data', data)
    receipts = []
    restore = m.install_native_dataset_audit(c, lambda *a: receipts.append(a))
    try:
        data.utils._REGISTRY[GPT]([128, 8, 0], None)
        receipt = receipts[-1][1]
        assert receipt['fixed_sequence_exposure_weights'] == list(NEW)
        assert receipt['native_train_epochs'] == 1
        assert all(x > 0 for x in receipt['first_global_batch_source_samples'].values())
        train.datasets[2].document_index = [0, 1, 0, 1]
        with pytest.raises(ValueError, match='repeat'):
            data.utils._REGISTRY[GPT]([128, 8, 0], None)
    finally:
        restore()
    assert data.utils._REGISTRY[GPT] is original


def test_native_builder_and_observer_use_actual_mix(monkeypatch, tmp_path):
    import sys
    from types import ModuleType, SimpleNamespace
    m = module()
    c = m.load_config(ROOT/'midtraining/configs/calibration.yaml')
    c['dataset']['weights'] = list(NEW)
    c['checkpoint_save'] = 'checkpoints/midtraining/fixture'
    manifest = prepared(tmp_path)['result']
    manifest['weights'] = dict(zip(m.KEYS, NEW))
    class Config:
        def __init__(self, **kw): self.__dict__.update(kw)
    config = ModuleType('megatron.bridge.training.config')
    config.GPTDatasetConfig = config.CheckpointConfig = Config
    config.runtime_config_update = lambda c: None
    monkeypatch.setitem(sys.modules, 'megatron.bridge.training.config', config)
    monkeypatch.setattr(m, 'helper', lambda *a: SimpleNamespace(
        make_config=lambda c, ds, out, **kw: SimpleNamespace(dataset=ds())))
    cfg, _ = m.make_native_config(c, manifest, tmp_path)
    assert cfg.dataset.blend_per_split[0][1] == list(NEW)
    assert cfg.dataset.blend_per_split[1][1] == list(NEW)
    assert cfg.checkpoint.save_optim and cfg.checkpoint.save_rng
    calls = []
    monkeypatch.setattr(m, 'helper', lambda *a: SimpleNamespace(install=lambda *a, **kw: calls.append(kw)))
    m.install_native_dataset_audit(c, lambda *a: None)
    assert calls[0]['weights'] == NEW
    manifest['weights'] = dict(zip(m.KEYS, m.WEIGHTS))
    with pytest.raises(ValueError, match='weights.*manifest'):
        m.make_native_config(c, manifest, tmp_path)


def test_mix67_loader_accepts_exact_blend_and_companion_only(tmp_path):
    m = module()
    value = prepared(tmp_path)['result']
    value.update(schema_version=1, status='prepared_verified', sequence_length=4096,
                 source_order=list(m.KEYS), weights=dict(zip(m.KEYS, NEW)))
    path = tmp_path/'manifest.json'
    def save():
        path.write_text(json.dumps(value))
        (tmp_path/'success.json').write_text(json.dumps(dict(phase='prepared_verified',
            result=dict(manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))))
    save()
    manifest, digest = m.load_prepared_manifest(path)
    assert m.manifest_weights(manifest) == NEW
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    value['weights']['formal_logic'] = .04
    save()
    with pytest.raises(ValueError, match='weights'):
        m.load_prepared_manifest(path)
    (tmp_path/'success.json').write_text(json.dumps(dict(phase='failed', result=value)))
    with pytest.raises(ValueError, match='SHA256'):
        m.load_prepared_manifest(path)


def test_adapter_does_not_promote_failed_companion_status(tmp_path):
    m = module()
    value = prepared(tmp_path)
    value['status'] = 'failed'
    with pytest.raises(ValueError, match='status'):
        m.adapt_prepared_manifest(value)


def test_loader_rejects_failed_nested_companion_and_nonmap_weights(tmp_path):
    m = module()
    value = prepared(tmp_path)['result']
    value.update(schema_version=1, status='prepared_verified', sequence_length=4096,
                 source_order=list(m.KEYS), weights=dict(zip(m.KEYS, NEW)))
    path = tmp_path/'manifest.json'
    def save(status):
        path.write_text(json.dumps(value))
        (tmp_path/'success.json').write_text(json.dumps(dict(phase='prepared_verified',
            result=dict(status=status, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))))
    save('failed')
    with pytest.raises(ValueError, match='status'):
        m.load_prepared_manifest(path)
    value['weights'] = list(NEW)
    save('prepared_verified')
    with pytest.raises(ValueError, match='weights'):
        m.load_prepared_manifest(path)


def test_approved_mix_config_capacity_and_exact_manifest_match(tmp_path):
    m = module()
    c = m.load_config(ROOT/'midtraining/configs/longrun24h.yaml')
    c['dataset']['weights'] = list(NEW)
    m.validate_config(c)
    manifest = prepared(tmp_path, tokens=2_000_000_000)['result']
    manifest['weights'] = dict(zip(m.KEYS, NEW))
    manifest['sources']['formal_logic']['train']['tokens'] = 60_564_276
    assert m.dataset_weights(c, manifest) == NEW
    capacity = m.capacity_gate(manifest, 3200, 128, 4096)
    assert capacity['sources']['formal_logic']['fixed_sequence_exposure_weight'] == .03
    assert capacity['sources']['code_concepts']['fixed_sequence_exposure_weight'] == .67
    old = copy.deepcopy(manifest)
    old['weights'] = dict(zip(m.KEYS, m.WEIGHTS))
    with pytest.raises(ValueError, match='formal_logic'):
        m.capacity_gate(old, 3200, 128, 4096)
    with pytest.raises(ValueError, match='weights.*manifest'):
        m.dataset_weights(c, old)
    c['dataset']['weights'] = [.66, .1, .04, .2]
    with pytest.raises(ValueError, match='approved.*weights'):
        m.validate_config(c)
