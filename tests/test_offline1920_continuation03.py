"""Continuation03 sparse repair and immutable cache contract tests."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

def load_contract():
    path = ROOT / 'midtraining/training/offline_kd/prefix/continuation03/repair_contract.py'
    spec = importlib.util.spec_from_file_location('repair03', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_sparse_repair_uses_only_requested_native_global_cursors():
    module = load_contract()
    assert module.repair_plan([70, 240, 70]) == [
        {'iteration': 70, 'consumed_samples': 8960, 'microbatches': 16},
        {'iteration': 240, 'consumed_samples': 30720, 'microbatches': 16},
    ]
    for invalid in ([], [-1], [1143], [70.5], [True]):
        with pytest.raises(ValueError):
            module.repair_plan(invalid)

def test_overlay_requires_every_rank_for_repaired_iteration(tmp_path):
    module = load_contract()
    original = tmp_path/'original'; original.mkdir()
    repairs = tmp_path/'repairs'; repairs.mkdir()
    for rank in range(8):
        for iteration in range(3):
            (original/f'cp0_dp{rank}__{iteration}.tar').write_bytes(b'original')
    interval = repairs/'iteration-1'; interval.mkdir()
    for rank in range(7):
        (interval/f'cp0_dp{rank}__1.tar').write_bytes(b'repaired')
    overlay = tmp_path/'overlay'
    with pytest.raises(FileNotFoundError):
        module.build_overlay(original, repairs, overlay, [1], stop=3)
    assert not overlay.exists()
    (interval/'cp0_dp7__1.tar').write_bytes(b'repaired')
    manifest = module.build_overlay(original, repairs, overlay, [1], stop=3)
    assert len(manifest) == 24
    assert sum(row['regenerated'] for row in manifest) == 8
    for row in manifest:
        path = overlay/row['name']
        assert path.is_symlink()
        assert path.read_bytes() == (b'repaired' if row['regenerated'] else b'original')
    bad_only = tmp_path/'bad-only'
    manifest = module.build_overlay(original, repairs, bad_only, [1], stop=3,
        repaired_names={'cp0_dp7__1.tar'})
    assert sum(row['regenerated'] for row in manifest) == 1
    assert (bad_only/'cp0_dp0__1.tar').read_bytes() == b'original'
    assert (bad_only/'cp0_dp7__1.tar').read_bytes() == b'repaired'

def test_sparse_writer_attaches_to_unwrapped_native_output_layer():
    module = load_contract()
    class Native:
        output_layer = object()
    class Float16Wrapper:
        module = Native()
    class Saver:
        attached = None
        def attach_hooks(self, model):
            assert hasattr(model, 'output_layer')
            self.attached = model
    saver = Saver(); wrapped = Float16Wrapper()
    module.attach_native_writer(saver, wrapped, lambda model: model.module)
    assert saver.attached is wrapped.module


def test_student_preserves_original_resume_and_full_schedule():
    import ast
    root = ROOT/'midtraining/training/offline_kd/prefix/continuation03'
    source = (root/'native.py').read_text()
    ast.parse(source)
    assert 'start=0' in source
    assert 'optimizer_reset=True,rng_reset=True,cursor_reset=True' in source
    assert 'if resume:raise ValueError' in source
    assert "STOP=1143" in source
    assert "lr_decay_iters=C['train_iters']" in source
    assert "lr_warmup_iters=C['warmup_iters']" in source
    assert "lr_wsd_decay_iters=C['decay_iters']" in source
    assert "CachedKDConfig('read',str(CACHE),cache_id())" in source
    assert "CACHE=Path('/output/cache-overlay')" in source
    assert "load_optim=resume,load_rng=resume" in source
    assert "if not teacher and row!=expected[i]:raise ValueError" in source
    repair = (root/'repair_native.py').read_text()
    assert 'consumed_samples=row[\'consumed_samples\']' in repair
    assert "if sample != expected[sample['micro']]" in repair
    assert 'LogitsSaverHooks(str(cache), k=32, p=None, min_k=1, save_dtype=\'fp16\')' in repair
    assert 'with torch.no_grad():' in repair
    assert '.backward(' not in repair and 'optimizer.step' not in repair
