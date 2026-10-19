"""Reproduce the installed provider's cross-architecture mutation."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/midtraining/super35-kd-01/sdk-source/megatron.bridge/models/distillation_provider.py'


def test_finalizing_student_must_not_overwrite_teacher_num_layers():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DistillationProvider')
    setter = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__setattr__')
    # Exact native setter body in a small class, not a reimplementation of it.
    minimal = ast.ClassDef(name='NativeProvider', bases=[], keywords=[], body=[setter], decorator_list=[])
    namespace = {}; exec(compile(ast.fix_missing_locations(ast.Module(body=[minimal], type_ignores=[])), str(SOURCE), 'exec'), namespace)
    provider_class = namespace['NativeProvider']
    student = provider_class(); student.teacher = SimpleNamespace(num_layers=88, seq_length=4096)
    student.num_layers = 52
    assert student.teacher.num_layers == 52  # deterministic native bug reproduction
    student.teacher.num_layers = 88
    patch_path = ROOT / 'midtraining/patches/distillation_provider_shape_guard.py'
    assert patch_path.exists(), 'isolated native provider shape patch is missing'
    spec = importlib.util.spec_from_file_location('shape_patch', patch_path)
    patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
    patch.apply(provider_class, source_path=SOURCE)
    student.num_layers = 52
    assert student.num_layers == 52
    assert student.teacher.num_layers == 88
    student.seq_length = 8192
    assert student.teacher.seq_length == 8192
