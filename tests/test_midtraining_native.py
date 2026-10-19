"""CPU behavioral contracts; these are not native/GPU execution evidence."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / 'midtraining/native_kd.py'
    assert path.exists(), 'native LM+KD entrypoint missing'
    spec = importlib.util.spec_from_file_location('native_kd', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_completed_update_index_uses_native_preincrement_callback_cursor():
    from types import SimpleNamespace
    import pytest
    runner = load_runner()
    assert hasattr(runner, 'completed_update_index'), 'native callback step accounting is missing'
    ctx = SimpleNamespace(state=SimpleNamespace(train_state=SimpleNamespace(step=0)))
    assert runner.completed_update_index(ctx, []) == 1
    ctx.state.train_state.step = 1
    assert runner.completed_update_index(ctx, [1]) == 2
    with pytest.raises(RuntimeError, match='contiguous'):
        runner.completed_update_index(ctx, [])


def test_frozen_pilot_contract():
    runner = load_runner()
    c = runner.load_config(ROOT / 'midtraining/pilot.yaml')
    assert c['train']['train_iters'] == 4
    assert c['optimizer']['lr'] == 1e-5
    assert c['optimizer']['min_lr'] == 1e-6
    assert c['kd']['skip_lm_loss'] is False
    assert c['kd']['intermediate_layer_pairs'] == []
    assert c['kd']['kd_loss_scale'] == .5
    assert c['kd']['logit_kl_temperature'] == 1.
    assert c['checkpoint_integrity'] == 'not_checked_explicit_user_waiver'
    assert c['logger']['wandb_project'] is None


def test_shifted_all_token_lm_and_kd_mask():
    import torch
    runner = load_runner()
    row = runner.make_sample([4, 11, 7, 8, 9], 4, 16)
    assert row['tokens'].tolist() == [4, 11, 7, 8]
    assert row['labels'].tolist() == [11, 7, 8, 9]
    assert row['position_ids'].tolist() == [0, 1, 2, 3]
    assert torch.equal(row['loss_mask'], torch.ones(4))
    # EOS is a real corpus target, not padding; loss stays enabled.
    assert row['loss_mask'][0].item() == 1


def test_token_semantic_gate_rejects_same_size_reassigned_ids(tmp_path):
    import json
    import pytest
    runner = load_runner()
    student = tmp_path / 'student'; teacher = tmp_path / 'teacher'
    student.mkdir(); teacher.mkdir()
    payload = {'model': {'type': 'BPE', 'vocab': {'a': 0, 'b': 1}, 'merges': []},
               'added_tokens': [], 'normalizer': None, 'pre_tokenizer': None,
               'post_processor': None, 'decoder': None}
    for folder in [student, teacher]:
        (folder / 'tokenizer.json').write_text(json.dumps(payload))
    assert runner.tokenizer_gate(student, teacher)['token_id_semantics_equal']
    payload['model']['vocab'] = {'a': 1, 'b': 0}
    (teacher / 'tokenizer.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='token-ID semantics'):
        runner.tokenizer_gate(student, teacher)


def test_teacher_frozen_and_metrics_gate():
    import torch
    import pytest
    from types import SimpleNamespace
    runner = load_runner()
    teacher = torch.nn.Linear(2, 3).requires_grad_(False)
    assert runner.check_teacher(teacher)['frozen_parameters'] == 9
    teacher.weight.grad = torch.zeros_like(teacher.weight)
    with pytest.raises(RuntimeError, match='teacher gradient'):
        runner.check_teacher(teacher)
    ctx = SimpleNamespace(loss_dict={'lm loss': torch.tensor(2.),
        'logits distillation loss': torch.tensor(.4), 'total loss': torch.tensor(3.)},
        grad_norm=1.2, skipped_iter=False)
    assert runner.update_metrics(ctx)['KD Loss'] == pytest.approx(.4)
    ctx.grad_norm = 0.
    with pytest.raises(FloatingPointError, match='observed_metrics=.*Grad Norm'):
        runner.update_metrics(ctx)


def test_native_balancer_has_kd_gradient_not_static_raw_kl():
    """Execute the installed-source class body, avoiding optional CUDA imports."""
    import ast
    import types
    import torch
    source = ROOT / 'results/midtraining/super35-kd-01/sdk-source/modelopt/torch/distill/plugins/megatron.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LogitsAndIntermediatesLossBalancer')
    namespace = {'Tensor': torch.Tensor, 'mtd': types.SimpleNamespace(
        DistillationLossBalancer=torch.nn.Module,
        loss_balancers=types.SimpleNamespace(STUDENT_LOSS_KEY='student_loss'))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    lm = torch.tensor(2., requires_grad=True); kd = torch.tensor(.4, requires_grad=True)
    output = namespace['LogitsAndIntermediatesLossBalancer'](kd_loss_scale=.5)(
        {'student_loss': lm, 'LogitsKLLoss_0': kd})
    assert output['kd_loss'].item() == 3.
    output['kd_loss'].backward()
    assert lm.grad.item() == 1.
    assert kd.grad.item() == 2.5


def test_runtime_options_preserve_independent_architectures():
    from types import SimpleNamespace
    import pytest
    runner = load_runner()
    c = runner.load_config(ROOT / 'midtraining/pilot.yaml')
    student = SimpleNamespace(hidden_size=1024, num_layers=52, vocab_size=131072)
    teacher = SimpleNamespace(hidden_size=4096, num_layers=88, vocab_size=131072)
    runner.configure_provider(student, c)
    runner.configure_provider(teacher, c)
    runner.check_provider_shapes(student, teacher)
    assert student.expert_model_parallel_size == teacher.expert_model_parallel_size == 8
    teacher.hidden_size = student.hidden_size
    with pytest.raises(RuntimeError, match='teacher shape'):
        runner.check_provider_shapes(student, teacher)


def test_cli_help_and_schema_are_cpu_supported():
    import subprocess
    import sys
    script = str(ROOT / 'midtraining/native_kd.py')
    result = subprocess.run([sys.executable, script, '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--reload-checkpoint' in result.stdout
    result = subprocess.run([sys.executable, script, '--schema-only'], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'GPU not executed' in result.stdout


def test_native_config_wiring_and_student_only_reload(monkeypatch, tmp_path):
    """Dependency doubles test wiring only; parent must exercise actual SDK."""
    import sys
    import types
    runner = load_runner()
    c = runner.load_config(ROOT / 'midtraining/pilot.yaml')
    calls = []

    class Bag:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def finalize(self):
            pass
        def register_pre_wrap_hook(self, hook, **kwargs):
            self.hooks = getattr(self, 'hooks', []) + [hook]

    class Container(Bag):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.dist = Bag()

    class KDProvider(Bag):
        pass

    def convert(student, teacher, kd):
        student.__class__ = KDProvider
        student.teacher = teacher
        student.kd_config = kd
        calls.append('convert')
        return student

    cfgmod = types.ModuleType('megatron.bridge.training.config')
    for name in ['TrainingConfig', 'OptimizerConfig', 'SchedulerConfig', 'LoggerConfig',
                 'TokenizerConfig', 'CheckpointConfig', 'DistributedDataParallelConfig',
                 'ValidationConfig', 'RNGConfig']:
        setattr(cfgmod, name, Bag)
    cfgmod.ConfigContainer = Container
    cfgmod.runtime_config_update = lambda cfg: calls.append('finalize')
    monkeypatch.setitem(sys.modules, cfgmod.__name__, cfgmod)
    kdmod = types.ModuleType('megatron.bridge.models.distillation_provider')
    kdmod.DistillationProvider = KDProvider
    kdmod.convert_to_distillation_provider = convert
    monkeypatch.setitem(sys.modules, kdmod.__name__, kdmod)
    postmod = types.ModuleType('megatron.bridge.training.post_training.distillation')
    postmod.ModelOptDistillConfig = Bag
    monkeypatch.setitem(sys.modules, postmod.__name__, postmod)

    def provider(path, load_weights):
        calls.append((str(path), load_weights))
        return Bag(hidden_size=4096 if 'teachers' in str(path) else 1024,
                   num_layers=88 if 'teachers' in str(path) else 52, vocab_size=131072)

    monkeypatch.setattr(runner, 'provider_from_hf', provider)
    monkeypatch.setattr(runner, 'apply_shape_patch', lambda cls: {'status': 'fixture_only_no_sdk'})
    cfg = runner.make_config(c, Bag, tmp_path, config_only=True)
    assert isinstance(cfg.model, KDProvider)
    assert cfg.model.teacher.hidden_size == 4096
    assert cfg.model.kd_config.skip_lm_loss is False
    assert cfg.checkpoint.save == str(ROOT / 'checkpoints/midtraining/super35-kd-pilot-01')
    assert calls.index('finalize') < calls.index('convert')
    assert [x[1] for x in calls if isinstance(x, tuple)] == [False, False]
    calls.clear()
    cfg = runner.make_config(c, Bag, tmp_path, reload_checkpoint=tmp_path / 'checkpoint')
    assert not isinstance(cfg.model, KDProvider)
    assert len([x for x in calls if isinstance(x, tuple)]) == 1
    assert cfg.validation.skip_train is True
    assert cfg.checkpoint.load_optim is cfg.checkpoint.load_rng is False
    assert cfg.checkpoint.save is None


def test_native_runtime_is_audited_and_has_no_custom_training_loop():
    import ast
    runner = load_runner()
    assert callable(runner.run_native)
    assert callable(runner.make_dataset_provider)
    tree = ast.parse((ROOT / 'midtraining/native_kd.py').read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    pretrain = [n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'pretrain']
    assert len(pretrain) == 1
    assert isinstance(pretrain[0].args[1], ast.Name) and pretrain[0].args[1].id == 'forward_step_modelopt'
    assert 'callbacks' in {arg.arg for arg in pretrain[0].keywords}
    assert not [n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr in ('backward', 'step')]


def test_architecture_gate_catches_mirrored_attention_and_mamba_dimensions():
    from types import SimpleNamespace
    import pytest
    runner = load_runner()
    provider = SimpleNamespace(hidden_size=4096, num_attention_heads=32, mamba_num_heads=128)
    expected = runner.architecture(provider)
    provider.mamba_num_heads = 24
    with pytest.raises(RuntimeError, match='architecture'):
        runner.check_architecture(provider, expected)


def test_native_loss_mask_plumbing_and_eval_keeps_lm_loss():
    """Execute actual Bridge loss adapter with synthetic per-token criterion values."""
    import ast
    import types
    import typing
    import torch
    source = ROOT / 'results/midtraining/super35-kd-01/sdk-source/megatron.bridge/training/post_training/distillation.py'
    tree = ast.parse(source.read_text())

    class CriterionFixture:
        training = True
        def compute_kd_loss(self, student_loss, loss_reduction_fn):
            kd = loss_reduction_fn(torch.tensor([[1., 20., 3., 40.]]))
            return {'kd_loss': student_loss + kd, 'logits_loss': kd, 'intermediate_loss': kd * 0}

    namespace = {'torch': torch, 'Callable': typing.Callable, 'MegatronModule': object,
        'mtd': types.SimpleNamespace(DistillationModel=CriterionFixture),
        'parallel_state': types.SimpleNamespace(get_tensor_model_parallel_group=lambda: None)}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('loss_func_kd', '_mask_loss')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    mask = torch.tensor([[1., 0., 1., 0.]])
    original = lambda output: ((output * mask).sum(), mask.sum().to(torch.int32), {'lm loss': torch.tensor([2., 2.])})
    model = CriterionFixture()
    loss, count, report = namespace['loss_func_kd'](torch.ones(1, 4), mask, original, model)
    assert loss.item() == 6. and count.item() == 2
    assert report['logits distillation loss'].tolist() == [4., 2.]
    model.training = False
    loss, _, _ = namespace['loss_func_kd'](torch.ones(1, 4), mask, original, model)
    assert loss.item() == 2.


def test_flat_corpus_reader_never_cycles_training_and_explicitly_reuses_heldout(monkeypatch):
    import copy
    import sys
    import types
    from dataclasses import dataclass
    import numpy as np
    import pytest
    runner = load_runner()
    c = copy.deepcopy(runner.load_config(ROOT / 'midtraining/pilot.yaml'))
    c['sequence_length'] = 4
    c['train'].update(train_iters=2, global_batch_size=2)
    c['validation'].update(eval_iters=1, eval_global_batch_size=2)

    class Reader:
        sequence_lengths = np.array([7, 10])
        def __init__(self, *args, **kwargs):
            self.docs = [np.arange(7), np.arange(7, 17)]
        def get(self, doc, offset=0, length=None):
            return self.docs[doc][offset:offset + length]

    @dataclass
    class Provider:
        dataloader_type: str = 'single'

    indexmod = types.ModuleType('megatron.core.datasets.indexed_dataset')
    indexmod.IndexedDataset = Reader
    datamod = types.ModuleType('megatron.bridge.data.base')
    datamod.DatasetProvider = Provider
    monkeypatch.setitem(sys.modules, indexmod.__name__, indexmod)
    monkeypatch.setitem(sys.modules, datamod.__name__, datamod)
    provider = runner.make_dataset_provider(c)()
    context = types.SimpleNamespace(train_samples=4, valid_samples=4)
    train, valid, test = provider.build_datasets(context)
    assert train[1]['tokens'].tolist() == [4, 5, 6, 7]
    assert train[3]['labels'].tolist() == [13, 14, 15, 16]
    assert valid[2]['tokens'].tolist() == valid[0]['tokens'].tolist()
    assert test is None
    with pytest.raises(IndexError):
        train[4]
    reload_data = runner.make_dataset_provider(c, reload_only=True)()
    assert reload_data.build_datasets(context)[0] is None


def test_pilot_bounds_are_not_defined_by_mutable_default_file(monkeypatch, tmp_path):
    import yaml
    import pytest
    runner = load_runner()
    c = runner.load_config(ROOT / 'midtraining/pilot.yaml')
    c['train']['train_iters'] = 4000
    mutated = tmp_path / 'pilot.yaml'
    mutated.write_text(yaml.safe_dump(c))
    monkeypatch.setattr(runner, 'DEFAULT_CONFIG', mutated)
    with pytest.raises(ValueError, match='bounded'):
        runner.load_config(mutated)
