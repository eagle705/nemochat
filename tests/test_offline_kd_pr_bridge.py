"""Isolated Bridge/Core port tests; CPU fixtures are not native GPU training."""
import importlib
import sys
import os
import torch
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / 'midtraining/patches/offline_kd_pr'


def api():
    sys.path.insert(0, str(PATCH))
    importlib.invalidate_caches()
    assert (PATCH / 'bridge_cached_kd/runtime.py').exists(), 'native runtime binding missing'
    return importlib.import_module('bridge_cached_kd.runtime')


def test_native_modules_are_ported_without_megatron_training_imports():
    for name in ('utils_logits', 'logits_saver', 'cached_logits_loss'):
        path = PATCH / 'bridge_cached_kd' / (name + '.py')
        assert path.exists(), 'native Core writer/storage/reader port missing'
        assert 'from megatron.training' not in path.read_text()


def test_explicit_config_is_opt_in_core_objective_and_rejects_unsafe_modes():
    api()
    path = PATCH / 'bridge_cached_kd/integration.py'
    assert path.exists(), 'Bridge cached KD integration missing'
    m = importlib.import_module('bridge_cached_kd.integration')
    config = m.CachedKDConfig(mode='read', cache_dir='/cache', cache_id='teacher-data-v1')
    assert config.objective == 'megatron-lm-ghost-tail-fixed-alpha'
    assert config.alpha == 0.5
    for kwargs in ({'mode': 'online'}, {'alpha': -1}, {'alpha': float('nan')},
                   {'cache_id': ''}, {'top_k': 0}, {'save_dtype': 'fp64'}):
        with pytest.raises(ValueError):
            m.CachedKDConfig(**({'mode':'read', 'cache_dir':'/cache', 'cache_id':'x'} | kwargs))


def test_explicit_runtime_binding_restores_without_training_globals():
    m = api()
    runtime = m.CoreRuntime(seed=17, seq_length=8, train_samples=16,
                            global_batch_size=2, micro_batch_size=1,
                            blend=(['/data/a'], [1.0]), blend_per_split=None,
                            iteration=lambda: 3, num_microbatches=lambda: 2)
    with pytest.raises(RuntimeError, match='bound'):
        m.get_args()
    with m.bind_runtime(runtime):
        assert m.get_args().curr_iteration == 3
        assert m.get_args().seed == 17
        assert m.get_blend_and_blend_per_split(m.get_args()) == ((['/data/a'], [1.0]), None)
        assert m.get_num_microbatches() == 2
        with pytest.raises(RuntimeError, match='nested'):
            with m.bind_runtime(runtime):
                pass
    with pytest.raises(RuntimeError, match='bound'):
        m.get_args()

def test_bridge_step_requires_explicit_checkpoint_cache_binding():
    api()
    m = importlib.import_module('bridge_cached_kd.integration')
    assert hasattr(m, 'CachedKDForwardStep'), 'native Bridge forward/loss adapter missing'
    config = m.CachedKDConfig(mode='read', cache_dir='/cache', cache_id='immutable-cache-v1')
    runtime = api().CoreRuntime(17, 8, 16, 2, 1, None, None, lambda: 3, lambda: 2)
    step = m.CachedKDForwardStep(config, lambda state: runtime)
    with pytest.raises(ValueError, match='resume'):
        step.validate_resume(runtime)
    binding = {'cache_id': 'immutable-cache-v1', 'next_iteration': 3,
               'consumed_samples': 6, 'global_batch_size': 2, 'micro_batch_size': 1,
               'seed': 17, 'seq_length': 8, 'train_samples': 16,
               'blend': None, 'blend_per_split': None,
               'objective': config.objective, 'alpha': 0.5}
    step = m.CachedKDForwardStep(config, lambda state: runtime, resume_binding=binding)
    step.validate_resume(runtime)
    for key in binding:
        wrong = dict(binding, **{key: 'wrong'})
        bad = m.CachedKDForwardStep(config, lambda state: runtime, resume_binding=wrong)
        with pytest.raises(ValueError, match='binding'):
            bad.validate_resume(runtime)


@pytest.mark.skipif(os.environ.get('NATIVE_KD_SDK_TESTS') != '1', reason='requires pinned SDK CPU runtime')
def test_sdk_writer_reader_bridge_loss_backward_roundtrip(tmp_path):
    api()
    m = importlib.import_module('bridge_cached_kd.integration')
    assert hasattr(m, 'CachedKDForwardStep'), 'native Bridge forward/loss adapter missing'
    from megatron.core import parallel_state
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method='file://' + str(tmp_path / 'rdzv'), rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    try:
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.output_layer = torch.nn.Linear(4, 8)
                self.config = SimpleNamespace()
            def compute_language_model_loss(self, labels, logits):
                return torch.nn.functional.cross_entropy(logits.reshape(-1,8), labels.reshape(-1), reduction='none').reshape(3,1).T
            def forward(self, x):
                logits = self.output_layer(x)
                return self.compute_language_model_loss(torch.zeros(3,1,dtype=torch.long), logits)
        torch.manual_seed(17)
        x = torch.randn(3,1,4)
        teacher, student = Tiny(), Tiny()
        mask = torch.tensor([[1.,0.,1.]])
        state = SimpleNamespace(train_state=SimpleNamespace(step=0, consumed_train_samples=0),
                                cfg=SimpleNamespace(model=SimpleNamespace()))
        runtime = api().CoreRuntime(17, 3, 2, 2, 1, None, None, lambda:state.train_state.step, lambda:2)
        def common(state, iterator, model, return_schedule_plan=False):
            return model(next(iterator)), mask
        cfg = m.CachedKDConfig('write', str(tmp_path/'cache'), 'fixture-native', top_k=3, save_dtype='fp32')
        with m.CachedKDForwardStep(cfg, lambda s:runtime, common_forward=common) as writer:
            for iteration in range(2):
                state.train_state.step = iteration
                state.train_state.consumed_train_samples = iteration * 2
                for _ in range(2):
                    writer.cache_forward(state, iter([x]), teacher)
        state.train_state.step = 0
        state.train_state.consumed_train_samples = 0
        tars = list((tmp_path/'cache').glob('cp0_dp0__0.tar'))
        assert len(tars) == 1
        from bridge_cached_kd.utils_logits import iter_logprobs_tar_entries, decode_logprobs_payload, compute_dataset_hash
        with api().bind_runtime(runtime):
            entries = list(iter_logprobs_tar_entries(str(tars[0]), expected_hash=compute_dataset_hash()[0]))
        assert [e.iteration for e in entries] == [0]
        values, indices = decode_logprobs_payload(entries[0].data)
        assert len(values) == 2 and values[0].shape == (3,1,3)
        assert torch.allclose(values[0], teacher.output_layer(x).log_softmax(-1).topk(3,-1).values)
        readcfg = m.CachedKDConfig('read', str(tmp_path/'cache'), 'fixture-native', top_k=3)
        bad_runtime = api().CoreRuntime(17,3,2,1,1,None,None,lambda:0,lambda:1)
        with m.CachedKDForwardStep(readcfg, lambda s:bad_runtime, common_forward=common) as bad_reader:
            output, loss_fn = bad_reader(state, iter([x]), student)
            with pytest.raises(ValueError, match='microbatch mapping'):
                loss_fn(output)
        with m.CachedKDForwardStep(readcfg, lambda s:runtime, common_forward=common) as reader:
            losses=[]
            for _ in range(2):
                output, loss_fn = reader(state, iter([x]), student)
                total, n, report = loss_fn(output)
                with pytest.raises(RuntimeError, match='replayed'):
                    loss_fn(output)
                if not losses:
                    with pytest.raises(RuntimeError, match='completed iteration'):
                        reader.checkpoint_binding()
                assert n.item() == 2
                assert torch.allclose(total.detach(), 0.5 * (report['lm loss'][0] + report['logits distillation loss'][0]))
                total.backward()
                losses.append(total.item())
            assert losses[0] == losses[1]
            assert torch.isfinite(student.output_layer.weight.grad).all()
            assert student.output_layer.weight.grad.abs().sum() > 0
            binding = reader.checkpoint_binding()
            assert binding['next_iteration'] == 1 and binding['consumed_samples'] == 2
            student.eval()
            state.train_state.step = 1
            output, loss_fn = reader(state, iter([x]), student)
            loss, n, report = loss_fn(output)
            assert list(report) == ['lm loss']
        student.train()
        state.train_state.consumed_train_samples = 2
        with m.CachedKDForwardStep(readcfg, lambda s:runtime, resume_binding=binding, common_forward=common) as resumed:
            for _ in range(2):
                output, loss_fn = resumed(state, iter([x]), student)
                loss_fn(output)[0].backward()
            assert resumed.checkpoint_binding()['next_iteration'] == 2
        assert not teacher.output_layer._forward_hooks and not student.output_layer._forward_hooks
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('NATIVE_KD_SDK_TESTS') != '1', reason='requires pinned SDK CPU runtime')
def test_sdk_reader_rejects_missing_iteration_instead_of_silent_alignment(tmp_path):
    api()
    import io
    from collections import OrderedDict
    from bridge_cached_kd.logits_saver import LogitsSaverHooks
    from bridge_cached_kd.cached_logits_loss import TeacherTarDataset
    from bridge_cached_kd.utils_logits import compute_dataset_hash
    runtime = api().CoreRuntime(17, 3, 2, 2, 1, None, None, lambda:0, lambda:2)
    with api().bind_runtime(runtime):
        hash_, identifiers = compute_dataset_hash()
        buffer = io.BytesIO()
        torch.save({'values':[torch.zeros(3,1,1)], 'indices_low':[torch.zeros(3,1,1,dtype=torch.uint16)],
                    'bit_17':[torch.zeros(3,1,1,dtype=torch.bool)]}, buffer)
        LogitsSaverHooks._write_batched_tar(str(tmp_path/'cp0_dp0__1.tar'), OrderedDict([(1,buffer.getvalue())]),
                                           __import__('json').dumps({'hash':hash_,'identifiers':identifiers}).encode())
        dataset = TeacherTarDataset(str(tmp_path),0,0,1,start_iteration=0)
        with pytest.raises(RuntimeError, match='iteration gap'):
            next(iter(dataset))


def test_reviewable_patch_applies_to_exact_staged_sdk_without_default_changes(tmp_path):
    import hashlib
    import subprocess
    patch = PATCH / 'megatron-bridge-0.6.1-cached-kd.patch'
    assert patch.exists(), 'reviewable Bridge integration diff missing'
    sdk = tmp_path / 'sdk'
    target = sdk/'src/megatron/bridge/training/gpt_step.py'
    target.parent.mkdir(parents=True)
    original = (PATCH/'evidence/bridge-installed/training__gpt_step.py').read_bytes()
    target.write_bytes(original)
    subprocess.run(['git','apply','--check',str(patch)], cwd=sdk,check=True,capture_output=True)
    subprocess.run(['git','apply',str(patch)],cwd=sdk,check=True,capture_output=True)
    assert target.read_bytes().startswith(original)
    assert 'def make_cached_kd_forward_step(' in target.read_text()
    for source in (PATCH/'bridge_cached_kd').glob('*.py'):
        staged=sdk/'src/megatron/bridge/training/cached_kd'/source.name
        assert staged.read_bytes() == source.read_bytes()
    provenance = __import__('json').loads((PATCH/'evidence/integration-base.json').read_text())
    assert provenance['bridge_gpt_step_sha256'] == hashlib.sha256(original).hexdigest()
    assert len(provenance['bridge_base_sha']) == 40


@pytest.mark.skipif(os.environ.get('NATIVE_KD_SDK_TESTS') != '1', reason='requires pinned SDK CPU runtime')
def test_unmodified_sdk_writer_native_and_port_reader_interoperate(tmp_path, monkeypatch):
    api()
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.training.distillation import logits_saver as native_writer
    from megatron.training.distillation import utils_logits as native_storage
    from megatron.training.distillation import cached_logits_loss as native_reader
    from bridge_cached_kd.cached_logits_loss import TeacherTarDataset
    from bridge_cached_kd.utils_logits import compute_dataset_hash
    dist.init_process_group('gloo', init_method='file://'+str(tmp_path/'rdzv'),rank=0,world_size=1)
    parallel_state.initialize_model_parallel()
    runtime=api().CoreRuntime(17,3,1,1,1,None,None,lambda:0,lambda:1)
    # Explicit in-memory dependency injection in the test ONLY. No sys.modules
    # replacement, no installed source writes and no Core global-vars mutation.
    monkeypatch.setattr(native_writer,'get_args',api().get_args)
    monkeypatch.setattr(native_storage,'get_args',api().get_args)
    monkeypatch.setattr(native_storage,'get_blend_and_blend_per_split',api().get_blend_and_blend_per_split)
    monkeypatch.setattr(native_writer,'get_num_microbatches',api().get_num_microbatches)
    model=torch.nn.Module()
    model.output_layer=torch.nn.Identity()
    model.compute_language_model_loss=lambda labels,logits: logits.sum(-1).T
    logits=torch.randn(3,1,8)
    try:
        with api().bind_runtime(runtime):
            saver=native_writer.LogitsSaverHooks(str(tmp_path/'native'),3,save_dtype='fp32')
            saver.attach_hooks(model)
            model.output_layer(logits)
            path,writes,meta,msc=saver.take_pending_data()
            saver._write_batched_tar(path,writes,meta,msc)
            saver.remove_hooks()
            native_values,native_indices=next(iter(native_reader.TeacherTarDataset(str(tmp_path/'native'),0,0,1)))
            port_values,port_indices=next(iter(TeacherTarDataset(str(tmp_path/'native'),0,0,1)))
            assert torch.equal(native_values[0],port_values[0])
            assert torch.equal(native_indices[0],port_indices[0])
            assert compute_dataset_hash()[0] == saver.dataset_hash
            student=torch.randn(3,1,8,requires_grad=True)
            loss=native_reader.topk_kl_div(student.clone(),port_values[0],port_indices[0],1,0,None,True).sum()
            loss.backward()
            assert torch.isfinite(student.grad).all() and student.grad.abs().sum() > 0
        assert not model.output_layer._forward_hooks
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('NATIVE_KD_PATCH_APPLIED') != '1', reason='requires staged SDK patch')
def test_staged_sdk_bridge_factory_imports_real_dependencies_and_preserves_default():
    from megatron.bridge.training import gpt_step
    from megatron.bridge.training.cached_kd.integration import CachedKDConfig
    from megatron.bridge.training.cached_kd.runtime import CoreRuntime
    import ast
    assert '/scratch/staged/src/' in gpt_step.__file__
    config=CachedKDConfig('read','/cache','explicit-cache-id')
    runtime=CoreRuntime(17,3,1,1,1,None,None,lambda:0,lambda:1)
    step=gpt_step.make_cached_kd_forward_step(config, lambda state:runtime)
    assert step.common_forward is gpt_step._forward_step_common
    assert step.config is config
    before=ast.parse((PATCH/'evidence/bridge-installed/training__gpt_step.py').read_text())
    after=ast.parse(Path(gpt_step.__file__).read_text())
    assert ast.dump(ast.Module(body=after.body[:-1],type_ignores=[])) == ast.dump(before)
    step.close()


@pytest.mark.skipif(os.environ.get('NATIVE_KD_SDK_TESTS') != '1', reason='requires pinned SDK CPU runtime')
def test_cached_adapter_rejects_online_modelopt_teacher_instead_of_running_both(tmp_path):
    api()
    import modelopt.torch.distill as mtd
    from bridge_cached_kd.integration import CachedKDConfig, CachedKDForwardStep
    model = mtd.DistillationModel.__new__(mtd.DistillationModel)
    torch.nn.Module.__init__(model)
    state=SimpleNamespace(cfg=SimpleNamespace(model=SimpleNamespace()))
    step=CachedKDForwardStep(CachedKDConfig('read','/cache','id'),lambda s:None)
    import torch.distributed as dist
    from megatron.core import parallel_state
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'rdzv'),rank=0,world_size=1)
    parallel_state.initialize_model_parallel()
    try:
        with pytest.raises(ValueError, match='online ModelOpt'):
            step._setup(state,model)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get('NATIVE_KD_SDK_TESTS') != '1', reason='requires pinned SDK CPU runtime')
def test_writer_close_fails_on_partial_iteration_and_removes_hooks(tmp_path):
    api()
    from megatron.core import parallel_state
    import torch.distributed as dist
    from bridge_cached_kd.integration import CachedKDConfig,CachedKDForwardStep
    dist.init_process_group('gloo',init_method='file://'+str(tmp_path/'rdzv'),rank=0,world_size=1)
    parallel_state.initialize_model_parallel()
    model=torch.nn.Module()
    model.output_layer=torch.nn.Identity()
    original=lambda labels,logits: logits.sum(-1).T
    model.compute_language_model_loss=original
    runtime=api().CoreRuntime(17,3,2,2,1,None,None,lambda:0,lambda:2)
    state=SimpleNamespace(cfg=SimpleNamespace(model=SimpleNamespace()),
                          train_state=SimpleNamespace(step=0,consumed_train_samples=0))
    def common(state,iterator,model,return_schedule_plan=False):
        return model.output_layer(next(iterator)),torch.ones(1,3)
    try:
        with pytest.raises(RuntimeError,match='partial iteration'):
            with CachedKDForwardStep(CachedKDConfig('write',str(tmp_path/'partial'),'id'),
                                      lambda s:runtime,common_forward=common) as writer:
                writer.cache_forward(state,iter([torch.randn(3,1,8)]),model)
        assert not model.output_layer._forward_hooks
        assert model.compute_language_model_loss is original
        assert not list((tmp_path/'partial').glob('*.tar'))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
