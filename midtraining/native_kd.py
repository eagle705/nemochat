"""Bounded native Bridge LM+logits KD (no custom optimizer loop).

ModelOpt's installed balancer uses LM + scale*KD*stopgrad(LM/KD) for
positive losses; displayed total is 1.5*LM, not evidence that KL is .5*LM.
CPU tests/schema checks are not native model loading/training evidence.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name('pilot.yaml')
REVISION = '4ba52e8ffa79e9693936c1a819ac89f551ceeef6'


def load_config(path):
    import yaml
    c = yaml.safe_load(Path(path).read_text())
    reference = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if c != reference:
        raise ValueError('only the frozen approved four-update pilot configuration is supported')
    bounded = (c['train']['train_iters'] == 4 and c['train']['global_batch_size'] == 8
        and c['train']['micro_batch_size'] == 1 and c['train']['exit_duration_in_mins'] == 120
        and c['sequence_length'] == 4096 and c['world_size'] == 8
        and c['optimizer']['lr'] == 1e-5 and c['optimizer']['min_lr'] == 1e-6
        and c['scheduler']['lr_decay_iters'] == 4 and c['scheduler']['lr_warmup_iters'] == 1
        and c['scheduler']['lr_wsd_decay_iters'] == 1 and c['scheduler']['lr_decay_style'] == 'WSD'
        and c['checkpoint_integrity'] == 'not_checked_explicit_user_waiver'
        and c['checkpoint_save'] == 'checkpoints/midtraining/super35-kd-pilot-01'
        and c['kd'] == dict(skip_lm_loss=False, intermediate_layer_pairs=[], kd_loss_scale=.5,
                            logit_kl_temperature=1., logit_kl_topk=None)
        and c['student_hf'] == 'checkpoints/exports/nemochat_baby_base_hf_bf16_02'
        and c['teacher_revision'] == REVISION
        and c['teacher_hf'] == f'checkpoints/teachers/nemotron35-super-base/{REVISION}-text'
        and c['parallel'] == dict(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=1, expert_model_parallel_size=8, expert_tensor_parallel_size=1,
            sequence_parallel=False)
        and all(c['logger'][key] is None for key in ('wandb_project', 'wandb_entity', 'wandb_exp_name')))
    if not bounded:
        raise ValueError('bounded native pilot contract changed; separate review required')
    return c


def tokenizer_gate(student, teacher):
    """Compare tokenizer payload semantics, not names, vocab sizes or weight bytes."""
    a = json.loads((Path(student) / 'tokenizer.json').read_text())
    b = json.loads((Path(teacher) / 'tokenizer.json').read_text())
    fields = ('model', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder', 'added_tokens')
    different = [key for key in fields if a.get(key) != b.get(key)]
    if different:
        spec = importlib.util.spec_from_file_location('midtraining_tokenizer_alias_gate',
            ROOT / 'midtraining/patches/tokenizer_alias_gate.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        try:
            return adapter.check(student, teacher,
                ROOT / 'results/pretraining/nemochat_baby/tokenizer_migration/manifest.json', ROOT)
        except ValueError as exc:
            raise ValueError('teacher/student token-ID semantics differ without accepted reserved alias migration: '
                             + str(exc)) from exc
    if not isinstance(a.get('model', {}).get('vocab'), dict):
        raise ValueError('explicit BPE token-ID vocabulary required')
    return dict(token_id_semantics_equal=True, compared_fields=list(fields),
                student=str(student), teacher=str(teacher),
                checkpoint_integrity='not_checked_explicit_user_waiver')


def make_sample(values, sequence, vocab):
    import torch
    if len(values) != sequence + 1:
        raise ValueError('shifted corpus window must contain sequence_length+1 tokens')
    tokens = torch.tensor(values, dtype=torch.long)
    if int(tokens.min()) < 0 or int(tokens.max()) >= vocab:
        raise ValueError('corpus token ID outside shared vocabulary')
    return dict(tokens=tokens[:-1], labels=tokens[1:],
                position_ids=torch.arange(sequence), loss_mask=torch.ones(sequence, dtype=torch.float32))


def check_teacher(teacher):
    count = 0
    for parameter in teacher.parameters():
        if parameter.requires_grad:
            raise RuntimeError('teacher must be frozen by ModelOpt before optimizer creation')
        if parameter.grad is not None or getattr(parameter, 'main_grad', None) is not None:
            raise RuntimeError('teacher gradient detected')
        count += parameter.numel()
    if not count:
        raise RuntimeError('teacher has no local parameters')
    return dict(frozen_parameters=count, teacher_gradients_absent=True)


def completed_update_index(ctx, updates):
    # Pinned native train.py fires on_train_step_end before train_state.step += 1.
    step = int(ctx.state.train_state.step) + 1
    if step != len(updates) + 1:
        raise RuntimeError('native completed optimizer update cursor is not contiguous')
    return step


def update_metrics(ctx):
    values = {name: float(ctx.loss_dict[key]) for name, key in (
        ('LM Loss', 'lm loss'), ('KD Loss', 'logits distillation loss'), ('Total Loss', 'total loss'))}
    values['Grad Norm'] = float(ctx.grad_norm) if ctx.grad_norm is not None else float('nan')
    if ctx.skipped_iter or not all(math.isfinite(v) for v in values.values()) or values['Grad Norm'] <= 0:
        raise FloatingPointError('native optimizer update requires finite LM/KD/total and positive gradient; '
                                 + f'observed_metrics={values!r}; skipped_iter={bool(ctx.skipped_iter)}')
    return values


def configure_provider(provider, c):
    """Only shared execution settings; never transplant student architecture."""
    import torch
    for key, value in dict(c['parallel'], seq_length=c['sequence_length'],
            params_dtype=torch.bfloat16, pipeline_dtype=torch.bfloat16, bf16=True,
            calculate_per_token_loss=True, mtp_num_layers=0, cuda_graph_impl='none',
            cross_entropy_fusion_impl='native', apply_rope_fusion=False).items():
        setattr(provider, key, value)


def check_provider_shapes(student, teacher=None):
    if student.hidden_size != 1024 or student.num_layers != 52 or student.vocab_size != 131072:
        raise RuntimeError('student shape changed from approved BASE')
    if teacher is not None and (teacher.hidden_size != 4096 or teacher.vocab_size != 131072):
        raise RuntimeError('teacher shape changed; DistillationProvider attribute mirroring is unsafe')


def architecture(provider):
    names = ('hidden_size', 'num_layers', 'vocab_size', 'num_attention_heads', 'num_query_groups',
             'kv_channels', 'ffn_hidden_size', 'moe_latent_size', 'moe_ffn_hidden_size',
             'num_moe_experts', 'moe_router_topk', 'moe_shared_expert_intermediate_size',
             'mamba_num_heads', 'mamba_head_dim', 'mamba_num_groups', 'mamba_state_dim',
             'hybrid_layer_pattern', 'share_embeddings_and_output_weights')
    return {name: getattr(provider, name) for name in names if hasattr(provider, name)}


def check_architecture(provider, expected):
    actual = architecture(provider)
    if actual != expected:
        differences = {key: {'expected': expected.get(key), 'actual': actual.get(key)}
                       for key in set(actual) | set(expected) if actual.get(key) != expected.get(key)}
        raise RuntimeError('native architecture mutated by provider finalization/mirroring: ' + repr(differences))


def provider_from_hf(path, load_weights):
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    if load_weights:
        provider = AutoBridge.from_hf_pretrained(str(path), trust_remote_code=False,
                local_files_only=True).to_megatron_provider(load_weights=True)
        spec = importlib.util.spec_from_file_location('midtraining_student_embedding_import',
            ROOT / 'midtraining/patches/student_embedding_import.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        object.__setattr__(provider, '_nemochat_hf_import_receipt', adapter.attach(provider, path))
        return provider
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)
    return AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)


def apply_shape_patch(provider_class):
    spec = importlib.util.spec_from_file_location('midtraining_provider_shape_patch',
        ROOT / 'midtraining/patches/distillation_provider_shape_guard.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter.apply(provider_class)


def make_config(c, Dataset, output, config_only=False, reload_checkpoint=None):
    import torch
    from megatron.bridge.training.config import (ConfigContainer, TrainingConfig, OptimizerConfig,
        SchedulerConfig, LoggerConfig, TokenizerConfig, CheckpointConfig,
        DistributedDataParallelConfig, ValidationConfig, RNGConfig, runtime_config_update)
    student = provider_from_hf(ROOT / c['student_hf'], load_weights=not config_only and reload_checkpoint is None)
    configure_provider(student, c)
    teacher = None
    if reload_checkpoint is None:
        teacher = provider_from_hf(ROOT / c['teacher_hf'], load_weights=not config_only)
        configure_provider(teacher, c)
        teacher.finalize()
    check_provider_shapes(student, teacher)
    cfg = ConfigContainer(model=student, train=TrainingConfig(**c['train']),
        optimizer=OptimizerConfig(**c['optimizer'], params_dtype=torch.bfloat16),
        ddp=DistributedDataParallelConfig(**c['ddp']), scheduler=SchedulerConfig(**c['scheduler']),
        dataset=Dataset(dataloader_type='single', num_workers=c['dataset']['num_workers'], persistent_workers=False),
        tokenizer=TokenizerConfig(tokenizer_type='NullTokenizer', vocab_size=c['vocab_size'],
            padded_vocab_size=c['vocab_size'], use_tokenizer_vocab_size=False, null_tokenizer_eod_id=11),
        logger=LoggerConfig(**c['logger'], wandb_save_dir=str(output / 'sdk-logs'),
                            save_config_filepath=str(output / 'native-finalized-config.yaml')),
        checkpoint=CheckpointConfig(save=str(ROOT / c['checkpoint_save']) if reload_checkpoint is None else None,
            load=str(reload_checkpoint) if reload_checkpoint is not None else None,
            pretrained_checkpoint=None, save_interval=c['train']['train_iters'],
            save_optim=False, save_rng=False, load_optim=False, load_rng=False,
            finetune=reload_checkpoint is not None, exit_on_missing_checkpoint=True,
            async_save=False, fully_parallel_save=False, ckpt_format='torch_dist',
            save_tokenizer_assets=False, verify_integrity=False, most_recent_k=1),
        validation=ValidationConfig(**c['validation'], skip_train=reload_checkpoint is not None),
        rng=RNGConfig(**c['rng']), mixed_precision='bf16_mixed')
    cfg.dist.distributed_timeout_minutes = 30
    cfg.dist.distributed_timeout_seconds_after_init = 1800
    # Finalize the normal independent providers first. Never assign student shape
    # fields after conversion: upstream DistillationProvider mirrors assignments.
    runtime_config_update(cfg)
    student_architecture = architecture(student)
    teacher_architecture = architecture(teacher) if teacher is not None else None
    def guard_shapes(chunks):
        check_provider_shapes(cfg.model, teacher)
        check_architecture(cfg.model, student_architecture)
        if teacher is not None:
            check_architecture(teacher, teacher_architecture)
        return chunks
    # Register before KD conversion: registering afterward may mirror/share the
    # teacher hook list and recurse through _convert_hook in some SDK versions.
    student.register_pre_wrap_hook(guard_shapes, prepend=True)
    if teacher is not None:
        from megatron.bridge.models.distillation_provider import convert_to_distillation_provider, DistillationProvider
        from megatron.bridge.training.post_training.distillation import ModelOptDistillConfig
        patch_receipt = apply_shape_patch(DistillationProvider)
        cfg.model = convert_to_distillation_provider(student, teacher, ModelOptDistillConfig(**c['kd']))
        object.__setattr__(cfg.model, '_nemochat_patch_receipt', patch_receipt)
        assert isinstance(cfg.model, DistillationProvider)
        # Exercise the exact repeat finalization pretrain() will perform, then fail
        # closed before any weights are loaded if it altered teacher architecture.
        runtime_config_update(cfg)
    guard_shapes([])
    return cfg


def corpus_helpers():
    """Reuse the executed BASE corpus reader/metadata contract, not its trainer."""
    path = ROOT / 'results/pretraining/nemochat_baby/training/runner.py'
    spec = importlib.util.spec_from_file_location('base_corpus_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_preflight(c, reload_only=False):
    helpers = corpus_helpers()
    sequence = c['sequence_length']
    valid = helpers.indexed_metadata(ROOT / c['dataset']['validation_prefix'])
    helpers.require_capacity(valid['prepared_tokens'],
        c['validation']['eval_iters'] * c['validation']['eval_global_batch_size'], sequence)
    train = None
    if not reload_only:
        if (ROOT / c['dataset']['train_prefix']).resolve() == (ROOT / c['dataset']['validation_prefix']).resolve():
            raise ValueError('training and heldout prefixes must differ')
        train = helpers.indexed_metadata(ROOT / c['dataset']['train_prefix'])
        helpers.require_capacity(train['prepared_tokens'],
            c['train']['train_iters'] * c['train']['global_batch_size'], sequence)
    return dict(train=train, validation=valid, all_token_loss_mask=True,
                repeat_from_pretraining=True, within_pilot_training_cycling=False,
                validation_policy='fixed heldout window explicitly reused across evaluations')


def make_dataset_provider(c, reload_only=False):
    import numpy as np
    import torch
    from dataclasses import dataclass
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from megatron.bridge.data.base import DatasetProvider
    helpers = corpus_helpers()
    sequence = c['sequence_length']
    window = c['validation']['eval_iters'] * c['validation']['eval_global_batch_size']

    class Packed(torch.utils.data.Dataset):
        def __init__(self, prefix, samples, validation=False):
            self.reader = IndexedDataset(str(ROOT / prefix), multimodal=False, mmap=True)
            self.offsets = np.concatenate(([0], np.cumsum(self.reader.sequence_lengths, dtype=np.int64)))
            self.samples = samples
            self.validation = validation
            helpers.require_capacity(int(self.offsets[-1]), window if validation else samples, sequence)

        def __len__(self):
            return self.samples

        def __getitem__(self, i):
            if not 0 <= i < self.samples:
                raise IndexError(i)
            position = (i % window if self.validation else i) * sequence
            values = helpers.read_window(self.reader, self.offsets, position, sequence + 1)
            return make_sample(values, sequence, c['vocab_size'])

    @dataclass
    class NativeData(DatasetProvider):
        def build_datasets(self, context):
            required = c['train']['train_iters'] * c['train']['global_batch_size']
            if not reload_only and context.train_samples != required:
                raise ValueError('native sample budget differs from approved four-update pilot')
            train = None if reload_only else Packed(c['dataset']['train_prefix'], context.train_samples)
            valid = Packed(c['dataset']['validation_prefix'], context.valid_samples, True) if context.valid_samples else None
            return train, valid, None
    return NativeData


def snapshot(value):
    """JSON/YAML source-config serialization only; metrics reject nonfinite values."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'config_nonfinite_float': str(value)}
    if isinstance(value, dict):
        return {str(k): snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    if value is None or type(value) in (str, bool, int, float):
        return value
    return str(value)


def native_source_receipt():
    """Record executed native sources; never checksum the waived student weights."""
    import hashlib
    import inspect
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.gpt_step import forward_step_modelopt
    from megatron.bridge.models.distillation_provider import convert_to_distillation_provider
    from megatron.bridge.training.post_training.distillation import loss_func_kd
    from modelopt.torch.distill.plugins.megatron import adjust_distillation_model_for_mcore
    from modelopt.torch.distill.distillation_model import DistillationModel
    sources = []
    for entry in (pretrain, forward_step_modelopt, convert_to_distillation_provider,
                  loss_func_kd, adjust_distillation_model_for_mcore, DistillationModel):
        target = entry
        while target is not None:
            path = Path(inspect.getsourcefile(target))
            sources.append(dict(callable=target.__qualname__, path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            target = getattr(target, '__wrapped__', None)
    if not {'state', 'data_iterator', 'model'}.issubset(inspect.signature(forward_step_modelopt).parameters):
        raise RuntimeError('installed forward_step_modelopt API differs from inspected source')
    if 'callbacks' not in inspect.signature(pretrain).parameters:
        raise RuntimeError('installed native pretrain lacks callbacks')
    return dict(sources=sources, optimizer_loop='native Bridge pretrain',
                loss='native forward_step_modelopt -> loss_func_kd',
                scheduler='native SchedulerConfig; no per-step LR rewrite',
                checkpoint_integrity='not_checked_explicit_user_waiver')


def run_native(c, output, config_only=False, reload_checkpoint=None, diagnostics=False):
    import traceback
    import torch
    import torch.distributed as dist
    import yaml
    from megatron.core.utils import unwrap_model
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.gpt_step import forward_step_modelopt
    from megatron.bridge.models.distillation_provider import DistillationProvider

    rank = int(os.environ.get('RANK', '0'))
    output = Path(output).resolve()
    # Each invocation must have a fresh directory. Rank-specific exclusive claims
    # catch accidental restarts without racing the other ranks' shared mkdir.
    output.mkdir(parents=True, exist_ok=True)
    (output / f'rank-{rank}-claim').open('x').close()
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    def publish(kind, value):
        (output / f'rank-{rank}-{kind}.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

    data = data_preflight(c, reload_only=reload_checkpoint is not None)
    semantics = None if reload_checkpoint is not None else tokenizer_gate(ROOT / c['student_hf'], ROOT / c['teacher_hf'])
    cfg = make_config(c, make_dataset_provider(c, reload_checkpoint is not None), output,
                      config_only=config_only, reload_checkpoint=reload_checkpoint)
    if reload_checkpoint is None:
        assert isinstance(cfg.model, DistillationProvider)
    sources = native_source_receipt()
    publish('source', sources)
    if reload_checkpoint is None:
        publish('patches', cfg.model._nemochat_patch_receipt)
    publish('preflight', dict(data=data, tokenizer=semantics, teacher_revision=c['teacher_revision'],
        student_hf=c['student_hf'], student_native_lineage=c['student_native_lineage'],
        model_only_warmstart=True, optimizer_reset=True, rng_reset=True,
        reload_checkpoint=str(reload_checkpoint) if reload_checkpoint else None,
        checkpoint_integrity=c['checkpoint_integrity']))
    if rank == 0:
        cfg.to_yaml(str(output / 'native-config.yaml'))
        (output / 'stage-config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
        # Teacher/KD are intentionally omitted by upstream provider serialization.
        # Preserve their independent config explicitly without mutating ConfigContainer.
        if reload_checkpoint is None:
            from dataclasses import fields
            teacher_cfg = {field.name: getattr(cfg.model.teacher, field.name)
                           for field in fields(cfg.model.teacher) if not field.name.startswith('_')}
            (output / 'teacher-provider.json').write_text(json.dumps(snapshot(teacher_cfg), indent=2))
    if config_only:
        publish('config-only', dict(status='NATIVE_CONFIG_CONSTRUCTED', weights_loaded=False,
                                   optimizer_updates=0, teacher_hidden_size=4096 if reload_checkpoint is None else None))
        print('NATIVE_CONFIG_CONSTRUCTED', flush=True)
        return
    if int(os.environ.get('WORLD_SIZE', '1')) != c['world_size']:
        raise RuntimeError('native pilot/reload require exactly eight ranks')
    if reload_checkpoint is None and (ROOT / c['checkpoint_save']).exists():
        raise ValueError('refusing existing checkpoint output')

    updates = []
    saves = []
    evaluations = []
    started = []
    diagnostic_handles = []

    def all_rank_gate(check):
        error = None
        result = None
        try:
            result = check()
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        okay = torch.tensor(int(error is None), device='cuda', dtype=torch.int32)
        dist.all_reduce(okay, op=dist.ReduceOp.MIN)
        if not bool(okay):
            raise RuntimeError(error or 'another rank failed the native audit gate')
        return result

    def model_gate(ctx):
        model = unwrap_model(ctx.model[0])
        check_provider_shapes(model.config, None if reload_checkpoint is not None else model.teacher_model.config)
        if reload_checkpoint is not None:
            if hasattr(model, 'teacher_model'):
                raise RuntimeError('reload must build only the student, without a teacher')
            return dict(teacher_present=False)
        imported = getattr(cfg.model, '_nemochat_hf_import_receipt', {})
        if imported.get('alias_required') and imported.get('native_embedding_transform_calls', 0) < 1:
            raise RuntimeError('student learned embedding did not pass through the native HF import')
        return check_teacher(model.teacher_model)

    class Audit(Callback):
        def on_train_start(self, ctx):
            frozen = all_rank_gate(lambda: model_gate(ctx))
            if int(ctx.state.train_state.step) != 0:
                raise RuntimeError('BASE warmstart must reset native update cursor')
            publish('start', dict(rank=rank, world_size=dist.get_world_size(), **frozen))
            publish('hf-import', getattr(cfg.model, '_nemochat_hf_import_receipt', {}))
            if diagnostics:
                spec = importlib.util.spec_from_file_location('midtraining_kd_finite_probe',
                    ROOT / 'midtraining/patches/kd_finite_probe.py')
                probe = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(probe)
                model = unwrap_model(ctx.model[0])
                path = output / f'rank-{rank}-activation-observations.jsonl'
                diagnostic_handles.extend(probe.install(model, 'student', path))
                diagnostic_handles.extend(probe.install(model.teacher_model, 'teacher', path))
                publish('diagnostic-mode', dict(observation_only=True, timing_not_comparable=True,
                                               native_loss_and_optimizer_unchanged=True))

        def on_train_step_start(self, ctx):
            if reload_checkpoint is not None:
                raise RuntimeError('reload evaluation must never enter the optimizer loop')
            all_rank_gate(lambda: model_gate(ctx))
            started[:] = [float(ctx.optimizer.param_groups[0]['lr'])]

        def on_train_step_end(self, ctx):
            publish('last-step-observation', dict(observation_only=True,
                native_step_before_increment=int(ctx.state.train_state.step),
                losses={str(key): repr(float(value)) for key, value in ctx.loss_dict.items()},
                grad_norm=repr(float(ctx.grad_norm)) if ctx.grad_norm is not None else 'None',
                skipped_iter=bool(ctx.skipped_iter), lr_used=started[0]))
            def gate():
                model_gate(ctx)
                return update_metrics(ctx)
            metrics = all_rank_gate(gate)
            step = completed_update_index(ctx, updates)
            updates.append(step)
            row = dict(rank=rank, update=step, metrics=metrics, lr_used=started[0],
                positive_lr_learning_update=started[0] > 0,
                scheduler_num_steps=int(ctx.scheduler.num_steps),
                processed_positions=step * c['train']['global_batch_size'] * c['sequence_length'],
                all_token_loss_mask=True, teacher_frozen_no_grads=True, native_optimizer=True)
            with (output / f'rank-{rank}-updates.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print('NATIVE_KD_UPDATE', json.dumps(row, allow_nan=False), flush=True)

        def on_eval_start(self, ctx):
            all_rank_gate(lambda: model_gate(ctx))

        def on_eval_end(self, ctx):
            def gate():
                model_gate(ctx)
                loss = float(ctx.total_loss_dict['lm loss'])
                if not math.isfinite(loss):
                    raise FloatingPointError('nonfinite native heldout LM Loss')
                return loss
            loss = all_rank_gate(gate)
            row = dict(rank=rank, update=int(ctx.state.train_state.step), lm_loss=loss,
                native_token_weighted=True, teacher_present=reload_checkpoint is None,
                optimizer_updates_this_process=len(updates), fixed_heldout_window=True)
            evaluations.append(row)
            publish('heldout', dict(evaluations=evaluations))

        def on_checkpoint_save(self, ctx):
            all_rank_gate(lambda: model_gate(ctx))
            step = int(ctx.state.train_state.step)
            saves.append(step)
            publish('save', dict(rank=rank, save_steps=saves, native_synchronous_save_completed=True,
                student_only_by_native_modelopt_sharded_state_dict=True,
                save_optim=False, save_rng=False, checkpoint_integrity=c['checkpoint_integrity']))

        def on_train_end(self, ctx):
            publish('train-end', dict(rank=rank, completed_optimizer_updates=len(updates), save_steps=saves))

    try:
        try:
            # Exact native distill() delegation, expanded only to attach supported
            # callbacks. No hand-written forward/loss/backward/optimizer/scheduler.
            pretrain(cfg, forward_step_modelopt, callbacks=[Audit()])
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        if not evaluations:
            raise RuntimeError('native entrypoint exited without finite heldout LM evaluation')
        if reload_checkpoint is None and (updates != list(range(1, 5)) or 4 not in saves):
            raise RuntimeError('native entrypoint did not complete four accepted updates and save at update four')
        if reload_checkpoint is not None and (updates or saves):
            raise RuntimeError('reload gate mutated training state')
    except BaseException:
        (output / f'rank-{rank}-error.log').write_text(traceback.format_exc())
        publish('entrypoint', dict(rank=rank, status='failed', completed_updates=len(updates), save_steps=saves))
        raise
    publish('entrypoint', dict(rank=rank, status='native_returned', native_exit_code=0,
        completed_updates=len(updates), save_steps=saves))
    publish('complete', dict(rank=rank, status='reload_eval_complete' if reload_checkpoint else 'pilot_complete',
        completed_updates=len(updates), save_steps=saves, finite_heldout_evaluations=len(evaluations),
        checkpoint_integrity=c['checkpoint_integrity'], sft_handoff_authorized=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config-only', action='store_true',
                        help='construct/finalize real native providers from local configs; no weights read')
    parser.add_argument('--schema-only', action='store_true', help='CPU YAML validation, not native configuration')
    parser.add_argument('--diagnostics', action='store_true',
                        help='opt-in activation observations; native training unchanged, timing perturbed')
    parser.add_argument('--reload-checkpoint', type=Path,
                        help='student-only native checkpoint reload plus heldout LM eval; no teacher or updates')
    args = parser.parse_args()
    c = load_config(args.config)
    if args.schema_only:
        print(json.dumps(dict(status='CPU schema valid; GPU not executed', config=c), indent=2))
        return
    if args.output is None:
        parser.error('--output is required for native config construction or execution')
    run_native(c, args.output, config_only=args.config_only, reload_checkpoint=args.reload_checkpoint,
               diagnostics=args.diagnostics)


if __name__ == '__main__':
    main()
