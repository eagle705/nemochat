#!/usr/bin/env python3
"""Native Bridge LM+KD new-corpus calibration and wall-bounded training.

Imports remain CPU/inert until explicit execution. No private optimizer loop.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('code_concepts', 'algorithmic', 'formal_logic', 'climbmix_replay')
WEIGHTS = (.6, .1, .1, .2)  # Legacy default; immutable calibration recipe.
MIX67_WEIGHTS = (.67, .1, .03, .2)
REVISION = '4ba52e8ffa79e9693936c1a819ac89f551ceeef6'
TEACHER_REPO = 'nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16'


def recognized_weights(value):
    """Only the legacy and explicitly user-approved recovery blends exist."""
    if isinstance(value, dict):
        if set(value) != set(KEYS):
            raise ValueError('approved source weights required')
        value = [value[key] for key in KEYS]
    if not isinstance(value, (list, tuple)) or tuple(value) not in (WEIGHTS, MIX67_WEIGHTS):
        raise ValueError('approved source weights required')
    return tuple(value)


def manifest_weights(manifest):
    # Legacy in-memory prepare_corpus adapter fixtures predate schema 1. Real
    # production input always traverses load_prepared_manifest, requiring weights.
    if 'weights' not in manifest and 'schema_version' not in manifest and 'status' not in manifest:
        return WEIGHTS
    if not isinstance(manifest.get('weights'), dict):
        raise ValueError('producer source weights map required')
    return recognized_weights(manifest.get('weights'))


def dataset_weights(c, manifest=None):
    weights = recognized_weights(c['dataset']['weights'])
    if manifest is not None and weights != manifest_weights(manifest):
        raise ValueError('dataset weights disagree with manifest')
    return weights


def helper(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config(path):
    import yaml
    c = yaml.safe_load(Path(path).read_text())
    validate_config(c)
    return c


def validate_config(c):
    if c['student_hf'] != 'checkpoints/exports/nemochat_baby_base_hf_bf16_02':
        raise ValueError('original BASE model-only warmstart required, not pilot trained weights')
    if (c['teacher_repo'] != TEACHER_REPO or c['teacher_revision'] != REVISION
            or c['teacher_hf'] != f'checkpoints/teachers/nemotron35-super-base/{REVISION}-text'):
        raise ValueError('exact approved Base teacher required')
    if (c['world_size'] != 8 or c['sequence_length'] != 4096 or c['vocab_size'] != 131072
            or c['train']['global_batch_size'] != 128 or c['train']['micro_batch_size'] != 1
            or c['parallel'] != dict(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                context_parallel_size=1, expert_model_parallel_size=8, expert_tensor_parallel_size=1,
                sequence_parallel=False)):
        raise ValueError('approved full-batch BF16 EP8 TP1 PP1 CP1 topology required')
    if (c['optimizer']['lr'] != 1e-5 or c['optimizer']['min_lr'] != 1e-6
            or c['optimizer']['clip_grad'] != 1 or not c['optimizer']['bf16']
            or c['kd'] != dict(skip_lm_loss=False, intermediate_layer_pairs=[], kd_loss_scale=.5,
                             logit_kl_temperature=1., logit_kl_topk=None)
            or c['scheduler']['lr_decay_style'] != 'WSD'
            or c['scheduler']['use_checkpoint_opt_param_scheduler'] is not False):
        raise ValueError('native LM+KD / BF16 / WSD / gradient clip contract changed')
    dataset_weights(c)
    if (c['dataset']['training_cycling'] is not False
            or c['checkpoint'] != dict(save_interval=250, most_recent_k=3, save_optim=True, save_rng=True)
            or c['checkpoint_integrity'] != 'not_checked_explicit_user_waiver'):
        raise ValueError('noncycling blend/full native checkpoint contract changed')
    stage = c['stage']
    expected_duration = dict(calibration=180, production=1440, duration_control=1)
    if stage not in expected_duration or c['train']['exit_duration_in_mins'] != expected_duration[stage]:
        raise ValueError('native stage duration mismatch')
    if stage == 'calibration' and c['train']['train_iters'] != 16:
        raise ValueError('calibration must use 16 native optimizer calls')
    if stage != 'production' and c['tracking']['mode'] != 'offline':
        raise ValueError('calibration/control must be offline')
    if stage == 'production' and ({key:c['tracking'].get(key) for key in ('mode','entity','project')}
            != dict(mode='online', entity='nvidia', project='nemochat')
            or c['logger']['wandb_entity'] != 'nvidia' or c['logger']['wandb_project'] != 'nemochat'):
        raise ValueError('approved online nvidia/nemochat destination required')
    horizon = c['train']['train_iters']
    if horizon is not None:
        schedule = c['scheduler']
        if (type(horizon) is not int or horizon < 2 or schedule['lr_decay_iters'] != horizon
                or not 0 < schedule['lr_warmup_iters'] < horizon
                or not 0 < schedule['lr_wsd_decay_iters'] < horizon):
            raise ValueError('native scheduler horizon must match the actual update budget')
        if stage == 'production' and (schedule['lr_warmup_iters'] != math.ceil(horizon*.01)
                or schedule['lr_wsd_decay_iters'] != math.ceil(horizon*.10)):
            raise ValueError('production horizon requires frozen 1% warmup/10% final decay')
    if not 0 < c['bounds']['watchdog_seconds'] <= 93600 or not 0 < c['bounds']['save_grace_seconds'] <= 1800:
        raise ValueError('outer watchdog/save grace exceed 26h cap')


def freeze_config(c, calibration=None):
    import copy
    c = copy.deepcopy(c)
    if c['stage'] == 'production':
        if calibration is None:
            raise ValueError('accepted native calibration required to freeze horizon')
        plan = derive_horizon(calibration)
        if dataset_weights(c) == MIX67_WEIGHTS:
            # Approved initial budget, not a new-mix timing measurement. Actual
            # native update receipts will supply this main run's warm timings.
            plan.update(old_profile_estimated_train_iters=plan['train_iters'],
                train_iters=3200, lr_decay_iters=3200, lr_warmup_iters=32, lr_wsd_decay_iters=320,
                timing_recipe_weights=list(WEIGHTS), production_weights=list(MIX67_WEIGHTS),
                fresh_new_mix_benchmark=False, actual_new_main_warm_steps_pending=True)
        c['train']['train_iters'] = plan['train_iters']
        for key in ('lr_decay_iters', 'lr_warmup_iters', 'lr_wsd_decay_iters'):
            c['scheduler'][key] = plan[key]
        c['projection'] = plan
    validate_config(c)
    return c


def force_duration_evaluation(elapsed_seconds, duration_minutes, completed, horizon):
    # Normal iteration completion owns exactly the native post-loop final pass.
    return completed < horizon and elapsed_seconds > duration_minutes * 60


TOKENIZER_SHA256 = 'd0432ad8b1c2984a2e2dca82e1933e3fd5628df6e95c51fba4c3ee9950dce299'


def corpus_tokenizer_gate(manifest, audit=None):
    t = manifest.get('tokenizer', {})
    if (t.get('repository') != TEACHER_REPO or t.get('revision') != REVISION
            or t.get('vocab_size') != 131072 or t.get('eos_id') != 11
            or t.get('asset_sha256', {}).get('tokenizer.json') != TOKENIZER_SHA256):
        raise ValueError('new corpus tokenizer exact Base backend identity required')
    if audit is None or audit.get('actual_prepared_bin_scan') is not True:
        raise ValueError('fresh actual new-corpus reserved-ID audit required')
    counts = audit.get('counts', {})
    if set(counts) != {str(i) for i in range(18,30)} or any(type(v) is not int or v != 0 for v in counts.values()):
        raise ValueError('new corpus contains ambiguous reserved token IDs 18..29; blocked pending explicit reviewed alias policy')
    return dict(**t, fresh_reserved_id_audit=audit, old_climbmix_absence_proof_reused=False)


def load_prepared_manifest(path):
    """Consume producer manifest.json only after companion success SHA readback."""
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    success = json.loads(path.with_name('success.json').read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if success.get('phase') != 'prepared_verified' or success.get('result',{}).get('manifest_sha256') != digest:
        raise ValueError('producer success/manifest SHA256 mismatch')
    if success.get('result', {}).get('status', 'prepared_verified') != 'prepared_verified':
        raise ValueError('producer companion result status is not prepared_verified')
    if (value.get('schema_version') != 1 or value.get('status') != 'prepared_verified'
            or value.get('source_order') != list(KEYS)
            or value.get('sequence_length') != 4096 or value.get('cycling') is not False):
        raise ValueError('producer manifest schema/source blend mismatch')
    manifest_weights(value)
    # Run the common prefix/split audit without silently inferring success from
    # prefix existence. The mandatory companion has already been checked above.
    adapt_prepared_manifest(dict(success, result=dict(value,
                               full_bounded_shards_scanned=True)))
    return value, digest


def audit_reserved_ids(manifest):
    """Fresh streaming int32 audit of ALL produced train and validation .bin.

    This is data verification, not a waived checkpoint payload audit. SHA256
    simultaneously binds scanned bytes to producer output; retained counts do
    not make a nonzero alias semantically safe. No tokenizer download/import.
    """
    import numpy as np
    totals = {str(i):0 for i in range(18,30)}
    rows = {}
    for key in KEYS:
        rows[key] = {}
        for split in ('train','validation'):
            entry = manifest['sources'][key][split]
            if entry.get('dtype') != 'int32' or entry.get('truncated_documents') != 0:
                raise ValueError('producer int32 complete-document contract required')
            prefix = ROOT/entry['prefix']
            counts = {str(i):0 for i in range(18,30)}
            count = 0
            digest = hashlib.sha256()
            with Path(str(prefix)+'.bin').open('rb') as stream:
                while True:
                    raw = stream.read(4*1024*1024)
                    if not raw:
                        break
                    if len(raw) % 4:
                        raise ValueError('partial int32 corpus token')
                    digest.update(raw)
                    ids = np.frombuffer(raw,dtype='<i4')
                    if ids.min() < 0 or ids.max() >= 131072:
                        raise ValueError('new corpus token ID out of shared vocabulary')
                    count += len(ids)
                    selected = ids[(ids >= 18) & (ids <= 29)]
                    unique, frequencies = np.unique(selected, return_counts=True)
                    for token, n in zip(unique.tolist(),frequencies.tolist()):
                        counts[str(token)] += n
            if (count != entry['tokens'] or count*4 != entry['bin_bytes']
                    or digest.hexdigest() != entry['bin_sha256']):
                raise ValueError('scanned corpus .bin SHA256/token count mismatch')
            idx = Path(str(prefix)+'.idx')
            if idx.stat().st_size != entry['idx_bytes'] or hashlib.sha256(idx.read_bytes()).hexdigest() != entry['idx_sha256']:
                raise ValueError('producer corpus .idx SHA256/length mismatch')
            rows[key][split] = dict(tokens=count, counts=counts, bin_sha256=digest.hexdigest())
            for token,n in counts.items():
                totals[token] += n
    return dict(actual_prepared_bin_scan=True, counts=totals, sources=rows,
                old_climbmix_absence_proof_reused=False)


def calibration_summary(rows):
    if len(rows) != 16 or [r['update'] for r in rows] != list(range(1, 17)):
        raise ValueError('calibration requires 16 contiguous full-batch updates')
    for r in rows:
        if (r['global_batch_size'] != 128 or r['sequence_length'] != 4096
                or r['profiled'] or not r['positive_gradient']
                or not math.isfinite(r['seconds']) or r['seconds'] <= 0
                or not math.isfinite(r['lr_used']) or r['lr_used'] < 0
                or r['peak_memory_bytes'] <= 0):
            raise ValueError('full-batch unprofiled finite positive-gradient timing required')
    warm = rows[8:]
    seconds = [r['seconds'] for r in warm]
    return dict(warm_updates=[r['update'] for r in warm], excluded_cold_updates=list(range(1, 9)),
        mean_step_seconds=statistics.mean(seconds), median_step_seconds=statistics.median(seconds),
        min_step_seconds=min(seconds), max_step_seconds=max(seconds),
        stdev_step_seconds=statistics.pstdev(seconds), peak_memory_bytes=max(r['peak_memory_bytes'] for r in warm),
        global_batch_size=128, sequence_length=4096, profiled=False,
        zero_lr_optimizer_calls=sum(r['lr_used'] == 0 for r in rows),
        positive_lr_learning_updates=sum(r['lr_used'] > 0 for r in rows),
        sustained_24h_evidence=False)


def derive_horizon(calibration):
    if (calibration['warm_updates'] != list(range(9, 17)) or calibration['profiled']
            or calibration['global_batch_size'] != 128 or calibration['sequence_length'] != 4096):
        raise ValueError('accepted full-batch calibration required')
    seconds = calibration['mean_step_seconds']
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive measured mean required')
    updates = math.floor(86400 * .9 / seconds)
    if updates < 16:
        raise ValueError('measured rate cannot support the long-run horizon')
    return dict(train_iters=updates, lr_warmup_iters=max(1, math.ceil(updates * .01)),
        lr_wsd_decay_iters=max(1, math.ceil(updates * .10)),
        lr_decay_iters=updates, assumed_overhead_fraction=.10, target_wall_seconds=86400,
        projection_only=True, learning_only_24h=False)


def adapt_prepared_manifest(value):
    """Adapter for prepare_corpus.py success.json; never accept progress.json.

    Producer schema: phase=prepared_verified, result.sources[key][split]
    {prefix,tokens,documents,reader_verified}; audit carries split exclusion.
    """
    if value.get('phase') != 'prepared_verified':
        raise ValueError('immutable prepared_verified success manifest required')
    if value.get('status', 'prepared_verified') != 'prepared_verified':
        raise ValueError('producer companion status is not prepared_verified')
    result = value['result']
    if result.get('status', 'prepared_verified') != 'prepared_verified':
        raise ValueError('producer result status is not prepared_verified')
    if result.get('cycling') is not False or result.get('full_bounded_shards_scanned') is not True:
        raise ValueError('producer complete noncycling source audit required')
    if set(result['sources']) != set(KEYS):
        raise ValueError('exact new Specialized sources plus ClimbMix replay required')
    audit = result['audit']
    if any(audit.get(k) != 0 for k in ('train_heldout_token_document_overlap', 'training_duplicate_token_documents')):
        raise ValueError('split overlap/duplicate audit failed')
    prefixes = set()
    for key in KEYS:
        for split in ('train', 'validation'):
            entry = result['sources'][key][split]
            prefix = (ROOT / entry['prefix']).resolve()
            if (entry.get('reader_verified') is not True or entry['tokens'] <= 0
                    or entry['documents'] <= 0 or prefix in prefixes):
                raise ValueError('distinct producer-verified nonempty train/heldout prefixes required')
            prefixes.add(prefix)
            for suffix in ('.bin', '.idx'):
                if not Path(str(prefix)+suffix).is_file():
                    raise ValueError('native corpus pair missing: ' + str(prefix))
    return result


def capacity_gate(manifest, updates, batch, sequence, cursor_samples=0):
    """Budget each native blended source, including builder's 0.5% surplus.

    Extra two sample reserve covers weighted integer rounding. GPTDataset shift
    consumes one further token. Cursor is explicit; no automatic epoch/cycling.
    """
    if updates <= 0 or cursor_samples < 0:
        raise ValueError('positive updates/nonnegative cursor required')
    samples = updates * batch + cursor_samples
    sources = {}
    for key, weight in zip(KEYS, manifest_weights(manifest)):
        required_samples = math.ceil(math.ceil(samples * weight) * 1.005) + 2
        required_tokens = required_samples * sequence + 1
        available = manifest['sources'][key]['train']['tokens']
        if available < required_tokens:
            raise ValueError(f'capacity insufficient for {key}: {available} < {required_tokens}; '
                             'fresh data expansion required, never repeat the calibration pack')
        if manifest['sources'][key]['validation']['tokens'] < sequence + 1:
            raise ValueError('heldout capacity insufficient for ' + key)
        sources[key] = dict(required_samples=required_samples, required_tokens=required_tokens,
                            prepared_tokens=available, fixed_sequence_exposure_weight=weight)
    return dict(sources=sources, no_training_repeat=True, cursor_samples=cursor_samples,
                processed_positions=updates*batch*sequence, builder_surplus_fraction=.005)


def no_repeat_gate(dataset):
    children = getattr(dataset, 'datasets', None)
    if children is not None:
        return dict(sources=[no_repeat_gate(child) for child in children], native_train_epochs=1)
    # MCore GPTDataset document_index concatenates the shuffled documents of
    # every epoch. Do not trust sufficient tokens alone: inspect the built index.
    if not hasattr(dataset, 'document_index') or not hasattr(dataset, 'indices'):
        raise ValueError('native repeat guard cannot inspect GPTDataset document_index')
    if len(dataset.document_index) != len(dataset.indices):
        raise ValueError('native GPTDataset would repeat training documents across epochs')
    return dict(native_train_epochs=1, documents=len(dataset.indices))


def completion_status(stage, updates, horizon, elapsed_seconds, final_save, final_heldout):
    if updates <= 0 or not final_save or not final_heldout:
        raise ValueError('accepted updates, final save and final-update heldout required')
    if stage == 'calibration':
        if updates != 16:
            raise ValueError('calibration needs exactly 16 updates')
        return 'calibration_complete_reload_pending'
    if stage == 'duration_control':
        if elapsed_seconds < 60:
            raise ValueError('duration control did not reach native one-minute deadline')
        return 'duration_control_complete_reload_pending'
    if elapsed_seconds >= 86400:
        return 'duration_complete_reload_pending'
    if updates == horizon:
        return 'iteration_budget_complete_short_of_wall_target'
    raise ValueError('early exit before wall target or iteration budget')


def evaluation_budget_with_startup(samples, eval_iters, eval_batch):
    # Native default allocates periodic + final. Reserve one additional first
    # zero-LR-call heldout pass; no second forced evaluation at normal end.
    train, valid, test = samples
    return [train, valid+eval_iters*eval_batch, test]


def evaluation_interval_for_step(original, completed, horizon, elapsed, duration):
    if completed == 1:
        return 1
    if force_duration_evaluation(elapsed,duration,completed,horizon):
        return completed
    return original


def make_native_config(c, manifest, output, config_only=False, reload_checkpoint=None):
    """Reuse accepted HF/shape adapter, replacing only data and checkpoint policy."""
    from megatron.bridge.training.config import GPTDatasetConfig, CheckpointConfig, runtime_config_update
    kd = helper('longrun_accepted_kd', 'midtraining/native_kd.py')
    paths = [[str(ROOT / manifest['sources'][key][split]['prefix']) for key in KEYS]
             for split in ('train', 'validation')]
    weights = dataset_weights(c, manifest)
    blend = [(paths[0], list(weights)), (paths[1], list(weights)), None]
    if reload_checkpoint is not None:
        blend[0] = None
    ds = GPTDatasetConfig(seq_length=4096, random_seed=c['rng']['seed'], blend=None,
        blend_per_split=blend, split=None, num_dataset_builder_threads=1,
        num_workers=0, dataloader_type='single', persistent_workers=False,
        eod_mask_loss=False, reset_position_ids=False, reset_attention_mask=False,
        skip_getting_attention_mask_from_dataset=True, add_extra_token_to_sequence=True,
        path_to_cache=str(output/'dataset-cache'))
    cfg = kd.make_config(c, lambda **kwargs: ds, output, config_only=config_only,
                         reload_checkpoint=reload_checkpoint)
    cfg.checkpoint = CheckpointConfig(
        save=str(ROOT/c['checkpoint_save']) if reload_checkpoint is None else None,
        load=str(reload_checkpoint) if reload_checkpoint is not None else None,
        pretrained_checkpoint=None, save_interval=c['checkpoint']['save_interval'],
        save_optim=reload_checkpoint is None, save_rng=reload_checkpoint is None,
        load_optim=False, load_rng=False, finetune=reload_checkpoint is not None,
        exit_on_missing_checkpoint=True, async_save=False, fully_parallel_save=False,
        ckpt_format='torch_dist', save_tokenizer_assets=False, verify_integrity=False,
        most_recent_k=3)
    runtime_config_update(cfg)
    return cfg, kd


def install_native_dataset_audit(c, publish, reload_only=False):
    """Load the isolated process-local native observer, never edit installed SDK."""
    return helper('longrun_native_dataset_observer',
                  'midtraining/patches/native_dataset_observer.py').install(
        c, publish, reload_only, no_repeat_gate=no_repeat_gate,
        evaluation_budget_with_startup=evaluation_budget_with_startup,
        keys=KEYS, weights=dataset_weights(c) if c else WEIGHTS)


def run_native(c, output, config_only=False, reload_checkpoint=None):
    import traceback
    import torch
    import torch.distributed as dist
    import yaml
    from megatron.core.utils import unwrap_model
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.training.pretrain import pretrain
    from megatron.bridge.training.gpt_step import forward_step_modelopt, forward_step
    from importlib.metadata import version

    versions = {key: version(package) for key, package in (
        ('bridge', 'megatron-bridge'), ('modelopt', 'nvidia-modelopt'), ('transformers', 'transformers'))}
    if (not versions['bridge'].startswith('0.6.1') or not versions['modelopt'].startswith('0.46')
            or versions['transformers'] != '5.12.1'):
        raise RuntimeError('pinned native runtime version mismatch: ' + repr(versions))
    validate_config(c)
    if not c['train']['train_iters'] or not c['checkpoint_save']:
        raise ValueError('launcher-frozen executable config required')
    output = Path(output).resolve()
    rank = int(os.environ.get('RANK', '0'))
    output.mkdir(parents=True, exist_ok=True)
    (output/f'rank-{rank}-claim').open('x').close()
    def publish(kind, value):
        target = output/f'rank-{rank}-{kind}.json'
        partial = target.with_suffix('.json.partial')
        partial.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
        partial.replace(target)
    manifest_path = ROOT/c['dataset']['manifest']
    manifest, manifest_digest = load_prepared_manifest(manifest_path)
    dataset_weights(c, manifest)
    reserved_audit = audit_reserved_ids(manifest)
    publish('new-corpus-reserved-id-audit', reserved_audit)
    tokenizer = corpus_tokenizer_gate(manifest, reserved_audit)
    capacity = capacity_gate(manifest, c['train']['train_iters'], 128, 4096)
    if not config_only and int(os.environ.get('WORLD_SIZE', '1')) != 8:
        raise RuntimeError('exactly eight native ranks required')
    if reload_checkpoint is None and (ROOT/c['checkpoint_save']).exists():
        raise FileExistsError('fresh checkpoint destination required')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'online' if c['tracking']['mode'] == 'online' and reload_checkpoint is None and not config_only else 'disabled'
    if os.environ['WANDB_MODE'] == 'online':
        # Secret file is mounted only at execution and never put in Engine Env,
        # payload receipts, CLI argv or config. Parser rejects executable syntax.
        env = helper('longrun_wandb_env', 'results/bridge_wandb/wandb_preflight.py').read_wandb_env(
            Path('/run/nemochat-wandb.env'))
        key = env.get('WANDB_API_KEY')
        if not key:
            raise ValueError('authorized online W&B key unavailable')
        os.environ['WANDB_API_KEY'] = key
        os.environ['WANDB_RUN_ID'] = c['tracking']['run_id']
        os.environ['WANDB_RESUME'] = 'never'
        os.environ['WANDB_DISABLE_CODE'] = 'true'
        os.environ['WANDB_DISABLE_GIT'] = 'true'
    cfg, kd = make_native_config(c, manifest, output, config_only, reload_checkpoint)
    # Existing migration gate validates old student vs teacher embedding lineage;
    # it must NEVER stand in for the independent NEW corpus tokenizer receipt.
    semantics = None if reload_checkpoint else kd.tokenizer_gate(ROOT/c['student_hf'], ROOT/c['teacher_hf'])
    publish('preflight', dict(versions=versions, capacity=capacity, tokenizer=tokenizer,
        student_teacher_tokenizer_lineage=semantics, manifest_sha256=manifest_digest,
        student_hf=c['student_hf'], teacher_repo=c['teacher_repo'], teacher_revision=REVISION,
        model_only_warmstart=reload_checkpoint is None, optimizer_reset=True, rng_reset=True,
        clock='GlobalState.start_time includes initialization', checkpoint_integrity=c['checkpoint_integrity']))
    publish('source', kd.native_source_receipt())
    if rank == 0:
        cfg.to_yaml(str(output/'native-config.yaml'))
        (output/'consumed-stage-config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
        if reload_checkpoint is None:
            (output/'teacher-kd-config.json').write_text(json.dumps(dict(
                architecture=kd.architecture(cfg.model.teacher), kd=c['kd'],
                teacher_hf=c['teacher_hf'], shape_patch=cfg.model._nemochat_patch_receipt), indent=2))
    if config_only:
        publish('config-only', dict(status='NATIVE_CONFIG_CONSTRUCTED', optimizer_updates=0, weights_loaded=False))
        return

    updates, saves, evaluations = [], [], []
    native_elapsed = [0.]
    timer = {}
    original_interval = cfg.validation.eval_interval
    duration = cfg.train.exit_duration_in_mins
    eval_deadline = []

    def all_rank_gate(function):
        error, result = None, None
        try:
            result = function()
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        good = torch.tensor(int(error is None), device='cuda', dtype=torch.int32)
        dist.all_reduce(good, op=dist.ReduceOp.MIN)
        if not good.item():
            raise RuntimeError(error or 'another rank failed native audit')
        return result

    def model_gate(ctx):
        model = unwrap_model(ctx.model[0])
        if reload_checkpoint is not None:
            if hasattr(model, 'teacher_model'):
                raise RuntimeError('reload must be teacher-free')
            return dict(teacher_present=False)
        kd.check_provider_shapes(model.config, model.teacher_model.config)
        imported = getattr(cfg.model, '_nemochat_hf_import_receipt', {})
        if imported.get('alias_required') and imported.get('native_embedding_transform_calls', 0) < 1:
            raise RuntimeError('actual student native embedding transform missing')
        return kd.check_teacher(model.teacher_model)

    class Audit(Callback):
        def on_train_start(self, ctx):
            frozen = all_rank_gate(lambda: model_gate(ctx))
            if int(ctx.state.train_state.step) != 0:
                raise ValueError('warmstart optimizer/scheduler/cursor must reset')
            publish('start', dict(rank=rank, world_size=8, **frozen,
                actual_initial_lr=float(ctx.optimizer.param_groups[0]['lr']), native_optimizer=True))
            publish('hf-import', getattr(cfg.model, '_nemochat_hf_import_receipt', {}))

        def on_train_step_start(self, ctx):
            all_rank_gate(lambda: model_gate(ctx))
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timer['lr'] = float(ctx.optimizer.param_groups[0]['lr'])
            timer['start'] = time.perf_counter()

        def on_train_step_end(self, ctx):
            torch.cuda.synchronize()
            seconds = time.perf_counter()-timer['start']
            duration_tensor = torch.tensor([seconds, time.time()-ctx.state.start_time],
                                           device='cuda', dtype=torch.float64)
            dist.all_reduce(duration_tensor, op=dist.ReduceOp.MAX)
            seconds, native_elapsed[0] = [float(v) for v in duration_tensor.tolist()]
            def check():
                model_gate(ctx)
                return kd.update_metrics(ctx)
            metrics = all_rank_gate(check)
            step = kd.completed_update_index(ctx, updates)
            row = dict(update=step, seconds=seconds, lr_used=timer['lr'], metrics=metrics,
                global_batch_size=128, sequence_length=4096, profiled=False,
                positive_gradient=True, peak_memory_bytes=int(torch.cuda.max_memory_allocated()),
                positive_lr_learning_update=timer['lr'] > 0, teacher_frozen_no_grads=True,
                native_optimizer=True, scheduler_num_steps=int(ctx.scheduler.num_steps),
                processed_positions=step*128*4096)
            if not math.isfinite(timer['lr']) or timer['lr'] < 0:
                raise FloatingPointError('invalid actual optimizer LR')
            updates.append(row)
            with (output/f'rank-{rank}-updates.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            cfg.validation.eval_interval = evaluation_interval_for_step(original_interval, step,
                c['train']['train_iters'], native_elapsed[0], duration)
            if force_duration_evaluation(native_elapsed[0], duration, step, c['train']['train_iters']):
                # Native step increments after this callback. The supported
                # in-loop evaluation consumes its reserved early-final window.
                publish('duration-stop-requested', dict(update=step, native_elapsed_seconds=native_elapsed[0]))

        def on_eval_start(self, ctx):
            all_rank_gate(lambda: model_gate(ctx))
            eval_deadline.append(cfg.train.exit_duration_in_mins)
            # Evaluation owns a bounded watchdog grace, not an altered training
            # deadline. Restore BEFORE native checkpoint_and_decide_exit.
            cfg.train.exit_duration_in_mins = None

        def on_eval_end(self, ctx):
            def check():
                model_gate(ctx)
                value = float(ctx.total_loss_dict['lm loss'])
                if not math.isfinite(value):
                    raise FloatingPointError('nonfinite heldout LM Loss')
                return value
            loss = all_rank_gate(check)
            step = int(ctx.state.train_state.step)
            evaluations.append(dict(update=step, lm_loss=loss, native_token_weighted=True,
                teacher_present=reload_checkpoint is None, fixed_heldout_pool_reused=True))
            publish('heldout', dict(evaluations=evaluations))
            cfg.train.exit_duration_in_mins = eval_deadline.pop()
            cfg.validation.eval_interval = original_interval
            native_elapsed[0] = time.time()-ctx.state.start_time

        def on_checkpoint_save(self, ctx):
            all_rank_gate(lambda: model_gate(ctx))
            step = int(ctx.state.train_state.step)
            saves.append(step)
            native_elapsed[0] = time.time()-ctx.state.start_time
            publish('save', dict(save_steps=saves, save_optim=True, save_rng=True,
                native_synchronous_save_completed=True, checkpoint_integrity=c['checkpoint_integrity']))

    restore = install_native_dataset_audit(c, publish, reload_only=reload_checkpoint is not None)
    try:
        try:
            pretrain(cfg, forward_step_modelopt if reload_checkpoint is None else forward_step,
                     callbacks=[Audit()])
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        if reload_checkpoint is not None:
            if updates or saves or not evaluations:
                raise ValueError('teacher-free reload requires finite eval and zero updates/saves')
            status = 'reload_eval_complete'
        else:
            last = len(updates)
            status = completion_status(c['stage'], last, c['train']['train_iters'], native_elapsed[0],
                last in saves, any(e['update'] == last for e in evaluations))
            if c['stage'] == 'calibration':
                publish('calibration', calibration_summary(updates))
        publish('complete', dict(status=status, rank=rank, completed_updates=len(updates),
            positive_lr_learning_updates=sum(r['lr_used'] > 0 for r in updates),
            zero_lr_optimizer_calls=sum(r['lr_used'] == 0 for r in updates),
            save_steps=saves, final_heldout_update=evaluations[-1]['update'],
            finite_heldout_evaluations=len(evaluations), native_elapsed_seconds=native_elapsed[0],
            native_exit_code=0, teacher_free=reload_checkpoint is not None,
            training_24h_accepted=False, remote_tracking_verified=False))
    except BaseException:
        (output/f'rank-{rank}-error.log').write_text(traceback.format_exc())
        publish('failed', dict(completed_updates=len(updates), save_steps=saves, training_24h_accepted=False))
        raise
    finally:
        restore()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('calibration.yaml'))
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preview', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--config-only', action='store_true')
    parser.add_argument('--reload-checkpoint', type=Path)
    args = parser.parse_args(argv)
    c = load_config(args.config)
    if not args.execute and not args.config_only:
        print(json.dumps(dict(status='inert_preview', config=c, native_training_executed=False), indent=2))
        return
    if args.output is None:
        parser.error('--output required')
    run_native(c, args.output, args.config_only, args.reload_checkpoint)


if __name__ == '__main__':
    main()
