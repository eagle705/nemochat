#!/usr/bin/env python3
"""Versioned C LM+KD1000 native SFT; inert without parent release.

Parent owns GPU authorization, cooperative locks and Engine launch. The source
is C LM+KD1000, never BASE; model-only load resets all training state.
"""
import argparse
import functools
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01'
CONFIG = Path('sft/configs/recipe-v1.yaml')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def stage_plan(config, manifest, *, pilot=False, duration_control=False):
    if config['source_checkpoint'] != SOURCE:
        raise ValueError('additional SFT requires C LM+KD1000 model-only source')
    if config['checkpoint_integrity'] != 'not_checked_explicit_user_waiver':
        raise ValueError('explicit checkpoint byte-audit waiver required')
    if (manifest.get('status') != 'passed' or manifest.get('schema') != 'nemochat-nemotron-sft-v1'
            or manifest.get('format') != 'official-super35-selected-assistant-v1'):
        raise ValueError('passed official Super3.5 complete-source prepared data required')
    sequence = config['sequence_length']
    if sequence not in (4096,16384) or manifest['sequence_length'] != sequence:
        raise ValueError('approved native context must match prepared 4096 or 16384')
    if (config.get('context_parallel_size',1),config.get('expert_model_parallel_size',8)) != (1,8):
        raise ValueError('validated runtime requires CP1 EP8; CP2 requires separate native acceptance')
    if sequence == 16384 and (config.get('excluded_datasets') != ['nvidia/Nemotron-SFT-Long-Context-Transduction-v1'] or
            manifest.get('excluded_datasets') != config['excluded_datasets']):
        raise ValueError('16K whole-dataset exclusion policy required')
    if manifest['train_heldout_overlap'] or manifest['kmmlu_exact_question_matches'] or manifest['training_repetition']:
        raise ValueError('data leakage/repetition forbidden')
    if manifest['kmmlu_reserved_questions'] != 34871:
        raise ValueError('complete prior KMMLU dev/test reservation required')
    n = manifest['train_dialogues']
    mode = config.get('execution_mode', 'one_epoch')
    if mode not in ('one_epoch', 'longrun', 'duration_control', 'calibration'):
        raise ValueError('unknown execution_mode')
    extended = mode != 'one_epoch'
    if mode == 'duration_control' and not (pilot and duration_control):
        raise ValueError('duration-control requires explicit --pilot --duration-control')
    if duration_control and mode != 'duration_control':
        raise ValueError('duration-control config required')
    if extended:
        if mode in ('longrun','calibration') and pilot:
            raise ValueError('production longrun cannot use pilot bypass')
        if not 0 < config['max_hours'] <= (5/60 if duration_control else 48):
            raise ValueError('duration-control limited to 5 minutes; longrun limited to 48 hours')
        epoch_limit = 1000 if duration_control else config.get('authorized_max_epochs',3)
        if type(epoch_limit) is not int or not 1 <= epoch_limit <= 1000:
            raise ValueError('explicit bounded authorized epoch ceiling required')
        if type(config['epochs']) is not int or not 1 <= config['epochs'] <= epoch_limit:
            raise ValueError('explicit bounded epochs required')
        if config['epochs'] > 1 and config.get('allow_repetition') is not True:
            raise ValueError('explicit repetition permission required')
        if not 1 <= config['max_updates'] <= 100000 or not 1 <= config['watchdog_grace_minutes'] <= 60:
            raise ValueError('bounded updates and watchdog grace required')
    if pilot:
        if not 32 <= n <= 128 or manifest['heldout_dialogues'] < 1:
            raise ValueError('pilot requires 32..128 unique train examples and real heldout')
        batch = 8
    else:
        if not (n >= 50000 and (extended or n <= 100000)) or manifest['heldout_dialogues'] < min(1000, n // 200):
            raise ValueError('main requires 50000..100000 unique dialogues and reliable heldout')
        if manifest.get('prior_heldout_reservation_file') != config['prior_heldout_reservation'] or not manifest.get('prior_reserved_prompt_groups'):
            raise ValueError('main must preserve entire prior heldout reservation with --reserve-heldout')
        batch = 128
    fixed = dict(micro_batch_size=1,epochs=1,max_hours=24,max_updates=1000,save_interval=250,
                 save_optim=True,save_rng=True,most_recent_k=3,artifact_upload='forbidden',
                 learning_rate=1e-5,min_learning_rate=1e-6,weight_decay=.01,seed=42,
                 reasoning_policy='nonthinking_omit_reasoning_content')
    if extended:
        for key in ('epochs', 'max_hours', 'max_updates'):
            fixed.pop(key)
    elif pilot and sequence == 16384 and 0 < config['max_hours'] <= .5:
        fixed.pop('max_hours')
    if any(config.get(k) != v for k,v in fixed.items()):
        raise ValueError('bounded native continuation contract drift')
    available = math.ceil(n/batch)*config['epochs']
    updates = min(available, config['max_updates']) if extended else math.ceil(n/batch)
    if mode == 'longrun':
        if (not isinstance(config.get('estimated_iteration_seconds'),(int,float)) or
                not math.isfinite(config['estimated_iteration_seconds']) or config['estimated_iteration_seconds'] <= 0):
            raise ValueError('fresh sequence-specific native calibration required before freezing duration')
        projected_seconds = updates*config['estimated_iteration_seconds']/(1-config['assumed_overhead_fraction'])
        if projected_seconds < config['max_hours']*3600:
            raise ValueError('duration budget insufficient for 48h target; expand unique corpus or explicitly approve more epochs')
    result = dict(config, pilot=pilot, execution_mode=mode, global_batch_size=batch, updates=updates,
                  available_updates=available, unique_dialogues=n,
                  planned_exposures=min(config['epochs'], updates//math.ceil(n/batch))*n +
                      min(n, (updates % math.ceil(n/batch))*batch))
    if not extended and result['updates'] > (16 if pilot else config['max_updates']):
        raise ValueError('update budget exceeded')
    return result


def verify_prepared_shapes(data, manifest):
    import numpy as np
    sequence = manifest['sequence_length']
    excluded_keys = {name.removeprefix('nvidia/') for name in manifest.get('excluded_datasets',[])}
    receipt = dict(sequence_length=sequence)
    for split in ('train','heldout'):
        n = manifest[split+'_dialogues']
        tokens = np.load(Path(data)/(split+'-tokens.npy'),mmap_mode='r')
        masks = np.load(Path(data)/(split+'-masks.npy'),mmap_mode='r')
        if tokens.shape != (n,sequence+1) or masks.shape != (n,sequence):
            raise ValueError('prepared shifted array shape mismatch')
        count = 0
        with (Path(data)/(split+'-dialogues.jsonl')).open() as handle:
            for line in handle:
                row = json.loads(line)
                if row['source_key'].split('--',1)[0] in excluded_keys:
                    raise ValueError('physically excluded dataset found in prepared dialogues')
                count += 1
        if count != n:
            raise ValueError('prepared dialogue metadata count mismatch')
        receipt[split+'_loss_tokens_beyond_4096'] = int(masks[:,4096:].sum(dtype=np.int64))
    if sequence == 16384 and receipt['train_loss_tokens_beyond_4096'] <= 0:
        raise ValueError('16K requires real supervised targets beyond 4096')
    return receipt


def verify_inputs(manifest_path, tokenizer_manifest, *, hash_data=True):
    import hashlib
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    migration = json.loads(Path(tokenizer_manifest).read_text())
    if (migration.get('status') != 'passed' or migration.get('vocab_size') != 131072 or
            migration.get('allowed_changed_ids') != list(range(18,30)) or
            manifest.get('tokenizer_revision') != migration.get('new_revision') or
            manifest.get('tokenizer_path') != migration.get('tokenizer_path')):
        raise ValueError('passed exact migrated-tokenizer lineage required')
    if len(migration.get('datasetReservedCounts',{})) != 5 or any(
            v.get('status') != 'passed' or sum(v.get('stored_counts',{}).values())
            for v in migration['datasetReservedCounts'].values()):
        raise ValueError('complete reserved-ID audit required')
    from nemochat.data.nemotron_sft import PINS, pinned_sources
    registered = manifest.get('sources', [])
    excluded = set(manifest.get('excluded_datasets',[]))
    if excluded and excluded != {'nvidia/Nemotron-SFT-Long-Context-Transduction-v1'}:
        raise ValueError('unknown excluded dataset policy')
    if {s['dataset'] for s in registered} != {'nvidia/'+name for name in PINS}:
        raise ValueError('all five pinned datasets must be registered, seed reserved separately')
    allowed = {(s['dataset'],s['path']):s for s in pinned_sources(ROOT/'results/pretraining/nemochat_baby/nemotron_sft/discovery')}
    trained = set()
    for source in registered:
        expected = allowed.get((source['dataset'],source['path']))
        if not expected or any(source.get(k) != expected[k] for k in expected):
            raise ValueError('pinned source metadata drift')
        stats = manifest['per_source'][source['key']]
        if source['dataset'] in excluded:
            if any(stats.get(k,0) for k in ('accepted','train','heldout','supervised_tokens')):
                raise ValueError('excluded dataset cannot enter train or heldout')
        elif source['purpose'] == 'reserved_rl_seed':
            if stats.get('accepted',0) or stats.get('reserved_rl_rows',0) < 1:
                raise ValueError('RL seed must remain reserved without SFT targets')
        elif stats.get('train',0):
            trained.add(source['dataset'])
    if trained != {'nvidia/'+name for name in PINS if 'Seed' not in name} - excluded:
        raise ValueError('each requested SFT dataset requires actual TRAIN examples')
    if manifest.get('prior_heldout_reservation_file'):
        reservation = {json.loads(line)['prompt_group'] for line in Path(manifest['prior_heldout_reservation_file']).open()}
        train_groups = {json.loads(line)['prompt_group'] for line in (manifest_path.parent/'train-dialogues.jsonl').open()}
        if reservation & train_groups or len(reservation) != manifest['prior_reserved_prompt_groups']:
            raise ValueError('prior heldout reservation leaked or changed')
    for name, info in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('prepared manifest path traversal')
        path = manifest_path.parent/name
        if path.stat().st_size != info['bytes']:
            raise ValueError('prepared file size drift')
        if hash_data:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda:handle.read(4*1024**2),b''):
                    digest.update(block)
            if digest.hexdigest() != info['sha256']:
                raise ValueError('prepared file hash drift')
    shapes = verify_prepared_shapes(manifest_path.parent,manifest) if manifest['sequence_length'] == 16384 else {}
    return dict(data_files_verified=len(manifest['files']),data_hashes_verified=hash_data,
                checkpoint_bytes_read=False,tokenizer_revision=migration['new_revision'],**shapes)


class PreparedRows:
    """Explicit seeded epochs, each rounded with zero-loss native tail rows."""
    def __init__(self, data, split, samples, batch=128, *, epochs=1, seed=42, window=None, sequence_length=4096):
        import numpy as np
        self.tokens = np.load(Path(data)/(split+'-tokens.npy'),mmap_mode='r')
        self.masks = np.load(Path(data)/(split+'-masks.npy'),mmap_mode='r')
        self.n = len(self.tokens)
        self.samples, self.split, self.batch = samples, split, batch
        self.epoch_samples = math.ceil(self.n/batch)*batch
        self.sequence_length = sequence_length
        if not self.n or self.tokens.shape[1] != sequence_length+1 or self.masks.shape != (self.n,sequence_length):
            raise ValueError('nonempty aligned shifted arrays required')
        if split == 'train' and not 1 <= samples <= epochs*self.epoch_samples:
            raise ValueError('explicit epoch sample budget exceeded')
        self.orders = [np.random.default_rng(seed+epoch).permutation(self.n) for epoch in range(epochs)] if split == 'train' else [np.arange(min(self.n, window or self.n))]
        self.fixed_window = split != 'train' and window is not None
        if self.fixed_window:
            from collections import defaultdict, deque
            groups = defaultdict(list)
            with (Path(data)/(split+'-dialogues.jsonl')).open() as handle:
                for index,line in enumerate(handle):
                    groups[json.loads(line)['source_key']].append(index)
            if sum(map(len,groups.values())) != self.n:
                raise ValueError('heldout metadata/array row mismatch')
            queues = [deque(np.random.default_rng(seed).permutation(groups[key]).tolist()) for key in sorted(groups)]
            selected = []
            while len(selected) < min(window,self.n):
                for queue in queues:
                    if queue and len(selected) < min(window,self.n):
                        selected.append(queue.popleft())
            self.orders = [np.asarray(selected)]
        self.order = self.orders[0]
        self.window_samples = math.ceil(len(self.order)/batch)*batch

    def coverage(self, updates):
        import numpy as np
        positions = updates*self.batch
        if not 0 <= positions <= self.samples:
            raise ValueError('performed prefix outside planned epoch budget')
        counts = None
        loss, exposures, unique = 0, 0, set()
        for epoch, order in enumerate(self.orders):
            take = min(self.n, max(0, positions-epoch*self.epoch_samples))
            selected = order[:take]
            for start in range(0,len(selected),256):
                loss += int(self.masks[selected[start:start+256]].sum(dtype=np.int64))
            exposures += take
            unique.update(map(int, selected))
        return dict(loss_tokens=loss, exposures=exposures, unique_dialogues=len(unique),
                    tail_rows=positions-exposures, processed_positions=positions*self.sequence_length)
    def __len__(self):
        return self.samples
    def __getitem__(self, index):
        import numpy as np
        import torch
        if not 0 <= index < self.samples:
            raise IndexError(index)
        epoch, offset = divmod(index, self.epoch_samples) if self.split == 'train' else (0, index % (self.window_samples if self.fixed_window else len(self.order)))
        if (self.split == 'train' and offset >= self.n) or (self.fixed_window and offset >= len(self.order)):
            tokens = torch.full((self.sequence_length+1,),11,dtype=torch.long)
            mask = torch.zeros(self.sequence_length)
        else:
            row = self.orders[epoch][offset]
            tokens = torch.tensor(self.tokens[row].astype(np.int64))
            mask = torch.tensor(self.masks[row],dtype=torch.float32)
        return dict(tokens=tokens[:-1],labels=tokens[1:],loss_mask=mask,position_ids=torch.arange(self.sequence_length))


def next_update_lr(update, horizon, remaining, iteration_seconds, scheduled_lr, minimum_lr):
    return minimum_lr if update == horizon or remaining <= iteration_seconds else scheduled_lr


def final_evaluation_due(update, elapsed_seconds, plan):
    return update == plan['updates'] or elapsed_seconds >= plan['max_hours']*3600


def in_loop_final_evaluation_due(update, elapsed_seconds, plan):
    """Duration exits need an in-loop eval; normal exits get the SDK final eval."""
    return update < plan['updates'] and final_evaluation_due(update, elapsed_seconds, plan)


def suspend_evaluation_deadline(train_config):
    """Finish bounded native validation within outer save grace, not more training."""
    limit = train_config.exit_duration_in_mins
    train_config.exit_duration_in_mins = None
    return limit


def restore_evaluation_deadline(train_config, limit):
    train_config.exit_duration_in_mins = limit


def masked_sum(output, mask):
    import torch
    values, mask = output.float().reshape(-1), mask.float().reshape(-1)
    if values.numel() != mask.numel() or not bool(torch.isfinite(values).all()):
        raise FloatingPointError('invalid/nonfinite native token losses')
    total = (values*mask).sum()
    count = mask.sum().to(torch.int)
    return total,count,{'lm loss':torch.stack([total.detach().clone(),count.to(total.dtype)])}


def cuda_memory_receipt(cuda):
    return dict(cuda_allocated_bytes=cuda.memory_allocated(),cuda_reserved_bytes=cuda.memory_reserved(),
                cuda_peak_allocated_bytes=cuda.max_memory_allocated(),cuda_peak_reserved_bytes=cuda.max_memory_reserved())


def configure_provider_runtime(provider, config):
    """Change runtime context only after frozen Baby parameter-shape validation."""
    provider.seq_length = config['sequence_length']
    if hasattr(provider,'max_sequence_length'):
        provider.max_sequence_length = config['sequence_length']
    provider.context_parallel_size = config.get('context_parallel_size',1)
    provider.expert_model_parallel_size = config.get('expert_model_parallel_size',8)


def make_config(c, m, Dataset, output, online=False):
    import torch
    import yaml
    from megatron.bridge.training.config import (ConfigContainer,TrainingConfig,OptimizerConfig,SchedulerConfig,
        LoggerConfig,TokenizerConfig,CheckpointConfig,DistributedDataParallelConfig,ValidationConfig,RNGConfig)
    parent = module(ROOT/'results/pretraining/nemochat_baby/training/runner.py','nemotron_frozen_baby')
    original = yaml.safe_load((ROOT/'results/pretraining/nemochat_baby/training/train.yaml').read_text())
    kd = module(ROOT/'midtraining/training/native_kd.py','sft_after_kd_hf_helpers')
    provider = kd.provider_from_hf(ROOT/SOURCE,load_weights=c.get('load_hf_weights',True))
    kd.configure_provider(provider,dict(parallel=dict(tensor_model_parallel_size=1,pipeline_model_parallel_size=1,context_parallel_size=1,expert_model_parallel_size=8),sequence_length=c['sequence_length']))
    kd.check_provider_shapes(provider)
    configure_provider_runtime(provider,c)
    # Architecture, EP8 and 131072 embedding rows remain identical. NullTokenizer
    # is native vocabulary metadata only; all real text used migrated official assets.
    cfg = ConfigContainer(model=provider,
        train=TrainingConfig(train_iters=c['updates'],micro_batch_size=1,global_batch_size=c['global_batch_size'],exit_duration_in_mins=c['max_hours']*60),
        optimizer=OptimizerConfig(lr=c['learning_rate'],min_lr=c['min_learning_rate'],weight_decay=.01,bf16=True,
            params_dtype=torch.bfloat16,use_distributed_optimizer=True,overlap_param_gather=False,clip_grad=1.,adam_beta1=.9,adam_beta2=.999,adam_eps=1e-8),
        ddp=DistributedDataParallelConfig(**original['ddp']),
        scheduler=SchedulerConfig(lr_decay_style='cosine',lr_decay_iters=c['updates'],lr_warmup_iters=max(1,math.ceil(c['updates']*.03)),
            start_weight_decay=.01,end_weight_decay=.01,weight_decay_incr_style='constant',use_checkpoint_opt_param_scheduler=False),
        dataset=Dataset(dataloader_type='single',num_workers=0,persistent_workers=False),
        tokenizer=TokenizerConfig(**original['tokenizer']),
        logger=LoggerConfig(log_interval=1,wandb_project='nemochat' if online else None,wandb_entity='nvidia' if online else None,
            wandb_exp_name=c['name'] if online else None,wandb_save_dir=str(output/'sdk-logs')),
        checkpoint=CheckpointConfig(load=None,pretrained_checkpoint=None,finetune=False,load_optim=False,load_rng=False,
            load_main_params_from_ckpt=False,exit_on_missing_checkpoint=True,save=str(output/'checkpoint'),save_interval=250,
            save_optim=True,save_rng=True,async_save=False,fully_parallel_save=False,ckpt_format='torch_dist',
            save_tokenizer_assets=False,verify_integrity=False,most_recent_k=3),
        validation=ValidationConfig(eval_iters=max(1,math.ceil(min(m['heldout_dialogues'],c.get('validation_max_dialogues',m['heldout_dialogues']))/8)),eval_interval=c.get('validation_interval',100),eval_global_batch_size=8,eval_micro_batch_size=1),
        rng=RNGConfig(seed=42),mixed_precision='bf16_mixed')
    cfg.dist.distributed_timeout_minutes = 30
    cfg.dist.distributed_timeout_seconds_after_init = 1800
    # HF provider hook owns model-only restore; native checkpoint loader stays disabled.
    cfg._checkpoint_load_required = False
    return cfg,parent


def run(args, c, manifest):
    import torch
    import torch.distributed as dist
    from dataclasses import dataclass
    from megatron.bridge.training.callbacks import Callback
    from megatron.bridge.data.base import DatasetProvider
    from megatron.bridge.training.pretrain import pretrain
    rank = int(os.environ.get('RANK','0'))
    data, batch = args.manifest.parent,c['global_batch_size']
    @dataclass
    class FrozenData(DatasetProvider):
        def build_datasets(self, ctx):
            if ctx.train_samples != c['updates']*batch:
                raise ValueError('native sample cursor drift')
            return (PreparedRows(data,'train',ctx.train_samples,batch,epochs=c['epochs'],seed=c['seed'],sequence_length=c['sequence_length']),
                    PreparedRows(data,'heldout',ctx.valid_samples,8,window=c.get('validation_max_dialogues'),sequence_length=c['sequence_length']) if ctx.valid_samples else None,None)
    cfg,parent = make_config(c,manifest,FrozenData,args.output,args.online)
    guard = parent.guard_native_sources()
    def publish(kind, value):
        (args.output/f'rank-{rank}-{kind}.json').write_text(json.dumps(value,indent=2,allow_nan=False))
    if rank == 0:
        publish('source-guard',guard)
        (args.output/'requested-config.json').write_text(json.dumps(parent.config_snapshot(cfg.to_dict()),indent=2,allow_nan=False))
    if args.config_only:
        print('NATIVE_NEMOTRON_SFT_CONFIG_CONSTRUCTED',flush=True)
        return
    if (args.output/'checkpoint').exists():
        raise ValueError('fresh output only; checkpoint exists')
    if args.online and (not os.environ.get('WANDB_RUN_ID','').startswith('nemochat-baby-nemotron-sft-') or os.environ.get('WANDB_RESUME') != 'never'):
        raise ValueError('separate stable Nemotron SFT tracking identity required')
    from wandb.sdk.wandb_run import Run
    def forbid(*a,**kw):
        raise PermissionError('artifact/code/checkpoint uploads forbidden')
    Run.log_artifact = Run.log_code = Run.save = forbid
    completed, saves, counts, current = [0],[],[0,0],{}
    def loss(mask, output):
        total,count,metrics = masked_sum(output,mask)
        if output.requires_grad:
            counts[0] += int(count)
            counts[1] += int(mask.numel())
        return total,count,metrics
    def forward(iterator, model):
        item = next(iterator)
        device = torch.cuda.current_device()
        output = model(input_ids=item['tokens'].to(device),labels=item['labels'].to(device),
            position_ids=item['position_ids'].to(device),attention_mask=None)
        return output,functools.partial(loss,item['loss_mask'].to(device))
    class Audit(Callback):
        def on_train_start(self,ctx):
            current['native_start_time'] = ctx.state.start_time
            if dist.get_world_size() != 8 or ctx.state.train_state.step != 0 or ctx.state.train_state.consumed_train_samples != 0 or ctx.scheduler.num_steps != 0:
                raise RuntimeError('all8 model-only C LM+KD1000 warmstart must reset training state')
            native_step = ctx.optimizer.step
            def checked_step(*a,**kw):
                square = torch.zeros((),device='cuda')
                for chunk in ctx.model:
                    for parameter in chunk.parameters():
                        grad = getattr(parameter,'main_grad',None)
                        if grad is None:
                            grad = parameter.grad
                        if grad is not None:
                            square += grad.detach().float().square().sum()
                okay = torch.tensor(int(bool(torch.isfinite(square)) and float(square)>0),device='cuda',dtype=torch.int)
                dist.all_reduce(okay,op=dist.ReduceOp.MIN)
                if not bool(okay):
                    raise FloatingPointError('all8 finite positive pre-optimizer gradients required')
                return native_step(*a,**kw)
            ctx.optimizer.step = checked_step
            receipt = getattr(cfg.model,'_nemochat_hf_import_receipt',{})
            if (receipt.get('native_embedding_transform_calls',0) < 1 and receipt.get('alias_required')):
                raise RuntimeError('learned C embedding must pass actual native import transform')
            publish('hf-import',receipt)
            publish('load',dict(rank=rank,source=SOURCE,source_midtraining_update=1000,stage_step=0,model_only=True,
                optimizer_scheduler_rng_cursor_reset=True,checkpoint_integrity=c['checkpoint_integrity'],checkpoint_bytes_read=False))
            if ctx.state.wandb_logger:
                ctx.state.wandb_logger.config.update({'nemotron_sft':dict(source_checkpoint=SOURCE,data=manifest,stage_step=0,pilot=c['pilot'])})
        def on_train_step_start(self,ctx):
            current['step_started'] = time.monotonic()
            counts[:] = [0,0]
            update,old = int(ctx.state.train_state.step)+1,ctx.scheduler.num_steps
            if old != (update-1)*batch:
                raise RuntimeError('scheduler cursor mismatch')
            try:
                ctx.scheduler.num_steps = update*batch
                for group in ctx.optimizer.param_groups:
                    group['lr'] = ctx.scheduler.get_lr(group)*group.get('lr_mult',1.)
            finally:
                ctx.scheduler.num_steps = old
            remaining = c['training_stop_timestamp']-time.time()
            current['terminal_update'] = (update == c['updates'] or remaining <= c['conservative_iteration_seconds'])
            for group in ctx.optimizer.param_groups:
                group['lr'] = next_update_lr(update,c['updates'],remaining,c['conservative_iteration_seconds'],group['lr'],c['min_learning_rate']*group.get('lr_mult',1.))
            current['lr'] = float(ctx.optimizer.param_groups[0]['lr'])
        def on_train_step_end(self,ctx):
            value = float(ctx.loss_dict['lm loss'])
            norm = float(ctx.grad_norm) if ctx.grad_norm is not None else float('nan')
            okay = torch.tensor(int(not ctx.skipped_iter and math.isfinite(value) and math.isfinite(norm) and norm>0),device='cuda',dtype=torch.int)
            dist.all_reduce(okay,op=dist.ReduceOp.MIN)
            if not bool(okay):
                raise FloatingPointError('invalid native optimizer update')
            update = int(ctx.state.train_state.step)+1  # callback is preincrement
            if ctx.scheduler.num_steps != update*batch:
                raise RuntimeError('scheduler progression mismatch')
            completed[0] += 1
            tally = torch.tensor(counts,device='cuda',dtype=torch.int64)
            dist.all_reduce(tally)
            row = dict(rank=rank,update=update,loss=value,grad_norm=norm,lr=current['lr'],
                global_loss_tokens=int(tally[0]),global_processed_positions=int(tally[1]),
                native_elapsed_seconds=time.time()-ctx.state.start_time,
                iteration_seconds=time.monotonic()-current['step_started'],
                native_optimizer=True,finite_positive_gradient=True)
            row.update(cuda_memory_receipt(torch.cuda))
            current['last_update_elapsed_seconds'] = row['native_elapsed_seconds']
            if current.get('terminal_update') and update < c['updates']:
                cfg.train.exit_duration_in_mins = max(0,row['native_elapsed_seconds']/60-1e-6)
            if c.get('execution_mode') != 'one_epoch':
                final_due = torch.tensor(int((current.get('terminal_update',False) and update < c['updates']) or in_loop_final_evaluation_due(update,row['native_elapsed_seconds'],c)),device='cuda')
                dist.all_reduce(final_due,op=dist.ReduceOp.MAX)
                if bool(final_due):
                    # Pinned train.py evaluates after this callback/step increment,
                    # before checkpoint_and_decide_exit; use its real config field.
                    cfg.validation.eval_interval = update
            with (args.output/f'rank-{rank}-updates.jsonl').open('a') as handle:
                handle.write(json.dumps(row,allow_nan=False)+'\n')
        def on_eval_start(self,ctx):
            if c.get('execution_mode') != 'one_epoch':
                current['eval_deadline'] = suspend_evaluation_deadline(ctx.state.cfg.train)
        def on_eval_end(self,ctx):
            if 'eval_deadline' in current:
                restore_evaluation_deadline(ctx.state.cfg.train,current.pop('eval_deadline'))
            value = float(ctx.total_loss_dict['lm loss'])
            if not math.isfinite(value):
                raise FloatingPointError('nonfinite native heldout CE')
            with (args.output/f'rank-{rank}-heldout.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(rank=rank,update=int(ctx.state.train_state.step),heldout_ce=value,
                    heldout_ppl=math.exp(value),native_token_weighted=True,heldout_dialogues=manifest['heldout_dialogues'],
                    evaluated_unique_dialogues=min(manifest['heldout_dialogues'],c.get('validation_max_dialogues',manifest['heldout_dialogues'])),
                    window_selection='seeded-source-stratified-roundrobin' if c.get('validation_max_dialogues') else 'legacy-cyclic',
                    fixed_heldout_window_reused=True),allow_nan=False)+'\n')
        def on_checkpoint_save(self,ctx):
            saves.append(int(ctx.state.train_state.step))
            publish('save',dict(rank=rank,save_steps=saves,save_optim=True,save_rng=True,native_synchronous_save_completed=True))
        def on_train_end(self,ctx):
            publish('complete',dict(rank=rank,updates=completed[0],step=int(ctx.state.train_state.step),save_steps=saves))
    code = 0
    native_system_exit = False
    try:
        pretrain(cfg,forward,callbacks=[Audit()])
    except SystemExit as exc:
        native_system_exit = True
        code = exc.code or 0
        if code:
            raise
    except BaseException:
        code = 1
        raise
    finally:
        if 'eval_deadline' in current:
            restore_evaluation_deadline(cfg.train,current.pop('eval_deadline'))
        elapsed = time.time()-current.get('native_start_time',time.time())
        reason = 'unexpected_exit'
        if code == 0 and completed[0] == c['updates']:
            reason = 'completed_budget'
        elif code == 0 and native_system_exit and elapsed >= c['max_hours']*3600:
            reason = 'native_duration_limit'
        if completed[0]:
            rows = PreparedRows(data,'train',c['updates']*batch,batch,epochs=c['epochs'],seed=c['seed'],sequence_length=c['sequence_length'])
            publish('coverage',rows.coverage(completed[0]))
        publish('entrypoint',dict(rank=rank,native_exit_code=code,updates=completed[0],save_steps=saves,
            termination_reason=reason,elapsed_seconds=elapsed,
            native_duration_scope='SDK GlobalState.start_time including initialization, evaluation, saves',
            last_update_elapsed_seconds=current.get('last_update_elapsed_seconds')))


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--duration-control',action='store_true')
    parser.add_argument('--online',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--config-only',action='store_true')
    parser.add_argument('--release',type=Path,help='parent-owned authenticated benchmark/data/calibration release')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    plan = stage_plan(config,manifest,pilot=args.pilot,duration_control=args.duration_control)
    plan['load_hf_weights'] = not args.config_only
    verification = verify_inputs(args.manifest,config['tokenizer_manifest'])
    if args.validate_only:
        print(json.dumps(dict(status='CPU contract/data valid; GPU unrun',plan=plan,verification=verification),indent=2))
        return
    if not args.output:
        parser.error('--output required')
    if not args.config_only:
        if not args.release:
            raise PermissionError('parent-reviewed actual benchmark producer contract/release required')
        release = json.loads(args.release.read_text())
        if release.get('status') != 'parent_accepted_all_prerequisites' or release.get('source_hf') != SOURCE:
            raise PermissionError('C-only parent release required')
        from controls import accept_benchmark, DEADLINE
        accept_benchmark(release['benchmark_proof'],release['producer_contract'])
        if not release.get('data_ready') or not release.get('new_corpus_reserved_id_audit_passed'):
            raise PermissionError('stable data and new-corpus tokenizer audit required')
        if time.time() >= DEADLINE.timestamp()-1800:
            raise TimeoutError('absolute 2026-10-19 noon KST deadline margin exhausted')
        if plan['training_stop_timestamp'] > DEADLINE.timestamp()-1800:
            raise ValueError('absolute deadline save/eval buffer violated')
        plan['max_hours'] = min(plan['max_hours'],(plan['training_stop_timestamp']-time.time())/3600)
        if plan['max_hours'] <= 0:
            raise TimeoutError('no training time before absolute stop')
    run(args,plan,manifest)


if __name__ == '__main__':
    main()
