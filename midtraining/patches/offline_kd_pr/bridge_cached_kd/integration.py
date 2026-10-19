"""Opt-in Bridge forward-step integration for native Megatron-LM cached KD.

No ModelOpt conversions, no online teacher in read mode, no default overrides.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CachedKDConfig:
    mode: str
    cache_dir: str
    cache_id: str
    alpha: float = 0.5
    top_k: int = 32
    top_p: float | None = None
    min_k: int = 1
    save_dtype: str = 'fp16'
    decode_threads: int = 1
    prefetch_factor: int = 2
    msc_prefetch_depth: int = 2
    objective: str = 'megatron-lm-ghost-tail-fixed-alpha'

    def __post_init__(self):
        if self.mode not in ('write', 'read'):
            raise ValueError('mode must be write or read')
        if not self.cache_dir or not self.cache_id:
            raise ValueError('cache_dir and immutable cache_id are required')
        if not math.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError('alpha must be finite and in [0, 1]')
        if self.objective != 'megatron-lm-ghost-tail-fixed-alpha':
            raise ValueError('only the native Core objective is implemented')
        if self.top_k < 1 or self.min_k < 1 or self.min_k > self.top_k:
            raise ValueError('invalid top_k/min_k')
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError('top_p must be in (0, 1]')
        if self.save_dtype not in ('fp16', 'bf16', 'fp32'):
            raise ValueError('invalid save_dtype')
        if min(self.decode_threads, self.prefetch_factor) < 1 or self.msc_prefetch_depth < 0:
            raise ValueError('invalid prefetch configuration')


class CachedKDForwardStep:
    """A Bridge ForwardStepCallable; construct explicitly, close deterministically.

    runtime_factory(state) supplies Core dataset identity and live iteration.
    Checkpoint bindings are explicit caller-owned sidecars, not silently added to
    Bridge TrainState. A restored run must supply its matching binding.
    """

    def __init__(self, config, runtime_factory, *, resume_binding=None, common_forward=None):
        self.config = config
        self.runtime_factory = runtime_factory
        self.resume_binding = resume_binding
        self.common_forward = common_forward
        self._model = None
        self._runtime = None
        self._capture = None
        self._saver = None
        self._reader = None
        self._next_iteration = None
        self._microbatch = 0
        self._outstanding = False
        self._closed = False

    def _binding(self, runtime, iteration):
        extra = {}
        if runtime.data_parallel_size != 1:
            extra = dict(data_parallel_size=runtime.data_parallel_size,
                         expert_parallel_size=runtime.expert_parallel_size,
                         dataset_identity=runtime.dataset_identity)
        return dict(**extra, cache_id=self.config.cache_id, next_iteration=iteration,
                    consumed_samples=iteration * runtime.global_batch_size,
                    global_batch_size=runtime.global_batch_size,
                    micro_batch_size=runtime.micro_batch_size,
                    seed=runtime.seed, seq_length=runtime.seq_length,
                    train_samples=runtime.train_samples, blend=runtime.blend,
                    blend_per_split=runtime.blend_per_split,
                    objective=self.config.objective, alpha=self.config.alpha)

    def validate_resume(self, runtime):
        iteration = runtime.iteration()
        if iteration < 0:
            raise ValueError('negative resume iteration')
        if iteration and self.resume_binding is None:
            raise ValueError('resume requires checkpoint cache binding')
        if self.resume_binding is not None:
            import json
            # Native GPTDataset blends contain tuples; durable JSON sidecars do
            # not. Compare canonical serialized values, never drop identity fields.
            expected = json.loads(json.dumps(self._binding(runtime, iteration)))
            actual = json.loads(json.dumps(self.resume_binding))
            if actual != expected:
                raise ValueError('checkpoint cache binding does not match runtime')


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(require_complete=exc_type is None)

    def _setup(self, state, model):
        import multiprocessing
        from megatron.core import parallel_state
        from megatron.core.utils import unwrap_model
        from .runtime import bind_runtime, validate_topology
        from .cached_logits_loss import CachedLogitsKDLoss, StudentLogitsCapture
        from .logits_saver import LogitsSaverHooks
        if self._closed:
            raise RuntimeError('cached KD adapter is closed')
        bare = unwrap_model(model)
        from modelopt.torch.distill import DistillationModel
        if isinstance(bare, DistillationModel):
            raise ValueError('cached KD cannot run an online ModelOpt teacher')
        if self._model is not None:
            if bare is not self._model:
                raise RuntimeError('model chunk changed; PP/VPP not implemented')
            return
        cfg = state.cfg.model
        if any(getattr(cfg, key, None) for key in
               ('mtp_num_layers', 'recompute_granularity', 'overlap_moe_expert_parallel_comm',
                'virtual_pipeline_model_parallel_size', 'sequence_packing_scheduler')):
            raise NotImplementedError('MTP/recompute/VPP/packing scheduler/overlap are gated')
        if getattr(cfg, 'cuda_graph_impl', 'none') not in ('none', None):
            raise NotImplementedError('CUDA graphs are gated')
        sizes = [parallel_state.get_tensor_model_parallel_world_size(),
                 parallel_state.get_context_parallel_world_size(),
                 parallel_state.get_data_parallel_world_size(),
                 parallel_state.get_pipeline_model_parallel_world_size()]
        ep = parallel_state.get_expert_model_parallel_world_size()
        if (sizes[0], sizes[1], sizes[2], sizes[3], ep) not in ((1,1,1,1,1), (1,1,8,1,8)):
            raise NotImplementedError('only TP1/CP1/PP1 DP1/EP1 or DP8/EP8 integration is supported')
        if multiprocessing.get_start_method() != 'fork':
            raise NotImplementedError('DataLoader runtime transport currently requires Linux fork')
        if not hasattr(bare, 'output_layer'):
            raise ValueError('model requires a native output_layer')
        runtime = self.runtime_factory(state)
        self.validate_resume(runtime)
        validate_topology(tp=sizes[0], cp=sizes[1], dp=sizes[2], pp=sizes[3], ep=ep,
                          gbs=runtime.global_batch_size, mbs=runtime.micro_batch_size,
                          microbatches=runtime.num_microbatches())
        if runtime.mtp_num_layers or runtime.data_parallel_size != sizes[2] or runtime.expert_parallel_size != ep:
            raise ValueError('runtime/actual parallel topology mismatch')
        if sizes[2] == 8 and not runtime.dataset_identity:
            raise ValueError('DP8 requires immutable full sampler/dataset identity')
        if state.train_state.consumed_train_samples != runtime.iteration() * runtime.global_batch_size:
            raise ValueError('consumed samples do not match cache cursor')
        if runtime.iteration() != state.train_state.step:
            raise ValueError('runtime iteration must equal Bridge train_state.step')
        with bind_runtime(runtime):
            if self.config.mode == 'write':
                from .utils_logits import storage_glob_with_caching
                import torch.distributed as dist
                # All ranks inspect BEFORE any rank can publish a shard.
                existing = storage_glob_with_caching(self.config.cache_dir, '*.tar')
                if dist.is_initialized():
                    dist.barrier()
                if existing:
                    raise FileExistsError('writer requires a fresh native cache directory')
                self._saver = LogitsSaverHooks(self.config.cache_dir, self.config.top_k,
                    p=self.config.top_p, min_k=self.config.min_k, save_dtype=self.config.save_dtype)
                self._saver.attach_hooks(bare)
            else:
                self._capture = StudentLogitsCapture()
                self._capture.attach_hooks(bare)
                self._reader = CachedLogitsKDLoss(self.config.cache_dir,
                    decode_threads=self.config.decode_threads, prefetch_factor=self.config.prefetch_factor,
                    msc_prefetch_depth=self.config.msc_prefetch_depth)
        self._model = bare
        self._runtime = runtime
        self._next_iteration = runtime.iteration()

    def _common(self, state, iterator, model):
        common = self.common_forward
        if common is None:
            from megatron.bridge.training.gpt_step import _forward_step_common
            common = _forward_step_common
        return common(state, iterator, model, False)

    def _check_cursor(self, state):
        if self._outstanding:
            raise RuntimeError('previous deferred loss has not been consumed')
        if state.train_state.step != self._next_iteration or self._runtime.iteration() != self._next_iteration:
            raise ValueError('cache cursor iteration mismatch; skip/rerun is not supported')
        if self._runtime.num_microbatches() * self._runtime.micro_batch_size * self._runtime.data_parallel_size != self._runtime.global_batch_size:
            raise ValueError('microbatch ramp-up is unsupported')

    def _advance(self):
        self._microbatch += 1
        if self._microbatch == self._runtime.num_microbatches():
            self._microbatch = 0
            self._next_iteration += 1

    def cache_forward(self, state, data_iterator, model):
        """Teacher-only native forward; never call a backward/optimizer step.

        Core hooks require training=True. The producer explicitly controls
        train/eval stochasticity and must freeze that policy in cache lineage.
        """
        import torch
        from .runtime import bind_runtime
        if self.config.mode != 'write':
            raise ValueError('cache_forward requires write mode')
        self._setup(state, model)
        if not model.training:
            raise ValueError('Core writer hooks require training=True')
        self._check_cursor(state)
        with bind_runtime(self._runtime), torch.no_grad():
            output, _ = self._common(state, data_iterator, model)
            self._advance()
            if self._microbatch == 0:
                self.flush()
        return output

    def __call__(self, state, data_iterator, model, return_schedule_plan=False):
        import torch
        from .runtime import bind_runtime
        if self.config.mode != 'read':
            raise ValueError('writer must use cache_forward without optimizer/backward')
        if return_schedule_plan:
            raise NotImplementedError('schedule plans are gated')
        self._setup(state, model)
        if model.training:
            self._check_cursor(state)
        with bind_runtime(self._runtime):
            output, mask = self._common(state, data_iterator, model)
        if not model.training:
            def eval_loss(output_tensor):
                loss = (output_tensor.reshape(-1).float() * mask.reshape(-1).float()).sum()
                count = mask.sum().detach().to(torch.int)
                return loss, count, {'lm loss': torch.stack((loss.detach(), count))}
            return output, eval_loss
        logits = self._capture.pop()  # bind THIS microbatch, not a later hook global
        iteration, microbatch = self._next_iteration, self._microbatch
        self._outstanding = True
        used = False

        def loss_function(output_tensor):
            nonlocal used
            if used:
                raise RuntimeError('cached KD loss closure cannot be replayed')
            used = True
            with bind_runtime(self._runtime):
                # Core normalizes logits in-place; clone prevents corruption of
                # saved LM-loss activations and makes FP32 leaf inputs safe.
                kd_tokens = self._reader(logits.clone(), iteration=iteration, microbatch_idx=microbatch)
                if len(self._reader._current_values) != self._runtime.num_microbatches():
                    raise ValueError('native cache microbatch mapping differs from fixed GBS/MBS')
            if output_tensor.shape != mask.shape or kd_tokens.shape != mask.shape:
                raise ValueError('cache/student/loss-mask shape mismatch')
            lm = (output_tensor.float() * mask.float()).sum()
            kd = (kd_tokens.float() * mask.float()).sum()
            # Native Core TP summation is identity under the explicitly gated TP1.
            total = (1-self.config.alpha) * lm + self.config.alpha * kd
            if not torch.isfinite(total):
                raise FloatingPointError('nonfinite cached KD loss')
            count = mask.sum().clone().detach().to(torch.int)
            report = {name: torch.stack((value.detach(), count)) for name, value in
                      [('lm loss',lm), ('logits distillation loss',kd), ('total loss',total)]}
            self._outstanding = False
            self._advance()
            return total, count, report
        return output, loss_function

    def checkpoint_binding(self):
        if self._runtime is None or self._microbatch or self._outstanding:
            raise RuntimeError('checkpoint binding requires a completed iteration')
        return self._binding(self._runtime, self._next_iteration)

    def flush(self):
        if self._saver is not None:
            path, writes, meta, msc = self._saver.take_pending_data()
            self._saver._write_batched_tar(path, writes, meta, msc)

    def close(self, *, require_complete=True):
        from .runtime import bind_runtime
        partial_write = self._saver is not None and self._microbatch != 0
        try:
            if self._saver is not None:
                with bind_runtime(self._runtime):
                    self.flush()
        finally:
            for hooks in (self._capture, self._saver):
                if hooks is not None:
                    hooks.remove_hooks()
            if self._reader is not None:
                loader_iter = self._reader._dataloader_iter
                if loader_iter is not None and hasattr(loader_iter, '_shutdown_workers'):
                    loader_iter._shutdown_workers()  # pinned PyTorch private lifecycle API
                self._reader._dataloader_iter = None
                self._reader._current_values = None
                self._reader._current_indices = None
            self._closed = True
        if require_complete and partial_write:
            raise RuntimeError('teacher cache closed with a partial iteration; not ready')
