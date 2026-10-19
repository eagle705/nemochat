# Core-first cached KD integration for Megatron-Bridge

**Isolated, opt-in integration patch; not activated in training.** The requested architecture is Megatron-LM writer → native tar/zstd storage → native reader → Bridge forward/loss closure, not vLLM-first or a new JSON production cache.

[한국어](README.ko.md) · [Design and gates](FEASIBILITY.md) · [PR draft](PR_DRAFT.md)

## Reviewable implementation

- `megatron-bridge-0.6.1-cached-kd.patch`: applyable diff against installed Bridge **0.6.1**, base **`2a8fcdd05181ffcef261192f71ff5910d0005bf2`**. Adds `training/cached_kd/` and one explicit factory to `training/gpt_step.py`; every existing default/online ModelOpt statement is unchanged.
- `bridge_cached_kd/{logits_saver,utils_logits,cached_logits_loss}.py`: substantive installed Core port. Preserves global-normalized teacher top-K, uint16+17th-bit indices, sentinel handling, tar metadata, zstd/torch payloads, storage abstraction, native reader/prefetch and ghost-tail KL. These are runtime code, not AST-extracted stubs.
- `bridge_cached_kd/runtime.py`: explicit dataset/iteration/microbatch dependencies, scoped without modifying `megatron.training` global args. Fork workers inherit the bound dataset identity; spawn is gated.
- `bridge_cached_kd/integration.py`: validated `CachedKDConfig`, teacher-only `cache_forward` (no backward/optimizer), student main-head hook, per-forward differentiable logits binding, deferred native-scheduler loss tuple, fixed-alpha reduction, LM-only evaluation, deterministic hook cleanup, cursor checks and explicit checkpoint binding.
- `build_patch.py`: reproducible diff/provenance generation. `evidence/integration-base.json` pins source bytes, image and base SHA; `evidence/bridge-installed/` contains exact read-only installed seam snapshots.
- `tests/test_offline_kd_pr_bridge.py`: implementation and SDK interoperability tests. Original `tests/test_offline_kd_pr.py` remains unchanged.

## Objective: deliberately Megatron-LM-compatible

`objective='megatron-lm-ghost-tail-fixed-alpha'`, temperature **1**, explicit alpha (default **0.5**, matching the native callable constructor, not the native CLI's 1.0):

`total = (1-alpha)*masked_LM_sum + alpha*masked_KL_with_ghost_tail_sum`

Return `(total, num_tokens, reports)`; the Bridge scheduler owns backward/normalization, optimizer and LR scheduling. The native tail clamp is 1e-8. This is **not full-vocabulary KL when K<V**, not conditional top-K, and **not the existing ModelOpt detached magnitude balancer**. No optional ModelOpt-compatible cached mode is implemented. Existing online ModelOpt remains separate and unchanged; mixing its `DistillationModel` into the cached adapter is rejected.

## Reproduce from repository root

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_offline_kd_pr.py tests/test_offline_kd_pr_bridge.py
python3 midtraining/patches/offline_kd_pr/build_patch.py
python3 midtraining/patches/offline_kd_pr/run_cpu_sdk.py --label review-fresh-01 --apply-patch --test tests/test_offline_kd_pr.py tests/test_offline_kd_pr_bridge.py
```

Use a new label each time. The SDK runner uses the existing pinned image, UID1001/GID985, `runc`, `NetworkMode=none`, `DeviceRequests=[]`, read-only repository and only a fresh owned `$TMPDIR` scratch bind writable. It reads SDK archives from the **stopped** same-image container `2f1750f9d121488f45a55d1ae66b8390b6ed758e2125e66747c010fe97bc2e92`. Root-only image source ancestors require staging exact source bytes in scratch. A **real host `libcuda.so.1` copy** satisfies Transformer Engine's eager dynamic link; there are no GPU devices, fake libraries, import stubs, downloads or installed SDK edits. The patch is applied only to that scratch copy. The runner records exact Engine ID/scope/exit and preserves warnings/failures under `evidence/<label>.{json,txt}`.

Verified tests include actual unmodified SDK `LogitsSaverHooks` → unmodified SDK `TeacherTarDataset` and port reader interoperability, port writer → reader → student loss/backward, LM-only eval, explicit reader cursor resume, iteration-gap and GBS/MBS rejection, partial-write rejection, and staged Bridge factory import. The SDK test supplies narrow **in-memory test dependency injections** for native CLI getters; it does not replace modules or mutate installed source/global args. This is real native serialization with a tiny CPU teacher fixture, **not actual Nemotron teacher/model training**. Final execution counts and receipts are in `evidence/integration-acceptance.json`.

## Explicit API and remaining limits

After applying to a reviewed SDK copy, import `CachedKDConfig` from `megatron.bridge.training.cached_kd.integration` and use `gpt_step.make_cached_kd_forward_step(config, runtime_factory, resume_binding=...)` in a `with` block as Bridge's `forward_step_func`. The factory binds the real `_forward_step_common`; the CPU end-to-end test injects only the model-forward/batch seam. No `ConfigContainer` or CLI fields are invented: serialize the separate dataclass and frozen runtime identity explicitly.

Native saver hooks require `model.training=True`; `cache_forward` rejects eval mode and runs under `no_grad`. A real producer must explicitly freeze dropout/stochastic-layer policy and prove equivalence to the intended frozen teacher, rather than assume no_grad implies eval behavior.

`runtime_factory(state)` must return `CoreRuntime` with the exact native sampler seed, sequence length, total sample budget, blend/blend_per_split, fixed GBS/MBS, `iteration=lambda: state.train_state.step` and native microbatch count. A nonzero start requires the matching `checkpoint_binding()` from a **completed cache iteration**; persist it alongside the successful native optimizer checkpoint, not after a merely computed loss. Automatic Bridge checkpoint sidecar transaction/save/reload is **not implemented**. `cache_id` is a caller-supplied immutable lineage handle, not a cryptographic authentication of teacher/cache bytes. The Core dataset hash does not authenticate exact teacher/tokenizer/packing semantics; a reviewed producer manifest remains mandatory before production.

The integration currently **rejects distributed TP/CP/DP/PP**, VPP, MTP, recomputation, schedule-plan overlap and CUDA graphs; the underlying Core TP/DP remapping code is preserved, not certified for this adapter. Native-model GPU forward/backward/optimizer, actual checkpoint reload, packed/CP/Mamba reset semantics, rerun/skip recovery, MSC remote publication, exact Base teacher load/parity, BF16 tolerance, throughput and quality remain unverified. The adapter cannot be used to activate the parent's multi-GPU main plan.

`cache_contract.py`, `source_controls.py`, `proof.py` and their prior JSON/CPU evidence remain **fixture/reference only**, not proposed production storage. vLLM adapters/evidence and historical feasibility notes remain optional future material under `evidence/legacy-vllm-feasibility*.md`; vLLM is not an acceptance dependency. Full-corpus full-K caching remains prohibited by capacity: the existing arithmetic records about **439.8 TB for FP16 values alone**. Only a separately authorized bounded tiny full-K comparison is sensible.

No active recovery/main source, corpus/controller/ledger, running container, GPU job, commit or remote PR was changed.
