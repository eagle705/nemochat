# DRAFT — Port Megatron-LM native cached-logits KD into Bridge

[한국어](PR_DRAFT.ko.md) · [Implementation and commands](README.md) · [Detailed gates](FEASIBILITY.md)

**Local review only. No commit, remote branch or PR was created. No installed SDK or active training change was made.** An actual applyable diff now exists and was applied only to a disposable SDK copy.

## Target and motivation

Backport/review base: installed **Megatron-Bridge 0.6.1, `2a8fcdd05181ffcef261192f71ff5910d0005bf2`**, image pinned in `evidence/integration-base.json`. Rebase onto upstream main is separate work: the earlier inspected upstream SHA `2311cf8c32ee895ab96db8391996c8f3732e1e7d` is not this patch base.

Follow Megatron-LM's existing offline KD design nearly verbatim: native `LogitsSaverHooks`, tar/zstd storage and reader, output-head capture and ghost-tail KL. Avoid a second JSON production schema or an unrelated vLLM-first serving pipeline. Preserve all default/online ModelOpt call sites.

## Changes in the diff

1. Add `training/cached_kd/{logits_saver,utils_logits,cached_logits_loss}.py`, ported from the exact installed Core snapshots with copyright retained. Preserve native tensor/storage/math paths; replace CLI-only args/print/microbatch imports with explicit Bridge runtime dependencies. Add missing-iteration rejection to the reader.
2. Add scoped `CoreRuntime` and validated `CachedKDConfig`; no invented `ConfigContainer` fields or global CLI initialization. The caller freezes the exact sampler/blend/sample budget and live Bridge step/microbatch getters.
3. Add `CachedKDForwardStep`: teacher-only `cache_forward` under no_grad (never optimizer/backward), synchronous complete-iteration native flush, fresh-directory guard, partial-iteration close error, student hook/captured-logits lifetime, native sum/count/report tuple and fixed-alpha objective, LM-only eval, failure/replay/cursor checks, hook/worker cleanup and explicit checkpoint binding.
4. Append `gpt_step.make_cached_kd_forward_step(...)`, explicitly binding `_forward_step_common`. Existing source/AST before this factory is identical, including online ModelOpt. Simultaneous cached adapter plus `DistillationModel` is rejected.

The supported objective is **`megatron-lm-ghost-tail-fixed-alpha`**, T=1, `(1-alpha)*LM + alpha*KD`, explicit default alpha=0.5. K<V is coarse-tail KL, not full KL or conditional top-K. This is a declared policy distinct from the current detached ModelOpt balancer. No optional cached ModelOpt mode is included.

## Verified acceptance slice

- [x] Strict RED→GREEN implementation transcripts under `evidence/bridge-tdd-*`, with failed environment attempts retained.
- [x] Native source writer/storage/reader are first-class runtime code, not stub imports or selected AST-only production implementations.
- [x] Actual unmodified SDK writer→unmodified SDK reader and port reader interoperate on a real native tar/zstd tensor cache in the pinned CPU image. Narrow in-memory getter injection is test-only and restored.
- [x] Port writer→reader→tiny student loss/backward, masked counts/reports, LM-only evaluation, cursor resume, mismatch/gap/replay and partial-write rejection.
- [x] `git apply --check` and apply on the exact staged installed SDK. Real staged Bridge factory import, dependency imports and exact default/ModelOpt source preservation.
- [x] Existing JSON fixture/source-control suite retained and rerun. JSON/vLLM are optional reference/future material, not native integration acceptance requirements.

Exact final counts, CPU-only Engine identities and exit receipts are recorded in `evidence/integration-acceptance.json`. SDK warnings remain visible. These checks do not execute a native Nemotron model or the full Bridge scheduler/optimizer/checkpoint loop.

## Explicitly incomplete, not merge-ready production claims

- [ ] Real Base teacher checkpoint/view/tokenizer/model load, no ignored tensors and real GPU output/loss/gradient/update parity at declared BF16 tolerances.
- [ ] Distributed TP/CP/DP/EP/PP/VPP and native CP/MBS/GBS/sample ownership; MTP/recompute/overlap/graphs. The adapter rejects these unsupported topologies/features despite retaining Core helpers.
- [ ] Production sampler/packing/position/mask/Mamba-reset provenance and complete immutable cache producer manifest. Native dataset hash and caller cache_id are not teacher/tokenizer-byte authentication.
- [ ] Atomic successful native optimizer-checkpoint plus cache-sidecar transaction/reload. `checkpoint_binding()` is explicit and unit/reader-resume tested; it is not automatically persisted by Bridge TrainState.
- [ ] Rerun rewind, skipped-step recovery, spawn/restart transport, remote MSC atomic readiness, adversarial native-cache hardening and concurrent-writer publication.
- [ ] Matched native whole-update performance, first-use dump+I/O+student GPU-hours, long-run stability and heldout quality. No CPU-derived speedup claim.

Future comparisons must isolate **caching at the same Core objective** from changing ModelOpt fullKL/detached balancing to ghost-tail/fixed-alpha. Tiny full-K versus top32 and online on-the-fly versus cached teacher scores need separately frozen conditions. Do not cache full-corpus full-K: prior arithmetic is about 439.8 TB for FP16 values alone. Keep the existing online plan unchanged pending parent review and separately authorized GPU gates.

## Handoff

Owned changes are only this folder and `tests/test_offline_kd_pr_bridge.py`; original `tests/test_offline_kd_pr.py` remains unchanged. Parent owns ledger/controller/corpus and any adoption decision. Runtime snapshots are primary source evidence; this isolated patch is a review artifact, not authorization to edit the pinned SDK or start training.
