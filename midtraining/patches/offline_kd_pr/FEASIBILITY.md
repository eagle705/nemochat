# Feasibility: faithful Megatron-LM cached KD in Bridge

[한국어](FEASIBILITY.ko.md) · [Implementation/commands](README.md)

## Frozen practical scope

The primary target is **installed Bridge 0.6.1 at `2a8fcdd05181ffcef261192f71ff5910d0005bf2`**, not upstream main `2311cf8c32ee895ab96db8391996c8f3732e1e7d`. The installed and upstream `gpt_step.py` differ substantially. Exact installed config/state/train/pretrain/loss/forward seams were read from a stopped pinned-image container; `evidence/bridge-installed/` and `integration-base.json` preserve their source basis. Core evidence is `installed-core-provenance.json`; current upstream references remain in `upstream_refs.json` for later rebase, not interchangeable baselines.

This change is a runnable isolated port plus applyable diff, **not production distributed support**. It does not alter the frozen recovery/native15 sources, online ModelOpt configuration, corpus/controller/ledger or installed packages.

## Source-to-implementation map

| Native Core source | Retained behavior / narrow Bridge adaptation |
|---|---|
| `logits_saver.py` | Same main output-layer hook, global log-sum-exp/top-K, FP16/BF16/FP32, top-P/sentinels, 17-bit IDs, `_buffer_iteration`, `take_pending_data`, `_write_batched_tar`. Bridge producer flushes once per complete iteration synchronously, bounding pending buffers; no copied async-checkpoint integration claim. |
| `utils_logits.py` | Same tar member names, metadata hash, zstd/torch decoding, local/MSC storage, ordered discovery and prefetch. Only CLI getter/blend dependencies are rebound. |
| `cached_logits_loss.py` | Same `TeacherTarDataset`, DP remapping helpers, `CachedLogitsKDLoss`, global student normalizer and ghost-tail math. Added contiguous-iteration checking so a missing iteration cannot silently shift the stream. |
| `pretrain_hybrid.py` / native `LossFuncCallable` | Mirror masked LM/KD sums, count/report keys, LM-only eval, fixed-alpha semantics. Use an instance-owned logits capture rather than reading the active global hook at deferred-loss time. |
| Bridge `gpt_step._forward_step_common` | Explicit factory binds this real batch/model-forward seam. Existing `forward_step` and `forward_step_modelopt` remain byte-for-byte unchanged. Bridge scheduler receives the normal three-value loss tuple; no custom optimizer/trainer is added. |
| Bridge `TrainState` / pretrain loop | `train_state.step` is the zero-based cache iteration and consumed samples must agree at attachment. Explicit sidecar binding validates cache ID/objective/alpha, dataset identity, fixed GBS/MBS, next iteration and consumed samples. Automatic checkpoint transaction integration is not implemented. |

The port replaces `megatron.training` imports with a scoped `CoreRuntime`; it does **not** call `set_global_variables`, invent Bridge CLI arguments, or mutate installed Core getters. The standalone SDK compatibility test injects narrow getter callables in memory only, with pytest restoration. Linux fork inheritance is explicit; spawn/persistent restart transport is not solved.

## Wire format is first-class, unchanged

Local file `cp{CP}_dp{DP}__{last_iteration}.tar` starts with `_meta.json` and contains `{iteration}.pt.zst`. The zstd-compressed torch dictionary has lists `values`, `indices_low` (uint16) and `bit_17` (bool), one tensor per microbatch, `[S,B,K]`. Decoding reconstructs global integer IDs. Local native publication uses `.tmp` plus `os.replace`; there is no new JSON per-token production cache or conversion dependency. Only trusted tensor caches should be read.

Native hash fields are seed, sequence length, sample budget and normalized blend identifiers. They **do not** bind exact teacher revision/view, tokenizer bytes, sample tokens, masks, reset/packing semantics or backend kernels. `cache_id` is an explicit external lineage handle checked against the checkpoint binding, not a digest embedded/authenticated by stock native tar metadata. Production acceptance still needs a reviewed immutable producer manifest and successful completion receipt. Single-writer/fresh-directory assumptions are explicit; overwrite races, remote atomic readiness and whole-corpus capacity are not certified.

## Objective and compatibility

The selected objective is native **ghost-tail forward KL + fixed alpha** at temperature 1. Teacher top-K remains normalized over the full vocabulary, and the student normalizer also covers full vocabulary. The omitted probability mass forms one bucket, with the native 1e-8 clamp. This does not recover within-tail KL. `top_k=V` is useful only for a bounded tiny parity cache; the 17-bit native vocabulary limit is 131072.

The adapter defaults alpha to **0.5**, the native callable default; native CLI load alpha defaults to **1.0**. The selection is declared, not silently inherited from a different entrypoint. The existing online ModelOpt full-KL detached `LM/KD` magnitude normalization is **not equivalent**. Prior source-control tests retain the CE/KL entropy counterexample. No ModelOpt-compatible offline mode is included, and no quality comparison may label a switch from the online objective as a pure caching change.

## Executed and pending evidence

Executed in the exact pinned no-GPU CPU image: actual unmodified SDK writer→SDK reader and port-reader round trip; port writer/reader with real CPU tensors and gradient-bearing student; loss/count/report checks; LM-only eval; explicit cache-reader resume from the next iteration; mismatch, replay, gap and partial-write rejection; scratch SDK `git apply --check`/apply plus real Bridge factory/dependency import. Default/ModelOpt preservation is an exact source/AST check, **not an executed full default training regression**. CPU fixture forwarding is injected; the actual native hybrid model and full scheduler/optimizer loop were not executed. Warnings and failed import attempts are retained rather than hidden.

Remaining gates:

1. Real native-model GPU forward/backward/optimizer and checkpoint save/reload, including successful optimizer-update/sidecar transaction binding. Loss closure completion is not optimizer success.
2. Distributed TP/CP/DP/EP/PP/VPP, identical MBS/CP layout, fixed GBS/sample ownership and native DP remap divisibility. Underlying helpers are retained, while this adapter deliberately rejects non-singleton TP/CP/DP/PP.
3. MTP/recompute/overlap/CUDA graphs (rejected); packed attention/position/Mamba reset policy; rerun/skip/restart and DataLoader spawn. Fail-closed cursor errors do not implement rewind/recovery.
4. Exact Base teacher revision/view/tokenizer acceptance and no ignored tensors; BF16 score/loss/gradient/update parity against the same declared Core objective. No real teacher weights were loaded here.
5. Remote MSC behavior/atomic completion, corrupt/truncated native payload hardening beyond inherited reader behavior, real cache size/I/O/H2D and full-update time.
6. First-use-inclusive dump+student GPU-hours, reuse break-even and heldout quality. Existing synthetic CPU timings are not production projections.

Existing arithmetic in `evidence/storage-estimate.json` gives about **439.8 TB for full FP16 values alone** for the considered full-day workload, excluding indices and overhead. Full-corpus full-K is not feasible under the previously reported storage budget. Native top32 storage estimates are illustrative, not measured compression. Keep online/default behavior unchanged unless a separately authorized matched-objective comparison justifies adoption.

## Optional future vLLM path

The prior adapters/source snapshots remain available, with the former investigation archived in `evidence/legacy-vllm-feasibility.md` and `.ko.md`. No further vLLM investigation or engine launch was performed for this integration. External producers would have to emit the native wire format and separately establish exact teacher-output/native-sampler parity; they are not acceptance requirements for the Core-first port.
