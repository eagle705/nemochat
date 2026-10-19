# Training systems lab — execution and review evidence

Latest status: reviewed CPU/offline control scope. Historical pending notes below refer to earlier snapshots; [final whole-source review](WHOLE_REPOSITORY_REVIEW.md) records repaired descriptors, fresh execution and independent approval. Native GPU/distributed/target-quality gates remain unverified.

## Scope

New agent-assisted standalone dense control and analytical linear forward/backward planner. This does not complete native hybrid GPU training, Nsight/Triton, distributed correctness, WSD/WSM target integration or model-quality evaluation.

## Baseline

Before implementation: `.venv/bin/python -m pytest -q` returned **520 passed, 1 skipped, 25 subtests passed**. CUDA availability was false. Existing untracked repository work was preserved; no staging/commit/push or external side effects occurred.

## Parent-executed artifacts

Fresh local outputs: `/tmp/nemochat-systems-parent/`.

- `baseline/`: YAML runner + actual CPU dense optimizer updates; success.
- `candidate/`: same seed/workload with activation checkpointing; success.
- `cuda-blocked/`: CUDA template; blocked with exit 3, no CPU substitution.
- `profile/trace.json`: separate CPU profiler update; phase labels data/forward/backward/optimizer present. A profiler allocator warning reported pre-existing allocation accounting; no CPU peak-memory claim is made.
- `comparison.json`: descriptive CPU A/B; experimental-only, CPU allocator memory unavailable. Initial small-sample checkpointing candidate was slower; not a meaningful GPU forecast or statistical finding.
- `linear.json`: declared small shape, analytical GEMM estimates only; repeat output write rejected with exit 2.

## Integration findings

A differential regression exposed divergent YAML/direct-JSON validation (10 failing contract cases). Both paths now use the shared strict validator; the 11-case differential suite passes. Typed OOM/cooperative-timeout propagation and strict bounded JSON input handling were also corrected. The original independent review identified incomplete comparison metadata as an additional blocker; required implementation/runtime/protocol descriptors are being hardened separately. The latest focused parent suite returned **143 passed, 1 skipped**; this is not the final whole-repository count.

A real one-second-budget invocation returned **timeout / exit 4**, with preflight exhaustion explicitly recorded. Imports/preflight are not hard-preempted, and this is documented rather than claiming a strict OS sandbox.

## Review and final regression

Pending independent verdict and final post-fix execution. Do not treat this draft as approval. AGENTS.md update was blocked by a protected-instruction-file approval timeout; it was not retried or bypassed. README and NemoChatAgent.md provide the new entry-point documentation.

## Runtime-path migration reference

The historical paths, timestamps and results above remain original evidence. For new runs follow the [repository-root runtime layout](RUNTIME_LAYOUT.md); locate relocated local artifacts through the runtime migration manifest's old → new mappings. Preserve receipt remote paths/revisions/hashes and inventory-relative payload paths; never overwrite existing artifacts.
