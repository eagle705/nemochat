# Pretraining performance lab: Nsight Systems → Triton fusion

For new CLI runs, use repository-root paths and fresh artifacts; see [runtime layout](RUNTIME_LAYOUT.md). Inputs belong in `datasets/`, checkpoints in `checkpoints/`, reports/traces/plots in `results/`, and text/W&B logs in `logs/`. Create required parent directories, not final no-overwrite output directories. Checked-in fixtures/configs and historical evidence paths remain unchanged.

## Status and purpose

This is an accepted pretraining work package, not an implemented GPU capability or a measured speedup. The current host cannot communicate with an NVIDIA driver (`nvidia-smi -L` failed); `nsys` is not on PATH. CPU tests cannot validate CUDA timing, Triton execution, native distributed training or GPU memory.

The goal is to learn evidence-driven performance engineering: identify launch-bound regions with Nsight Systems, implement a small optional Triton fusion, validate forward/backward correctness, and measure whether the entire training update improves. Preserve the [WSD/WSM experiment](PRETRAIN_WSD_WSM_EXPERIMENT.md).

## Executable standalone control

The repository now adds a generated-token dense Transformer full-update control, a strict YAML runner and linear forward/backward operator estimates. See [training tools](TRAINING_PERFORMANCE_TOOLS.md), [agent contract](AGENT_EXPERIMENT_CONTRACT.md) and [learning sequence](TRAINING_SYSTEMS_LAB.md). These do not implement native hybrid training, Triton fusion or GPU/Nsight evidence. Historical pending-status statements below apply to the native P0–P3 gates, not to the new CPU control.

## Execution sequence and gates

### P0 — Reproducible GPU baseline

- Obtain authorized GPU access; pin container, driver/runtime, PyTorch, Triton, Bridge/MCore/Transformer Engine and source revisions. Check installed `nsys --help` before constructing a launch command.
- Establish a working native NemoChat forward/backward/update and checkpoint reload before optimizing. Do not treat the tiny HF CPU control as the target training path.
- Freeze model/checkpoint, materialized real batches, seed, precision, sequence length, GBS, MBS/accumulation, parallel layout, routing/capacity/drop policy and recomputation. Replayed batches isolate performance; they are not corpus training evidence.
- Record rank-aware processed/loss-token deltas without multiplying by TP/CP/EP replicas. Record completed optimizer-update wall time, training tokens/s, end-to-end tokens/s and rank-max memory separately.
- Measure allocated/reserved CUDA peak memory and GPU-wide usage separately. Reset allocator peaks after warmup and report per-rank maxima. External library allocations may not appear in PyTorch allocator metrics.
- Compile/autotune/warm up before steady-state timing. Predeclare warmup, measurement windows and independent repeats. Use synchronized window boundaries for GPU completion, without forcing synchronization between every phase or update and destroying overlap.

Acceptance: immutable resolved configuration, raw measurements and a usable native training baseline. No optimization claims yet.

### P1 — Nsight Systems diagnosis

- Mark data/H2D, forward, backward, optimizer and checkpoint with NVTX. Add a narrow candidate-module range where needed; CPU ranges do not themselves measure GPU execution duration.
- Capture a short steady-state window, not initialization/JIT or a full rental. Start with one rank for readability, then capture all relevant ranks for distributed bottlenecks. Avoid output collisions by using distinct run/rank paths.
- Inspect CPU launch/API activity, CUDA kernels, transfers, synchronization, stream dependencies, collectives and GPU idle gaps together. Use correlation information to map a suspected kernel sequence back to its source module.
- CPU time spent waiting in synchronization is not proof of CPU compute overhead. A GPU gap can be data, communication or dependency waiting rather than launch starvation. Check other streams/ranks and the critical path.
- Select a candidate only after finding repeated short kernels or unnecessary intermediate memory traffic. Verify the installed SDK is not already using a fused implementation.

Acceptance: saved `.nsys-rep`, source mapping and a written hypothesis with alternative explanations. A negative finding is valid; do not manufacture a fusion target.

Read [Nsight analysis with Codex](NSIGHT_CODEX_GUIDE.md) for version checks, CSV/SQLite export, schema-aware read-only analysis, a ready-to-use task, and GUI confirmation.

### P2 — Small fusion experiment

- First learning candidate: elementwise SwiGLU activation (`silu(gate) * up`), excluding its GEMMs. It is a proposed educational candidate, not an observed NemoChat bottleneck.
- Create an isolated readable PyTorch reference and collect actual shapes, strides and dtypes from the native path. Cover small shapes and tails as well as representative pretraining shapes.
- Compare eager reference, compiled reference where supported, and custom Triton. Label unsupported compile paths rather than silently dropping that baseline. CUDA Graph is an optional separate launch-overhead comparison, not interchangeable with fusion.
- Implement forward and backward, initially under an explicit experimental option. Document FP32 intermediates, supported dtype/layout, aliasing/in-place behavior, noncontiguous input policy and unsupported shapes. Default training remains unchanged; unsupported cases use a tested reference fallback or an actionable rejection.
- Avoid attention/GEMM/dispatch fusion as the first exercise. Do not alter routing, token dropping, model math or RNG semantics to get faster results.

Acceptance: standalone GPU tests and unprofiled forward/backward benchmarks, including compilation excluded from steady-state timing. No end-to-end claim from microbenchmarks.

### P3 — Native training A/B and rollout gate

- Run baseline and fusion from the same raw checkpoint and batch sequence in separate output directories. Keep WSD horizon/LR/token accounting and raw optimizer/RNG/data-cursor resume state intact. Performance pilot branches are not extra production WSD or WSM trajectories.
- Check output, input gradients and relevant parameter gradients with dtype-specific tolerances declared before observing results. Record maximum errors; do not loosen thresholds merely to pass. Include FP32 reference, actual training dtype, boundary shapes, supported layouts and large/small values. Use higher-precision/autograd references for derivative checks where feasible.
- Verify short training loss/gradient behavior, finite values, checkpoint serialization and same-implementation resume. Baseline-vs-fused numeric equivalence need not be bitwise; declare its tolerance separately from resume guarantees.
- Re-profile to test the hypothesis (launch count, intermediate traffic, GPU gaps). Then disable profilers for final repeated full-update and memory measurements. Report distributions, regression cases, instrumentation scope and variability, not only the best run.
- For distributed integration inspect rank imbalance and overlap; a one-rank gain is not an eight-GPU gain. For memory include backward temporaries and saved tensors, not forward alone.
- Reject or retain experimental-only status if correctness fails, unsupported inputs break the contract, or total training performance regresses. An operator gain with no update gain is a valid reported outcome, not a success claim.

Acceptance: correctness evidence, comparable unprofiled update measurements and an explicit adopt/reject/experimental decision. Switch a production default only after separate review of native training and resume evidence.

## Required artifacts and proposed locations

These paths are proposed implementation work, not existing runnable tools:

- `src/nemochat/kernels/`: reference + optional Triton forward/backward; keep CPU imports working without CUDA/Triton.
- `scripts/benchmark_pretrain_fusion.py`: isolated comparisons and JSON output.
- `tests/test_pretrain_fusion.py`: CPU reference/contract tests and explicitly CUDA-gated correctness tests.
- `configs/performance/`: versioned shape/measurement protocol without credentials.
- Ignored `results/performance/<unique-run>/`: resolved config, raw timing/memory, correctness errors, traces and comparison report. Never overwrite earlier runs.

Track W&B stage `pretrain` with separate implementation/benchmark-role labels; record parent checkpoint, source revision and immutable artifact handles. Do not mix replay/microbenchmark metrics with real training throughput or feed profiler-slowed numbers into the budget planner.

Every experiment report must contain: observation → hypothesis → change → correctness → operator result → full-update result → memory trade-off → adoption decision → remaining gaps. Preserve negative results.

## Test and evidence matrix

| Gate | CPU host | Authorized CUDA host | Native multi-GPU path |
|---|---|---|---|
| Reference math, input contract, optional imports | Required when implemented | Required | Reuse |
| Triton forward/backward and layout/dtype coverage | Not validated by skip | Required | Required for actual inputs |
| Isolated timing and memory | Not available | Required | Not evidence of full-update speedup |
| Nsight critical-path hypothesis | Not available | Required | Relevant ranks required |
| Full-update A/B, checkpoint and resume | CPU control only | Native GPU required | Required before distributed claim |

At document creation all CUDA/Triton/Nsight and native training gates remain pending. No fusion code, GPU integration or performance improvement is claimed.

## Upstream references to consult during implementation

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

Resolve version-specific support in the installed stack before adopting commands or kernels from a tutorial.
