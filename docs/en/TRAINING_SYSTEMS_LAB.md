# Agent-assisted training systems reference lab

## Purpose and current boundary

NemoChat teaches training engineering, not agent application development. A coding agent operates a bounded, explicit experiment contract; a human learns how workload shape, memory, launches and optimizer updates interact. Start with the standalone dense Transformer control before moving to the existing native hybrid plan. Generated tokens are replay inputs, not a meaningful corpus or model-quality benchmark.

The [YAML contract](AGENT_EXPERIMENT_CONTRACT.md) and [benchmark tools](TRAINING_PERFORMANCE_TOOLS.md) implement the local entry points. The existing [native performance lab](PRETRAIN_PERFORMANCE_LAB.md) remains the acceptance gate for Nsight, Triton, full hybrid training and distributed/resume claims.

## Learning sequence

1. **Freeze the experiment.** Validate YAML, inspect dry-run argv and resource limits, save a resolved config. No arbitrary shell commands, downloads, paid resource acquisition or uploads are configured by this runner.
2. **Predict the work.** List projection shapes and forward/backward FLOPs. Distinguish logical operand traffic from actual DRAM traffic and optimizer/activation costs.
3. **Measure a full update.** Exclude warmup; measure repeated forward/backward/optimizer windows. CUDA window boundaries synchronize completion; do not synchronize every phase and destroy overlap. Keep profiler activity outside final timed windows.
4. **Explain a candidate.** On a GPU, inspect timeline, API/kernel correlation, data, communication and dependencies. CPU phase ranges and a sum of kernel durations do not establish a GPU critical path.
5. **Change one thing.** Start with activation checkpointing, precision or microbatch size; do not silently change global-batch semantics, WSD policy, routing or tokens.
6. **Validate separately.** Finite losses/gradients are sanity checks, not baseline/candidate numerical equivalence. Before adoption, verify output/input/parameter gradients, short loss behavior and native checkpoint/resume.
7. **Compare and decide.** Save raw repeats and memory. A speed ratio is descriptive and noisy, not proof of a GPU gain, statistical significance or an adoption authorization. Preserve regressions and experimental-only decisions. Comparison requires complete implementation/runtime and timing/model/optimizer descriptors; matching omitted fields cannot establish compatibility. Imported descriptors are still assertions, not cryptographic authentication of a remote run.

## Versioned control experiments

- `configs/experiments/dense_cpu.yaml`: unprofiled baseline.
- `configs/experiments/dense_cpu_checkpointing.yaml`: same workload/protocol with recomputation enabled; comparable to the baseline.
- `configs/experiments/dense_cpu_profile.yaml`: separate one-update trace after unprofiled windows; not a GPU trace.
- `configs/experiments/dense_cuda.yaml`: GPU control template, unavailable on a CPU-only host; preview is not execution evidence.

These are small mechanics controls. Larger workload configurations need their own fixed measurement protocol and authorized GPU access.

## Linear forward/backward operator planner

```bash
.venv/bin/python scripts/plan_training_operator.py --config configs/performance/linear_unmeasured.json --output runs/my-linear-estimate.json
```

The parent directory must exist; output is created exclusively. This small tool does not run a GPU or infer its specifications.

For `Y = X @ W`, `X:[M,K]`, `W:[K,N]`:

- Forward GEMM: M × N × K, FLOPs `2*M*N*K`.
- Input gradient `dY @ W.T`: M × K × N, same FLOPs.
- Weight gradient `X.T @ dY`: K × N × M, same FLOPs.
- Each GEMM logical operand count: `(M*K + K*N + M*N) * dtype_bytes`.
- Both gradients are assumed required. No bias, activation, attention, optimizer, accumulation read, workspace, routing or collective traffic is included.

All three operand counts happen to match here; the shapes and cache/tile behavior do not. Logical FLOPs/byte is a teaching estimate, **not a measured DRAM roofline point**. Activation checkpointing adds recomputation in the full update; this isolated planner does not account for it.

### Strict JSON inputs

Required: `schema_version:1`, positive integer `batch_tokens`, `in_features`, `out_features` (each ≤1,000,000), `dtype` (`float32`, `bfloat16`, `float16`) and `provenance:{kind,description}`. Kind is `source`, `measurement` or `synthetic`; description is nonempty. Unknown fields, duplicates and nonfinite constants fail. Input is bounded to 64 KiB.

Optional effective ceilings:
- `effective_compute_flops_per_second`
- `effective_bandwidth_bytes_per_second`

Both must describe the exact device, precision, shape and execution conditions; they are not automatically obtained from a product specification.

Optional `measurements` maps any of `forward`, `input_gradient`, `weight_gradient` to exactly:
- `dram_bytes`: finite positive bytes for the same operator scope.
- `observed_seconds`: finite positive operator duration.
- `description`: profiler/version/device/precision/window and traffic-counter derivation.

Only supplying actual traffic inputs enables the operator roofline report. Missing ceilings yield `measurement_required`; timing below a modeled bound yields `assumptions_inconsistent`. Measurements are user-supplied and not authenticated. Synthetic arithmetic fixtures require `--allow-synthetic` and must never be presented as measured traces. Do not add individual bounds to claim an end-to-end update bound.

## Ready-to-use agent task

```text
Read README, NemoChatAgent.md and the training systems lab docs.
Validate configs/experiments/dense_cpu.yaml and inspect dry-run.
Run a baseline in a new directory. Do not rent GPUs or access the network.
Propose one controlled change and state the hypothesis before executing.
Use another new directory; preserve the config and all raw results.
Compare only compatible workloads/protocols and label CPU evidence as CPU.
Report observation, hypothesis, change, sanity checks, distributions,
memory availability, experimental-only decision and missing acceptance gates.
Do not claim numerical equivalence from finite gradients or adopt a default.
```

The runner limits configured resources and subprocess runtime; it is not an OS/container security sandbox and cannot control independent actions taken by an external coding agent.
