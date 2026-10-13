# Offline dense full-update performance tools

## Implemented scope

`src/nemochat/training_perf.py` implements a small **standalone educational dense causal Transformer**, not the native NemoChat hybrid Latent MoE/Bridge/MCore path. It trains from seeded generated token batches with real cross-entropy, backward and AdamW updates. No downloads, actual corpus, tokenizer, model-quality evaluation, Triton, distributed training or checkpoint/resume are involved. Native hybrid training remains unimplemented. This control does not change WSD/WSM defaults.

- `CPU-model-executed` means actual CPU control execution, **not GPU performance evidence**.
- `GPU-smoke-executed` means this same standalone control on one CUDA device, **not native hybrid or distributed validation**.
- CUDA absence/unsupported bfloat16 produces `status: blocked`, exit **3**, no CPU fallback and no execution scope. OOM returns exit **3** (`failure_kind: out-of-memory`); cooperative budget exhaustion returns exit **4** (`time-budget-exceeded`). Both retain measurement `status: failed`; the experiment runner reports blocked/timeout respectively and preserves the precise reason. Other execution failures return exit **1** (`nonfinite-or-no-update`, `execution-error`). Success returns exit **0**.
- CUDA timing, allocator memory and CUDA profiler capture require authorized hardware and remain pending on a CPU-only host. CPU profiler traces are locally exercisable.

## Direct CPU smoke

Save this complete JSON as `/tmp/dense-control.json` (no credentials). Unknown/missing keys, booleans used as integers, nonfinite/nonpositive rates and invalid counts are rejected. The direct JSON and YAML runner share the [same bounded schema](AGENT_EXPERIMENT_CONTRACT.md): name is 1–64 ASCII letters/digits/`_.-` starting with a letter/digit; seed is 0–2^32−1; vocabulary is 2–65536; learning rate is in (0,1]; CPU requires float32 and GPU budget 0, CUDA requires GPU budget 1; max_seconds is 1–3600. All per-field and combined resource bounds apply. JSON config is at most 32 KiB and comparison input at most 8 MiB per file. Duplicate keys and non-finite numbers (including exponent overflow) are rejected; malformed UTF-8/JSON, parser recursion and huge-integer errors return clean CLI rejection before creating output.

```json
{
  "schema_version": 1,
  "experiment": {"name": "dense-cpu-smoke", "seed": 42},
  "workload": {
    "device": "cpu", "dtype": "float32", "batch_size": 2,
    "sequence_length": 16, "hidden_size": 32, "ffn_size": 64,
    "layers": 1, "vocab_size": 128, "learning_rate": 0.001,
    "activation_checkpointing": false
  },
  "measurement": {"warmup_steps": 1, "measured_steps": 3, "repeats": 2, "profile": false},
  "resources": {"max_seconds": 60, "max_gpu_count": 0}
}
```

```bash
.venv/bin/python scripts/benchmark_training.py --help
.venv/bin/python scripts/benchmark_training.py \
  --config /tmp/dense-control.json --output /tmp/dense-baseline/measurement
# Repeat with activation_checkpointing=true in a separate JSON, all else equal:
.venv/bin/python scripts/benchmark_training.py \
  --config /tmp/dense-checkpoint.json --output /tmp/dense-candidate/measurement
.venv/bin/python scripts/compare_training_runs.py \
  --baseline /tmp/dense-baseline/measurement/result.json \
  --candidate /tmp/dense-candidate/measurement/result.json \
  --output /tmp/dense-comparison.json
.venv/bin/python -m pytest -q tests/test_training_perf.py
```

Always use fresh paths. Benchmark refuses an existing directory, even an empty one. Comparison exclusively creates a fresh JSON file. The parent experiment runner calls exactly `python scripts/benchmark_training.py --config output/resolved_config.json --output output/measurement`; direct use invokes the same full bounded validator as the runner. A measurement directory contains `resolved_config.json`, `result.json`, and optional `trace.json`. Invalid input is rejected before creating output; a runtime failure can leave a structured unsuccessful artifact and partial repeats, which comparisons reject.

## Model and fixed-workload protocol

The readable reference has token and learned position embeddings, single-head causal `F.scaled_dot_product_attention` with no dropout, pre-attention/pre-FFN layer norms, residuals, GELU FFN, final layer norm and an untied vocabulary projection. Every block optionally uses `checkpoint(..., use_reentrant=False)`. Hidden size must remain divisible by 4 as an explicit bounded-control contract, not because this single-head model needs four heads. The `B*4*S*S*L` attention guard is a conservative fourfold safety upper bound. CPU parameters/activations use `float32`; CUDA accepts `float32` or supported `bfloat16`; CE and gradient-norm reductions use float32. This is not mixed-precision master-weight training; AdamW states follow parameter dtype and there is no GradScaler.

A CPU-local generator creates B x (S+1) tokens per batch; inputs are the first S and next-token targets the final S. Every update processes **B*S tokens** and has **B*S loss-bearing tokens**, without padding, masking or token dropping. Generated batches are replay, not unique corpus coverage. Batch and parameter SHA256 fingerprints are diagnostics, not proof against maliciously edited artifacts.

Each repeat resets seed, model and AdamW state, replays identical warmup and measured batches, and excludes warmup from elapsed time and counters. Generated batch creation, initialization and fingerprinting are outside timing. The timed window includes data slicing/H2D, zero-grad, forward CE, backward, device-side finite/nonzero-gradient checks and optimizer updates. It includes the overhead of these checks, not profiling. Finite losses/gradients/updated parameters and an actual measured parameter change are required for success; finite checks do not prove mathematical equivalence or useful learning.

CUDA synchronizes immediately before/after the measured window, **not after each phase or update**. Loss/gradient flags are collected device-side and read only afterwards. Warmup diagnostics are allowed to synchronize outside timing. There is no CPU-phase timing used to infer GPU kernel duration. The CPU thread count is limited to at most two and restored on success/failure; caller CPU RNG state is restored and selected CUDA device RNG is forked/restored. No multi-GPU RNG/reproducibility guarantee is claimed; other devices are not seeded. Deterministic fixed inputs/initialization do not guarantee bitwise CUDA results on all kernels; runtime deterministic/TF32/matmul settings are recorded rather than silently changed.

`max_seconds` covers setup, repeats and optional profiling and is **cooperative** at update boundaries; an in-flight operator cannot be hard-interrupted. For strict process deadlines use the parent's subprocess timeout. Tiny windows emphasize timer/setup variability; increase warmup, steps and independent runs before drawing practical conclusions.

## Memory, environment and profiles

CUDA allocator allocated and reserved peaks are separate per-repeat fields, reset **after warmup**. Unused CUDA cached blocks are released between repeats outside timing. Initial model/live AdamW state remains part of peak memory; backward temporaries/checkpoint saved tensors occur in measured updates. PyTorch allocator metrics exclude some external allocations and are not GPU-wide usage. CPU memory fields are **null/unavailable**, never zero-as-a-measurement or mislabeled GPU memory.

Artifacts record Python/PyTorch/runtime/cuDNN versions, platform, thread count, precision flags, implementation hash and CUDA device name/capability/total memory/multiprocessors when applicable. No environment-variable dump or credentials are collected. Driver version is explicitly not queried; pin/record it separately for serious GPU experiments. Equal recorded environment metadata does not prove an idle/exclusive GPU, matching clock/power/thermal state or absence of host contention.

`measurement.profile=true` performs one additional update from separate fresh/warmed state **after unprofiled measurements**. `torch.profiler` records CPU (plus CUDA on a CUDA host), shapes and memory and exports Chrome `trace.json`. `record_function` labels are `data`, `forward`, `backward`, `optimizer`. The trace is diagnostic and never contributes throughput. A CPU trace cannot establish CUDA launch gaps, GPU critical paths, overlapping-stream behavior or Nsight findings. CUDA profile execution is a separate hardware gate; use [the pretraining lab](PRETRAIN_PERFORMANCE_LAB.md) and [Nsight guide](NSIGHT_CODEX_GUIDE.md) for native GPU diagnosis.

## Analytical operator estimates, not a hardware roofline

For each QKV, attention-output, FFN-up/down and output-head projection, report dimensions M=B*S, N=output width, K=input width, layer-instance count and **2*M*N*K forward FLOPs**. The byte model is element-width * (M*K + K*N + M*N): each input/weight read once and output written once. This is an idealized **minimum-byte lower bound**, not measured DRAM traffic or whole-update memory. The FLOPs/byte ratio uses that bound; actual traffic, cache reuse, layouts, intermediates and SDPA kernel choices are not observed.

The projection-only forward+backward estimate is three times forward (forward, dInput and dWeight GEMMs); checkpointing adds a nominal block-projection forward recomputation, not head recomputation. Non-reentrant early stopping may differ from this nominal budget. Attention QK/AV, softmax, embeddings, norms, GELU, CE, optimizer and finite-check work are explicitly excluded. No achieved FLOPs/s or hardware headroom/roofline claim is made. Such a claim requires explicit precision-appropriate hardware ceilings, bandwidth and measured traffic, with matching operator/timing scope. CPU observations cannot supply GPU ceilings or a GPU timeline.

## Compatibility and decision gate

Comparison requires successful schema-1 artifacts, complete finite repeat counters, correct token/update ratios, finite diagnostics and passed correctness flags. Both must have the same workload/seed/batch identity/initial model, measurement protocol (including profile choice), implementation hash, device/environment and evidence scope. The **only workload difference allowed is activation checkpointing**; experiment names and resource budgets may differ. Incomplete/failed/blocked runs and CPU-vs-GPU comparisons are rejected.

Report raw throughput distributions (min/median/mean/max/stdev), ratio of medians, ordered repeat ratios and separate allocated/reserved memory distributions/deltas on CUDA. Ordered ratios are descriptive, not randomized paired trials or significance evidence. CPU memory trade-off is unavailable. The decision is always **experimental-only**, even when faster: output/gradient equivalence across the intended dtype/shape domain, checkpoint/resume, native hybrid integration and hardware/profile evidence require separate gates. A tiny CPU checkpoint equivalence test is not general CUDA or native correctness. Never adopt solely on speed; report regressions, variability and no improvement honestly.
