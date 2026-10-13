# Agent experiment contract (schema 1)

This is a **custom offline PyTorch dense control**, not native NemoChat hybrid,
Bridge/MCore, MoE, Mamba, distributed training, model-quality validation or an
established speedup. No model/tokenizer/data downloads, remote tracking or network
operations are required. Keep the native GPU integration gates separate.

## Commands

From the repository root, with PyYAML installed in the interpreter:

```bash
.venv/bin/python scripts/run_experiment.py validate --config configs/experiments/dense_cpu.yaml
.venv/bin/python scripts/run_experiment.py dry-run --config configs/experiments/dense_cpu.yaml --output runs/dense-preview
.venv/bin/python scripts/run_experiment.py run --config configs/experiments/dense_cpu.yaml --output runs/dense-new
# CUDA template: validation/preview do NOT assert CUDA availability.
.venv/bin/python scripts/run_experiment.py dry-run --config configs/experiments/dense_cuda.yaml
```

`validate` and `dry-run` require only Python + PyYAML, not torch/CUDA; they read the
source and print JSON, perform no hardware checks and **write no output**, even
when `--output` names an existing path. Omitting output in a preview prints a
placeholder path. `run` requires torch and `--output`: its final directory must
not exist (including symlinks). Parent directories may be created. Reservation is
exclusive; do not reuse failed or blocked directories. No automatic cleanup or
retry overwrites old artifacts. The CPU example is intentionally tiny: batch 2,
sequence 16, hidden 32, FFN 64, one layer, vocabulary 128, warmup 1, measured 3,
repeats 2, no profile, 60 seconds, zero GPUs. All values are explicit: no defaults.

## Exact fields and limits

Every mapping requires exactly the listed keys. No unknown or missing key is
accepted, at any level. Integers exclude YAML booleans. Booleans must be actual
booleans. Schema version is integer `1`.

| Section | Field | Allowed value |
|---|---|---|
| root | schema_version | 1 |
| experiment | name | 1–64 ASCII letters/digits/`_ . -`; first character letter/digit |
| experiment | seed | integer 0–4,294,967,295 |
| workload | device | `cpu` or `cuda` |
| workload | dtype | `float32` or `bfloat16`; CPU accepts only float32 |
| workload | batch_size | integer 1–64 |
| workload | sequence_length | integer 1–4096 |
| workload | hidden_size | integer 1–4096, divisible by 4 (explicit bounded-control contract, not an architectural necessity) |
| workload | ffn_size | integer 1–16384 |
| workload | layers | integer 1–24 |
| workload | vocab_size | integer 2–65536 |
| workload | learning_rate | finite numeric value in (0, 1]; integer 1 also accepted |
| workload | activation_checkpointing | boolean |
| measurement | warmup_steps | integer 0–100 |
| measurement | measured_steps | integer 1–1000 |
| measurement | repeats | integer 1–20 |
| measurement | profile | boolean |
| resources | max_seconds | integer 1–3600 |
| resources | max_gpu_count | integer 0 or 1; CPU requires 0, CUDA requires 1 |

Let B=batch, S=sequence, H=hidden, F=FFN, L=layers, V=vocabulary.
Additional mandatory bounds:

- `(warmup_steps + measured_steps) * repeats <= 2000`.
- Estimated matrix parameters `L*(4*H*H + 2*H*F) + 2*V*H <= 100000000`.
- Each of `B*S*V`, `B*S*max(H,F)*L`, `B*4*S*S*L <= 16777216`.

The model uses **single-head** causal attention. `B*4*S*S*L` is a conservative
fourfold safety upper bound, not the actual head count or a measured allocation.

These conservative element/parameter guards are **not a byte-accurate peak-memory
promise**: optimizer state, gradients, allocator behavior, profiling and library
workspaces also consume memory. OOM can still block a valid experiment. The GPU
budget is a visibility/count constraint, not a VRAM quota or scheduler reservation.
CUDA visibility is limited to the first inherited visible GPU (or GPU 0 if no
visibility list exists); CPU workers have empty CUDA visibility. GPU availability
and bfloat16 support are checked only on `run`.

## YAML and execution safety

Input is UTF-8, at most 32768 bytes, at most eight mapping/sequence nesting levels,
one document, unique string mapping keys. A SafeLoader rejects duplicate keys;
explicit tags (including benign explicit tags), aliases and anchors are rejected.
No executable tags, merge aliases, arbitrary commands, module paths, environment
settings, downloads, external dataset references or shell expressions are fields.
YAML implicit scalar typing still applies: use `true`/`false` for booleans and
quote strings if necessary. Non-finite numeric values are rejected.

Direct benchmark JSON uses the same field validator, a 32768-byte input bound,
and rejects duplicate keys and all non-finite numbers (including exponent overflow).
Comparison inputs have an 8 MiB bound per artifact and the same strict JSON rules.
Malformed UTF-8/JSON, parser recursion and oversized integer conversion errors
produce clean CLI rejection without output creation.

The source is read once; SHA256 covers its **exact bytes**. Validation produces a
frozen dataclass with recursively immutable mappings; exported dictionaries are
copies. The resolved JSON contains exactly the agreed schema, without metadata
that might break the worker. It is written once and marked read-only (0444), not a
security boundary against its owner/root. Source SHA256 is recorded in result.json.

The sole experiment target is invoked using an argv list, the current Python
interpreter, `shell=False`, and repository cwd:

```text
<current-python> <repository>/scripts/benchmark_training.py --config <output>/resolved_config.json --output <output>/measurement
```

No config can select another target. No `.env` is read and no environment dump is
written. The worker inherits the caller's environment, except GPU visibility and
offline/disabled tracking flags (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
`WANDB_MODE`); these flags are not a network sandbox for arbitrary modified code.
Only trusted local repository code is executed. The custom worker uses synthetic
local tensors and must not add network operations.

The worker gets the remaining `max_seconds` budget after local torch/hardware
preflight. On a subprocess timeout the runner sends SIGTERM then SIGKILL to the
worker's entire new POSIX process group, waits/reaps the leader, and reports timeout.
Cleanup may take up to an additional second; local preflight/import time is not
hard-preempted. Descendants that deliberately escape the group are not sandboxed.
This runner targets the documented Linux/POSIX environment, not Windows.

## Artifacts, statuses and evidence

An admitted run reserves its directory before checking runtime dependencies and
writes `resolved_config.json`, `result.json` and `report.md`. Spawned runs also
write `stdout.log`, `stderr.log` and any worker-produced `measurement/` artifacts.
A preflight-blocked run need not contain logs/measurements. Catastrophic storage
errors can prevent reports; the CLI returns input/output error rather than
pretending a result exists. Ctrl-C cleans up a spawned process group but is not a
completed experiment result.

| CLI exit | Status | Meaning |
|---|---|---|
| 0 | success / valid / dry_run | Worker completed, or configuration-only operation completed |
| 1 | failed | Nonzero worker exit other than 3 or 4, or ordinary runtime exception |
| 2 | invalid | Malformed/unsupported contract, missing run output, existing output, input/output error |
| 3 | blocked | Missing torch/CUDA/bfloat16 support, worker exit 3, or detectable OOM |
| 4 | timeout | Resource deadline exhausted or benchmark timeout |

Typed worker `failure_kind: out-of-memory` maps to worker exit 3 and runner
**blocked**, with the precise reason retained from JSON stdout. Cooperative
`time-budget-exceeded` maps to worker exit 4 and runner **timeout**, even without
stderr. Worker measurement artifacts retain `status: failed` and the failure kind.
As a fallback, OOM classification recognizes `out of memory`/`OutOfMemoryError` in
the last 65536 stderr bytes; an unexplained OS kill is **failed**, not invented OOM evidence.
Timeout takes precedence over partial OOM logs. result.json records status, exit
code, worker return code (null when not spawned), reason, source SHA256, resolved
config, exact argv, elapsed seconds and evidence label. Reports do not fabricate
throughput/loss when the worker fails. Inspect worker measurement scope separately.

All reports carry `custom_offline_dense_control; native hybrid NOT validated`.
A configuration preview is **configuration-only evidence**; passing unit tests is
**CPU contract/runner evidence**; a real completed CPU/CUDA worker is **measured
custom dense control evidence** on its actual device. None proves native hybrid
compatibility, distributed correctness, WSD/WSM resume parity, model quality or
full-training improvement. Profiled runs must not be relabeled as unprofiled
throughput A/B measurements. See [performance lab](PRETRAIN_PERFORMANCE_LAB.md)
for native follow-up gates.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_experiments.py
.venv/bin/python scripts/run_experiment.py dry-run --config configs/experiments/dense_cpu.yaml
```

The suite exercises exact fields, unsafe/malformed YAML, resource bounds, immutable
resolution, CLI no-write previews, exclusive reservation, all run statuses and real
local POSIX descendant termination. Mocked worker status tests are not performance
measurements. Run the fixed real worker separately on a fresh directory once it is
available; a skipped CUDA path remains unvalidated.
