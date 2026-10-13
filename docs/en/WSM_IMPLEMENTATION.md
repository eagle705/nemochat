# Official Megatron WSM integration

## Contract and source

This launcher runs NVIDIA's **unmodified official implementation**, not a custom
Hugging Face tensor average. Authoritative PR:
https://github.com/NVIDIA/Megatron-LM/pull/5114 (merged; API checked).
Pinned head: `ca4826e9555032b8710dfb4a4e5693c46c1613dd`.
Source: https://github.com/NVIDIA/Megatron-LM/blob/ca4826e9555032b8710dfb4a4e5693c46c1613dd/tools/checkpoint/weighted_merge.py

The upstream tool derives model FQNs, global shapes, dtypes, chunks and byte/object
extra-state from public PyTorch DCP metadata. Inputs must be compatible
**same-layout `torch_dist` Megatron checkpoints**: matching model keys, shapes,
dtypes and chunk layouts. It streams floating model tensors through CPU fp32
accumulators and saves in the source dtype. This is not a topology-changing
conversion or an HF checkpoint importer. The local HF controls are not its inputs.

The wrapper checks the exact Git HEAD, rejects modified/untracked Megatron or
checkpoint-tool files, and compares the tool bytes to `git show` at the pin. It
requires a new, nonoverlapping output root whose basename is not `iter_<digits>`
(the upstream checkpoint-path mode would write a tracker in its parent), a nonnegative target, a positive
window/interval/count and explicit `--trusted-inputs`. DCP metadata and torch
objects involve pickle/code-execution risk, **even during dry-run**. A clean Git
checkout is not a sandbox; use only trusted code, dependencies and checkpoints.
The official tool validates layout and publishes its checkpoint transactionally.
The wrapper never substitutes a different averaging algorithm on failure.

## Isolated installation (outside this repository)

Prerequisites: Git and [uv](https://docs.astral.sh/uv/getting-started/installation/)
on PATH; uv can provision the requested Python. Run repository commands from your
local checkout root, with the project `.venv` already set up.

Tested runtime: Python **3.12.13**, torch **2.14.1+cpu**, NumPy **2.5.3**,
packaging **26.3**, PyYAML **6.0.3**, Triton **3.8.0**. No project dependency changes
were needed. These are the actually exercised versions, not minimum compatibility
claims. Upstream imports `typing.override`, so Python 3.11 failed. Although the
merge itself uses CPU and constructs no model, this revision's package initializer
transitively imports Triton. Installing its wheel solved the import without any
GPU or CUDA kernels being exercised. TE/Apex/absl warnings are expected here.

```bash
CACHE=${HOME}/runs/wsm-upstream
PIN=ca4826e9555032b8710dfb4a4e5693c46c1613dd
mkdir -p "$CACHE"
git init "$CACHE/Megatron-LM"
git -C "$CACHE/Megatron-LM" remote add origin https://github.com/NVIDIA/Megatron-LM.git
git -C "$CACHE/Megatron-LM" fetch --depth 1 --filter=blob:none origin "$PIN"
git -C "$CACHE/Megatron-LM" sparse-checkout set megatron/core tools/checkpoint
git -C "$CACHE/Megatron-LM" checkout --detach FETCH_HEAD
uv venv --python 3.12.13 "$CACHE/venv312"
uv pip install --python "$CACHE/venv312/bin/python" 'torch==2.14.1+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$CACHE/venv312/bin/python" 'numpy==2.5.3' \
  'packaging==26.3' 'PyYAML==6.0.3' 'triton==3.8.0' 'pytest==9.1.1'
```

The installation example assumes a fresh cache. If it already exists, validate
the checkout/environment instead of repeating `remote add` or overwriting it. A sparse, shallow external checkout
provides the official dependency graph, not a huge vendored copy in this project.

## Stable sliding-window comparisons

Use **one stable training trajectory**. At every scheduled milestone evaluate
(1) its untouched raw target and (2) WSM from its sliding window, using identical
evaluation data/tokenizer/context. There are **no separate decay-training branches**.
`configs/pretrain/wsm.json` records an illustrative policy, not a runnable Megatron
training config or an automatically loaded wrapper config. Repeat identical window
and spacing arguments for each target; preserve all raw checkpoints.

```bash
# From your local NemoChat checkout root.
CACHE=${HOME}/runs/wsm-upstream
.venv/bin/python scripts/merge_checkpoints.py \
  --upstream "$CACHE/Megatron-LM" --python "$CACHE/venv312/bin/python" \
  --checkpoint-root /trusted/run/checkpoints --output /trusted/run/wsm-at-5000 \
  --target-iteration 5000 --window-iterations 1000 \
  --min-iteration-interval 100 --min-checkpoints 2 --trusted-inputs --dry-run
# Run again without --dry-run after inspecting the metadata validation.
```

The inclusive range is `[max(0, target-window_iterations), target]`. Upstream
selects chronological checkpoints and greedily filters **backward** by minimum
interval, preserving the target. Coefficients are based on **selected positions**,
not iteration distances or tokens. For n selected checkpoints, chronological
weights are `sqrt(1/n)`, then `sqrt((i+1)/n)-sqrt(i/n)` for interior i, and
`1-sqrt((n-1)/n)` at the newest checkpoint (n=1 gives 1). These are discrete
coefficients of upstream's **minus-sqrt** schedule, not `1-sqrt(i/n)` normalized as
raw weights. Irregular spacing does not produce token-weighted coefficients.
The wrapper explicitly copies common state and tensor/byte `_extra_state` from
the **target**, rather than relying on upstream's default extra-state index 0.

By default optimizer/non-model DCP entries fail closed. Only explicit
`--ignore-non-model-state` maps to the supported upstream
`--merge-ignore-non-model-state`. It discards all outside-model DCP entries, not
only optimizer tensors. The target common state is still copied: this does **not**
make its optimizer state mathematically consistent with averaged weights. Treat
outputs as evaluation artifacts, not verified resumable training checkpoints.

## Token-window accounting limitation

The official CLI only understands **iteration windows and iteration spacing**.
No `--window-tokens` exists. Never equate an iteration window with a fixed token
budget without recorded training accounting. For a desired token width W at
milestone t, use actual cumulative training-token metadata C(k) to locate the
first saved checkpoint with `C(k) >= C(t)-W`, then pass `t-k` as the wrapper's
iteration width. Record the requested W, chosen k, selected checkpoints and
actual span `C(t)-C(k)`. Selection spacing and missing checkpoints can further
change that span. This is a boundary conversion, not token-based coefficients.

For fixed, full-length batches only, nominal processed tokens can be derived as
`consumed_train_samples * seq_length` from the **actual saved training arguments**;
verify batch/packing/padding/loss-mask semantics. Variable lengths, curriculum,
packing or excluded labels require recorded per-step token increments. Do not
substitute tokenizer corpus counts or accepted preprocessing tokens for consumed
training tokens, and do not invent a missing batch size.

Concrete accounting evidence in the repository's *HF CPU control*,
`runs/cpu-train-uninterrupted/run.json`: steps 1–4 have cumulative
`loss_tokens_seen = [22,45,76,98]`. At step 4 a requested loss-token window of 50
has threshold 48; choosing the first saved step at/above that threshold gives
step 3, iteration width 1, and actual loss-token span **22**. Choosing step 2
instead spans **53** and exceeds the requested width. Context length 64 is not
64 supervised tokens per step. These are real control metadata values demonstrating
conversion/rounding only; they are **not Megatron processed-token measurements**
and those HF checkpoints cannot be passed to this DCP merger. Production WSM
must record the corresponding actual Megatron training metadata.

## Executed evidence and reproducibility

```bash
CACHE=${HOME}/runs/wsm-upstream
FIXTURE_PARENT=$(mktemp -d "${TMPDIR:-/tmp}/nemochat-wsm.XXXXXXXX")
# The parent exists; the fixture child must be new. -O also exercises optimized guards.
"$CACHE/venv312/bin/python" -O scripts/verify_wsm_merge.py \
  --upstream "$CACHE/Megatron-LM" --work-dir "$FIXTURE_PARENT/fixture"
NEMOCHAT_WSM_UPSTREAM="$CACHE/Megatron-LM" \
  "$CACHE/venv312/bin/python" -m pytest tests/test_wsm.py -q
```

`--work-dir` must be new. The pytest integration is explicitly skipped unless
`NEMOCHAT_WSM_UPSTREAM` is set; it must run under the isolated interpreter.
Review-blocker verification: **25 tests passed** in the isolated environment;
the normal project environment ran **24 passed, 1 skipped**. The official fixture
also passed all eight cases under `python -O`. Reports and per-case stdout/stderr
are saved in the chosen fixture directory, including expected failures.

Host-only historical artifacts (not repository files or portable prerequisites):
`/home/ubuntu/runs/wsm-fixture-5/verification.json` is the earlier 10-test run;
`/home/ubuntu/runs/wsm-review-opt-qSfc76JH/fixture/verification.json` records
the optimized review-blocker run. Do not assume either path exists elsewhere.

Trusted fixtures at iterations 10/20/30 have model tensors filled with 1/3/9,
one integer tensor extra-state, one byte/object extra-state and an optimizer
entry. Fixtures use the public synchronous DCP writer with flattening disabled
plus official Megatron sidecar writers. Megatron's *training* save strategy was
also attempted but calls CUDA synchronization on CPU; it was not patched or
used to claim a CPU training checkpoint save. The **merge implementation itself
is official and unmodified**. Independent public DCP reload verifies weighted
model values, both extra-state forms copied from iteration 30, and removal of
optimizer entries only after explicit opt-in. Both the upstream
`weighted_merge_provenance` in `common.pt` and wrapper
`nemochat_wsm_provenance.json` are checked. The wrapper records source SHA256,
revision, command, runtime package versions, boundaries and opt-in policy.

Eight integration cases: optimizer default rejection; no-write dry-run; official
merge; overwrite rejection; missing target rejection; insufficient checkpoint
count rejection; backward interval filtering preserves target; layout mismatch
rejection. Launcher unit tests add invalid arguments, missing trust, overlap,
wrong revision and rejection of alternative schedule branches. They also verify
`iter_<digits>` output-root rejection leaves a parent tracker unchanged without
any subprocess invocation (including dry-run), and deliberately corrupt state
under optimized Python to prove the verifier fails closed. Reported extra-state
iteration is read from the independently reloaded tensor, not a constant.

**Evidence boundary:** this establishes pinned source + CPU fixture wiring only.
It does not establish target-model WSM quality, actual Transformer Engine
interpretation, distributed multi-rank performance, GPU training, Bridge export,
or optimizer/RNG-resume correctness. No WSM quality or speedup claim is made.
