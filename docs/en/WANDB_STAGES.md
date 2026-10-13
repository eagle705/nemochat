# W&B stage tracking: implemented scope and safe operation

For new CLI runs, use repository-root paths and fresh artifacts; see [runtime layout](RUNTIME_LAYOUT.md). Inputs belong in `datasets/`, checkpoints in `checkpoints/`, reports/traces/plots in `results/`, and text/W&B logs in `logs/`. Create required parent directories, not final no-overwrite output directories. Checked-in fixtures/configs and historical evidence paths remain unchanged.

## What exists (and what does not)

W&B is an optional observer, not a training backend. Install the `tracking` extra
(`python -m pip install -e '.[tracking]'` in a virtual environment); the declared
SDK range is `wandb>=0.30,<0.31`. Offline verification used **0.30.0**.

| Stage / run type | Current integration | Evidence scope |
|---|---|---|
| `data` / `stage` | `scripts/prepare_data.py` | Bounded data preparation, counters and dataset metadata |
| `pretrain` / `stage` | `scripts/train_cpu.py` | CPU tiny-HF control: updates, save and resume; not NemoChat/Bridge training |
| `pretrain` / `evaluation` | `scripts/base_eval.py` | Measured CPU HF inference; report retains partial/capped CORE scope |
| `pretrain` / `evaluation` | `scripts/gym_eval.py` | Imported saved Gym results; fixtures explicitly labeled; no judge/model API call |
| `longcontext` / `schedule-plan` | `scripts/lc_schedule.py` | Schedule and cursor emission only, no model training |
| `midtrain`, `sft`, `rlvr`, `mopd` | Registration vocabulary only | No corresponding training entrypoint implemented |

`checkpoint-merge` is a valid run-type label, not proof of a tracked merge
entrypoint. `configs/tracking/stages.json` describes registration/integration
availability; it is **not an execution ledger or a `--wandb-config` input**.
GPU/distributed training and all-stage execution are not demonstrated here.

## Opt-in, configuration and identity

Tracking defaults to `disabled`. Disabled runs and `RANK != 0` do not import the
SDK or write tracking receipts. The rank gate reads `RANK` (default `0`), not
`LOCAL_RANK`, and does not implement distributed synchronization.

Shared CLI options are `--wandb-mode {disabled,offline,online}`, `--wandb-config`,
`--wandb-id`, `--wandb-entity`, `--wandb-project`, `--wandb-group`,
`--wandb-parent-id`, `--wandb-checkpoint-id`, `--wandb-dir`, `--wandb-resume`.
For mode, CLI overrides `NEMOCHAT_WANDB_MODE`; JSON cannot enable tracking.
For identity/directory fields the precedence is CLI, then matching
`NEMOCHAT_WANDB_<FIELD>` environment variable, then JSON, then wrapper defaults.
Environment field names include `PARENT_ID`, `CHECKPOINT_ID` and `DIR`.

Use `configs/tracking/default.json` as the options template. Supported JSON
options are `run_id`, `entity`, `project`, `group`, `parent_id`, `checkpoint_id`,
`directory`. A strict allowlist rejects unknown/reserved keys (including `mode`,
`stage`, `run_type`, `resume`, `config` and credentials) with an actionable
`ValueError`, without echoing supplied keys or values. Values must be nonempty
strings; null is allowed only for optional identity fields, not project/directory.
Do not insert stage registry data. JSON cannot activate or resume tracking.
The wrapper assigns a random run ID if none is provided. Group, parent and
checkpoint IDs are user-supplied lineage metadata, not automatic provenance
validation or model recovery.

Example with an already available local model (replace paths):

```bash
mkdir -p logs/wandb
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/train_cpu.py --model checkpoints/pretrained/my-model \
  --texts datasets/my-texts.jsonl --output checkpoints/cpu/my-offline-run \
  --steps 2 --context-length 32 \
  --wandb-mode offline --wandb-config configs/tracking/default.json \
  --wandb-id cpu-example --wandb-dir logs/wandb
```

W&B offline mode does not make model/data acquisition offline. Use local inputs
and HF offline environment flags when no downloads are permitted. Offline SDK
records remain local; do not run `wandb sync` without separate authorization.
Online mode requires explicit CLI/environment opt-in and `WANDB_API_KEY` in the
environment; the wrapper does not invoke interactive login. Never put the key in
JSON, source code, CLI examples, run names, paths or artifact metadata.

## Metrics and artifacts

Training logs global update, cumulative loss-token count, LM cross-entropy,
learning rate/phase, invocation input/loss-token counts and update timing.
Input tokens include the reset prefix; loss tokens exclude it. Throughput is
loss tokens per measured update second, not end-to-end GPU throughput.
Successful finish records `wall_seconds_this_invocation`, not cumulative lifetime
wall time; failed cleanup skips this optional summary write.
Data preparation logs preparation counters, not optimizer-consumed training
tokens. Nested scalar metrics are flattened with `/`; raw arrays stay in local
reports. Evaluation evidence distinguishes measured inference, user-attested
imports and synthetic fixtures.

Artifacts are **metadata-only**: local checkpoint/dataset/report paths and
identity/counter metadata. No weights, dataset bytes or report files are added
by the wrapper. HF references validate an immutable 40-character lowercase
commit and repository ID and do not download/upload weights.

The sanitizer recursively redacts recognized credential **keys** in config,
metrics, artifact and run identity metadata (including local receipts), while
preserving tokenizer and token-budget keys. This is a **key-only, non-authentication
guarantee**, not a general secret scanner or URL validator: strings, URL userinfo
and query authentication, paths, free-text identity values and unrecognized
credential key spellings are not guaranteed safe. Never supply credential-bearing
URLs or other secrets under ordinary metadata keys. The options JSON allowlist
prevents arbitrary options from being forwarded; it does not inspect secret values
in explicitly supplied training config. Review these before online use. Git/code
collection and system statistics
are disabled, but SDK diagnostic files and requirements metadata still exist.

## Failure and RNG semantics

Init/log/config/artifact/finish SDK exceptions propagate; there is no silent
fallback to a successful untracked run. A failed logging operation remains failed
even when its caller catches the exception. Context-body failures finish with
exit code 1. Local `tracking-<id>-<invocation>.json` receipts report initializing,
running, finished or failed and include stage/type/lineage and the SDK directory.
Receipts are observational, not transactionally committed training checkpoints.
Missing online credentials are rejected before a receipt is created. Receipt I/O
failures propagate when they are the primary failure. Secondary failed-receipt
writes and finish cleanup failures do not replace an existing init/operation/body
exception; generic exception notes disclose cleanup failure without logging supplied
values. Initialization failures after SDK success attempt SDK finish with exit code 1.
Failure is sticky, and failed paths do not write a successful receipt. If receipt
I/O remains unavailable, the receipt can be missing, stale or incomplete; it is
not proof of success. SDK finish and local receipts are not an atomic transaction.
A checkpoint already saved is not rolled back when tracking fails.

SDK operations save/restore Python random, already-imported NumPy legacy RNG and
already-imported **CPU** torch RNG. This does not preserve CUDA RNG, independent
NumPy Generators or RNG modules first imported inside the operation. Do not claim
GPU RNG invariance from CPU evidence.

## Two distinct kinds of resume

`train_cpu.py --resume CHECKPOINT` verifies the checkpoint inventory and unchanged
training contract, then restores optimizer, CPU torch RNG, global step, data
cursor and loss-token count. `--steps` means additional updates. A WSD run must
keep the same global schedule contract/horizon. Existing output paths are refused.
This is local model recovery, not recovery from W&B artifacts.

With enabled tracking, checkpoint resume automatically requires an explicit
`--wandb-id`; `--wandb-resume` also requires an explicit ID. Online sends
`resume="must"`; offline sends no SDK resume policy and starts a **new offline
segment with the same ID**. Offline history is not automatically reopened or
merged. On resume, existing receipts for that ID in the selected local directory
must match run ID, stage, run type, entity and project; conflicts raise before SDK
initialization. This validates available local stage identity only, not checkpoint
contents, remote history, or optimizer/data continuity. Missing local history is
not invented or treated as verified. An explicit ID alone does not prove continuity:
the wrapper does not check that it matches a checkpoint's prior tracking ID.
Preserve correct parent
and checkpoint identity yourself. Logging timing may differ even when model and
optimizer tensors match exactly.

## Verification receipts (local evidence, not all-stage execution)

Artifacts under `/tmp/nemochat-wandb-smoke` were inspected during review:

- `tests/test_tracking.py`: **19 passed** using SDK 0.30.0. Registration/failure/
  RNG unit tests use a fake SDK; the dedicated offline test uses the real SDK.
- `review-tests/test_real_offline_sdk0`: a finished receipt and real `.wandb`
  record exist. `review-tests.network` contains **zero AF_INET/AF_INET6 connect
  calls** under `strace -f`; no remote W&B calls were observed.
- Existing `continuous` and `resumed` checkpoints both end at update 2; `first`
  ends at update 1. All 28 saved model tensors, optimizer state and CPU torch RNG
  are exactly equal between continuous and resumed final checkpoints.
- Existing data, CPU training/evaluation, fixture Gym comparison and schedule-plan
  receipts were finished; their reviewed network traces also contained zero
  Internet connect calls. Gym fixtures and schedule emission are not training.

This environment did not permit `unshare -n`; verification used offline settings
and syscall tracing, not an enforced network namespace. No online test was run.
Repair verification: `.venv/bin/python -m pytest tests/test_tracking.py -q` returned
**46 passed** with actual SDK 0.30.0 (two real-offline tests; other tests use fixtures).
The full `tests/` suite returned **358 passed, 1 skipped, 25 subtests passed**.
This rerun was not syscall-traced and does not establish a new network-isolation
claim. Tracking regressions cover strict options validation,
original exceptions under simultaneous receipt/finish failures, post-init failed
cleanup, sticky failure and local stage/type/entity/project continuity. Real SDK
offline tests exercise logging/artifact metadata, same-ID separate resume segments,
and a failed context body; none execute GPU or unimplemented stage training.
Remaining limitations include CUDA/distributed RNG/rank coverage, arbitrary
secret values, unavailable receipt storage and checkpoint-to-tracking identity
validation. No GPU RNG or all-stage execution claim follows from these tests.
