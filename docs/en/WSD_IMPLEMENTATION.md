# WSD learning-rate policy: CPU HF mechanics control

## Scope

`scripts/train_cpu.py` now supports an explicit warmup/stable/linear-decay (WSD) policy and optional periodic optimizer-bearing checkpoints. It still trains **a tiny Hugging Face control**, not NemoChat, Megatron-Core or NeMo Bridge. This is a pretraining-policy integration seam, not an implementation or validation of the target model's pretraining.

The experiment intent is **one WSD training trajectory plus PR5114 WSM stable-phase sliding-window evaluation**, not decay branches. WSM is a separate offline weight-merging/evaluation operation owned by the checkpoint tooling. This scheduler does not average weights, spawn branches, replace training weights with a merge, or run evaluation. Refer to [the pretraining experiment](PRETRAIN_WSD_WSM_EXPERIMENT.md) and the separate WSM implementation documentation for that workflow.

## Small explicit implementation

- `src/nemochat/training/wsd.py`: frozen `WSDConfig` and direct `learning_rate_for_update` function. No scheduler factory or hidden counter.
- `scripts/train_cpu.py`: assign LR before each global optimizer update; log the actual LR and phase; capture complete update-boundary checkpoints.
- `tests/test_wsd.py`: boundaries, finite values, CLI rejection, legacy constant contract, exact local-HF split resume at every phase boundary, periodic resume and collision protection.

## Boundary convention

Let `u` be the **one-based global optimizer update**, `W/S/D` the explicit phase lengths, `T = W + S + D`, `p` the peak learning rate, and `r` the floor ratio.

| Updates | Phase | LR used for that optimizer update |
|---|---|---|
| `1 <= u <= W` | warmup | `p * u / W` |
| `W < u <= W + S` | stable | `p` |
| `W + S < u <= T` | decay | `p * (r + (1-r) * (1-(u-W-S)/D))` |

`W` and `D` may be zero; their phases are skipped. `S` must be positive. A one-update decay immediately reaches the floor. The final update has LR `p*r`; the default `r=0` means its gradient does not move weights, although AdamW's state still advances. This discrete convention is intentional. There is no initial zero-LR warmup update. Update `W` is tagged **warmup**, not stable, even though its LR equals `p`.

`--schedule-total-steps` must equal the phase-length sum. It is the global horizon, **never** inferred from `--steps`; `--steps` remains additional updates for the current invocation. Requests exceeding the horizon fail before model loading. All phase lengths must be nonnegative integers, peak LR must be positive and finite, and the floor ratio must be finite in `[0,1]`. Linear decay is the only implemented decay shape; it is not an assertion of numerical parity with a specific external WSD scheduler.

## Resume and capture contract

The default remains `--lr-schedule constant`. Its original contract keys are unchanged: data SHA256, context length, learning rate, model and revision. Existing constant-LR checkpoints remain resumable only in explicitly unverified mode with `--allow-unverified-legacy-resume` when they lack an inventory. WSD adds the exact policy dictionary under `contract.lr_schedule`, including shape, phase lengths, global horizon and floor ratio. Any changed data/config/policy is rejected. Repeat all WSD arguments when resuming. Horizon extension is intentionally not exact-contract resume.

The next LR is a pure function of `saved_step + 1`, so there is no mutable scheduler counter to restore. Safe-loaded optimizer tensors, Torch CPU RNG, chunk cursor and consumed target-token count are restored as before. LR/phase are recorded in each stdout JSON line and `run.json` history. Historical manifests without these new log fields remain compatible, subject to the explicit legacy opt-in above.

Use `--checkpoint-every N --checkpoint-dir ROOT` together. Saves occur **after completed global updates divisible by N**, with deterministic child names `ROOT/step-NNNNNNNN`. The final `--output` checkpoint is still always written; it may duplicate the final periodic checkpoint. Capture saves model, tokenizer, optimizer, the exact Torch RNG and manifest cursor/token counts. Capture preserves RNG even if serialization consumes it, so taking snapshots does not change the trajectory. The root and final output may not contain one another. Existing target directories fail preflight and `mkdir(exist_ok=False)` prevents overwriting at write time.

This is a single-writer CPU contract. Checkpoints are not transactional: failed serialization may leave a partial directory (without a complete manifest). Do not reuse that directory. Python/NumPy RNG, distributed sharded optimizers, accelerator RNG and dataloader-worker state are not exercised or captured because this control does not use them. The deterministic chunk cursor is not a production distributed dataloader implementation.

## Checkpoint identity integrity

New `run.json` manifests include `checkpoint_format: cpu-hf-sha256-v1` and `artifacts_sha256`. The SHA256 inventory covers **every flat checkpoint file except `run.json` itself**: all model weight shards and shard indexes, model/generation configuration, tokenizer vocabulary/merges/serialized tokenizer, tokenizer configuration and special/added-token files when present, and `optimizer_rng.pt`. The original constant-LR `contract` is unchanged.

Resume verifies the complete inventory **before loading the tokenizer, model or optimizer/RNG**. Missing/changed/extra files, omitted inventory entries, missing required model/tokenizer/optimizer artifacts, malformed hashes, unsafe names, symlinks/directories and indexes pointing outside the verified shard set are rejected. A new-format manifest with missing/incomplete inventory fails even with the legacy flag. An old manifest lacking both format and inventory requires `--allow-unverified-legacy-resume` and prints an explicit warning; it is **not identity-verified**. Outputs from that legacy invocation receive a new inventory, which does not retroactively verify the parent.

This inventory detects accidental corruption or artifact/token-ID replacement, **not malicious manifest rewriting and not authentication**. `run.json` is not signed or externally trusted; an actor rewriting files and hashes (or removing both format fields to masquerade as legacy) is outside this guarantee. Use trusted, single-writer, immutable checkpoint directories during verification/loading; protection against concurrent mutation is not provided. Do not interpret `weights_only=True` as checkpoint authentication.

These are **HF CPU checkpoints, not Megatron distributed checkpoints (DCP)**. Even the stable snapshots cannot be passed to the Megatron DCP WSM merger. A target-stack WSM run needs compatible Megatron DCP checkpoints and its own checks; these files only exercise HF scheduling/save/resume mechanics.

## Integrity rerun: clean-clone commands

A clean clone uses the checked-in text fixture, **not** untracked historical `runs/` artifacts. With project dependencies installed in `.venv`, choose a fresh root:

```bash
ROOT=$(mktemp -d /tmp/nemochat-cpu-identity-XXXXXXXX)
WSD_ARGS="--texts tests/fixtures/texts.jsonl --context-length 32 --lr-schedule wsd --warmup-steps 2 --stable-steps 4 --decay-steps 2 --schedule-total-steps 8 --min-lr-ratio 0.1"
.venv/bin/python -m pytest -q tests/test_wsd.py
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 4 --output "$ROOT/split"
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 4 --resume "$ROOT/split" --output "$ROOT/resumed"
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 8 --checkpoint-every 2 --checkpoint-dir "$ROOT/snapshots" --output "$ROOT/continuous"
.venv/bin/python scripts/verify_cpu_resume.py "$ROOT/resumed" "$ROOT/continuous"
```

Integrity rerun artifacts: `/tmp/nemochat-cpu-identity-hX2aU1zi/{split,resumed,continuous,snapshots/step-00000002,snapshots/step-00000004,snapshots/step-00000006,snapshots/step-00000008}`. Invocation logs: `split.log`, `resumed.log`, `continuous.log` in that root. Actual targeted result: **53 passed in 6.78s**. Full-suite rerun: **222 passed, 1 skipped, 25 subtests passed in 16.40s**. All seven new inventories and exact combined 4+4 versus 8 history were also verified. Actual pretrained tiny-HF 4+4 versus 8 result: `PASS: exact model, optimizer/RNG, and run cursor/token-count equality (CPU control only)`. These new artifacts have inventories; no legacy flag was used. Tests include token-ID swap before loader calls, configuration/weights/optimizer/special-token tampering, inventory omissions/missing/extra files, unsafe paths/symlinks, multi-shard resume, legacy refusal and explicit opt-in exact constant resume.

## Historical executed commands and artifacts

Executed from `/home/ubuntu/nemochat` with the **existing** `.venv/bin/python` (`torch 2.14.1+cpu`, `transformers 4.57.6`). The pretrained model was `sshleifer/tiny-gpt2` at default immutable revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`. Historical artifact root: `/tmp/nemochat-wsd-20261007` (temporary, not committed, created before inventories). The counts and outputs below describe that earlier execution, not the integrity rerun. `runs/cpu-train-*` are local historical compatibility fixtures, not available in a clean clone. Resuming those old artifacts now requires the explicit unverified legacy flag shown below. Choose a new root on reruns: checkpoint paths deliberately cannot be overwritten.

```bash
.venv/bin/python -m pytest -q tests/test_wsd.py
# 34 passed in 4.92s
.venv/bin/python -m pytest -q
# Initial snapshot: 152 passed, 23 subtests passed in 16.32s
# Final rerun after concurrent additions: 173 passed, 23 subtests passed in 15.19s

WSD_ARGS="--texts tests/fixtures/texts.jsonl --context-length 32 --lr-schedule wsd --warmup-steps 2 --stable-steps 4 --decay-steps 2 --schedule-total-steps 8 --min-lr-ratio 0.1"
ROOT=/tmp/nemochat-wsd-20261007
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 8 --checkpoint-every 2 --checkpoint-dir "$ROOT/snapshots" --output "$ROOT/continuous"
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 4 --output "$ROOT/split"
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 4 --resume "$ROOT/split" --allow-unverified-legacy-resume --output "$ROOT/resumed"
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 4 --resume "$ROOT/snapshots/step-00000004" --allow-unverified-legacy-resume --output "$ROOT/periodic-resumed"
.venv/bin/python scripts/verify_cpu_resume.py "$ROOT/resumed" "$ROOT/continuous"
.venv/bin/python scripts/verify_cpu_resume.py "$ROOT/periodic-resumed" "$ROOT/continuous"

.venv/bin/python scripts/train_cpu.py --texts tests/fixtures/texts.jsonl --steps 4 --output "$ROOT/constant-continuous"
.venv/bin/python scripts/verify_cpu_resume.py "$ROOT/constant-continuous" runs/cpu-train-uninterrupted
.venv/bin/python scripts/train_cpu.py --texts tests/fixtures/texts.jsonl --steps 2 --resume runs/cpu-train-2 --allow-unverified-legacy-resume --output "$ROOT/legacy-resumed"
.venv/bin/python scripts/verify_cpu_resume.py "$ROOT/legacy-resumed" "$ROOT/constant-continuous"
```

Every verifier returned:

```text
PASS: exact model, optimizer/RNG, and run cursor/token-count equality (CPU control only)
```

Observed uninterrupted WSD history:

| Global update | Phase | LR | Target tokens seen | LM CE |
|---:|---|---:|---:|---:|
| 1 | warmup | 0.00005 | 22 | 10.815328598022461 |
| 2 | warmup | 0.0001 | 45 | 10.814092636108398 |
| 3 | stable | 0.0001 | 76 | 10.826976776123047 |
| 4 | stable | 0.0001 | 98 | 10.818061828613281 |
| 5 | stable | 0.0001 | 121 | 10.812102317810059 |
| 6 | stable | 0.0001 | 152 | 10.824250221252441 |
| 7 | decay | 0.00005500000000000001 | 174 | 10.814067840576172 |
| 8 | decay | 0.00001 | 197 | 10.811110496520996 |

Each checkpoint contains `model.safetensors`, HF model/tokenizer configuration, `optimizer_rng.pt` and `run.json`. Periodic captures exist at steps 2, 4, 6 and 8. Only **4 and 6** are stable-phase checkpoints under this convention: a stable-window evaluation must exclude warmup step 2 and decay step 8. These two HF snapshots are not inputs to the Megatron DCP WSM merger; they can only support a separate HF-format wiring probe; they are not evidence that WSM was run by this scheduler or improved anything. Merged evaluation exports must not be used as optimizer-bearing resume checkpoints. The checkpoint worker's own merge compatibility checks and evidence apply separately.

## Evidence boundaries

| Claim | Evidence |
|---|---|
| WSD discrete LR boundaries/validation | Unit tests |
| Save/resume at phase boundaries | Actual random local tiny-HF fixture tests |
| Real pretrained tiny-HF forward/backward | Eight-update CPU run above |
| Exact split/periodic resume and old default behavior | Tensor-exact checkpoint comparisons above |
| WSM merge/evaluation | Separate tooling; not exercised by this scheduler |
| NemoChat/Bridge/MCore/B300 training | Not tested |
| Model quality improvement | Not claimed |

Transformers emitted its existing `loss_type=None` fallback warning and used `ForCausalLMLoss`; execution and comparisons succeeded. Tiny-model CE on the handwritten smoke corpus is not a quality benchmark. Full pretraining, distributed state synchronization and evaluation attribution still require target-stack verification.

## Runtime-path migration reference

The historical paths, timestamps and results above remain original evidence. For new runs follow the [repository-root runtime layout](RUNTIME_LAYOUT.md); locate relocated local artifacts through the runtime migration manifest's old → new mappings. Preserve receipt remote paths/revisions/hashes and inventory-relative payload paths; never overwrite existing artifacts.
