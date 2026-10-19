# Native new-corpus LM+KD long run

한국어: [LONGRUN.ko.md](LONGRUN.ko.md).

## Execution boundary

These files implement the native lifecycle, not evidence that a 24-hour run has happened. The parent owns source review, actual pinned-image configuration, calibration, duration-control execution, production-data expansion, the supervisor and online tracking readback. This implementation does not launch services, acquire data, modify the accepted pilot or resume an existing writer. CPU fixture tests must not be reported as native GPU validation.

- Runner: `midtraining/training/longrun_native.py`; launcher: `midtraining/training/launch_longrun.py`.
- Offline full-batch calibration: `calibration.yaml`, **16 native updates**, GB128 / MB1 / sequence4096 / eight ranks / EP8 TP1 PP1 CP1 / BF16.
- Production template: `longrun24h.yaml`, exact display name **nemochat baby midtraining super35 24h**. The iteration and WSD horizons deliberately remain null until actual calibration is accepted.
- Start is **original BASE HF model-only warmstart**, `checkpoints/exports/nemochat_baby_base_hf_bf16_02`, resetting optimizer, scheduler, RNG and cursor. The modified operational-control student is never selected.
- Teacher is exact Super-3.5 **VL Base BF16**, revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`, through the existing reviewed text view. Accepted embedding-import, native shape-guard and provider helpers are reused without changing their files.
- Native `pretrain(cfg, forward_step_modelopt, callbacks=...)` owns forward/loss/backward/optimizer/scheduler/save. ModelOpt's balancer is LM + scale*KD*stopgrad(LM/KD), **not raw 0.5 KL**. Gradient clipping remains native norm1. Finite large losses/gradients are retained; no fabricated adversarial numeric threshold or per-step LR rewrite exists.

## Data, alias and no-repeat gates

The launcher consumes the actual producer **manifest.json**, not a fragment or progress report. The companion `success.json` must have `phase=prepared_verified` and `result.manifest_sha256` equal to the actual manifest SHA256. Source order, 60/10/10/20 fixed-length exposure weights, sequence length, eight distinct train/validation prefixes, reader verification and zero overlap/duplicates are required. See [producer schema](CORPUS.md#native-consumer-manifest-contract-schema-1).

Repository mounts preserve `/raid/yoon/nemochat` as the same absolute path in the container; absolute native-prefix identities are not rewritten. All `.bin` files are streamed as little-endian int32 and counted for **every token ID18..29 across all eight source/split files**, simultaneously checking vocabulary range, producer SHA256/token counts, and `.idx` SHA256/size. The host audit is saved under `results/midtraining/longrun-corpus-audits/<run-id>/reserved-id-audit.json`, including on a semantic-policy block; native ranks save their fresh audits separately. Exact Base tokenizer repository/revision/backend asset SHA is also required. **Any nonzero ID18..29 blocks execution pending explicit reviewed alias semantics. The old ClimbMix absence audit is not evidence for new text.** The old helper gate remains only as student/teacher embedding-lineage evidence.

Training uses actual Bridge `GPTDatasetConfig`, whose registry delegates to `BlendedMegatronDatasetBuilder(GPTDataset, ...)`. Before construction, each source must cover its weighted sample budget, integer-rounding reserve, builder0.5% surplus and shifted-label extra token. A process-local, restored-on-exit wrapper observes the original native registry function; it never replaces blending or sample construction. After native construction, every training child must have one document-index epoch, and all four sources must appear in the planned horizon and first actual global batch. The native config and dataset/source receipts are persisted. Training cannot silently cycle.

The 10M initial preparation pack covers **calibration only**. Sixteen GB128 updates consume 8,388,608 training positions; production needs a separately sufficient per-source capacity after timing. Scarce data fails before Engine create. Additional acquisition requires the parent's new reviewed recipe generation, not repetition of this pack. ClimbMix20% is replay already consumed by BASE, not novel acquisition.

Validation is a separate explicitly reused heldout pool, never training exposure. EvalGB8 / MB1 / one iteration. The native sampler reserves periodic + final windows; the observation adapter adds one startup window and records the native versus actual reserved sample counts. After optimizer call1, the supported native in-loop validation records startup heldout Loss and its actual LR context. A zero-LR call is not counted as positive-LR learning. Normal horizon completion uses the SDK's existing post-loop final pass, without an extra forced final evaluation. Early native-duration expiry instead requests the existing in-loop validation before the SDK's duration checkpoint/exit. Eval callbacks temporarily disable the native deadline only during validation and restore it before the native save/exit decision; outer bounds remain active. Both lifecycle branches still require real pinned-image GPU exercise.

## Calibration, schedule and honest completion

CUDA synchronization brackets whole native update timing. Every recorded time is the maximum across all eight ranks; peak allocated memory is recorded per rank. Calls1..8 are the declared cold window, **all calls9..16** are the eight warm, unprofiled measurements. No diagnostic pilot timing is reused. The accepted receipt includes mean/median/min/max/population-stdev, memory, actual LR-used values, finite positive gradients, frozen/no-grad teacher, contiguous completed optimizer calls and separate zero-LR versus positive-LR counts.

Production `train_iters=floor(86400*0.90/mean_warm_step_seconds)`. Ten percent is an explicitly assumed save/eval/data overhead allowance, not measured sustained performance. Native WSD is frozen from that actual horizon: peak1e-5, min1e-6, warmupceil1%, final linear decayceil10%; scheduler horizon must equal the update budget. The full frozen stage YAML is actually passed in Engine argv and saved beside `native-config.yaml` / `native-finalized-config.yaml`; teacher/KD config is preserved separately.

Native `exit_duration_in_mins=1440` measures from **GlobalState.start_time including initialization**, not 24 learning-only hours. Native duration completion, iteration-budget completion, config construction, tracking registration and accepted training remain distinct. If the iteration horizon finishes before86400 native seconds, the receipt is `iteration_budget_complete_short_of_wall_target`, with `training_24h_accepted=false`. A duration claim with less than86400 seconds is rejected. A finished W&B session or successful Engine create/exit is never enough.

## Save, reload, tracking and supervisor

Native synchronous full optimizer/RNG `torch_dist` saves occur every250 updates, most_recent_k3, plus native final save. Comparable full save was measured at approximately137GB: disk preflight reserves three retained + one in-flight +16GB headroom. Configured retention is not a claim that rotation has already been exercised. Existing checkpoints/results are never overwritten. Full-state resume is intentionally **not exposed** by this fresh-run launcher: do not pass a saved model as a new BASE or claim exact resume. Native full state remains available for a separately reviewed resume adapter.

After clean exact Engine exit, all eight ranks must prove updates, finite heldout at the final update and synchronous optimizer/RNG save. The launcher then starts a **separate teacher-free native reload/evaluation process**, requiring zero optimizer calls/saves and finite heldout before issuing `*_save_reload_accepted`. Checkpoint payload integrity remains `not_checked_explicit_user_waiver`; data checks do not revive weight forensics.

Production online destination is the already-authorized **nvidia/nemochat**. The existing secure `read_wandb_env` parser reads a read-only credential-file mount; no key is put in argv, Engine Env, payload/config receipts or logs. Calibration/control are offline. Stable run ID is explicit. The launcher does not fabricate a run URL or claim remote metric readback; parent verification remains `remote_tracking_verified=false` until separately checked. No extra preparation tracking run is created.

The cooperative all-eight lock is the same existing `results/pretraining/nemochat_baby/training/gpu-all8.lock`; idle/recovery state is checked while holding it and the lock stays held through save/reload. Engine GPU IDs are explicitly0..7. Engine restart policy is no; mounts/capabilities/memory are bounded. Native trainer and controller are separate cgroups. Native24h +30min save grace, timeout kill grace, and a separate30min reload fit below26h; use a parent service RuntimeMaxSec93600 and ExecStopPost exact-owned cleanup. Config-only is600s; calibration native180min plus30min save grace, then30min reload. Cleanup checks the exact Engine ID, exact name and exact labels before stopping it and reads the resulting state back. No broad Docker/GPU cleanup is used.

## Parent command API

Host-safe commands (no Engine/GPU/network):

```bash
.venv/bin/python midtraining/training/launch_longrun.py --help
.venv/bin/python midtraining/training/launch_longrun.py --preview
.venv/bin/python midtraining/training/longrun_native.py --preview
.venv/bin/python -m pytest tests/test_midtraining_longrun.py -q
```

After actual review, the parent freezes a JSON map with **exactly** the `SOURCES` allowlist paths and their real SHA256 values. CLI execution requires that review; the launcher does not approve itself. Any shared-source change invalidates it.

Inside a fresh, independent parent-supervised service with `NEMOCHAT_LONGRUN_SUPERVISED=1`, invoke:

```bash
.venv/bin/python midtraining/training/launch_longrun.py --config midtraining/configs/calibration.yaml \
  --run-id super35-calibration-01 --reviewed-sources /ABS/PARENT-REVIEW.json --execute
```

Replace `--execute` by `--config-only` and use another fresh run ID for real native configuration without weights/updates. For the actual one-minute GlobalState-duration control, use another fresh ID with `--duration-control --execute`. This tests an early exit even if initialization itself exceeds one minute; it is not one minute of learning. A nonzero/timeout exit is failed, not accepted. Results are under `results/midtraining/longrun/<run-id>/controller/accepted.json`.

Main, after calibration/control acceptance and sufficient new-generation capacity:

```bash
.venv/bin/python midtraining/training/launch_longrun.py --config midtraining/configs/longrun24h.yaml \
  --run-id super35-longrun-01 --manifest /ABS/PRODUCTION/manifest.json \
  --calibration /ABS/CALIBRATION/controller/accepted.json \
  --duration-control-receipt /ABS/CONTROL/controller/accepted.json \
  --calibration-compatibility /ABS/PARENT-COMPATIBILITY.json \
  --reviewed-sources /ABS/PARENT-REVIEW.json --execute
```

For a different manifest, compatibility must equal this nonsecret parent-reviewed record, with actual digests (not placeholders): `status=parent_accepted_same_recipe_expansion`, `calibration_manifest_sha256`, `production_manifest_sha256`, `same_provider_tokenizer_sequence_batch_topology=true`, `heldout_exclusions_preserved=true`. This explicitly acknowledges an extrapolation to expanded data; it is not sustained24h validation.

ExecStopPost API: `.venv/bin/python midtraining/training/launch_longrun.py --run-id EXACT-RUN --cleanup-owned`. Parent must read back service MainPID/argv, exact Engine state, advancing native update rows and the real remote run before reporting started/learning. Source/manifest readiness, configured duration and successful CPU tests alone do not meet that gate.
