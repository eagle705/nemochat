# Native SFT runtime02 — inert until parent release

This generation owns only `sft/archive/runtime02/` and
`results/midtraining/sft-after-kd-runtime02/`. It reuses, without modifying,
`native_sft_v1.py` and `recipe-v1.yaml`. No GPU training has been executed by
preparing this controller. The old failed data publication is not repaired,
rebased, or substituted. No benchmark is rerun.

## Parent review and start

```bash
sft/archive/runtime02/start.sh                 # inert preview
sft/archive/runtime02/start.sh --preview main # inert preview
# Only AFTER reviewing this source freeze and independent DATA READY:
sft/archive/runtime02/start.sh --calibration --release /ABSOLUTE/PARENT-RELEASE.json
# If automatic_main_after_calibration=false, use after calibration service exits:
sft/archive/runtime02/start.sh --run main --release /ABSOLUTE/PARENT-RELEASE.json
```

Set `automatic_main_after_calibration: true` in the parent release for the
already-authorized calibration → operational reload → fresh original-C main
sequence in one durable calibration supervisor. Otherwise calibration and
reload finish without starting main. The main-only command refuses an active
calibration service. No repeat attempt, automatic restart, or implicit resume
is allowed in this generation. Never delete execution claims to bypass failure.

`source-freeze.json` is a startup proposal for parent review, not launch
permission. Parent must bind its SHA to a separately authored release. A
missing READY or release fails before service/GPU creation. Do not copy the
CPU test fixtures into production readiness paths.

## Parent release contract

The independent DATA worker owns the raw `READY.json` and audit/provenance
receipts. Its schema need not be rewritten to match the launcher. Parent
reviews the actual raw receipt, binds its byte SHA, and provides this explicit
normalization in `parent-release.json` (all paths absolute):

```json
{
  "status": "parent_accepted_runtime02",
  "source_hf": "checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01",
  "source_freeze_sha256": "SHA_OF_RUNTIME02_SOURCE_FREEZE_JSON",
  "ready_path": "/ABSOLUTE/DATA-ACCEPTANCE/READY.json",
  "ready_sha256": "SHA_OF_RAW_READY",
  "benchmark_acceptance_sha256": "SHA_OF_EXISTING_PARENT_FINAL_ACCEPTANCE_JSON",
  "checkpoint_integrity": "not_checked_explicit_user_waiver",
  "automatic_main_after_calibration": true,
  "data_contract": {
    "ready_sha256": "SHA_OF_RAW_READY",
    "manifest_sha256": "78ab81fbbd9c0b13c55392b7d87c9e7035ad75778abad0dd3795d64121125c21",
    "files": {
      "EVERY_EXACT_MANIFEST_FILENAME_7_ENTRIES": {
        "expected_sha256": "UNCHANGED_MANIFEST_EXPECTATION",
        "observed_sha256": "ACTUAL_DATA_WORKER_OBSERVATION",
        "bytes": 123
      }
    },
    "buffered_mmap_consumer_accepted": true,
    "new_sft_reserved_id_audit_passed": true,
    "fresh_c_reserved_id_audit_passed": true,
    "source_provenance_accepted": true,
    "readonly_data_release": true,
    "audit_receipt": {"path": "/ABSOLUTE/ACTUAL-AUDIT.json", "sha256": "ACTUAL_SHA"},
    "provenance_receipt": {"path": "/ABSOLUTE/ACTUAL-PROVENANCE.json", "sha256": "ACTUAL_SHA"}
  }
}
```

The raw READY must itself have status `accepted_existing_full03_data`.
Normalization is a parent attestation of the raw audited observations, not an
automatically invented success status. The seven manifest entries are:
`heldout-masks.npy`, `train-tokens.npy`, `train-masks.npy`,
`train-dialogues.jsonl`, `heldout-dialogues.jsonl`, `heldout-tokens.npy`,
`source-schema-rejections.jsonl`. All observed digests must match unchanged
manifest expectations. Audit/provenance receipt bytes are authenticated again.
`source_provenance_accepted` covers the full shape/mask/target census, exclusion
of the whole Long-Context-Transduction dataset, all source revisions, selected
assistant masks, nonthinking official template, and all heldout reservations.
`fresh_c_reserved_id_audit_passed` must cover C, not an older BASE audit.

**Direct PASS plus buffered FAIL is not release for this mmap consumer.** Parent
must not set `buffered_mmap_consumer_accepted=true` in that case. No silent
transport switch, row quarantine, rehash/rebase, or fallback is implemented.
A new reviewed guarded transport would require a new source freeze. Ordinary
read-path changes after bulk acceptance remain a disclosed host/storage risk;
this implementation does not claim continuous per-row byte authentication.

Bulk data acceptance is independent of the checkpoint-forensics waiver. To
avoid eight simultaneous hundreds-of-GB rescans before calibration, the native
wrapper checks manifest SHA, each file's current size, tokenizer lineage and
real native dataset headers/shapes, relying on the parent's independently
accepted full-byte/mask/reservation audit. It never waives the input gate.

## Native contract

- Original C LM+KD1000 HF source only. Fresh native optimizer/scheduler/RNG and
  consumed-sample cursor for both calibration and main. No B/BASE source.
- BF16, 8 GPUs, EP8/TP1/PP1/CP1, seq16384, GB128/MB1, LR1e-5/min1e-6,
  AdamW WD.01, seed42, native Bridge pretrain/DDP/optimizer/checkpoint/loss.
- Full one-epoch native `train_iters=34135`, sampler allocation, cosine horizon
  and evaluation sample allocation are preserved during calibration and main.
  `exit_interval` limits actual calls (16 for calibration; actual measured main
  cap). There is no 16-call corpus/sampler layout and no data cycling. The
  inherited terminal-update minimum-LR policy is applied to the execution cap.
- Fixed seeded/source-stratified 128-dialogue validation window, eval16 batches
  of8, interval250, native final evaluation. Full34135 iteration allocation
  prevents native validation iterator under-allocation when an early final
  evaluation is requested. Repeated validation is not training repetition.
- Native full-state save16, periodic250 thereafter, retention3, final save.
  Calibration then operationally reloads model/optimizer/scheduler/RNG and
  sample cursor at16 for validation without any new optimizer call. The main
  never uses that learned checkpoint; it reloads original C model-only.
- Existing version-gated HF embedding import adapter is unchanged. Added
  destination coverage observer fails on unmapped local parameters/persistent
  buffers, alongside all6289 HF index keys having native source mappings and
  an observed `[131072,1024]` embedding transform. Actual all-rank finite
  positive gradients, contiguous optimizer calls, final save/heldout and clean
  Engine exit are acceptance requirements. There is no random missing-key
  fallback and no handwritten optimizer loop or installed SDK edit.
- Checkpoint payload SHA/CRC remains `not_checked_explicit_user_waiver`.
  Metadata/config/source and data-audit hashes are still checked. No weight
  forensics, extra checkpoint copies, HF push, export or online W&B is added.

## Timing and absolute deadline

Hard deadline: **2026-10-19 12:00 KST**, not twelve hours from launch. Training
stop policy:11:30 KST, reserving1800s for native final evaluation/save and exit.
A new optimizer step is anticipated one conservative measured step before that
stop; native duration includes initialization and evaluation. Late/pathological
steps may still overrun the estimate and must not be reported as on-time success.

Main budget uses the mean of all-rank-max actual native iteration seconds for
updates9..16, conservatively at least the slowest observed step and divided by
0.85 (explicit15% unmeasured periodic-save/eval/data overhead assumption), with
300s fresh model startup allowance. Its cap is bounded by34135 unique updates
and remaining wall time. This is a projection, not a promise to finish at noon.
The absolute native stop remains authoritative if the projection is wrong.

Calibration native runtime is at most90min, plus15min save grace; operational
reload is bounded by30min. At least3h of training clock is required before
calibration. Main native duration is recomputed from the current absolute stop.
The independent user-systemd service has RuntimeMaxSec ending120s before noon,
with bounded60s owned-Engine stop and120s service cleanup. These outer limits
are failure safeguards, not successful final saves. If save/eval cannot finish
inside the available margin, preserve failure/partial checkpoints and report it.

## Ownership and operational boundary

- User-systemd has no explicit User/Group or restart; actual MainPID, UID,
  `/proc` full argv and starttime are recorded and verified. Root-only paths
  inside the pinned NeMo image require container User0:0 + DAC_OVERRIDE,
  matching the actual v1 config probe; the host supervisor remains the user.
- Shared `gpu-all8.lock` is held through accepted jobs and cleanup. Live idle
  checks run before each real calibration/reload/main Engine start.
- Exact Engine ID/name/task label checks retain inherited image labels.
  ExecStopPost cleans only recorded runtime02 scope; an exclusive claim plus
  exact deterministic name/label reconciles CREATE-before-receipt races.
  No container is removed; no other experiment/service is stopped.
- Root/data/source mounts are read-only, output/cache mount is runtime02-only
  writable, networking disabled, no restart, GPU IDs0..7 explicit. Fresh HOME,
  Triton/Inductor/CUDA caches per segment; no copied compiled artifacts.
- Fixture tests exercise real pidfd exit/timeout/stale-identity behavior,
  fail-closed release/7-file/direct-only guards, budget/source/epoch contracts,
  Engine ownership/cleanup races and executable CLI refusal. They do not
  establish GPU restoration, learning, reload, deadline final save or quality.
- A no-device/no-weight pinned-image ConfigContainer construction is real
  CPU SDK evidence. The first config probe retained a root-permission failure;
  the root-readable followup exited0. Optional backend import warnings remain
  in logs. See HANDOFF.json for the final reviewed probe/source identities.
