# C LM+KD1000 → deadline-bound SFT (CPU handoff v1)

[한국어](README.ko.md)

## Outcome

**GPU optimizer updates by this preparation: 0. No training/calibration was launched.**
The deadline is exactly **2026-10-19 12:00:00 KST (+09:00)**, not October 20.
No promise of completing a save/export by noon is made without native calibration.

The three-model GPU prerequisite was independently read back after parent-reported
completion. `results/midtraining/sft-after-kd-01/benchmark-acceptance.json` binds the
actual final marker, reports, source SHA, clean Engine states and inactive service.
`benchmark_gate.py` invokes the producer's exact full GPU validators, checks all
per-question receipts, verifies both PNGs, and rejects CPU/partial reports. Since
the producer publishes its marker before unlocking, service exit is required;
**the future launcher must also acquire `gpu-all8.lock` and check live idle**.

## Actual data readiness — BLOCKED, not absent

Existing approved full-03 data are physically prepared at
`datasets/nemotron_sft/nemochat_baby/prepared-full-03/data`:

| Split | Rows (NPY headers) | Fresh complete mask sum | Targets beyond 4096 |
|---|---:|---:|---:|
| Train | 4,369,280 | 3,219,012,753 | 898,456,733 |
| Heldout | 45,090 | 33,153,239 | 9,258,249 |

Fresh full scans checked every stored mask: uint8, binary, totals matching the
manifest; shifted uint32 token headers have width 16385 and mask width 16384.
Finite capacity at GB128 is **34,135** updates, one unique epoch, without cycling.
`cpu-readiness.json` preserves file sizes and one fresh matching heldout-mask SHA.
It does **not** certify every 400+GB prepared file hash or token content.

The historical exact CPU producer exited 0 and counted/source-mask audit passed,
but its completion publisher failed `stable prepared file byte/hash gate failed`.
The full `completion-ready.json` is still **missing**. The error did not name the
file; do not guess it was a particular file, alter expected hashes, substitute an
old pilot, or treat the manifest's `status=passed` as launch readiness. Revalidate
stable full data against immutable expectations before release. Root-cause/storage
repair is not established by the fresh mask scan.

Approved recipe is **16K CP1 / EP8 BF16 / microbatch1 / GB128**, Agentic v4 +
IF/Chat v4 + BBQ. Entire Long Transduction is deferred from both splits, RL seeds
remain unlabeled reservations. Preserve all14 prior prompt groups, 34871 KMMLU
exclusions, complete nontruncated examples, structured tools and official full-render
selected-assistant masks. IFC masks align with all messages, not assistant-only
indices. Do not download new data, fabricate labels or repeat epochs.

## Source and tokenizer

Only `checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01` is allowed;
lineage is C LM+KD native `offline2b-prefixtrain-1000-01/checkpoint/iter_0001000`.
B LM-only, original BASE, previous SFT391 and calibration weights are forbidden main
sources. Actual native HF import uses the existing pinned embedding-name adapter
`midtraining/patches/student_embedding_import.py` with native index-key mapping
coverage and observed learned-embedding transform; no embedding row permutation.
SFT optimizer/scheduler/RNG/cursor start fresh at zero; KD/teacher are not active SFT
objectives. Retain model-only HF and resumable full native saves as distinct artifacts.

Original vocab131072/EOD11 is preserved. Accepted migration manifest renames exactly
IDs18–29 to Super3.5 aliases, but its old corpus scan does not cover the new C corpus.
A new corpus-bound reserved-ID audit remains **pending**, along with complete current
SFT input-ID verification. Do not silently declare semantic alignment from the old
10B ClimbMix receipt. Use official pinned tokenizer/template, trust_remote_code=False,
no vocabulary resize or model-weight audits. Weight SHA/CRC/direct forensics remain
`not_checked_explicit_user_waiver`; operational restore/update/save are still required.

## Native code/config and controls

- `native_sft_v1.py`: task-local snapshot of proven selected-turn native Bridge SFT,
  changed only for C HF import/reset/lineage, bounded consumed-row coverage, cloned
  logged loss storage, absolute-stop and actual terminal-update minimum LR controls.
- `recipe-v1.yaml`: inert versioned native recipe. Update/time budget fields are null
  until fresh C16K GB128 native calibration (15 updates; >=10 warm measurements).
- `config_probe.py`: real pinned image ConfigContainer construction succeeded,
  Engine03 exit0 with **no DeviceRequests, no weights and no optimizer calls**.
  `config-probe-03/native-config.json` is the SDK-created native configuration.
  Failed first libcuda-free and abstract-provider attempts are preserved. Probe03
  mounted driver shared libraries read-only, but no GPU devices or allocations.
- `controls.py`: exact deadline, finite-capacity budget, normalized all-three score
  gate and real pidfd process identity/wait mechanics.
- `controller.py`, `start.sh`: **inert conditional preview only, not armed scheduling**.
  `--start` deliberately refuses missing stable data. Parent launcher/service wiring
  and immutable startup allowlist must be finished/reviewed before actual scheduling.
  The preview's systemd argv has Restart=no and no changes to existing services.

Production budget is min(finite one-epoch capacity, remaining clock / conservative
fresh warm timing), with at least1800s held for validation/save/optional export.
Full native sampler and heldout budgets must remain provisioned even under early
native duration stop. The planned final/early-stop optimizer call uses minimum LR;
record actual per-call LR and verify terminal save/all-rank clean exit. These new
absolute-stop/GPU lifecycle paths are **CPU-tested, GPU-unrun**, not a noon guarantee.
Final resumable optimizer/RNG checkpoint is required; HF export is optional and no
new HF push is authorized by this preparation.

## Parent action / remaining gates

1. Review CPU handoff and exact existing data failure; authenticate stable full data.
2. Complete fresh C-corpus/SFT reserved-ID audit and bind tokenizer receipt.
3. Finish/review inert service→owned launcher integration, exact source-freeze and
   scoped cleanup. Reacquire cooperative lock and fresh GPU idle before calibration.
4. Restore C, measure actual finite updates and save/heldout/clean-exit; freeze budget.
5. Start independent main from C again (not calibration weights), respect absolute
   noon deadline with save/eval buffer, and verify actual resumable terminal save.

No extra quality threshold is imposed on the accepted benchmark. No root ledger,
shared SDK, existing dataset, existing training run, publication or git commit was changed.
