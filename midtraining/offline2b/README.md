# Bounded native offline KD (DP8 / EP8)

This opt-in path leaves legacy online/ModelOpt training and the installed SDK unchanged. Production release belongs to the parent. Merely starting a controller is not model-training acceptance.

## Current approved profile: approximately 1.920B unique positions

The user accepted the same-mixture, no-cycling option: **3,662 calls / 1,919,942,656 positions**, original BASE warmstart, unchanged67/10/3/20 blend. Use `--mode production1920`; the legacy `production` and `gate` budgets remain unchanged. WSD warmup37/final decay366, LR1e-5/min1e-6, and the native model/forward/loss/adapter/topology are unchanged.

The new corpus is `datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json`. Seven reviewed Algorithmic documents containing reserved aliases are excluded whole, with no ID rewriting; every prepared training/validation file must still stream-audit IDs18..29 to zero. No new raw acquisition or training repetitions are authorized.

Parent release requires an exact `budget_compatibility` dictionary bound to the actual prior GPU gate source map, with changes confined to reviewed launch/supervisor controls, and `corpus_approval` bound to the actual manifest SHA plus the exact clean, owned CPU producer Engine. The executed gate is reused, not relabeled as a new benchmark. A live owner's scratch directory is kept active during the finite job to avoid idle-workspace expiry. Approved request: `results/midtraining/offline1920-parent-01/request.json`.

Corpus materialization and producer acceptance precede teacher-cache generation; student learning starts only after all native teacher shards are ready. Runtime preparation is not production learning.

## Original 2B recipe (capacity-blocked; preserved for compatibility)

- Original BASE: `checkpoints/exports/nemochat_baby_base_hf_bf16_02`, not smoke-trained weights.
- Teacher: NVIDIA Nemotron 3.5 Super VL **Base** BF16, revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`, existing reviewed text view. Teacher uses plain AutoBridge, frozen BF16, dropout zero, no DDP, no optimizer.
- Eight physical GPUs, TP1/PP1/CP1, DP8/EP8; GBS128, MBS1, **16 local microbatches**, sequence4096, seed43.
- Core-native tar writer, reader, DP mapping, top32 and ghost-tail fixed-alpha objective: `primaryTotal = .5 LM + .5 KD`, temperature1. Router auxiliary losses remain native model-side additions, not part of this primaryTotal metric. Native raw metric keys remain `lm loss`, `logits distillation loss`, `total loss`.
- Native Bridge `pretrain`, native Adam, clipping1, BF16, full optimizer/RNG checkpoints. WSD LR1e-5 → min1e-6. Production3815 calls = 2,000,158,720 processed positions; warmup38, final decay382. This is a token/update budget, not a 24-hour promise.
- Approved mixture67/10/3/20, no training cycling; actual native one-epoch index guard and per-source0.5% reserve. Production exclusively requires `datasets/midtraining/super35-kd-01/offline2b-v1/data/manifest.json` with authenticated producer success. No pilot fallback.
- Top32 cache ceiling600GiB, data64GiB; combined finite watchdog72h. Exceeding a watchdog is failure, never accepted shortened2B training.
- Checkpoint byte integrity: `not_checked_explicit_user_waiver`; no checkpoint payload hashes/copies. Operational loading, finite positive gradients, skip checks and actual updates remain mandatory.

## Real distributed gate

From repository root:

```bash
.venv/bin/python midtraining/offline2b/launch.py --mode gate \
  --output results/midtraining/offline2b-gate-01 --start
```

Use a **fresh** output generation after any failure; existing outputs are immutable evidence. All GPU stages hold `results/pretraining/nemochat_baby/training/gpu-all8.lock` and reject occupied GPUs. The user-systemd service inherits UID/GID (no `User=`/`Group=`), verifies its MainPID/UID/exact argv and checks source hashes. Native phases are separate pinned-image containers with network disabled. Only the new output and task-owned scratch are writable. `ExecStopPost` stops only recorded, matching-owner containers.

The gate uses the existing pilot **only for four unique native global batches**. Dump, first student run and resume all build the **same full512 training-sample budget** and identical validation budget; seed equality alone is not relied upon. Every actual training batch's tokens, labels, positions and masks are compared against the teacher's per-rank data SHA. First student run saves at2 and4; metadata-only resume selection points to checkpoint2, then native optimizer/RNG/scheduler restore executes updates3–4 again. These two extra gate updates are reported separately, not charged as production data.

Acceptance requires clean exact Engine exits, eight rank receipts, real positive finite nonskipped updates, all-rank masked-token counts, native synchronous full-state save, atomic all-rank cache-cursor transaction, native disjoint heldout LM evaluation, and resumed positive updates. `report.json` is published only after the whole chain. CPU tests and provider loading alone are not this acceptance.

## Parent production release

Preview does not create or start a GPU job. The parent must review source, real gate receipts and fresh corpus acceptance. A parent-authored JSON review has `source_sha256` (the complete current preview mapping) and `gate_output` (absolute accepted gate output). Do not synthesize a review or edit historical freezes to pass.

```bash
.venv/bin/python midtraining/offline2b/launch.py --mode production \
  --output results/midtraining/offline2b-production-01 \
  --review results/midtraining/offline2b-runtime-01/parent-review.json
# Parent only, after review: append --start to the exact command above.
```

The runner writes consumed `run-config.json` and SDK-native YAML snapshots in its new output. Checkpoint transaction sidecars bind cache identity, full sampler budget, seed, exact corpus identity, topology and next iteration. A native save without its committed sidecar is not a resumable accepted cursor. Production corpus capacity or producer failures must block before loading the full teacher; never change mix, cycle the10M pack or switch to online KD.

## Executed gate evidence

`results/midtraining/offline2b-gate-02/report.json` is `gate_accepted`: all eight ranks completed four positive-LR native updates, saved full state at2/4, restored optimizer/RNG/cursor at2 and executed two additional positive resumed updates. All actual sample digests matched the teacher. There were32 native cache shards (172,247,040 bytes), with524,288 all-rank masked tokens per update. Native heldout LM and raw router auxiliary metrics are retained in the Engine logs. This is operational resume evidence, not bitwise trajectory equivalence or model-quality validation.

Independent readback, exact Engine IDs/outcomes, source freeze, parent start argv and CPU results are under `results/midtraining/offline2b-runtime-01/`. The final focused CPU run was99 passed/6 SDK-only skipped; GPU acceptance was executed separately, not inferred from those skips. Gate01 was intentionally stopped before any student update after detecting excessive compiler-worker fan-out and a CPU JSON-sidecar regression; its evidence remains intact. The fixed launch caps Inductor compile threads at1, and EP8 real optimizer groups did **not** need the EP1 TE-Adam compatibility wrapper.

At the final readback all owned containers had exited, the user-systemd unit was inactive with result success, all eight GPUs were idle, and the cooperative lock could be reacquired. Production has **not** been released. Its fresh manifest was still absent; parent review and accepted full-corpus capacity remain required.

## Verification

```bash
.venv/bin/python -m pytest tests/test_offline2b_runtime.py tests/test_offline_kd_pr_bridge.py \
  tests/test_offline_kd_pr.py tests/test_offline_kd_gpu_smoke.py -q
```

Initial adapter and generated patch snapshots/hashes are preserved under `results/midtraining/offline2b-runtime-01/initial*`. Native source evidence remains untouched. No commit, public push, network download or GPU reset is part of this path.
