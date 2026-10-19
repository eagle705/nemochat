# Native SFT runtime04 — sealed RAM only

CPU-tested executable, **not GPU training evidence**. No production supervisor or GPU Engine was started by this worker. Frozen v1/runtime02 and their failed original buffered path remain untouched. No original array/dialogue fallback exists.

## Parent handoff (no renewed user authorization needed)

1. Review `results/midtraining/sft-after-kd-runtime04/HANDOFF.json`, source freeze, tests and pinned CPU receipts.
2. After producer supplies complete `READY.sealed.json`, review its all7 SFT and all16 C exact-byte proofs, migrated tokenizer/provenance, bounds/masks/target totals, alias18..29/C-absence audit and pinned CPU same-FD full readback. Do not promote a partial snapshot.
3. Run the actual public-client CPU adapter integration, **once**, after READY exists (this worker does not poll producer files):

```bash
.venv/bin/python midtraining/sft_after_kd/runtime04/integrate_snapshot.py \
  --ready /raid/yoon/nemochat/results/midtraining/sft-after-kd-sealed03/READY.sealed.json
```

This creates an owned pinned-image **no-device** eight-process CPU Engine. It receives seven descriptors through the real producer client and authenticates actual shifted train/heldout inputs. It never loads a model/optimizer. Failed outputs are retained; no implicit retries/overwrites. Parent must use a reviewed new generation/path if it fails.

4. Fill `parent-release.template.json` in the runtime04 result directory, producing `parent-release.json`. Bind exact raw READY SHA, snapshot ID, live holder identity, runtime freeze file SHA, producer freeze file SHA, benchmark receipt SHA and `actual-sealed-cpu-proof.json` SHA. Every normalization flag means parent actually reviewed that raw proof; the template has **false** flags and is deliberately rejected. The producer freeze path must name the file whose SHA the producer bound in READY. No fabricated buffered compatibility.
5. Validate and start the already-authorized calibration:

```bash
bash midtraining/sft_after_kd/runtime04/start.sh --calibration \
  --release /raid/yoon/nemochat/results/midtraining/sft-after-kd-runtime04/parent-release.json
```

`automatic_main_after_calibration=true` explicitly permits the same durable calibration supervisor to run native full-state operational reload/eval and then reimport **original C** afresh for main. False requires `start.sh --run main --release ...` after clean calibration supervisor exit. Neither path uses calibration learned weights for main. Do not launch runtime02 or an original-mmap consumer.

## Execution and ownership

- Default `bash .../start.sh` and `--preview calibration|main` are inert.
- Production still uses frozen Bridge pretrain, native DDP/distributed AdamW/scheduler/masked CE/checkpoint and full-parameter/learned-embedding import coverage. Missing destinations cannot stay random.
- Original C model-only source; fresh optimizer/scheduler/RNG/native sampler cursor in calibration and main. Checkpoint SHA/CRC is `not_checked_explicit_user_waiver`; no weight payload hashes are performed.
- BF16 seq16384, CP1 EP8 TP1 PP1, GB128 MB1. The seeded v1 row permutation, stratified heldout selection and zero-loss tail padding are retained exactly. One unique epoch: native sampler and cosine horizon **34135**, calibration execution cap16. The 34135 horizon remains for main; no silent cap-rescaled cosine. As in v2, the actual terminal update is explicitly forced to minimum LR, which is separate from the full native cosine horizon.
- All eight actual calibration optimizer receipts, finite positive gradients/LR, native save and full-state reload at step16/scheduler2048/cursor2048 are mandatory. Warm timings use all-rank maxima for updates9..16, plus explicit 15% overhead/300s fresh-main import assumptions. These are projections, not a noon guarantee.
- Absolute training stop **2026-10-19 11:30 KST**, save/evaluation/owned cleanup by **12:00 KST**. Preserve existing native final evaluation/full-state save and outer watchdog/ExecStopPost controls.
- Exact supervisor MainPID/UID/full argv, permanent execution claim, all8 cooperative lock/live idle checks, exact pinned Engine ID/name/owned labels, no automatic restart. Owned Engine prefix `nemochat-sft-sealed03-r04-`, peer label `nemochat.sealed03=v1` (also on CPU Engines); inherited image labels allowed. Cleanup never enumerates/kills unrelated jobs. Readonly same-absolute-path socket-directory mount, no Docker socket in containers. Root+DAC_OVERRIDE remains necessary for the pinned root-only SDK, no privilege escalation/other capabilities.

## Input authentication and failure granularity

`sealed_input.Snapshot` imports only producer `sealed_snapshot03.client.open_snapshot` and retains every OpenedSnapshotFile while its numpy views/datasets exist. All tokens, masks, row catalogs **and heldout dialogue metadata** come from received O_RDONLY seal15 FDs. Actual file size, dtype, shape and readonly view checks are mandatory. Producer client owns SCM frame/holder/header identity validation and host-peer-descendant policy.

Each selected native item carries row/source-index/role/snapshot identity. Guard reconstructs the expected row from the unchanged dataset order and source index, requires actual int64 shifted token/label overlap and vocabulary bounds, exact float32 0/1 mask and arange positions, then compares SHA256(domain-zero-byte + LEuint64 row + contiguous LEuint32 full token row + uint8 mask) to the sealed raw[N,32] catalog. Native synthetic tail rows must be token11/zero loss. Train/eval role is independently checked against native model.training.

Clone first, verify those exact clones, stage them once to CUDA, synchronize all-rank MIN success, then pass the **same tensor objects** through frozen native forward `.to(same device)`. Local loader/guard/staging exceptions participate in MIN before any rank forwards that microbatch. One bad rank is all-rank fatal, no quarantine/replacement/resampling or optimizer/LR advance for the failed accumulated update. Earlier good microbatch forwards in the same accumulation **may already have happened** (including model-side forward state); later failure aborts the process and suppresses that update. This is microbatch pre-forward integrity, not a rollback transaction across an entire global batch. No bad-input/RNG continuation is attempted.

## Evidence and remaining gates

- `python -m unittest discover -s midtraining/sft_after_kd/runtime04 -p 'test_*.py' -v`: lifecycle/release/row/header/identity CPU tests.
- `verify_cpu.py`: already-executed pinned eight-process Gloo + real memfd/SCM_RIGHTS fixture, nine rank7 faults rejected by all ranks before forward. Its client stub is **test-local only**, never production fallback.
- `verify_sdk.py`: already-executed pinned no-device/no-weight SDK construction; readonly host driver libraries permit SDK import without GPU device access. Initial missing-libcuda failure remains preserved.
- `integrate_snapshot.py`: actual producer integration remains parent-run, cannot be claimed from fixtures.
- Actual all8 model import, native learning16, finite gradients, synchronous save, full-state reload, fresh-C main, final heldout/save and on-time complete.json remain **unrun** until parent release.
