# runtime05 — short private TMPDIR checkpoint fix

CPU preparation only. **A renewed user authorization and explicit parent release are required before any new GPU learner.** Default `start.sh` is inert; no service, learner or automatic retry is installed by preparation. The release template is NOT_RELEASED and `automatic_main_after_calibration=false`.

## Scope and lineage

Clone only runtime04's small working source. Keep runtime04's frozen source and failed save untouched. Attempt04 executed 16 genuine positive-LR updates on all eight ranks, but failed its first save with `multiprocessing.SyncManager` → `SocketListener.bind` → `OSError: AF_UNIX path too long`; zero committed resumable checkpoints. Those unsaved updates cannot be spliced into runtime05 counters or curves.

Every runtime05 container starts with `TMPDIR=TMP=TEMP=/nct` **in its Engine process environment**, before Python imports/caches tempfile or spawns workers. `/nct` is a new container-private tmpfs (`rw,exec,nosuid,nodev,mode=1777,size=8g`), with execution allowed for compiled JIT shared objects. No host global /tmp mount/use. Host subprocess scratch stays in the existing task scratch. HOME, Inductor, Triton and CUDA caches remain fresh generation/stage paths, never copied compiled artifacts. Readonly root, no-new-privileges, DAC_OVERRIDE-only addition and no-device CPU proofs remain intact.

Unchanged native contracts: original C `checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01`, model-only fresh optimizer/scheduler/RNG/cursor for calibration and main, native Bridge pretrain/DDP/distributed AdamW/clipping/checkpoint APIs; complete embedding/destination import; BF16 seq16384, CP1 EP8 TP1 PP1, GB128 MB1, one unique epoch/native cosine horizon 34135. Existing sourceguard authenticates the frozen local wrapper plus unchanged four SDK source checks. FD-only real sealed public client, row/dtype/label/mask/position hashes and all-rank MIN remain unchanged. Holder unit is still `nemochat-sft-sealed03-holder.service`; accepted peer prefix is still `nemochat-sft-sealed03-`. Runtime05 owns only its `r05-` services/Engines and result directory.

Deadline unchanged: stop training **2026-10-19 11:30 KST**, finish save/evaluation/cleanup **12:00 KST**. No base/Mid KD rerun; no model payload SHA/CRC/audit. Checkpoint integrity remains `not_checked_explicit_user_waiver`.

## CPU execution and acceptance

`finalize_cpu.py --freeze-only` freezes the allowlisted small source/metadata. It refuses drift on an existing freeze. Run once per owned output:

```bash
.venv/bin/python midtraining/sft_after_kd/runtime05/verify_storage.py --attempt 01
.venv/bin/python midtraining/sft_after_kd/runtime05/verify_cpu.py
.venv/bin/python midtraining/sft_after_kd/runtime05/verify_sdk.py
.venv/bin/python midtraining/sft_after_kd/runtime05/integrate_snapshot.py --ready /raid/yoon/nemochat/results/midtraining/sft-after-kd-sealed03/READY.sealed.json
.venv/bin/python midtraining/sft_after_kd/runtime05/finalize_cpu.py
```

`verify_storage.py` uses the pinned image and identical /nct/readonly/capability isolation, no DeviceRequests. It exercises real torch multiprocessing Manager.Queue put/get/proxy close/shutdown and a spawned child inheriting /nct; records actual socket paths and byte lengths; reproduces the long-path failure in a private negative-control subprocess; compiles/runs a C executable and loads a shared library from /nct. The tiny storage test uses CPU fixture tensors, Gloo, real MCore ShardedTensor and TorchDistSaveShardedStrategy `torch_dist` synchronous save plus native load; no SHA/CRC. Observing SyncManager.start records but does not replace the native queue implementation. This proves the storage integration seam, **not a full model/optimizer save or native learner reload**. Failed bounded fixture attempts and logs are retained, never overwritten.

`verify_sdk.py` executes actual post-install_runtime sourceguard on the pinned SDK with five denied negative controls. Readonly host driver library mounts permit import without GPU devices. `verify_cpu.py` is an eight-process sealed-input fixture, explicitly not real producer integration. `integrate_snapshot.py` separately runs the real public producer client over the existing READY/holder in eight CPU ranks and binds the new runtime freeze. It does not recapture or modify the producer. See HANDOFF.json for actual passed gates, exact counts and remaining work.

## Observed CPU storage limitation (not a pass)

Bounded storage attempts are preserved: 01 exposed the pinned spawned AutoProxy Queue's absent `_close` finalizer (close its actual IPC connection instead); 02 proved Docker tmpfs defaults can remain noexec unless `exec` is explicit (fixed by adding `exec`); 03 passed genuine Queue/spawn/socket-budget/compiled executable/shared-object checks and reached the actual MCore synchronous save, then failed at pinned `filesystem_async.py:248 preload_tensors` calling `torch.cuda.synchronize()` with `RuntimeError: No CUDA GPUs are available`. The tiny native save did **not** complete and native load was **not** executed. No CUDA monkeypatch or SDK edit is used. `collect_storage_blocker.py` summarizes that exact retained failed Engine and unchanged executed fixture source without another attempt. Parent must set `cpu_native_storage_blocker_acknowledged=true` explicitly after reviewing the bound blocker receipt before any separately authorized GPU calibration. This does not relax the required actual full-model GPU save16/reload gate.

## Single explicit restart, ONLY after approval

Parent reviews renewed user authorization and the CPU receipts, fills `parent-release.template.json` as `parent-release.json` with status `parent_accepted_runtime05`, exact READY/snapshot/live-holder/source/producer canonical-freeze, sourceguard/regression/storage/real-FD proof digests, and explicitly reviewed normalization flags. The producer digest authenticates canonical `files` JSON, not the envelope file byte hash; holder start_ticks is canonical. The untouched template cannot release anything.

```bash
bash midtraining/sft_after_kd/runtime05/start.sh --calibration \
  --release /raid/yoon/nemochat/results/midtraining/sft-after-kd-runtime05/parent-release.json
```

This is a fresh-C 16-update native calibration, not recovery of attempt04. Require actual all-eight finite updates, committed full-state save16, native operational full-state reload/evaluation before any main. With template policy false, main needs a separate explicit `start.sh --run main --release ...` after accepted calibration and clean supervisor exit. Parent may only set automatic main policy true with corresponding explicit authorization; main always imports original C afresh, never calibration learned weights. Full native GPU save/reload remains unproven until that future authorized calibration actually succeeds.
