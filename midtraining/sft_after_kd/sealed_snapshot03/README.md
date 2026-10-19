# sealed_snapshot03 — CPU-only immutable full03 transport

This generation captures the unchanged 23 manifest-bound originals (seven SFT, eight C bin, eight C idx) into RAM, then derives two raw32 per-row SHA256 catalogs. No original source writes, downloads, GPU allocation, cache eviction, model-weight reads, or original-mmap release.

## Commands

From `/raid/yoon/nemochat`, use the existing `.venv/bin/python`:

- `-m unittest discover -s midtraining/sft_after_kd/sealed_snapshot03 -p 'test_*.py' -v`
- `-m midtraining.sft_after_kd.sealed_snapshot03.holder fixture` — tiny genuine pinned-image SCM_RIGHTS fixtures, no original inputs
- `-m midtraining.sft_after_kd.sealed_snapshot03.driver start` — authorized one-generation durable user-systemd start
- `-m midtraining.sft_after_kd.sealed_snapshot03.driver status`

The `nemochat-sft-sealed03-holder.service` is independent of this agent session, with MemoryMax768G / MemoryHigh640G / CPUQuota800%, no explicit User/Group and no restart. The holder enforces a 3600-second capture-stage deadline and absolute `2026-10-19T12:15:00+09:00` RAM lifetime. Parent cleanup is `systemctl --user stop nemochat-sft-sealed03-holder.service`. A STARTED receipt forbids silently repeating an original acquisition.

## Gates and evidence

At least 1TiB MemAvailable is required immediately before systemd start and capture. Authenticate original manifests, historical producer exit0 and exact producer-source/tokenizer chain. Each source gets at most three read views: aligned direct capture, then only on mismatch buffered+fresh direct observations with initial-candidate page vote. Every final sealed descriptor must match the immutable original full SHA, with unchanged source stats; voting is not a replacement authority.

All descriptors have seal mask15 and O_RDONLY. Array shape/dtype, full token bounds/mask values/target totals, complete dialogue census, classified SFT aliases, and C int32 index/bin linkage and full absence18..29 are audited from those exact authenticated RAM bytes. Row catalogs use the protocol's actual-NUL domain; see PROTOCOL.md.

A real locally present digest-pinned NeMo Docker image receives all descriptors via SCM_RIGHTS and independently hashes full stream and mmap views. It runs finite CPU-only with no network, read-only root/mounts, caps dropped, no-new-privileges, no devices, bounded CPU/memory/PIDs; its exact owned container is removed and absence read back. Same-UID arbitrary host peers are rejected. Owned pinned-image init and verified descendants are accepted, including native torchrun children; inherited image labels are permitted while owned labels are checked.

Only all-gates-pass creates `results/midtraining/sft-after-kd-sealed03/READY.sealed.json`, schema `nemochat-sealed-full03-v1`, status `accepted_sealed_snapshot_full03`, explicit `input_mode=sealed_scm_rights`. It contains25 files including two derived catalogs, not23. The live holder must remain matching; a stale READY is not authorization. Any unresolved hash/semantic gate closes all RAM and writes precise BLOCKED evidence without rebasing/retry. Original mmap remains BLOCKED. GPU release is parent-only. Checkpoint integrity is `not_checked_explicit_user_waiver`; neither host repair nor weight verification is claimed.

Source freeze includes all producer/client/consumer/tests/docs files. Do not edit them after service start. Durable results include request/source freeze, fixture receipts (including preserved setup failures), focused tests, STARTED exact MainPID/argv readback, per-file source views/stat/hash audits, CPU-consumer container receipt, progress, cgroup metrics and holder-state heartbeat.
