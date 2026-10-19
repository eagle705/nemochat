# FD handoff CPU spike 02

## Verdict: VALIDATED — single-file FD handoff only

Receipt: `results/midtraining/sft-after-kd-fd-probe-02/PROBE.json`.
The executable run took 23.215 seconds. It did not launch SFT, use GPUs, read weights, mark READY, expand production RAM, or modify original sources/manifests. Runtime02 and original mmap inputs remain BLOCKED.

## Proven

- Pinned image `nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf`, UID/GID 1001, runc, root read-only, network none, capabilities ALL dropped, no-new-privileges, no device requests, 8 GiB memory/swap cap and four CPU quota. Holder affinity four CPUs and address-space cap 8 GiB.
- Only a short private UID1001-owned mode0700 socket directory mounted `/handoff:ro`. Single Python entrypoint authenticated using host SO_PEERCRED UID1001 and PID equality with Docker State.Pid.
- AF_UNIX SOCK_SEQPACKET + SCM_RIGHTS; identity request and response bind role, name, byte size, immutable SHA, manifest SHA, shape and dtype. Client rejects truncation or anything except exactly one descriptor, seals15, O_RDONLY.
- Genuine tiny NumPy fixture passed actual pinned Docker consumer; wrong client identity denied by host without FD, wrong response identity denied by client, sealed O_RDWR denied, unsealed O_RDONLY denied. Host descriptor validation controls also rejected wrong access/seals.
- Only one real acquisition, exactly three streamed observational views of original heldout file. Buffered SHA failed; first O_DIRECT SHA matched; second O_DIRECT SHA failed. Page voting: 721485 pages, 36 majority pages, zero ties. This variability is NOT a host-health fix or an independent oracle. Candidate accepted only after exact original SHA matched, not by voting confidence.
- Original immutable SHA `19a28db000dabf1bae97cc85c2abbf6c38ac2837f84b08baf71686724182aaf4` matched sealed candidate, host readonly mmap, Docker full stream and mmap hashes, and postconsumer host readback. All15 seals persisted. Host kept actual candidate alive through clean consumer exit.
- Actual container NumPy1.26.4 `np.load('/proc/self/fd/N', mmap_mode='r', allow_pickle=False)` produced shape45090x16385, uint32; full-array min2/max131070 within [0,131072). Array/mmap readonly and mutation attempts denied; host sealed write/grow/shrink attempts denied.
- Source metadata unchanged. Six final-run container CIDs independently read back absent; all socket dirs removed and tracked holder FDs verified closed. No surviving diagnostic candidate.

## Issues and evidence

Initial tiny pass hit an overly strict cleanup label equality assertion because the image injects inherited labels. `tiny-setup-failure.json` preserves that attempt; its exact CID was removed and independently verified before rerunning tiny tests. Cleanup now checks the owned label key/value subset, never broad engine cleanup. No real source reads occurred in the failed setup attempt. Old ram_snapshot_probe code and evidence were not changed. No rejected proc-fd bind was retried.

`PROBE.json` is a single-acquisition guard: do not remove it or rerun the original file acquisition without new explicit authorization. Local tests are safe and source-read-free:

```sh
cd /raid/yoon/nemochat/sft/archive/fd_handoff_probe02
python3 -m unittest -v test_probe.py
```

## Production recommendation — not implemented

Production still requires exact sealed snapshots for all seven files, C-corpus provenance/semantic/reserved-ID proofs, source freeze, an FD-aware native adapter, per-row authentication, and reviewed physical/cgroup memory and descriptor lifetime budgets. This one-file NumPy proof does not validate original mmap input paths, all seven files, native training, C-corpus semantics, reserved IDs, checkpoints, or host health. No production modifications are authorized by this spike.
