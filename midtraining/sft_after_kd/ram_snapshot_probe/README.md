# Bounded sealed-RAM snapshot diagnostic

## Verdict: PARTIAL

A single real heldout-tokens.npy acquisition (three full observational views, 2,955,198,728 bytes each) recovered the immutable full-file expected SHA in a sealed memfd. Buffered view failed; both aligned O_DIRECT views matched. Only 34 of 721,485 pages differed; majority voting changed exactly those 34 pages, with no ties. No zero-is-wrong rule was used.

Docker handoff failed on the tiny sealed-memfd control before any consumer process executed: runc rejected the read-only `/proc/<holder>/fd/<fd>` bind with EINVAL. The container was removed and absence verified. The identical real-file bind was not retried. Therefore this is NOT end-to-end validation, and no SFT release is authorized.

The real memfd was sealed with WRITE/GROW/SHRINK/SEAL (15), and all three mutation attempts failed EPERM. Its full positional-read SHA matched the original immutable digest. A read-only mmap on the original O_RDWR holder fd failed EPERM; the memfd was closed during cleanup. A later tiny-only control established that reopening the sealed fd with O_RDONLY permits read-only mmap. This remedy was NOT re-tested on the real file: the three-source-view budget had already been exhausted.

## Files and evidence

- `probe.py`: runnable diagnostic; DO NOT rerun without a fresh explicit bounded authorization.
- `test_probe.py`: four controls (complementary corruption/gold digest, zeros, ties, common-corruption rejection), all passed after observed red/green.
- Results directory: `PROBE.json`, per-disagreement page hashes, tiny Docker failure receipt, tiny mmap control and preflight errors.

## Established workflow / pitfalls

- Stream at most three source views chunk-wise into ONE candidate memfd; do not retain three full-file copies.
- Select only bytes observed at the same page offset. Majority votes are observational, not an oracle; only the original entire-file expected SHA is acceptance authority.
- Direct views use anonymous aligned buffers and preadv; partial final EOF reads are accounted exactly.
- Seal before readback; reopen the sealed fd O_RDONLY before mmap. Do not infer that original O_RDWR descriptor + ACCESS_READ is sufficient on this host.
- Docker daemon accepted container config and UID/cap/network/device/memory controls, but its runc bind of this memfd failed. No mounts, proc settings, admin changes or alternate unreviewed handoffs were attempted.
- Docker API minimum was 1.44; host Python omits fcntl seal symbols. Constants were resolved from installed Linux UAPI headers, not guessed.

## Unexercised / production requirements

Real-file consumer SHA/mmap, full token bounds, and semantic/source audit were not completed. Header bytes parse as NPY v1, little-endian uint32 shape (45090, 16385), consistent with manifest and byte size; that does not substitute for token validation.

Full production remains blocked and requires parent/user review, an approved working sealed-RAM consumer handoff, exact immutable SHA checks for ALL data files, explicit memory/cgroup/lifetime/accounting controls at full scale, training consumer validation and existing audit gates. This probe does not fix the host, certify weights, create new training data, or establish golden-producer semantics.

No source writes, downloads, weight reads, GPU requests, global cache changes, READY marker or SFT launch occurred. All owned container(s) and memfds were cleaned up. Only the two delegated ownership directories were modified.
