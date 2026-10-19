# Full03 read-only data acceptance diagnosis

This task owns only this directory and `results/midtraining/sft-after-kd-data-acceptance-01/`.
It never reads model-weight payloads, allocates GPUs, rewrites prepared inputs,
changes expected digests, evicts page cache, or publishes the old completion marker.

## Protocol and evidence

- `verify.py`: paired buffered and aligned O_DIRECT immutable chunk snapshots,
  hashes, first-disagreement evidence, token bounds/reserved-ID measurements,
  mask census and JSON parsing. Direct EOF is an explicitly short aligned request,
  never buffered fallback. Four streaming workers and 8 MiB buffers.
- `serial_check.py`: independent GNU `dd bs=4M` → `sha256sum` verification of
  all seven SFT prepared files plus all eight C `.bin`/`.idx` pairs. Both read
  modes, both process exits, expected sizes, and before/after file identity are
  recorded. Three pipelines; no source writes or file copies.
- `dd_audit.py`: complete fresh C token census with SHA computed from the exact
  same GNU direct-dd immutable stream; a wrong digest invalidates authentication
  of that census even if observed IDs stay in bounds and reserved counts are zero.
- `provenance.py`: retained producer IDs/actual clean exit, historical source
  checksums, original counted audit/source-schema receipts, exact tokenizer assets
  and complete backend comparisons. Tokenizer assets inside checkpoint directories
  are explicitly allowed; model payloads are never opened.
- `range_evidence.py`: repeat implicated 4096-byte ranges through independent dd.
  Saved bytes and digests distinguish a stable buffered zero page from direct
  content, while preserving initial transient direct anomalies.

Launch CPU processes with `CUDA_VISIBLE_DEVICES=''`, BLAS/OMP threads set to one,
`taskset -c 0-7`, and the existing `.venv/bin/python`. Use tracked background
processes only while the worker stays active; wait for completion before returning.
A tool foreground deadline interrupted the initial paired scan after 420 seconds;
its log/results are retained. An initial NumPy-private-header incompatibility
invalidated its NPY semantic census, not its full-file hash observations. The
public versioned header functions are regression-tested and all four arrays
receive a separate complete follow-up census in `array-audit02/`.

## Release rules

No READY is produced while either read protocol disagrees with immutable expected
bytes, a relevant census lacks authentication, or the real buffered/mmap consumer
is unsafe. A passing reread does not erase a failed prior observation and does not
repair the host. O_DIRECT itself is not a validated recovery when independent
full direct reads also disagree.

Reserved IDs 18–29 are **measured, not blanket-forbidden in new SFT**: the approved
new aliases may occur legitimately under the pinned official tokenizer/template.
Prior-trained C data needs fresh authenticated absence for the original-alias
meaning transition. No embedding permutation or vocabulary resize is performed.
The original counted audit includes every row/mask but only 216 independent
source-offset re-renders; do not label that an exhaustive second rendering.

The historical publisher raised an exception without logging filename/digest.
Its exact old failing file cannot honestly be reconstructed from that log. New
receipts identify precise freshly failing files instead of inventing old evidence.

A blocked outcome is the requested deliverable when correctness cannot be
reconciled. Never substitute a smaller pilot, re-render from new raw, rebase SHA,
or create trusted per-row digests from unauthenticated observed bytes.
