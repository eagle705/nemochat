# Approved offline1920 native materialization

`materialize.py` is the actual CPU producer; `prepare.py` remains the immutable older capacity census. The approved recipe is **3662 optimizer calls**, 4096 sequence, GB128, MB1, 1,919,942,656 processed positions, 67/10/3/20 blend, no cycling or acquisition. Native per-source reserve is `(ceil(ceil(3662*128*weight)*1.005)+2)*4096+1`.

| Source | Required unique tokens |
|---|---:|
| Code Concepts | 1,292,808,193 |
| Algorithmic | 192,966,657 |
| Formal Logic | 57,901,057 |
| ClimbMix Replay | 385,921,025 |

The capacity SHA is `467608152ca9288b561d5111d9f39d33cd26d546ea05d876024755e13f9593db`; scarce Code has only 262,453 tokens of measured headroom. The seven census-identified Algorithmic raw documents are authorized **whole-document exclusions**, never ID remapping. Their text and token hashes are authenticated. New train/validation output must contain zero IDs18..29.

## Data path and acceptance

- Reauthenticate failed historical generations without promoting their whole output. Pin frozen selector sources, official 13-file raw pool, full new read-only catalog, immutable seed, tokenizer and complete heldout history.
- Reuse the three authenticated native prefixes document-for-document and rebuild fresh native indices. Only their remaining catalog tails receive BPE. Authenticate seven native cursor samples for each source.
- Replay preserves actual native IDs, excludes every decode/encode mismatch as a whole document, and always omits the known upstream clipped final document. Apply full benchmark, raw heldout and token reservations.
- Preserve all 32,397 old-plus-new grouped heldout records and 84,990 original ClimbMix text reservations; clone all four seed validation pairs byte-identically.
- Run a fresh complete native index/count/shape/range/EOS/global duplicate/heldout audit and a fresh streaming ID18..29 and byte-SHA audit on all eight produced pairs. Publish `manifest.json` and SHA-bound `success.json` only after every gate passes.

Output: `datasets/midtraining/super35-kd-01/offline1920-v1/data/`.
Evidence: `results/midtraining/offline1920-corpus-01/` (`approved-request.json`, `producer-review.json`, source freeze, Engine create/start/exit receipts). The parent request arrived after this immutable producer started: `producer-review.json` authenticates exact budget and all seven exclusion identities against it, binding the parent SHA to the already-running producer request/source SHA and exact container ID without changing running code. The manifest binds the producer-request SHA, completing this explicit hash chain. `progress.json` reports phase, source, active native tokens/documents, elapsed seconds and output bytes. It is **not** a readiness receipt.

## Execution and handoff

The one-shot `results/midtraining/offline1920-corpus-01/launch_cpu.py` freezes the source/request digests and creates a durable Docker Engine container, not a terminal child. Exact approved image, UID1001:GID985, CPU-only runc, network none, 12 CPUs, 64GiB memory/no swap, two-hour application timer, 16GB output cap, no capabilities, read-only root/repo and only fresh generation/scratch writable. Container/request identity is in `engine-create.json`; do not run the launcher twice.

```sh
.venv/bin/python -m pytest tests/test_offline2b_materialize.py tests/test_offline2b_corpus.py tests/test_midtraining_recover_corpus.py -q
python midtraining/offline2b/materialize.py             # inert preview
# Already executed once by the authorized materializer:
python results/midtraining/offline1920-corpus-01/launch_cpu.py
```

The parent must require the exact Engine clean exit (exit0, not OOM) plus `longrun_native.load_prepared_manifest()` success/manifest SHA agreement, then release its own 3662-call native training controller. This producer never launches training or reads model weight payloads. No failed/progress artifact is manually promoted. Legacy sources and census evidence remain read-only.
