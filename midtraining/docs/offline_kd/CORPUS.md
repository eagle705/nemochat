# Offline2B unique corpus gate

한국어: [CORPUS.ko.md](CORPUS.ko.md).

## Immutable training request

This is a **new offline KD** corpus generation, not the historical 3,200-update
online plan. The runtime starts the original
`checkpoints/exports/nemochat_baby_base_hf_bf16_02` (never the smoke-overfit export).
The corpus producer does not read model-weight payloads. Their explicit integrity
waiver does **not** waive tokenizer or dataset byte verification.

Sequence length 4096, global batch 128:
`ceil(2,000,000,000 / (4096 * 128)) = 3815` optimizer updates,
**2,000,158,720 processed input positions**. This is not the number of unique
prepared tokens or supervised targets. Source order and weights are fixed:
Code Concepts / Algorithmic / Formal Logic / previously consumed ClimbMix replay,
67% / 10% / 3% / 20%.

The native per-source preparation gate is
`(ceil(ceil(3815 * 128 * weight) * 1.005) + 2) * 4096 + 1` tokens,
including native 0.5% oversampling, rounding reserve and the shifted extra token:

| Source | Required unique prepared tokens |
|---|---:|
| Code Concepts | 1,346,818,049 |
| Algorithmic | 201,027,585 |
| Formal Logic | 60,317,697 |
| ClimbMix replay | 402,042,881 |

No cycling, implicit epochs, reduced budget, mix substitution, new downloads or
clipped documents are authorized. Uniqueness means exact token-document hashes
within this new generation, not semantic deduplication or novelty versus BASE.
ClimbMix is explicitly replay.

## Real historical failure and new Replay policy

The preserved recovery failed after 504.4558 seconds with
`ValueError('replay pinned tokenizer roundtrip mismatch')`, after 17,435,561
selected Replay tokens / 27,791 documents. Its entire manifest is **not accepted**.
The exact immutable failure SHA is
`8f5de4d23ecd7de41b842fc7aee4a12f0e28d609b5edd368ed20c34d83cfd041`.

The new producer checks each native token document with the pinned tokenizer.
A decode/encode mismatch excludes the **whole document**, increments an explicit
counter and saves bounded mismatch metadata. It never normalizes decoded text,
changes token IDs, fabricates text, bypasses the roundtrip requirement, or accepts
the known clipped upstream tail. Reserved IDs18..29, range/EOS violations or
corrupt bytes remain hard failures, not skippable Replay anomalies. Benchmark,
full heldout and cross-source duplicate checks still apply after roundtrip.
The real-run mismatch examples distinguish lossy replacement characters from
other noncanonical tokenizations; do not infer a cause from the exception alone.

## Inherited provenance, independently authenticated

`prepare.py` reads the historical helpers without editing them and authenticates
all active helper digests, all six frozen original production sources and all
eight frozen recovery sources. It verifies the original full scan, official
13-file allowlist (6,032,365,862 bytes), pinned official repository/revision,
benchmark inventory and tokenizer asset hashes. No network is used.

Only the valid full production catalog is opened, read-only and immutable:
`datasets/midtraining/super35-kd-01/production-24h-v1/data/catalog.sqlite`, SHA
`9d55e9cde8dfef7f61ad8d6b97271563f1245c4a34ce24574f36b6abb36e7eb1`.
The malformed pilot catalog is never read, hashed, repaired or substituted.

Published native parts from the failed recovery receive fresh bin/idx SHA256,
file lengths, native reader token/document/dtype/document-index checks, full
range/EOS/reserved-ID checks and full heldout/cross-source hash checks.
These new measurements are labelled as current digests, not invented historical
digests. Historical zero-skip selectors and seven deterministic first/last/interior
re-tokenized samples establish the first-N native cursor for **each source**.
Remaining eligible catalog rows are fully tokenized once to measure capacity;
no prefix cycling or billion-token re-tokenization is hidden.

The complete frozen old-plus-new reservation JSONL, all 84,990 original ClimbMix
raw heldout hashes, all original native Replay validation documents (including
its clipped tail as reserved, never trained), and all eight frozen validation
files remain excluded. Frozen validation bin/idx hashes must equal the seed.

## Executable, CPU-only census

From repository-root CWD:

```bash
.venv/bin/python -m pytest -q tests/test_offline2b_corpus.py
.venv/bin/python midtraining/data/offline_kd/prepare.py --preview
python midtraining/data/offline_kd/prepare.py --census \
  --reviewed-sources results/midtraining/offline2b-corpus-01/reviewed-sources-01.json \
  --output results/midtraining/offline2b-corpus-01/census-FRESH
```

Run the census only in the already-local pinned CPU image from `corpus.yaml`,
Transformers5.12.1, UID1001:GID985, `runc`, no devices/GPU, no network,
read-only repository and only this worker's results/scratch read-write.
The real Engine receipt records the exact image ID, source review, mounts,
64GiB hard memory/swap limit, 12-CPU limit and container identity.
`prepare.py` has a two-hour SIGALRM watchdog and a two-billion-token Replay scan
cap. It publishes no training manifest or success receipt on a capacity census.
No dataset bytes are rewritten before the unique-capacity gate is satisfied.

The full-source census for each Specialized source counts all eligible rows,
not a projection from average lengths. Replay stops at the target and is labelled
a **lower bound**, not a full 10B-token uniqueness survey. Source results are
published immediately in `capacity.json` so the parent can see shortages early.
`census-completed.json` means the census completed; it never means corpus ready.

## Parent/runtime handoff

`results/midtraining/offline2b-corpus-01/data-contract.json` freezes the plain
schema1 manifest and native `GPTDatasetConfig` fragment. Eventual acceptance needs
all four train and four frozen-validation native prefixes, complete per-file
counts/digests, tokenizer lineage, full heldout digest, zero reserved-ID counts,
zero train/heldout overlap, zero cross-source duplicates, native per-source
capacity, companion `success.phase=prepared_verified`, exact manifest digest in
`success.result.manifest_sha256`, and a clean exact Engine exit. Native sample-index
no-repeat, teacher-cache build and GPU release are separate parent gates.

Source one completed first: **Code has exactly 1,293,070,646 eligible unique tokens
in 2,697,481 complete documents, short by 53,747,403 tokens** versus the fixed
native reserve. This alone blocks the requested run without newly approved Code
data. These are measured full-source counts, not a token/byte projection.

The first real census then stopped correctly on forbidden IDs18..29 in the
Algorithmic tail (Engine exit1, no OOM). Its failure and source snapshot remain
immutable. A separate diagnostic generation scans the entire tail while counting
and excluding every forbidden-ID document, but **keeps the source compatibility
gate blocked pending explicit review**. Diagnostic continuation never authorizes
training, alias changes, or treating those documents as compatible. Default
selection remains a hard failure on such IDs. Final counts and occurrence/example
hashes are in the exact `capacity.json` receipt.

The full Specialized census now also confirms:

| Source | Eligible unique tokens | Documents | Shortfall |
|---|---:|---:|---:|
| Code Concepts | 1,293,070,646 | 2,697,481 | 53,747,403 |
| Algorithmic (forbidden-ID documents excluded) | 193,495,508 | 179,436 | 7,532,077 |
| Formal Logic | 60,564,276 | 408,328 | 0 |
| ClimbMix Replay (measured lower bound) | 402,043,247 | 628,502 | 0 |

The final CPU census completed with exact Engine exit0 (container
`421cb3930a963cfb39279cd7e4148fdd4d2df9a68cff3ac54f183822ab5c5de2`),
no OOM and no GPU/network access. Replay scanned 402,088,071 tokens in 628,536
documents and excluded 34 non-roundtrip whole documents. The historical failure
was reproduced at native document27791: first differing position15,
1,067 original versus1,054 re-encoded non-EOS tokens, **zero** Unicode replacement
characters. Thus this observed failure is not explained by lossy replacement
characters; the native token sequence differs from the current pinned encoder's
canonical sequence. The original token IDs were never rewritten.

All accepted candidate documents across sources have zero IDs18..29, zero
full-heldout token-document overlap and zero cross-source token-document duplicates.
The report binds all32,397 frozen old-plus-new reservations, all84,990 original
ClimbMix raw heldout hashes and exact frozen validation bytes. It records fresh
native train/validation bin/idx hashes and complete lineage.

Final capacity report SHA256:
`467608152ca9288b561d5111d9f39d33cd26d546ea05d876024755e13f9593db`.
The status is **blocked_insufficient_unique_source_capacity**, not corpus ready.
No `offline2b-v1` training manifest, success receipt or cached teacher data was
published, and this worker released no production training.

A metadata-only follow-up restores the original Replay source's explicit
`upstream_last_document_truncated:true` / `truncated_documents:1`; the zero adapter
counter in the immutable census02 source receipt must **not** be read as a claim
that upstream data was untruncated. The target selector already omitted that tail.
`replay-source-readback-03.json` re-read the actual native source in the same
CPU-only pinned image, rehashed its full40,001,077,252-byte bin and idx, matched
all census02 native counts/hashes, and exited0. Source snapshot02 and its full
census remain immutable; the current producer exposes the corrected provenance.
The focused corpus/recovery/production/tokenizer suites pass **62 tests**.

The seven rejected Algorithmic documents contain 4,735 tokens total, including
10 occurrences of ID19 (`<img>`) and one ID18 (`<image>`). They are never training
candidates. The explicit source compatibility block remains independent of the
capacity shortfall. The pinned official metadata lists **only one** Algorithmic
Parquet file, already completely scanned. Therefore an extra Code part alone
cannot satisfy this fixed recipe; Algorithmic needs 7,532,077 additional eligible
unique tokens from a separately authorized source/revision or a separately
approved recipe change. No such change or acquisition was made.

**Current outcome:** read the exact measured `capacity.json` and Engine exit
receipt under `results/midtraining/offline2b-corpus-01/`. A source shortfall is an
honest blocker requiring newly approved additional eligible data, not permission
to quietly shrink the run. No historical failed whole manifest is promoted.
