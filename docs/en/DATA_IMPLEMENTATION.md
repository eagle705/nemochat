# CPU-first ClimbMix data implementation

## Scope and rights

`src/nemochat/data/climbmix.py` and `scripts/prepare_data.py` implement bounded,
CPU-only selected-shard acquisition and local JSONL preparation. Optional
`transformers`, `requests`, and `pyarrow` imports are lazy. No snapshot downloader,
GPU, corpus-wide shuffle, upload, or training step runs.

The exact `SOURCE` object is required in ClimbMix configs:

- Shuffle: `karpathy/climbmix-400b-shuffle` at
  `915333b4f8b8684f39aeaafea600fea6f43fb703`.
- Origin: `nvidia/Nemotron-ClimbMix` at
  `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`.
- Retain **NVIDIA attribution and CC BY-NC 4.0** for this noncommercial educational
  project; the derivative MIT card is not a waiver. Review redistribution and
  trained-weight terms separately. `rights_acknowledged: true` records a user's
  review, not legal verification by the program.

Local files require explicit attribution and either `CC0-1.0` or
`CC-BY-NC-4.0`. The self-authored smoke fixture in
`src/nemochat/data/fixtures/smoke.jsonl` is dedicated to CC0-1.0. Arbitrary/unknown
source or license declarations are rejected; an assertion cannot establish rights.

## Commands and tokenizer contract

From the repository root, with dependencies installed:

```bash
.venv/bin/python -m pytest tests/test_data.py -q
# Alternatively, stdlib-only control/fixture tests:
python3 tests/test_data.py
.venv/bin/python scripts/prepare_data.py \
  --config configs/data/climbmix_smoke.json \
  --output /tmp/nemochat-climbmix-smoke-fresh
```

The supplied remote config explicitly chooses only `shard_00000.parquet`, a
256-token quota, one maximum shard, and a 100,000,000-byte download budget. A
small token quota still requires downloading that selected whole Parquet shard
(about 92 MB), not merely 256 tokens over the wire. Requests stream in 1 MiB
blocks, have connect/read timeouts, and check both advertised and actual byte
counts. Only one selected shard is cached at once; parsing uses 64-row batches.
No entire-corpus download is possible without an explicit list and budget.

Train cannot select `shard_06542.parquet`; validation must select that shard.
Only pinned names `shard_00000.parquet` through `shard_06542.parquet` are accepted.
The selected sequence is deterministic, not random or a claim of representative
sampling. `max_shards` and `token_quota` must be explicit positive integers.

Default tokenizer is
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` at
`2dc98e2afe4face0e4ce40972a915c45368bd34a`.
`AutoTokenizer.from_pretrained(..., trust_remote_code=False)` is used, without
model weights. GPT-2 at `607a30d783dfa663caf39e06633721c8d4cfcd7e` is allowed only
with `purpose: smoke`; `purpose: training` rejects it. Tokenization uses
`add_special_tokens=False`, then appends exactly one EOS. Manifest records EOS,
revision, and SHA256 of the serialized fast-tokenizer backend when available;
this is not a hash of every original tokenizer file. Injected unit-test tokenizers
are clearly labeled and must never be mistaken for actual Super output.

For offline JSONL, use a config such as:

```json
{
  "mode": "local_jsonl",
  "files": ["src/nemochat/data/fixtures/smoke.jsonl"],
  "local_license": "CC0-1.0",
  "local_attribution": "NemoChat self-authored smoke fixture",
  "rights_acknowledged": true,
  "split": "train",
  "purpose": "smoke",
  "token_quota": 64,
  "max_shards": 1,
  "quota_policy": "prefix",
  "reserved_hashes": {"train": [], "test": [], "validation": []},
  "reserved_manifests": []
}
```

Each JSONL line must be an object with a string `text`. Blank/malformed JSON
lines and non-string text fail; empty/whitespace-only documents are rejected.
Relative config paths resolve from the **process working directory**. Local file
size is capped (`max_input_bytes`, default 1 GiB per file), and document/JSONL-line
bytes are capped (`max_document_bytes`, default 16 MiB). Selected files are fully
hashed for provenance; only rows visited before the quota stops are validated.

## Quota and leakage semantics

- `accepted_tokens = accepted_text_tokens + accepted_eos_tokens` is the quota
  unit and equals the flat binary length in uint32 elements. No quota overrun.
- `prefix` (default) keeps a token prefix plus one EOS when a document exceeds
  the remaining budget. Each record retains its **full original text hash**,
  original token count, accepted token count, and `prefix_truncated` flag. It
  does not pretend the prefix is a complete source document. Prefixes can end
  inside UTF-8 byte-token sequences; this is a token pack, not decoded text.
- `complete` skips documents that do not fit. Both policies can underfill; a
  single remaining slot cannot hold nonempty text plus EOS. `quota_reached`
  reports this explicitly. Zero accepted records is an error, not an empty pack.
- `tokenized_text_tokens` counts ordinary tokens in visited eligible unique
  documents before quota truncation/skipping. It is neither the upstream
  advertised token count nor accepted tokens.
- `consumed_training_tokens` is **zero**: acquisition does not run a trainer.
  A later trainer must keep a separate ledger including repetition, packing,
  loss masks and actual optimizer consumption.
- `reserved_hashes` requires explicit train/test/validation lists, even if empty.
  Cross-split hashes are excluded; `reserved_manifests` imports the
  `record_hashes` of previous packs in other splits. Prepare heldout packs first,
  then pass their manifests to train, or otherwise freeze all cross-split hashes.
  Empty lists are an explicit unreserved smoke selection, **not** leakage safety.
- Exact SHA256 dedup is only within visited selected rows and supplied
  reservations. There is no corpus-wide dedup, near-duplicate removal, benchmark
  contamination clearance, or global concurrent split registry. Raw UTF-8 is
  hashed without whitespace or Unicode normalization.

## Output format and failure behavior

Fresh output directory contains:

- `tokens.u32`: contiguous **little-endian uint32** token IDs, text then EOS per
  accepted record. This is **not Megatron indexed `.bin/.idx`** and requires an
  explicit tested adapter before MCore training.
- `records.json`: record array with source shard/file and zero-based row,
  original text SHA256/UTF-8 bytes, token-element offset/count, truncation metadata,
  and SHA256 of each emitted record's token bytes. Byte offset is token offset * 4.
- `manifest.json`: source/rights/tokenizer/config, selected input byte hashes,
  counters, ordered record hashes, token-file SHA256 and records-file SHA256.

The shuffled text loses original URLs and may contain assembled segments rather
than pristine documents. Shard/row/hash provenance is not recovered original
NVIDIA document lineage. Artifacts do not include raw text or a Megatron adapter.

A sibling exclusive lock and same-filesystem private staging directory prevent
cooperating producers from overwriting. Existing output (including symlinks) is
rejected. Normal malformed-input, tokenizer, HTTP, download-budget and Parquet
errors remove staging and cached shards; publication is one directory rename.
SIGKILL/power loss can leave a stale lock/staging directory: inspect and remove
those manually only when no producer runs. Atomic publication is not an fsync
power-loss durability promise. Non-cooperating external filesystem writers are
outside the lock protocol. HTTP failures are surfaced, not silently retried or
replaced with synthetic output.

Memory grows with visited exact hashes and accepted record metadata; one complete
document is tokenized at a time. This readable implementation is for bounded
CPU pilots, not an out-of-core billion-token dedup framework. Cache/tokenizer
storage and final token bytes also need disk budgeting. Future work includes a
scalable ledger, near-duplicate/contamination auditing, a verified Megatron
adapter, and actual training consumption accounting.

## Execution evidence

The pure fixture suite passed 15 tests and 16 subtests; the stdlib unittest runner also passed all 15 tests. A real selected-shard
GPT-2 CPU smoke run read two rows, tokenized 323 ordinary tokens, accepted 254
ordinary tokens plus 2 EOS (256 total), truncated one record, and consumed zero
training tokens. The output was 1,024 bytes and its SHA256 matched its manifest.
The selected input was 92,291,918 bytes, SHA256
`054ddbd98abf30d773c54de578fc9d579bafeb6c14e04e97bd36aa90e825bf9b`.
Pinned Super tokenizer also loaded successfully with remote code disabled (EOS 11).
A real Super selected-shard pilot likewise accepted 254 ordinary tokens + 2 EOS
at quota 256, after tokenizing 317 ordinary tokens; token binary SHA256 was
`ecde7f64fa1a3b32ee925f2a084c587e814cd48e0a8ee58ee87410da614027f7`.
The local English/Hangul/duplicate/empty fixture with real Super accepted 34
ordinary tokens + 3 EOS, rejected one duplicate and one empty document, and
reported underfill (37/64). These are preparation/tokenizer checks, not GPU
training, evaluation, or full-corpus acquisition results.
