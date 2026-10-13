# Corpus size and training budget: measured, bounded, scope-labeled

## Run locally

From the repository root, using the CPU/test environment from the main setup:

```bash
python scripts/estimate_corpus.py --config configs/planning/corpus_local_smoke.json --output /tmp/corpus-local.json
python scripts/plan_training.py --config configs/planning/training_measured_example.json --output /tmp/training-plan.json
python scripts/plan_training.py --config configs/planning/training_unmeasured.json --output /tmp/unknown-plan.json
python -m pytest -q tests/test_planning.py
```

Use fresh output paths: both CLIs refuse overwrite. Input paths are relative to
**the working directory**, not the config directory. No corpus is downloaded.
`transformers` is needed for real tokenization; `pyarrow` is optional for Parquet.
Tokenizer loading is cache-only (`local_files_only=True`, remote code disabled).
A missing cached tokenizer is a failure, not a switch to a toy tokenizer.

The intended estimation tokenizer is
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` at immutable revision
`2dc98e2afe4face0e4ce40972a915c45368bd34a`, matching the data preparation policy.
Only `purpose: smoke` may use GPT-2 at
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. Super is still used in the supplied
local smoke. No weights or GPU inference are needed. A revision pin and backend
hash identify declared/cache tokenizer content; they do not authenticate a
mutable local cache or prove a later training job used it.

## Two explicit sampling paths

Each `strata` entry names **one source and one shard population**, a local
`.jsonl` or `.parquet` path, `population_documents`, `sample_documents`, and
`mode`. JSONL records must contain string `text` (or explicit `text_field`).
No implicit glob, streaming dataset download, or full-corpus tokenization occurs.

1. **`reservoir`**: read the entire explicitly selected local shard and use
   Algorithm R to sample documents uniformly without replacement. Only the
   reservoir is tokenized. Scanned count must equal the known shard population.
   `seed` is required; independent per-source/shard seeds are SHA256-derived,
   stable across stratum ordering. A huge corpus need not be read; selected local
   files do need to be read completely. If even that is too expensive, use path 2.
2. **`representative`**: the file already contains exactly `sample_documents`
   rows sampled elsewhere. Require `sampling_provenance`, describing the frame,
   source/shard sampling design, document selection, inclusion probabilities if
   known, and population-count provenance. The provided population is the
   population this sample represents, not the file's number of rows. The code
   consumes all rows and refuses to treat a prefix of a longer file as a sample.
   Representativeness is a user attestation, not verified evidence. Point
   estimates are conditional on that attestation; uncertainty is **null**.

For example, adapt the smoke manifest by changing one entry to:

```json
{
  "source": "my-source",
  "shard": "externally-sampled-frame-A",
  "path": "my-representative-sample.jsonl",
  "population_documents": 2000000,
  "sample_documents": 2000,
  "mode": "representative",
  "sampling_provenance": "User-attested source/shard frame and sample design; document-count provenance required"
}
```

Choose sampling allocations explicitly per stratum; the implementation does not
invent proportional quotas or missing populations. Duplicate strata/files and
sample sizes exceeding populations fail. `max_files`, `max_input_bytes`,
`max_document_bytes`, `max_sample_documents`, `max_sample_text_bytes`, and
`max_sample_tokens` are required positive bounds. File byte sizes are checked
before scanning, JSONL line reads are bounded, and all selected documents have
bounded text size. Sample text/token quotas fail rather than truncate documents.
Reservoir memory is bounded by selected document count × document byte limit;
final sample-byte quota is checked after selection. The on-disk byte bound is
not a Parquet decompression/RSS guarantee: Arrow may decode row-group pages even
with one-row batches. Keep local inputs immutable during estimation; concurrent
file replacement is outside the provenance/byte accounting trust boundary.

**A chosen shard is not an unbiased sample of all shards.** This implementation
only totals the named populations. It never multiplies sampled-shard totals by a
source-wide shard count. A full source estimate needs an externally established
representative design/population manifest. Missing shard-selection probabilities
cannot be repaired by a document-level confidence interval.

## Token/byte accounting and uncertainty

For stratum h with population N_h, sampled document text length l_i, and n_h rows:

- estimated text tokens = N_h × mean(l_i);
- estimated EOS tokens = N_h; total = text + EOS;
- estimated UTF-8 source text bytes = N_h × sampled mean bytes.

Encoding uses `add_special_tokens=False`, no BOS, one EOS per document. EOS has
**zero source-text bytes**, not the byte length of an EOS spelling. Reports give
sample hashes/row IDs, sampled docs/bytes/text and EOS counts, per-stratum min,
max, mean, p50, p95, p99 text/byte lengths, tokenizer identity/backend hash,
bounds, scope and conditional uncertainty. Sample percentiles are not estimated
population percentiles or token-weighted context distributions.

For complete fixed-shard reservoirs, the variance of each shard total is
`N_h² × (1 - n_h/N_h) × sample_variance(l) / n_h`; independent shard-stratum
variances are summed. This is **shard-aware stratification**, not pooled IID
variance over documents from selected clusters. It conditions on exactly these
shards; source-wide random cluster-selection uncertainty is not modeled. Census
variance is zero, and a non-census n=1 or any representative entry suppresses the
overall CI. The reported normal-approximate 95% interval can be poor for small
samples/heavy tails; it is not exact coverage and does not reflect population
count error, unselected shards, filters or deduplication. Do not treat the tiny
smoke interval as strong statistical evidence.

Required `scope` descriptions: `population`, `shard_selection`, `heldout`,
`filter`, `dedup`. The estimator measures **raw rows**, including empty strings
and repeated documents, with one hypothetical EOS each. It performs no heldout
subtraction, filtering, dedup, tokenizer-empty exclusion, quota truncation or
packing. The preparation pipeline excludes empty/reserved/duplicate documents;
therefore this estimate is not automatically its accepted training-token total.
For a filtered/heldout-free/deduplicated total, supply populations and samples
from that already processed frame and state how it was produced. Do not infer a
corpus-wide dedup rate from duplicates in a small sample.

## Time and cost inputs

`plan_training` uses simple direct functions, not an SDK or hardware simulator.

- `global_batch_size = data_parallel × micro_batch_size × gradient_accumulation_steps`.
  DP means actual data-parallel replicas, not total GPUs when TP/PP are active.
  A supplied GBS must match. The planner does not validate model/TP/PP/EP memory
  fit or infer a valid hardware topology.
- Each stage has its own `sequence_length`, `target_tokens`, optional
  `completed_tokens`, and a **measured** rate with `measurement_provenance`.
  Short-context rates must not be silently reused at long context.
- Supply `tokens_per_update` and a list of observed `step_seconds` for rate
  `tokens_per_update / mean(step_seconds)`, or supply `tokens_per_second`.
  Do not supply both. Token counts must use the same nonpadding/EOS accounting
  as the target. A supplied tokens/update may be below GBS × sequence length.
  With throughput-only inputs, absent tokens/update assumes fully packed
  GBS × length and explicitly labels that assumption for update/capacity rounding.
- `effective_wall` rates already include idle/utilization, checkpoint and eval
  costs: utilization must be 1 and separate checkpoint/eval seconds must be 0.
  Setup must be excluded from this measured rate and entered only once as
  `remaining_setup_seconds`.
- `core_compute` measurements exclude idle/checkpoint/eval. Require explicit
  `utilization` in (0,1] for **other** idle time only. Optional
  `checkpoint_seconds`/`checkpoint_every_updates` and
  `evaluation_seconds`/`eval_every_updates` add events exactly once. Events occur
  at divisible completed remaining updates, including the final one if divisible.
  Counters restart for remaining work, not a global resumed checkpoint cursor;
  use effective-wall rates or carefully adjusted inputs for a resumed schedule.

Rate sensitivity defaults to conservative/base/optimistic multipliers 0.8/1/1.2,
not measured confidence bounds. Supply alternate ordered multipliers with base
exactly 1. Final updates round up, and extra rounded token capacity is shown.

Prices have explicit `price_unit: node` or `gpu`. Cluster hourly price is
price × nodes, additionally × GPUs/node only for GPU-unit prices. All allocated
resources are billed during setup, idle, checkpoint and evaluation wall time.
`elapsed_seconds`, optional `spent_cost` (otherwise billed elapsed time),
`time_limit_seconds`, and `budget_limit` define remaining quota. Completed stage
tokens are separate progress inputs. Capacity fits whole optimizer updates via
binary search including periodic overhead, proceeds stage-by-stage, and cannot
skip an unfinished earlier stage. Reports include available remaining time and
budget, projected remaining/full time and cost, signed slack, limits fit and
capacity. Exhausted quota/setup-consuming quota yields zero capacity, not a
negative or magically faster schedule. Free-price runs remain constrained by time.

Optional `storage` takes user-estimated `checkpoint_bytes`, positive
`retain_checkpoints`, `existing_checkpoints` (default 1), `eval_artifact_bytes`,
and `retain_eval_artifacts` (default 1). It estimates retained bytes and a peak
upper bound allowing one transient checkpoint. Sizes must cover whatever model,
optimizer/resume state the user intends to retain. Dataset/cache/replica storage,
existing evaluation artifacts and object-store billing are not modeled. Storage
is not a verified disk-availability check.

**Unknown throughput stays unknown.** `training_unmeasured.json` names B300 but
returns `measurement_required` and null time, cost, and capacity in all three
scenarios. GPU names do not supply throughput. `training_measured_example.json`
is explicitly an *illustrative arithmetic fixture*, not an observed hardware
benchmark. Replace its rates and provenance before making a purchase decision.

## Exercised evidence

The local smoke was actually run with the cached pinned **Super** tokenizer,
without downloading corpus/model weights or using an injected test tokenizer:
three handwritten CC0 fixture rows form the entire declared population, two
rows (rows 2 and 3) sampled with seed 1729. The sample contained 45 text tokens
+ 2 EOS, 136 UTF-8 text bytes; the conditional three-document estimate was
67.5 text + 3 EOS = **70.5 tokens** and 204 source text bytes. These are only local fixture estimates,
not ClimbMix sizes or GPU throughput. Unit tests exercise deterministic/uniform
sampling, census/stratified variance, representative CI refusal, quota/invalid
inputs, actual local Parquet, overhead/capacity/price math, and the planner CLI.
