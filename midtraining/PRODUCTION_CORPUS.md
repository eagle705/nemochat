# Bounded production corpus expansion

한국어: [PRODUCTION_CORPUS.ko.md](PRODUCTION_CORPUS.ko.md).

## Boundary

`production_corpus.py` is a **local-only CPU producer**, not a downloader or trainer.
It reuses `prepare_corpus.py`'s actual Parquet scanner, `Catalog`, benchmark
matcher, preserved MCore indexed SDK, native writer and final token-document audit.
The pilot producer/configuration, accepted pilot outputs and native training inputs
are never edited. No service, GPU, checkpoint-weight load, remote tracking or
network acquisition is started. `--help`, default invocation and `--preview` do
not create outputs or import Transformers. Preview can read an accepted calibration
receipt to print quotas, but does not verify or authorize raw acquisition.

Host-safe commands, from `/raid/yoon/nemochat`:

```bash
.venv/bin/python midtraining/production_corpus.py --help
.venv/bin/python midtraining/production_corpus.py --preview
TMPDIR=/raid/yoon/nemochat/datasets/midtraining .venv/bin/python -m pytest -q \
  tests/test_midtraining_production_corpus.py tests/test_midtraining_corpus.py \
  tests/test_midtraining_longrun.py
```

## Measured horizon, not the 10M pilot quota

Execution requires the parent's actual `calibration_save_reload_accepted` receipt,
`all_ranks=8`, bound to the immutable pilot manifest SHA256
`52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308`.
The existing native helper derives
`train_iters=floor(86400*0.90/mean_step_seconds)` from warm updates9..16,
unprofiled GB128 / sequence4096. This is a projection with assumed10% overhead,
not sustained24h evidence. No pilot timing or arbitrary token budget substitutes
for that receipt.

For each fixed source weight `w` (Code-Concepts .6, Unconditional-Algorithmic .1,
Formal-Logic .1, ClimbMix replay .2), the production target is:

```text
samples = train_iters * 128
required_samples = ceil(ceil(samples * w) * 1.005) + 2
required_tokens = required_samples * 4096 + 1
```

The finished manifest must pass the actual `longrun_native.capacity_gate`.
Whole-document overshoot is allowed; truncation, cycling, source substitution and
weight drift are not. A scarce source fails with its actual selected unique train
tokens/documents and required quota in `failure.json`/`insufficient_capacity`.
Here “unique” means exact native-token documents within this selected generation,
not semantic uniqueness or novelty relative to BASE. ClimbMix remains previously
consumed replay. Production starts from ORIGINAL BASE, never calibration weights;
this producer does not select or load model checkpoints.

## Parent review and raw-file boundary

**Do not execute before the parent reviews these new files and dependencies.**
The original acquisition authorization was at most1GB, with three seed files
 totaling759,378,296 bytes. There is no authorization here to acquire the full
repository. Additional shards require a separately reviewed eligible-file list,
official digests, total byte cap and parent acquisition approval. The parent
obtains them independently; this program only verifies existing local files.

The user subsequently approved ten additional Code-Concepts shards
(`part_000000.parquet` through `part_000009.parquet`), totaling5,272,987,566 bytes;
with the original three seed shards the reviewed raw total is6,032,365,862 bytes.
This is not authorization for other files. Acquisition remains parent-owned.

`production_corpus.yaml` now has a7GB raw allowlist cap,12h preparation wall
limit,64GB output observation cap,2B replay scan tokens and1M tokens/document.
A larger reviewed generation can change **only this new YAML**, then refresh its
source digest and the parent review. Absolute implementation ceilings are24h,
256GB output,64GB raw allowlist,100B replay scan tokens and1M tokens/document.
These are safety ceilings, **not acquisition permissions**. Raw/output/cache
paths resolve under `/raid/yoon/nemochat/datasets/midtraining`; output must be
fresh and cannot be the datasets/midtraining root. Failed outputs are not resumed.
All HF/XDG caches are directed into the new output and offline mode is mandatory.

The review JSON contains these fields (placeholders deliberately cannot execute):

```json
{
  "approved": false,
  "all_intended_benchmarks_registered": false,
  "files": {
    "midtraining/production_corpus.py": "SHA256_AFTER_REVIEW",
    "midtraining/production_corpus.yaml": "SHA256_AFTER_REVIEW",
    "midtraining/prepare_corpus.py": "SHA256_AFTER_REVIEW",
    "midtraining/corpus.yaml": "SHA256_AFTER_REVIEW",
    "midtraining/longrun_native.py": "SHA256_AFTER_REVIEW",
    "results/pretraining/nemochat_baby/climbmix/corpus_sdk.py": "SHA256_AFTER_REVIEW"
  },
  "benchmark_manifests": [{
    "path": "results/pretraining/nemochat_baby/sft/data/raw/kmmlu-reserved-questions.json",
    "sha256": "61a804e227bd8aedbb4f1818c2559391f122627398e6f80085cd58c2525b5f0b",
    "count": 34871
  }],
  "metadata": {
    "path": "results/midtraining/super35-kd-01/corpus/hub-metadata.json",
    "sha256": "ACTUAL_LOCAL_OFFICIAL_METADATA_SHA256"
  },
  "raw_byte_cap": 6032365862,
  "raw_files": [{
    "key": "code_concepts",
    "path": "Nemotron-Pretraining-Code-Concepts/part_000060.parquet",
    "local_path": "datasets/midtraining/specialized-v1.1-13fa979/raw/Nemotron-Pretraining-Code-Concepts/part_000060.parquet",
    "size": 477960523,
    "sha256": "b3e69c8bb02be54d12e6773a8fde5594af30cd357e539b93326f02cde2ba6559"
  }],
  "calibration": {
    "path": "ACTUAL_CALIBRATION/controller/accepted.json",
    "sha256": "ACTUAL_ACCEPTED_RECEIPT_SHA256"
  }
}
```

The example raw list is intentionally incomplete. Supply **all three original
seed shards**, plus only explicitly approved extra shards. Every entry must match
`hub-metadata.json`'s repository/revision, `siblings[].rfilename`, `size` and
`lfs.sha256`; actual local length/SHA256 are checked before scanning. Only the
three allowed source directories and `part_XXXXXX.parquet` names are eligible;
no glob discovery, auto-expansion, missing-file download or duplicate paths.
All three source keys must be present. Repository/revision are inherited unchanged
from `corpus.yaml`: `nvidia/Nemotron-Pretraining-Specialized-v1.1` at
`13fa979be2e7f7e62913eee0ec5e97c8fd6e24af`.

The source map is exact: six dependencies, no missing/extra entries. Existing
original benchmark/tokenizer/ClimbMix lineage review is reused, including the
full34,871-question original KMMLU manifest and84,990 original ClimbMix raw
heldout hashes. The benchmark scope is the original registered inventory, **not
an all-benchmark or semantic decontamination claim**. Literal normalized substring
matching and the original Code-Concepts/Formal-Logic identity policies remain
unchanged. Expanding the benchmark inventory requires separate parent work;
this producer does not modify the frozen original policy.

The required top-level review fields are `approved`,
`all_intended_benchmarks_registered`, `files`, `benchmark_manifests`, `metadata`,
`raw_byte_cap`, `raw_files` and `calibration`, with nested fields exactly as shown
above. Only `files` is an exact-key allowlist; the top-level object does not reject
extra historical fields. `seed_catalog_sha256` is retired: it is not required,
read or verified, and supplying an old value confers no evidence. Refresh the
producer digest after this change before approval.

## Heldout preservation and actual data path

1. Verify seed manifest SHA and its `prepared_verified` success companion. Verify
   the entire old reservation file SHA/count, not merely the selected406k
   validation window. The old `catalog.sqlite` is a non-authoritative cache;
   it is never opened, hashed, repaired or copied by the production producer.
2. Before adding any new raw rows, install **every seed reservation content hash**
   and all original ClimbMix heldout hashes in `Catalog.original_reserved_hashes`.
   Rescan the complete explicit allowlist, including all seed shards. Shared
   question components propagate old heldout exclusion to answer variants before
   finalizing splits. Benchmark matches also exclude their entire components.
3. Preserve every old reservation row in the new `heldout-reservations.jsonl`,
   appending all additional deterministic heldout rows. Look up **every** old
   reservation's `content_hash` in the freshly scanned new `Catalog`; recompute
   exact SHA256 of its UTF-8 text before token-reserving it, even if not selected
   as validation. Any missing row or mismatched hash is fatal before training
   selection; an answer variant cannot substitute for the original text. The
   proof is canonical raw size/SHA256 plus mandatory complete seed-shard rescan
   plus exact coverage of all prior reservation hashes, not old-cache readability.
   Reserve all newly heldout token documents before any training
   selection, and reserve the entire original native ClimbMix validation pool.
4. Copy the four frozen pilot validation `.bin/.idx` pairs to fresh output paths,
   preserving exact digests and native readback. This is **explicit frozen heldout
   pool reuse**, not training cycling. No new validation quota is demanded from
   components now excluded because they were previously heldout.
5. Select unique complete specialized training documents with the fixed quotas;
   then select native ClimbMix training documents, omitting its known clipped
   final item. Require exact pinned-tokenizer decode/encode roundtrip. Token
   documents matching any reserved or already selected token document are skipped
   before filling quotas, including cross-source duplicates. Recheck decoded
   replay against benchmark literals and prior raw heldout hashes.
6. Read back all native outputs, audit cross-source exact token duplicates and
   train/heldout overlap, EOS11 and vocabulary[0,131072). Freshly stream **all eight
   .bin** files for IDs18..29 and actual SHA/counts; any nonzero count blocks success.
   `.idx` lengths/digests and complete-document totals are checked too.

Tokenizer is the exact Base tokenizer/asset digests inherited from `corpus.yaml`,
loaded local-only with `trust_remote_code=False` and no downloaded Python.
Transformers must be5.12.1 in the already pinned CPU image; host upgrading is not
part of this task. The process records that it cannot attest its container image;
parent retains the exact Engine image/exit receipt and CPU/RAM/no-network limits.

## Execution and artifacts

After real review/acquisition and accepted calibration, preview the measured plan:

```bash
.venv/bin/python midtraining/production_corpus.py --preview \
  --calibration /ABS/CALIBRATION/controller/accepted.json
```

Then **inside the parent-approved pinned CPU container**, at the same absolute
repository mount and with no GPUs or network access:

```bash
python midtraining/production_corpus.py --execute \
  --reviewed-sources /ABS/PARENT-PRODUCTION-REVIEW.json \
  --calibration /ABS/CALIBRATION/controller/accepted.json \
  --output datasets/midtraining/super35-kd-01/production-24h-v1
```

The program never starts that container. Use another fresh output generation on
failure, reusing independently verified local raw files. Parent should enforce
CPU/RAM/cgroup/disk quotas externally. SIGALRM bounds blocking preparation work;
SIGTERM/SIGINT produce failure receipts when Python can handle the signal. Output
size and progress persistence are checked at most once per five seconds during
ordinary progress calls, with forced checks initially and before success,
**not an OS-level hard disk quota**: writes since the last observation (up to the
five-second checking interval plus any blocking batch/document) and forensic
receipts may remain after a storage-limit failure. An OS kill/OOM may leave only
progress, never an accepted success. Outputs include the full SQLite catalog and reservation file,
four train/four frozen validation native pairs, manifest and timestamped progress /
failure / success receipts. No partial prefix implies readiness.

`manifest.json` uses the existing native schema1, source order and weights. Its
companion `success.json` must have phase`prepared_verified` and the actual manifest
SHA at `result.manifest_sha256`; `longrun_native.load_prepared_manifest` performs
final readback. The manifest carries the measured production plan, native capacity
gate, fresh reserved-ID audit and explicit frozen validation reuse. It does not
claim native sample-index no-repeat construction, actual24h training or model
quality. Those remain the parent's native gates, with the expanded-manifest
calibration compatibility receipt described in [LONGRUN.md](LONGRUN.md).

Tests exercise real local handwritten Parquet, the actual preserved native SDK,
whole-document writing/readback, raw reconstruction with missing/malformed old
caches, missing prior rows and reconstructed SHA mismatch failing closed,
old reservation/variant propagation, additional
heldout preservation, frozen validation digest reuse, capacity and duplicate
policies. A reversible **fixture tokenizer** is explicitly not a model-tokenizer
or corpus-scale result. Full production execution, sufficient scarce-source
capacity and the pinned runtime remain parent-owned verification gates.
