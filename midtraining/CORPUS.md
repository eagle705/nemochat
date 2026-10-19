# Bounded midtraining corpus preparation

한국어: [CORPUS.ko.md](CORPUS.ko.md). Recipe: [corpus.yaml](corpus.yaml).

## Contract and current boundary

This is a **10 million-token preparation pilot capacity**, not a long-run training budget. The production horizon is intentionally `null`. Preparation never cycles, silently substitutes sources, truncates documents, launches training, reads checkpoint weight payloads, enables W&B, or uploads anything. A prepared corpus is not evidence of a native optimizer update or model quality.

`--help`, the default invocation, and `--preview` are inert: they do not create directories, download data, import Transformers, or load a tokenizer. `--execute` requires a parent-reviewed source/benchmark receipt and Transformers **5.12.1** in the approved pinned CPU runtime. The host environment has Transformers 4.57.6; do not upgrade it for this task. Corpus-scale execution is parent-owned and remains pending; tests use explicitly handwritten fixtures.

## Frozen sources and experiment weights

Dataset: `nvidia/Nemotron-Pretraining-Specialized-v1.1`, revision **`13fa979be2e7f7e62913eee0ec5e97c8fd6e24af`**. Exact metadata evidence: `results/midtraining/super35-kd-01/corpus/hub-metadata.json`. License metadata is `cc-by-4.0`; the scanner additionally rejects rows whose `license` is not exactly that value. Retain NVIDIA/source attribution and the pinned recipe with derived artifacts; dataset metadata is not a legal clearance for all downstream uses.

| Source | Only allowed remote file | Exact bytes | Training token target | Blend weight |
|---|---|---:|---:|---:|
| Code-Concepts | `Nemotron-Pretraining-Code-Concepts/part_000060.parquet` | 477,960,523 | 6,000,000 | 60% |
| Unconditional-Algorithmic | `Nemotron-Pretraining-Unconditional-Algorithmic/part_000000.parquet` | 208,343,534 | 1,000,000 | 10% |
| Formal-Logic | `Nemotron-Pretraining-Formal-Logic/part_000000.parquet` | 73,074,239 | 1,000,000 | 10% |
| ClimbMix replay | Existing local native training prefix; no remote acquisition | 0 new remote bytes | 2,000,000 | 20% |

The three Parquet files total **759,378,296 bytes**. Their SHA256 pins, in table order:

```text
b3e69c8bb02be54d12e6773a8fde5594af30cd357e539b93326f02cde2ba6559
706bab9781f6541cb9f40b8af61242e45c6e3dabc0d2f9b3171f08571da37ccb
b8b641309dec27e836db6a2003d3045c6ed060c8d6a23e3a93100e1c8a3a450b
```

The **60/10/10/20 mixture is this experiment's design, not official NVIDIA sampling weights**. Weights mean fixed-length sequence/token exposure in the eventual native blend, not row proportions. Preparation selects complete documents until each token quota is crossed, explicitly reporting the overshoot. Counts include EOS. No document is silently cut to make an exact quota; documents above the 1,000,000-token safety limit are rejected rather than truncated. Validation has a separate 100,000-token target per source, with complete-document overshoot. Scarce train/validation capacity fails instead of cycling or borrowing another source's quota.

Only `text` is trained. Code-Concepts metadata is used for identity, not as an extra copy of `generated_solution`. Formal-Logic itself contains concatenated question/answer text, often MCQ-formatted: excluding the separate Multiple-Choice config does **not** mean that all MCQ formatting has been excluded.

**Not included:** Economics and the separately named Multiple-Choice subset, pending separate contamination/license review. Agentic, Instruction-Following and BBQ remain reserved for SFT and are not acquired or mixed here.

## Tokenizer, native storage and replay lineage

Tokenizer assets come from `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`, revision **`4ba52e8ffa79e9693936c1a819ac89f551ceeef6`**, already downloaded under `checkpoints/teachers/nemotron35-super-base-bf16-4ba52e8f`. The recipe pins SHA256 for `tokenizer.json` and `tokenizer_config.json`. Loading is local-only, `trust_remote_code=False`; EOS is **11**, token IDs must be in `[0, 131072)`. Every new text document uses `add_special_tokens=False`, `truncation=False`, followed by one EOS. This is full next-token LM input for native LM+KD, **not assistant-only SFT masking**.

Output is native MCore int32 `.bin`/`.idx`, one full document per indexed item. The preserved upstream reader/builder source is `results/pretraining/nemochat_baby/climbmix/corpus_sdk.py`, SHA256 **`add90e4c4ac95df509d89b7e246002daf291eab775038755a01d5e13d5dd9613`**. The task-local loader isolates only unavailable cloud/distributed imports and the CPU Dataset base, following the original corpus's existing approach; reader/builder methods are unchanged. No installed SDK edits or alternate training loop are introduced. Every published prefix is read back with that native reader; all token ranges/EOS, token/document totals, and output file SHA256 are checked.

ClimbMix reads existing prefixes:

- `datasets/climbmix/nemochat_baby/native-02/train/corpus`
- `datasets/climbmix/nemochat_baby/native-02/validation/corpus`

Lineage: `karpathy/climbmix-400b-shuffle@915333b4f8b8684f39aeaafea600fea6f43fb703`, upstream `nvidia/Nemotron-ClimbMix@5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`. BASE already consumed this source: the 2M replay tokens are **not claimed as newly acquired unique knowledge**. Original preparation status `datasets/climbmix/nemochat_baby/control/preparation-status-02.json` is pinned by SHA256 in YAML; source native file lengths are checked. Both original corpus tails were truncated by the earlier job. This preparation omits those final upstream items rather than claiming they are complete documents. It does not change the original files.

The existing validation split is never repartitioned. All its indexed token-document hashes are reserved even after the selected validation quota is reached. Replay selection decodes and re-encodes each candidate with the target tokenizer and requires exact token-ID roundtrip compatibility; there is no silent retokenization fallback. Training replay scans at most 20M tokens including the full existing validation scan and fails if the quota cannot be filled.

## Grouped heldout and benchmark policy

All rows in **all three acquired bounded shards** are scanned before quota selection. Exact UTF-8 content hashes deduplicate documents globally in this selected pool. Split identities are:

1. **Code-Concepts:** normalized `metadata.problem_prompt`, not row UUID, generated answer, function name alone or row number. Missing identity is a counted schema rejection.
2. **Formal-Logic:** parse each concatenated question's stem before `A.`/`A)` options and remove its `Answer:` line. Documents sharing any normalized question stem are unioned into one connected component. All answer variants and multi-question connections stay in the same split. Unknown formats are counted rejections, not a fabricated UUID grouping. Large connected components can prevent usable validation/train coverage; that must fail and be reviewed, not silently split apart.
3. **Unconditional-Algorithmic:** no source question identifier is present in the inspected metadata schema. Group only normalized full-text duplicates; this does **not** establish semantic problem/translation/paraphrase separation.

Normalization matches the existing SFT helper's NFC + whitespace removal + casefold policy. A component's stable ID is SHA256 of the fixed seed plus its lexicographically smallest question hash. Bucket `int(group_id, 16) % 100 == 0` reserves approximately 1% of **components**, not exactly 1% of tokens. Splits are deterministic for this complete, fixed bounded pool. Expanding the pool requires a new reviewed generation, because connected components may change.

`heldout-reservations.jsonl` persists the **entire** deterministic heldout reservation, including documents not materialized due to validation quota. `catalog.sqlite` retains scanned texts, dedup ownership, component IDs and splits for audit. The whole original ClimbMix reservation of **84,990** raw content hashes is verified by exact SHA/count and excluded from specialized training, propagating through shared problem components. It is never reduced to a current validation window.

Benchmark exclusion requires the complete existing **34,871-question KMMLU** manifest at `results/pretraining/nemochat_baby/sft/data/raw/kmmlu-reserved-questions.json`, pinned to revision `d61b3f19e552c576bf5960dd24289763edc36a88` and SHA256 **`61a804e227bd8aedbb4f1818c2559391f122627398e6f80085cd58c2525b5f0b`**. We reuse its question normalization, and check **literal normalized substring containment** against training text and Code-Concepts problem prompts, not just equality to an entire long document. A match excludes the whole connected component. Replay is checked on decoded text too. No benchmark answer is used to construct examples.

**No blanket decontamination proof:** this catches registered normalized literal question matches, not translations, semantic paraphrases, code-equivalent problems, answer-only fragments or unknown benchmarks. The parent must inventory *all intended evaluation benchmarks*, add any further `{"questions": [...]}` manifests to `policy.additional_benchmark_manifests`, and review their exact digests/counts before setting `all_intended_benchmarks_registered=true`. Merely having KMMLU locally does not establish that other CORE/evaluation tasks are covered. The executable rejects an incomplete/unapproved benchmark review. A final native-token document audit rejects training overlap with **all** token reservations and duplicate training token-documents across sources.

## Bounded acquisition and parent review

Acquisition permits only the three pinned HTTPS paths. Each file needs exact expected length and SHA256 before Parquet scanning. `.part` files survive interrupted transfers; resume requires an exact `206 Content-Range`. A server ignoring Range is rejected rather than appended or blindly replayed. At most three request attempts per source occur per execution. Existing complete cache files are reverified, not trusted by existence.

The shared acquisition cache has a single-writer lock and persistent transfer ledger. Payload bytes across retries/restarts are bounded to **1,000,000,000**; TLS/HTTP headers are not counted as dataset payload. Prior retained cache bytes are conservatively charged if needed. Invalid-digest partials are retained for review, not silently discarded and redownloaded. Preparation output must be fresh; failed/published prefixes are not resumed or overwritten. Restart preparation in a new output generation while reusing the verified acquisition cache.

The code enforces six hours wall time, 8,000,000,000 bytes combined output/cache storage, a 20M-token replay scan ceiling and repository `datasets/midtraining/` write scope. The parent must additionally enforce CPU/memory limits (suggested: 8 CPUs, 12 GiB RAM), no GPU device requests, and a pinned container image. The process records that image identity itself is **not** attested by its Python version check; retain the exact Engine image/exit receipt. No actual container/service is started by this module.

Runtime image:

```text
nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf
```

Mount the repository at the same absolute path `/raid/yoon/nemochat` to preserve the original receipt's prefix identities. Needs Python, Transformers 5.12.1, NumPy, PyArrow and PyYAML; executed versions are recorded. Do not install into the host or download Hub code.

The parent writes a review JSON **after source review**. It must contain exactly these reviewed `files` keys with real SHA256 values, `approved: true`, `all_intended_benchmarks_registered: true`, and one entry per declared benchmark manifest:

```json
{
  "approved": false,
  "all_intended_benchmarks_registered": false,
  "files": {
    "midtraining/prepare_corpus.py": "REPLACE_AFTER_REVIEW",
    "midtraining/corpus.yaml": "REPLACE_AFTER_REVIEW",
    "results/pretraining/nemochat_baby/climbmix/corpus_sdk.py": "REPLACE_AFTER_REVIEW"
  },
  "benchmark_manifests": [{
    "path": "results/pretraining/nemochat_baby/sft/data/raw/kmmlu-reserved-questions.json",
    "sha256": "61a804e227bd8aedbb4f1818c2559391f122627398e6f80085cd58c2525b5f0b",
    "count": 34871
  }]
}
```

This deliberately unapproved template is **not an authorization receipt**. Docs and tests are not used as runtime executable dependencies. Modifying code/YAML after review invalidates the source gate. Execution only consumes the canonical reviewed `midtraining/corpus.yaml`.

## Commands and receipts

From the repository root, host-safe:

```bash
.venv/bin/python midtraining/prepare_corpus.py --help
.venv/bin/python midtraining/prepare_corpus.py --preview
.venv/bin/python -m pytest -q tests/test_midtraining_corpus.py
```

Exact preparation command **inside the parent-approved CPU container**, with a fresh output:

```bash
python midtraining/prepare_corpus.py --execute \
  --reviewed-sources results/midtraining/super35-kd-01/corpus/reviewed-sources.json \
  --raw-directory datasets/midtraining/specialized-v1.1-13fa979/raw --local-only \
  --output datasets/midtraining/super35-kd-01/pilot-10m-v1
```

`progress.json` reports timestamp/PID, actual acquisition payload bytes, phase, completely scanned files/rows, exclusions/dedup, active selected token/document counters and elapsed time. These are different denominators; downloaded bytes are not prepared-token progress. `failure.json` preserves a genuine failure; only a successful final audit writes `success.json`. Missing success means not ready, even when some `.bin/.idx` files exist. Outputs also include the full heldout reservation, SQLite catalog and `bridge-dataset-fragment.yaml`.

The shared parent acquisition is reused at `datasets/midtraining/specialized-v1.1-13fa979/raw`; `--raw-directory` aliases `--cache`. `--local-only` refuses missing files rather than issuing a duplicate request. All HF cache/home state is under `datasets/midtraining/specialized-v1.1-13fa979/.hf-home`. The parent's completed seed receipt is `results/midtraining/super35-kd-01/corpus/seed-verified.json`; preparation still performs its necessary exact raw-file gates.

### Native consumer manifest contract (schema 1)

Default path: `datasets/midtraining/super35-kd-01/pilot-10m-v1/manifest.json`. Require the companion `success.json` with `phase == "prepared_verified"`, and verify the manifest SHA256 equals `success["result"]["manifest_sha256"]`. A manifest alone is not an accepted producer completion.

- `schema_version: 1`, `status: "prepared_verified"`, `sequence_length: 4096`, `cycling: false`, `production_training_budget: null`.
- `source_order`: `["code_concepts", "algorithmic", "formal_logic", "climbmix_replay"]`.
- `weights`: source-key → `0.6/0.1/0.1/0.2`; fixed-sequence token exposure, not raw-row weights.
- `sources[source_key]["train" or "validation"]`: `prefix` (absolute native prefix without extension), `tokens`, `documents`, `dtype: "int32"`, `reader_verified`, `bin_bytes`, `idx_bytes`, `bin_sha256`, `idx_sha256`, `target_tokens`, `overshoot_tokens`, `truncated_documents: 0`.
- `tokenizer`: recipe's repository/revision/local path/EOS/vocabulary/asset SHA256 object.
- `capacity_by_source`: source-key → conservative full-sequence capacity; `audit` contains actual train/validation token/document totals and zero-overlap/duplicate assertions.
- `heldout_reservations_sha256`, `companion_success_receipt`, `native_training_gate` preserve readiness linkage and pending native integration scope.

A 16-update calibration at global batch 128 and sequence length 4096 uses **8,388,608 processed positions** before any separately accounted evaluation. Twenty updates would request 10,485,760 positions and must not silently cycle this 10M preparation pilot. Calibration does not freeze a 24-hour corpus budget; use measured production-batch throughput to review a new, adequate source capacity.

The fragment uses the native Bridge `GPTDatasetConfig`/MCore blend representation, separate train/validation prefix lists, explicit weights, `seq_length=4096`, and `eod_mask_loss=False`; it is an integration fragment, not a new trainer or a runnable full training recipe. Exact pinned-image API construction and the native blend sample-index gate remain parent-owned. The receipt records conservative per-source full-sequence capacities `(prepared_tokens - 1) // 4096`; those are **capacity bounds, not an authorized update count**. Native datasets can create multiple epochs when requested too many samples. Before any training, the parent must freeze duration and sample requirements, verify the actual generated indices do not repeat/cycle this preparation pilot, and confirm full-token LM+KD behavior in a real native update/save/reload. No long-run horizon is inferred from 10M tokens.

## Verification limits

Strict RED→GREEN iterations cover inert CLI, fixed pins/caps, real local HTTP range resume/digest checks, full-shard fixture scanning, problem-component grouping, original reservation propagation, normalized benchmark containment, complete-document native writing/readback, quota exhaustion, unchanged replay splits/token roundtrip, full heldout reservation beyond selected quota, progress counters and final overlap audit. Parquet/native integration tests use handwritten content and a clearly labeled fixture tokenizer; they do not stand in for real dataset or GPU evidence.

Remaining gates: parent source/benchmark/license review; exact CPU image and tokenizer execution; bounded 759 MB real acquisition and full scanning/tokenization; sufficient grouped heldout capacity (especially Formal-Logic); actual final corpus receipts; native Bridge blend/config/index validation; then independently authorized native LM+KD training, save and reload. No GPU job, long download, online tracking, upload, commit, installed SDK edit or original-corpus rewrite was performed for this implementation.
