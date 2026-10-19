# Fast unique-data recovery: 67/10/3/20

한국어: [RECOVERY_MIX67.ko.md](RECOVERY_MIX67.ko.md).

## Scope and acceptance boundary

`recover_corpus.py` is an inert-by-default, local-only **CPU corpus producer**.
It starts no container, service, GPU, network acquisition, remote tracker or
training. It does not open/hash student or teacher model-weight payloads; the
explicit checkpoint-byte waiver remains recorded. Tokenizer assets and dataset
bytes **are** verified. Run from `/raid/yoon/nemochat`.

```bash
.venv/bin/python midtraining/recover_corpus.py --help
.venv/bin/python midtraining/recover_corpus.py --preview
.venv/bin/python -m pytest -q tests/test_midtraining_recover_corpus.py
```

The new YAML fixes 3,200 updates, global batch128, sequence4096 and weights
Code67% / Algorithmic10% / Formal3% / ClimbMix replay20%. Native per-source targets
include `ceil(ceil(3200*128*w)*1.005)+2` samples and one shifted token:

| Source | Required tokens | Reused published tokens | Additional preparation |
|---|---:|---:|---|
| Code | 1,129,705,473 | 1,011,675,137 | at least118,030,336, complete-document overshoot allowed |
| Algorithmic | 168,620,033 | 168,620,068 | copy all published documents |
| Formal | 50,593,793 | 0 | encode from the valid new catalog |
| Replay | 337,231,873 | 0 | original native ClimbMix, with checked tokenizer roundtrip |

These are exact-token-document uniqueness claims **within this generation**, not
semantic deduplication or novelty relative to BASE. ClimbMix is explicitly
previously consumed replay. Scarce Code tail or any other short pool fails;
no cycling, truncation, substitution, implicit epochs or quota reduction.
The old24.295-second calibration is historical legacy-mix timing, not a new-mix
benchmark or sustained24h measurement. Its accepted SHA/horizon is preserved.

## Why the fast cursor is valid

The origin is the immutable **failed** Formal-quota run, not an accepted corpus:
`datasets/midtraining/super35-kd-01/production-24h-v1/data/failure.json`, SHA
`5bb7bc7c94d16cb67a2f9588db7fa9e6b7162135169dd3240a114a91dab599ac`.
Its complete13-file scan and grouping are inherited; recovery does not rescan raw
records. Official raw bytes/SHA and Parquet footer/schema/row totals are rechecked.

The new production catalog is9,301,573,632 bytes, SHA
`9d55e9cde8dfef7f61ad8d6b97271563f1245c4a34ce24574f36b6abb36e7eb1`.
It is opened `mode=ro&immutable=1`, digest/integrity/required-table checked,
never repaired or modified. Missing/corrupt databases or transaction sidecars
are fatal. **The malformed pilot SQLite is never read or hashed.**

All three historical skip counters must be exactly zero (absent means zero):
`oversized_documents`, `reserved_token_documents_skipped`, and
`duplicate_token_documents_skipped`. The reviewed frozen producer SHA is
`5bd5afc188402bd0908364b3c6d00731a48fa50bef5a27953d2c0813960adcb8`.
Its selection uses the first eligible TRAIN documents ordered by catalog id.
Therefore the2,111,817th eligible Code TRAIN row is the last reused row.
Seven deterministic first/last/interior samples are re-tokenized and compared
with actual native documents, including exact UTF-8 content-hash checks.
The remainder of the1B-token Code prefix is copied by document IDs, **not BPE**.
Every reused native document still receives fresh range/EOS/reserved-ID,
old-selected membership, heldout and cross-source duplicate checks.

Historical source lineage uses **all six frozen files** under
`results/midtraining/super35-kd-01/corpus/production-source-snapshot-01`,
including its old `longrun_native.py`. Today's modified live native module is
reviewed independently; it cannot stand in for frozen history.
Parent measurements are pinned by
`results/midtraining/super35-kd-01/corpus/recovery-origin-parts-verified-01.json`.
Its Code/Algo SHAs are explicitly current measurements, not fabricated old
producer digests. Committed old `selected_tokens` contains exactly2,229,682
Code+Algo documents; the failed Formal transaction rolled back.

## Exact version1 parent-review contract

Execution requires the following **exact top-level fields**, with only
`expected_code_cursor` optional. Every `PIN` is an object
`{"path":"repository-relative-or-absolute-local-path","sha256":"64 lowercase hex"}`.
`approved` must become true only after parent review. `files` maps exactly the
eight active paths below to measured SHA256 strings:

```text
midtraining/production_corpus.py
midtraining/production_corpus.yaml
midtraining/prepare_corpus.py
midtraining/corpus.yaml
midtraining/longrun_native.py
results/pretraining/nemochat_baby/climbmix/corpus_sdk.py
midtraining/recover_corpus.py
midtraining/recovery_mix67.yaml
```

```json
{
  "schema_version": 1,
  "approved": false,
  "files": {"EACH_OF_THE_EIGHT_PATHS_ABOVE": "ACTUAL_CURRENT_SHA256"},
  "origin_failure": "PIN",
  "origin_catalog": "PIN",
  "origin_review": "PIN",
  "origin_parts": "PIN",
  "origin_snapshot": {"EACH_OF_THE_FIRST_SIX_PATHS_ABOVE": "PIN_TO_ITS_FROZEN_COPY"},
  "full_reservations": "PIN",
  "raw_verification": "PIN",
  "calibration": "PIN"
}
```

The placeholders intentionally cannot execute. Required pin targets:

- `origin_failure`: the immutable failure path above.
- `origin_catalog`: its sibling `catalog.sqlite`, with the new catalog SHA above.
- `origin_review`: `results/midtraining/super35-kd-01/corpus/production-reviewed-sources-01.json`.
  Its parsed object must equal `failure.source_review` exactly.
- `origin_parts`: the parent's `recovery-origin-parts-verified-01.json` above.
- `origin_snapshot`: exactly six original dependency names, each pointing to its
  corresponding frozen copy; SHA must equal `origin_review.files[name]`.
- `full_reservations`: the origin failure's sibling
  `heldout-reservations.jsonl`, **all old+new rows**, not just the published window.
- `raw_verification`: `results/midtraining/super35-kd-01/corpus/production-raw-verified-01.json`.
  Its13-file identity/size/SHA list must match the old reviewed raw allowlist
  (6,032,365,862 bytes), which is revalidated against official pinned metadata.
- `calibration`: exactly `origin_review.calibration`:
  `results/midtraining/longrun/super35-newdata-calibration-01/controller/accepted.json`,
  SHA `96e2a098793e6ac1ac98ba7f671a1062d58d0fba799d775c38c4a2ed0a984250`.

Optional `expected_code_cursor` is the expected final **catalog id**, not a count
or token offset. Omit it to derive and sample-verify the cursor. A supplied wrong
value fails closed. Do not pin active files until concurrent writers have stopped;
active dependency digests are checked again before success publication.

The complete old seed manifest SHA remains
`52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308`.
Its entire heldout reservation SHA/count and inclusion in the full new reservation
file are checked, independently of the seed SQLite. Original benchmark inventory,
question-group policy,34,871-question KMMLU manifest,84,990 original ClimbMix raw
heldout hashes, local Base tokenizer assets and original replay receipt checks
are reused from `prepare_corpus.verify_review`, not weakened.

## Execution and output

Inside the **already approved production CPU runtime**, use exactly this command
with a fresh output (the program does not launch the runtime):

```bash
python midtraining/recover_corpus.py --execute \
  --reviewed-sources results/midtraining/super35-kd-01/corpus/RECOVERY-REVIEW.json \
  --output datasets/midtraining/super35-kd-01/recovery-67-v1/data
```

The production image identity is
`nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf`;
retain its existing local dependency overlay with Transformers5.12.1. Do not
upgrade the host or download packages. Parent owns the image/Engine exit receipt,
no-network/no-GPU container and64GiB CPU-memory cgroup limit. The process records
that it cannot independently attest those external controls.

Producer limits:12h SIGALRM watchdog,64,000,000,000 observed output bytes,
2B replay-scan tokens and1M tokens/document. Storage observations occur at most
every five seconds plus forced preflight/final checks, not an OS hard disk quota.
A blocking operation or OS OOM kill may leave partials/progress only. Failed
outputs remain immutable; there is no automatic resume or deletion.

Outputs: fresh native four-train/four-validation `.bin/.idx` pairs,
`selection.sqlite` (new selected set and copied full reserved set), byte-identical
full `heldout-reservations.jsonl`, `code-cursor-proof.json`, `manifest.json`,
`progress.json`, and either `success.json` or `failure.json`. The origin catalog
and origin partial Formal bytes are never changed. All validation bytes match
the frozen seed. New Code skips any old selected hash; Formal may use its exact
eligible incomplete old subset without treating itself as a duplicate. Every
new source shares the **new** selected set. Replay omits the upstream clipped tail
and receives exact roundtrip, literal benchmark and full-heldout checks.

Final audits freshly inspect all eight outputs: native document/token counts,
range/EOS, bin/idx bytes and SHA, IDs18..29 zero, full train/heldout overlap zero,
cross-source token-document duplicates zero, and native67/10/3/20 capacity.
Manifest schema1 carries `inherited_full_scan_provenance`,
`recovery_did_rescan_raw:false`, honest `reused_parts_provenance`, frozen validation
reuse and the exact full Base tokenizer. Acceptance additionally requires
`success.phase == prepared_verified` and its `result.manifest_sha256` matching the
actual manifest; the native consumer is read back before returning success.
Native sample-index no-repeat and GPU training remain separate parent gates.

## Verification performed

Strict RED→GREEN slices exercised cursor inference, corruption/skip rejection,
real native materialization, failure/exhaustion, exact review schema, watchdog
and provenance binding. Fixtures use the preserved real indexed SDK and an
explicit handwritten reversible tokenizer, not model-tokenizer or scale evidence.
The focused recovery/corpus/production/mix67/longrun/tokenizer suites passed
91 tests. Compilation and `git diff --check` passed. A separate read-only check
validated the **actual** frozen six-file history and parent part receipt against
the immutable failed origin. No full recovery, service or GPU was launched by
this implementation task; production capacity is pending real parent execution.
