# Source organization and local release handoff

[한국어](../ko/SOURCE_HANDOFF.md)

## Current accepted milestone (2026-10-19)

The latest accepted native SFT checkpoint is **`iter_0001248`**, not step 1200
and not the earlier SFT905 snapshot. It resumed the native full state at 905
for another 343 updates: 159,744 conversations, 117,409,886 assistant targets,
and 2,617,245,696 processed positions **including padding**. These counters
are distinct; none is a claim of unique dataset tokens.

The already-published, **private**, model-only HF repository is
`eagle705/nemochat-10BA1B-BF16-SFT-1200`, immutable revision
`893c00ca610f25bde4c6b46050a224ecc9388385`. The repository's requested name
contains 1200; its actual source is 1248. The saved publication receipt records
53 payload files, 19,601,967,508 bytes, 12 weight shards and 41 small assets.
This source curation does not republish or modify that repository.

The frozen 400-question diagnostic (200 MMLU, 100 ARC-Challenge, 100 HellaSwag)
reports SFT1248 accuracy **22.5% / 35% / 48%**, heldout NLL
**1.399376922740647**, PPL **4.052674050965295**. It is not the full benchmarks
or an official unmodified harness execution. Six actual greedy generations all
hit their 128-token cap without EOS and exposed repetition, poor Korean and
invalid/incomplete JSON. Do not claim a generation-quality win. Native/HF logit
parity was **not run**; native payload SHA/CRC auditing was explicitly waived.

Read-only local evidence (excluded from Git):
- `results/midtraining/hf-sft1200-publication-01/publication_receipt.json`
- `results/midtraining/hf-sft1200-publication-01/parent-final-verification.json`
- `results/midtraining/sft1248-benchmark-gpu-01/summary.json`
- `results/maintenance/source-commit-20261019-01/` for this curation's manifests,
  redacted heuristic scan, exact commands, baseline/archive test outcomes and diff.

These are historical execution facts, not authorization to launch anything.

## Read source by responsibility

| Area | Entrypoints / contracts | Boundary |
|---|---|---|
| Dataset preparation | `scripts/posttraining/nemotron_sft.py`, `src/nemochat/data/nemotron_sft.py` | Author-selected assistant turns, full official rendering, reservations and no silent truncation |
| Settings | `configs/posttraining/nemotron_dataset_settings.yaml`, `src/nemochat/data/sft_dataset_settings.py` | Strict planner crosschecks **local frozen discovery/migration/reservation receipts**; a source-only checkout cannot assert operational readiness |
| Tokenizer transition | `scripts/posttraining/prepare_tokenizer.py`, `src/nemochat/tokenization/posttraining.py`, `configs/tokenizers/nemotron_super35.yaml` | Reserved-name-only IDs 18–29; no mapping, weights or scheduler changes |
| Native KD history | `midtraining/README.md`, `midtraining/offline2b/`, `midtraining/patches/offline_kd_pr/` | Frozen source variants and patched Bridge integration; not a generic portable trainer |
| Native SFT history | `midtraining/sft_after_kd/` | Runtime02–05, sealed-memory handoff and continuation/recovery variants retained byte-for-byte |
| Export/publication | `midtraining/publication/`, `src/nemochat/hub/base_checkpoint.py` | Model-only export is distinct from optimizer-bearing native resume; publication needs separate authorization |
| Dashboard | `dashboard/server.py`, `dashboard/README.md` | Read-only, loopback-only by default; captures/SDKs are optional external inputs |
| Pretraining wrappers | `scripts/pretraining/README.md` | Historical machine-bound entrypoints depend on ignored original runtime helpers |

The root README's original CPU implementation table describes the earlier tutorial
scope. This page supersedes it **only for the accepted execution milestones above**;
CPU scaffolding elsewhere must not be retroactively called GPU-validated.

## Versions, provenance and exclusions

The CPU tutorial extra remains **Transformers >=4.51,<5**. Accepted native/export
work used its separately pinned NeMo runtime with **Transformers 5.12.1**; do not
upgrade the host to merge those environments. Existing YAMLs pin dataset and teacher
revisions. The settings registry distinguishes three active SFT datasets, deferred
128K long-context data and unlabeled reserved RL seed data; 128K CP validation is
not complete. Tokenizer preparation requires the accepted local migration manifest,
not an arbitrary same-sized tokenizer.

No datasets, checkpoints, cached teacher outputs, credentials, raw runtime logs or
working receipts belong in this staged source tree. `LOGS.md` and all excluded
untracked originals remain local. The existing `uv.lock` metadata matches
`pyproject.toml`, but no uv workflow/lock intent was established, so it remains
unstaged; it is not the pinned GPU image lock.

Only minimal licensed upstream snapshots required by cached-KD CPU method checks
and patch reproduction are included. See
[`licenses/README.md`](../../midtraining/patches/offline_kd_pr/licenses/README.md).
The dashboard font must retain its OFL and provenance files. No blanket project,
dataset, tokenizer or weight license has been invented.

## Verification policy

Run the normal suite with `CUDA_VISIBLE_DEVICES='' WANDB_MODE=disabled
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest tests dashboard/tests -q`.
`tests/historical_contracts.json` separately identifies 88 exact run-bound test
functions (149 collected cases). A per-entry re-audit removed three portable
functions (seven cases): the pre-manifest opt-in guard, fully mocked statx
alignment failures, and the committed CPU-gate source existence check. They run
by default. Tokenizer loading now requires all four preparation-receipt assets
(`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `README.md`)
and verifies their bytes/hashes before importing Transformers; vocabulary and
EOS checks alone are not a compatible-migration proof.
Registered historical functions require explicit `NEMOCHAT_HISTORICAL_TESTS=1`; their original assertions and
frozen source guards remain unchanged. A skip is **not** a historical integration
pass. Missing source-only runtime receipts are not replaced by copied private data
or fabricated acceptance fixtures. See the maintenance report for actual counts,
known historical failures, the clean-HEAD baseline and filesystem/dependency limits.

Before local commits, independent review must inspect the staged allowlist/diff,
licenses, redacted scan findings and test report. The parent owns local commits;
no push, PR, public release, training, dependency upgrade or new download is part
of this handoff.
