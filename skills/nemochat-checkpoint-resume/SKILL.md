---
name: nemochat-checkpoint-resume
description: Verify checkpoint integrity and exact resume state.
version: 0.1.0
author: Joosung Yoon, Hermes Agent
platforms: [linux]
---

# NemoChat checkpoint and resume verification

## When to Use
- Saving, restoring, migrating or validating a training checkpoint.
- Do not equate a model-only HF export or WSM merge with optimizer-bearing resume state.

## Prerequisites
Read `AGENTS.md`, [WSD implementation](../../docs/en/WSD_IMPLEMENTATION.md) and [checkpoint pipeline](../../docs/en/HF_CHECKPOINT_PIPELINE.md). Use existing project Python and trusted checkpoint inputs. CPU examples use an already-cached pinned tiny model; enable offline settings when no download is authorized. Dataset/tokenizer provenance and all resume hyperparameters must match.

## How to Run
Through the agent's terminal tool, inspect CLI help before selecting parameters:

```bash
.venv/bin/python scripts/train_cpu.py --help
.venv/bin/python scripts/verify_cpu_resume.py --help
```

For an actual split-versus-continuous CPU test, select three fresh checkpoint paths and run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/train_cpu.py --texts tests/fixtures/texts.jsonl --steps 2 --output checkpoints/skill-split2
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/train_cpu.py --texts tests/fixtures/texts.jsonl --steps 2 --resume checkpoints/skill-split2 --output checkpoints/skill-resumed4
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/train_cpu.py --texts tests/fixtures/texts.jsonl --steps 4 --output checkpoints/skill-continuous4
.venv/bin/python scripts/verify_cpu_resume.py checkpoints/skill-resumed4 checkpoints/skill-continuous4
```

## Procedure
1. Inspect manifest/inventory and origin without loading untrusted pickle. Completion: exact checkpoint kind, immutable model revision, data hash and schedule contract identified.
2. Validate inventory before state loading. Keep raw state and inventory-relative paths intact. Completion: every covered payload hash matches; no legacy-integrity bypass silently enabled.
3. Execute split/continuous controls with identical seed/data/model/optimizer settings. WSD horizon is global; invocation `--steps` is additional updates. Completion: restored model, optimizer/RNG, cursor and token counts are exactly compared, not merely final loss.
4. Put prepared data in `datasets/`, state bundles in `checkpoints/`, reports in `results/` and text logs in `logs/`. Do not separate inventory-covered files by extension. Completion: paths are fresh, originals preserved and migration mappings recorded if moving artifacts.
5. For remote transfer, obtain explicit destination/action authorization. Require rights review, private-by-default repository, immutable revision and actual read-back; never echo credentials. Completion: transport verification is separate from native optimizer resume evidence.
6. Append commands and outcomes to the ledger. No implicit commit/push/upload. Completion: failure, skipped checks and unsupported native/distributed cases explicitly reported.

## Pitfalls
- A hash inventory detects accidental changes, not a malicious rewritten manifest.
- WSM merged weights are for evaluation, not replacement of raw resumable optimizer state.
- CPU RNG equality does not establish CUDA/distributed equality.
- Runtime roots do not automatically relocate the shared HF cache.

## Verification
Run `.venv/bin/python -m pytest -q tests/test_resume_verify.py tests/test_wsd.py tests/test_hf_checkpoint.py` via the terminal tool. Exercise actual resume when changing state semantics; synthetic transport tests are not proof of remote transfer.
