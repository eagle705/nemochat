# Repository-root runtime layout

[한국어](../ko/RUNTIME_LAYOUT.md) · [README](../../README.md)

## Four runtime roots

All relative CLI/config paths are resolved from the **repository-root working directory**, not from the config file or document directory. Run examples from the root unless a tutorial explicitly uses an absolute root path.

| Root | Contents | Examples |
|---|---|---|
| `datasets/` | Downloaded input shards/corpora and prepared/tokenized data, including generated evaluation inputs | `datasets/climbmix/<run>/`, `datasets/niah/<run>/fixtures.jsonl` |
| `checkpoints/` | Downloaded pretrained models/tokenizers, training-generated raw resumable snapshots, model-only exports/WSM merges, packaged and restored bundles | `checkpoints/pretrained/<model>/`, `checkpoints/cpu/<run>/`, `checkpoints/exports/<run>/` |
| `results/` | Experiment reports, raw predictions, metrics, receipts, profiler traces, screenshots and plots | `results/training_perf/<run>/`, `results/niah/<run>/`, `results/observability/<run>/` |
| `logs/` | Text/stdout/stderr logs and local/offline W&B logs | `logs/<run>/train.log`, `logs/wandb/` |

These are local runtime artifacts, not versioned source. Keep checked-in input configurations in `configs/`, and authored test/source fixtures in their existing locations. Do not move those fixtures into runtime roots. Use unique run names and never overwrite existing artifacts, even failed runs.

A CLI's `--output` means its own artifact type, **not always a report**: `prepare_data.py --output` belongs under `datasets/`; `train_cpu.py --output`, `--checkpoint-dir` and `--resume` belong under `checkpoints/`; evaluation/benchmark/report outputs belong under `results/`. Downloaded checkpoint inputs and training-generated checkpoints share a root but must have distinct subdirectories and provenance. Raw optimizer/RNG-bearing checkpoints remain the resume source; model-only exports and WSM merged weights are not substitutes. A long-context schedule cursor is a scheduling artifact under `results/`, not proof of completed training.

Some tools produce inseparable bundles (for example a CPU checkpoint also contains history/config/inventory, and an experiment directory contains a resolved config). Keep those contracts intact; do not split or edit inventory-covered payloads merely to separate file extensions.

## Explicit paths, not new configuration fields

Representative commands (choose fresh names; setup and any download authorization are separate):

```bash
mkdir -p results/planning logs/my-run
.venv/bin/python scripts/estimate_corpus.py \
  --config configs/planning/corpus_local_smoke.json \
  --output results/planning/my-corpus-estimate.json
.venv/bin/python scripts/train_cpu.py \
  --texts tests/fixtures/texts.jsonl --steps 2 \
  --output checkpoints/cpu/my-step-2 > logs/my-run/train.log 2>&1
.venv/bin/python scripts/train_cpu.py \
  --texts tests/fixtures/texts.jsonl --steps 2 \
  --resume checkpoints/cpu/my-step-2 --output checkpoints/cpu/my-resumed
.venv/bin/python scripts/run_experiment.py dry-run \
  --config configs/experiments/dense_cpu.yaml \
  --output results/training_perf/my-dense-control
```

Create **parent directories** when the tool requires them, not a final output directory a no-overwrite tool must reserve itself. Shell redirection can truncate logs: use a fresh log filename/run directory too. Experiment YAML schema 1 has no output-path field: pass `--output` explicitly; adding `output_dir`, dataset paths or invented environment variables would be rejected. W&B uses the existing `NEMOCHAT_WANDB_DIR=logs/wandb` or supported CLI/config directory setting; it remains disabled unless explicitly enabled. See [experiment contract](AGENT_EXPERIMENT_CONTRACT.md), [checkpoint pipeline](HF_CHECKPOINT_PIPELINE.md), and [W&B stages](WANDB_STAGES.md).

If a tutorial runs from its own folder, use an explicit root variable and absolute artifact paths, for example:

```bash
# In tutorials/nemo_lens/; adapt the root if using a different checkout.
REPO_ROOT=$(git rev-parse --show-toplevel)
# Keep the tutorial's documented interpreter/command; pass its existing --output:
# <tutorial command> --output "$REPO_ROOT/results/observability/<unique-run>"
# Use "$REPO_ROOT/datasets/...", "$REPO_ROOT/checkpoints/..." and
# "$REPO_ROOT/logs/..." for other supported paths/redirections.
```

## Hugging Face cache is separate

This convention does **not** automatically relocate the Hugging Face Hub/Transformers/Xet cache, model IDs used by `from_pretrained`, or an external upstream checkout/virtual environment. Existing cache defaults remain in force unless the user explicitly configures supported upstream settings such as `HF_HOME`. Use an explicit checkpoint download destination or local model path where the relevant CLI supports it. Do not invent a cache flag or assume `checkpoints/` replaces the shared HF cache. Data-preparation temporary download/cache behavior likewise remains tool-specific.

## Historical evidence and migration compatibility

Historical append-only logs, timestamps, measured results and receipt paths remain unchanged. Old `runs/`, `outputs/`, `exports/`, `figures/`, `/tmp/...` and host-absolute paths in verification records describe the original execution, not recommended destinations for a new run. See the relevant verification report and the migration manifest produced by the repository's runtime migration for old → new local path mappings. Do not rewrite historical evidence to pretend it originally used this layout.

A local runtime migration must preserve payload bytes and inventory-relative paths. Remote Hub receipt `path`, immutable `revision`, hashes and stage/run namespaces are persistent handles, not local directory suggestions: continue using their exact recorded values. Migration does not rename remote namespaces, reissue verified receipts, create missing temporary artifacts, or prove checkpoint resume. Existing artifacts must not be overwritten or deleted to make a new path available.

## Applied migration and verified logging

On this host, the physical migration is recorded in `results/runtime-layout-migration.json`: 720 original files (72,960,623 bytes) retain their original SHA-256 hashes, and 147 moves have live relative compatibility links. Both preservation and logging changes passed independent readonly review. See `results/runtime-layout-verification.json` for real layout smoke evidence, including exact state equality after resuming a relocated WSD checkpoint.

Experiment runner stdout/stderr are in repository-root `logs/experiments/<unique-id>/`, referenced by `result.log_directory`. Lens benchmark defaults to root `logs/nemo_lens/<output-name>/`, or explicit `--log-output`; result and log directories must be fresh and disjoint. W&B remains disabled by default and uses cwd-relative `logs/wandb` when enabled without an override. This layout task did not stage, commit or push any changes.

## Canonical stage-owned source (2026-10-19)

`pretraining/`, `midtraining/` (KD/continued pretraining only), `sft/`, `rl/` own stage source/configs/entrypoints. `datasets/` and `checkpoints/` remain payload-only. See [stage-layout mapping](STAGE_LAYOUT.md) and its original-blob manifest. Prior run paths in append-only journals are historical, not current source entrypoints. No operational RL trainer is claimed; native payloads and evidence remain in their original locations.
