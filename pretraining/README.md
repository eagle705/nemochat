# Pretraining

LM-only BASE stage. Run commands from the repository root.

- `entrypoints/`: original native preflight / launch / verification wrappers. These still require explicitly historical ignored producers in `results/pretrain_followup`; they are not a portable generic trainer.
- `configs/`: evaluation and WSM configuration.
- `publication/`: BASE export/publication and training curves, not SFT.
- Shared CPU controls remain `scripts/train_cpu.py`, `scripts/merge_checkpoints.py` and `src/nemochat/training/` because they are reusable mechanics rather than the native stage.

Future data belongs under `datasets/pretraining`, weights/state under `checkpoints/pretraining`, reports under `results/pretraining`. Existing runtime payloads are untouched. No new native/GPU qualification is implied by moving source.

See [migration and provenance](../docs/en/STAGE_LAYOUT.md).
