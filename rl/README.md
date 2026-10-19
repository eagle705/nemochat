# Reinforcement learning / RLVR boundary

There is **no operational RL trainer in the versioned reusable source tree**. Native historical RL attempts/receipts under ignored results do not constitute an implemented portable trainer.

- `evaluation/gym_eval.py`: actual saved official-result importer/comparator; does not execute a model, judge, environment or optimizer.
- `configs/gym_pretrain.json`: unmeasured evaluation manifest template, not benchmark success.
- Shared Gym protocol library: `src/nemochat/evaluation/gym_benchmarks.py` (also usable by BASE/SFT evaluation).
- RL design/rollout contracts: [stage plan](../docs/NEMOCHAT_PROJECT_PLAN.md), [source observability](../docs/en/RL_ROLLOUT_OBSERVABILITY.md), [Gym protocol](../docs/en/GYM_PRETRAIN_EVALUATION.md). Educational rollout controls remain under `tutorials/nemo_lens/rollout/` and are not an RL trainer.

`.venv/bin/python rl/evaluation/gym_eval.py --help`

Future RL inputs/checkpoints/reports belong under `datasets/rl`, `checkpoints/rl`, `results/rl`. Reserved RL seeds must not be synthesized into labels or included in SFT without explicit approval. No training is started by this layout change.
