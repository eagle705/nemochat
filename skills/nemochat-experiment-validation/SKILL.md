---
name: nemochat-experiment-validation
description: Validate bounded experiments and performance claims.
version: 0.1.0
author: Joosung Yoon, Hermes Agent
platforms: [linux]
---

# NemoChat experiment validation

## When to Use
- Before running YAML experiments, comparing throughput/memory, or changing training performance code.
- Do not use CPU controls to approve native GPU, distributed or model-quality gates.

## Prerequisites
Run from the repository root. Read `AGENTS.md`, the current task and [experiment contract](../../docs/en/AGENT_EXPERIMENT_CONTRACT.md). Use the existing project Python environment. Check Git state before editing; preserve unrelated work. New installs, paid GPU runs and external actions need explicit authorization.

## How to Run
Execute these through the agent's terminal tool. Hermes calls it `terminal`; other agents use their equivalent. Replace the output name with a fresh run name.

```bash
.venv/bin/python scripts/run_experiment.py validate --config configs/experiments/dense_cpu.yaml
.venv/bin/python scripts/run_experiment.py dry-run --config configs/experiments/dense_cpu.yaml --output results/training_perf/skill-preview
.venv/bin/python scripts/run_experiment.py run --config configs/experiments/dense_cpu.yaml --output results/training_perf/skill-new
```

## Procedure
1. Inspect script `--help`, YAML and workload identity. Completion: schema validation succeeds and requested work fits explicit time/GPU budgets. Preview is not execution or hardware availability proof.
2. Record the hypothesis and baseline/candidate configs; hold data, seed, precision, optimizer, sequence and effective batch constant except the intended variable. Completion: confounders and expected evidence are listed.
3. Execute fresh baseline/candidate outputs under `results/`, never overwrite failed runs. Read `result.json`, `report.md`, `measurement/` and the result's `log_directory` under `logs/`. Completion: exit codes and actual artifacts agree; failures remain failures.
4. Compare loss/gradients and resume where relevant before performance claims. For GPU work: Nsight diagnosis, correctness, then unprofiled full-update throughput/peak-memory A/B; a fused kernel microbenchmark alone is insufficient. Completion: each claim has its own measurement scope and limitations.
5. Run focused tests and append exact commands/artifacts to `NemoChatAgent.md`, preserving history. Completion: all changed code/config/doc files accounted for. Commit/push only if separately authorized.

## Pitfalls
- YAML schema has no arbitrary command, output-dir or dataset override fields. Use supported CLI flags.
- Tokens prepared, processed and loss-bearing differ. CPU throughput is not a B300 projection.
- Profiled time and unprofiled throughput differ; concurrent sums and kernel sums are not wall time.
- Maintain WSD and official stable-window WSM policy; no silent algorithm substitution.

## Verification
Run `.venv/bin/python -m pytest -q tests/test_experiments.py tests/test_training_perf.py tests/test_runtime_layout.py` through the terminal tool. Inspect actual receipts rather than copying old counts. See [training performance guide](../../docs/en/PRETRAIN_PERFORMANCE_LAB.md) for native gates.
