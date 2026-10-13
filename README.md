# NemoChat

Run CLI examples from the repository root. Use `datasets/` for downloaded/prepared data, `checkpoints/` for downloaded/exported/resumable checkpoints, `results/` for reports/traces/plots, and `logs/` for text/W&B logs. See [runtime layout and historical migration compatibility](docs/en/RUNTIME_LAYOUT.md) ([한국어](docs/ko/RUNTIME_LAYOUT.md)). Create required parent directories, choose fresh output paths, and keep checked-in fixtures/configs in place.

A **noncommercial educational training-systems reference lab**: understand GPU bottlenecks and run reproducible, agent-assisted experiments. The long-term NVIDIA SDK path remains a small Nemotron-inspired hybrid Latent MoE, followed by SFT/RLVR and Super-teacher distillation.

The focus is not another agent application or a generic training framework. It is **prediction → measurement → diagnosis → change → correctness → full-update comparison**, with machine-readable experiment contracts and explicit evidence boundaries.

## Current implementation boundary

This repository currently implements readable **Part I CPU-first building blocks**, not the full seven-stage GPU training system.

| Component | Implemented / exercised | Not established yet |
|---|---|---|
| ClimbMix acquisition | Pinned selected shards, bounded download, token quota, exact selected-pool dedup, provenance, uint32 flat pack | Curator GPU pipeline, near dedup, Megatron indexed format conversion |
| Base evaluation | Real single-process HF CE/PPL/BPB and nanochat-style task scoring, pinned CORE bundle | MCore vocab/TP/CP/EP adapter and exported hybrid parity |
| Long context | Tokenizer-measured NIAH inputs, real completion calls, scoring/PNG/SVG; resumable long/short schedule | 64K training, Bridge alternating dataloaders, Mamba packed reset and CP correctness |
| Model | Explicit A/B dimensions, analytical census, direct Bridge provider code | Provider instantiation in pinned GPU SDK, actual numel, forward/backward |
| CPU control / WSD | Tiny HF optimizer update, constant/WSD LR, periodic snapshots, exact optimizer/RNG/data-cursor resume | Bridge scheduler parity, NemoChat training or evidence of final model quality |
| Training systems lab | Strict YAML runner, offline dense full-update control, separate profiler capture, compatible A/B and linear forward/backward estimates | CUDA/Nsight execution, measured DRAM roofline, native hybrid/distributed integration and default-adoption gates |
| WSM checkpoint merge | Pinned official Megatron PR #5114 launcher; actual CPU synthetic DCP merge and independent value checks | Actual NemoChat hybrid checkpoint reload/export or benchmark improvement |
| NeMo Gym results | Pinned official-result importer and raw/merged comparison with compatibility checks | Official benchmark execution against a NemoChat model; no measured target-model scores |
| Stages [5][6][7] | Documented design and artifact contracts | SFT/RLVR/MOPD implementation |

There is intentionally no generic trainer/plugin/backend registry. Start at a script, then read its small module. English code/docstrings; separate [Korean walkthrough](docs/ko/CPU_IMPLEMENTATION.md).

## Operations extensions (reviewed CPU/offline scope)

- [Large-corpus sampling and measured budget planning](docs/en/CORPUS_AND_BUDGET.md): bounded selected-population reservoir samples with the pinned target tokenizer; no claim that hand-selected shards represent the entire corpus.
- [HF checkpoint packaging and transfer](docs/en/HF_CHECKPOINT_PIPELINE.md): model-only exports vs trusted native training bundles, immutable revisions and inventory verification. Local packaging/restore and mocked remote tests exercised; real Hub round trip remains pending.
- [Operations and stage boundaries](docs/en/OPERATIONS_PIPELINE.md): stage-specific counters, checkpoint persistence and W&B lineage. Real offline tiny-HF tracking exercised; online and distributed/GPU stage execution remain unverified.

Optional dependencies: `.venv/bin/python -m pip install -e '.[hub,tracking]'`.

```bash
mkdir -p results/planning
.venv/bin/python scripts/estimate_corpus.py --config configs/planning/corpus_local_smoke.json --output results/planning/my-corpus-estimate.json
.venv/bin/python scripts/plan_training.py --config configs/planning/training_measured_example.json --output results/planning/my-budget-plan.json
.venv/bin/python scripts/hf_checkpoint.py --help
```

These examples use a three-document handwritten corpus fixture and illustrative user-supplied timing inputs, **not a corpus-wide survey or B300 benchmark**. Use fresh output paths. W&B is disabled unless explicitly enabled with `--wandb-mode offline` or authorized online configuration on supported scripts.

## Agent-assisted training systems lab

Start with one readable, offline **dense Transformer control**, not the unimplemented native hybrid path. YAML fixes the workload and measurement protocol; the runner rejects unknown fields and arbitrary commands, reserves fresh artifacts and enforces a subprocess time budget.

```bash
.venv/bin/python scripts/run_experiment.py validate --config configs/experiments/dense_cpu.yaml
.venv/bin/python scripts/run_experiment.py dry-run --config configs/experiments/dense_cpu.yaml
.venv/bin/python scripts/run_experiment.py run --config configs/experiments/dense_cpu.yaml --output results/training_perf/my-dense-control
.venv/bin/python scripts/run_experiment.py run --config configs/experiments/dense_cpu_checkpointing.yaml --output results/training_perf/my-dense-checkpointing
.venv/bin/python scripts/compare_training_runs.py --baseline results/training_perf/my-dense-control/measurement/result.json --candidate results/training_perf/my-dense-checkpointing/measurement/result.json --output results/training_perf/my-dense-comparison.json
.venv/bin/python scripts/plan_training_operator.py --config configs/performance/linear_unmeasured.json --output results/planning/my-linear-estimate.json
```

Use a fresh output directory. These commands run actual offline optimizer updates or analytical estimates; generated token replay is **not corpus training or model-quality evidence**. CPU throughput is not a GPU projection. CUDA configurations require a working GPU and separate authorization for rented resources.

- [Agent YAML contract](docs/en/AGENT_EXPERIMENT_CONTRACT.md) ([Korean](docs/ko/AGENT_EXPERIMENT_CONTRACT.md))
- [Training benchmark and A/B tools](docs/en/TRAINING_PERFORMANCE_TOOLS.md) ([Korean](docs/ko/TRAINING_PERFORMANCE_TOOLS.md))
- [Training systems learning sequence and linear roofline inputs](docs/en/TRAINING_SYSTEMS_LAB.md) ([Korean](docs/ko/TRAINING_SYSTEMS_LAB.md))

## NeMo Lens tutorial and modeler-facing Nsight Bridge

Two independent opt-in lab extensions live in dedicated folders:

- [NeMo Lens tutorial](tutorials/nemo_lens/README.md) ([Korean](tutorials/nemo_lens/README.ko.md)): real SDK execution, local trace UI startup/navigation and repeated overhead comparisons. Do not presume enabled or disabled instrumentation is free; CPU measurements are not native MoE GPU overhead.
- [NeMo Lens–Nsight Bridge](plugins/nemo_lens_nsight_bridge/README.md) ([Korean](plugins/nemo_lens_nsight_bridge/README.ko.md)): MoE operation meaning and verified Python source snippets, with opt-in Lens/NVTX instrumentation and a Nsight configuration-plugin path. No disconnected web viewer, raw kernel renaming or name-only bottleneck diagnosis. Native plugin loading, CUDA event/source correlation and custom GUI panels require separate Nsight/GPU validation.

The Lens [rollout/tool bottleneck lab](tutorials/nemo_lens/rollout/README.md) ([Korean](tutorials/nemo_lens/rollout/README.ko.md)) covers generation/admission, tool client/server, verification and concurrent tail analysis. [Native RL/Gym source map](docs/en/RL_ROLLOUT_OBSERVABILITY.md) ([Korean](docs/ko/RL_ROLLOUT_OBSERVABILITY.md)) separates the local educational control from actual distributed RL/vLLM coverage.

Read [scope and acceptance gates](docs/en/OBSERVABILITY_TRACKS.md) ([Korean](docs/ko/OBSERVABILITY_TRACKS.md)) and the folders' actual verification reports. The [whole-repository review and commit gates](docs/en/WHOLE_REPOSITORY_REVIEW.md) ([Korean](docs/ko/WHOLE_REPOSITORY_REVIEW.md)) records the current complete-source pass, repairs and final commit scope. Preserve default training, main dependencies, WSD/WSM and resume behavior. These educational controls are not the implemented native hybrid training system.

## Install on a CPU host

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[cpu,test]'
.venv/bin/python -m pytest -q
```

Bridge/Core/Transformer Engine are **not** installed in this CPU environment. Keep their future GPU container separate. The implementation verification report records the versions actually tested; dependency ranges above are not a complete GPU environment lock.

## Read in this order

1. `scripts/model_config.py` -> `models/dimensions.py`: why A/B are about 13.8B total / 2B active.
2. `scripts/prepare_data.py` -> `data/climbmix.py`: source validation -> selected shard -> tokenize -> quota -> atomic flat-pack output.
3. `eval/metrics.py` -> `eval/token_bytes.py` -> `eval/hf_eval.py`: CE/BPB and exact ByteLevel raw bytes.
4. `scripts/base_eval.py` -> `eval/core.py`: continuation likelihood, teacher-forced token checks and baseline centering.
5. `scripts/niah.py` -> `long_context/niah.py` and `plot.py`: frozen inputs -> actual predictions -> scores -> figures.
6. `scripts/lc_schedule.py`: long-only then long/4K alternation, explicitly scheduling only.
7. `scripts/train_cpu.py`: concrete optimizer/save/resume control, not a GPU training substitute.
8. `models/bridge_provider.py`: explicit pinned-API configuration to validate when the GPU environment exists.

All package paths above are relative to `src/nemochat/`.

## Run actual CPU inference

The smoke model is intentionally tiny and poor. Its scores below do not measure NemoChat capability. The input JSONL is a hand-written integration fixture, not a benchmark.

```bash
.venv/bin/python scripts/model_config.py --variant A
.venv/bin/python scripts/base_eval.py \
  --model sshleifer/tiny-gpt2 \
  --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be \
  --texts tests/fixtures/texts.jsonl \
  --core --core-tasks copa --max-per-task 3 \
  --context-length 128 --output results/eval/my-base-smoke
```

The result contains raw task likelihoods, counts and explicit **partial task subset** scope. Do not present its COPA-only score as full CORE. BPB uses document-local BOS-prefixed chunks; keep the protocol fixed and do not compare it numerically to nanochat's packed-stream BPB without matching context/data semantics.

## Exercise tiny optimizer save/resume

```bash
.venv/bin/python scripts/train_cpu.py \
  --texts tests/fixtures/texts.jsonl --steps 2 --output checkpoints/cpu/my-step-2
.venv/bin/python scripts/train_cpu.py \
  --texts tests/fixtures/texts.jsonl --steps 2 \
  --resume checkpoints/cpu/my-step-2 --output checkpoints/cpu/my-resumed
```

This writes model **safetensors** plus optimizer/RNG tensors loaded with `weights_only=True`, a data/config contract, and a versioned SHA256 inventory of all checkpoint artifacts. Resume checks the inventory **before loading** model/tokenizer/optimizer state. It detects accidental changes, not malicious manifest rewriting. Historical checkpoints without an inventory require explicit `--allow-unverified-legacy-resume` and remain unverified. Reusing the short fixture is intentional for mechanics testing, not claims about diverse pretraining data. Never overwrite run directories.

## Prepare a bounded real ClimbMix subset

```bash
.venv/bin/python scripts/prepare_data.py \
  --config configs/data/climbmix_smoke.json --output datasets/climbmix/my-smoke
```

Review the explicit noncommercial rights acknowledgment in the config before running. This downloads **one selected shard**, capped at 100MB for this smoke, and keeps only a 256-token flat pack; temporary shard/cache is removed. Network bytes can be much larger than selected training bytes. GPT-2 is smoke-only; training uses pinned Super. The flat `.u32` pack is **not** an MCore `.bin/.idx` dataset. No training consumes tokens in this preparation command.

## NIAH with a real completion server

First start this loopback-only CPU control in a separate terminal:

```bash
.venv/bin/python scripts/serve_cpu.py --port 8766
```

Then verify `/health` and run:

```bash
curl --fail http://127.0.0.1:8766/health
.venv/bin/python scripts/niah.py generate \
  --tokenizer sshleifer/tiny-gpt2 --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be \
  --context-lengths 128 256 --depths 0 .5 1 --samples 1 \
  --generation-tokens 32 --output datasets/niah/my-smoke/fixtures.jsonl
.venv/bin/python scripts/niah.py run \
  --tokenizer sshleifer/tiny-gpt2 --revision 5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be \
  --input datasets/niah/my-smoke/fixtures.jsonl --output results/niah/my-smoke/predictions.jsonl \
  --endpoint http://127.0.0.1:8766/v1/completions --model sshleifer/tiny-gpt2
.venv/bin/python scripts/niah.py score \
  --input results/niah/my-smoke/predictions.jsonl --output results/niah/my-smoke/cells.jsonl \
  --csv results/niah/my-smoke/cells.csv
.venv/bin/python scripts/niah.py plot \
  --input results/niah/my-smoke/predictions.jsonl --output-prefix results/niah/my-smoke/heatmap
```

Use a vLLM endpoint with its **matching tokenizer** later. Current protocol is plain completions, not automatic chat wrapping. Input generation is synthetic by design, but measured figure scores require actual model replies. Unverified token usage, errors and unsupported cells are separate from incorrect answers. One sample/cell is smoke only; the figure includes counts/Wilson intervals. The repeated generic haystack is a recall fixture, not realistic agentic coding evidence.

## Long/short schedule (no training)

```bash
.venv/bin/python scripts/lc_schedule.py \
  --long-only-steps 2 --alternating-steps 4 --effective-tokens 131072 \
  --cursor results/lc/my-smoke/cursor.json --output results/lc/my-smoke/steps.jsonl --count 6
```

The cursor describes emitted schedule steps, **not completed GPU optimizer updates**. Do not treat its advancement as a training checkpoint. Effective batch counts must be mapped to the actual MCore DP/CP/TP groups when integrating training.

## WSD training control and stable-phase WSM

The selected experiment keeps one WSD training trajectory and compares each stable milestone's raw checkpoint with a historical-window WSM merge. It does **not** launch decay branches at every evaluation point.

CPU scheduler smoke (choose new output paths on every run):

```bash
.venv/bin/python scripts/train_cpu.py \
  --texts tests/fixtures/texts.jsonl --context-length 32 \
  --lr-schedule wsd --warmup-steps 2 --stable-steps 4 --decay-steps 2 \
  --schedule-total-steps 8 --min-lr-ratio 0.1 --steps 8 \
  --checkpoint-every 2 --checkpoint-dir checkpoints/cpu/my-wsd-snapshots \
  --output checkpoints/cpu/my-wsd-final
```

The eight-step horizon remains fixed when resuming; `--steps` is additional work for this invocation. The CPU snapshots are **HF checkpoints**, not inputs to the Megatron DCP merger. See [WSD boundaries/resume](docs/en/WSD_IMPLEMENTATION.md).

`merge_checkpoints.py` invokes the official pinned Megatron implementation on compatible trusted `torch_dist` checkpoints. It requires a **separate Python 3.12 runtime** at the inspected upstream pin; do not install its dependency stack into the CPU tutorial environment. Follow [WSM setup and tested commands](docs/en/WSM_IMPLEMENTATION.md) ([Korean explanation](docs/ko/WSM_IMPLEMENTATION.md)). Actual official CPU fixture tests can be enabled with:

```bash
# These paths describe the tested host; adapt to your external checkout/environment.
NEMOCHAT_WSM_UPSTREAM=/home/ubuntu/runs/wsm-upstream/Megatron-LM \
  /home/ubuntu/runs/wsm-upstream/venv312/bin/python -m pytest tests/test_wsm.py -q
```

The default project test suite deliberately skips that opt-in integration when the external runtime is absent. Passing it proves merge mechanics, not hybrid model compatibility or a performance gain. Preserve optimizer-bearing raw checkpoints for training resume; merged weights are evaluation artifacts.

## Pretraining performance engineering (planned GPU work)

Pretraining now includes an explicit Nsight Systems → small Triton fusion → forward/backward correctness → full-update A/B lab. Preserve the WSD/WSM recipe and default execution path. The standalone dense control and YAML runner provide local executable measurement scaffolding. No Triton kernel, native hybrid GPU integration or measured GPU improvement is established.

See [execution and test gates](docs/en/PRETRAIN_PERFORMANCE_LAB.md) ([Korean](docs/ko/PRETRAIN_PERFORMANCE_LAB.md)).

For agent-assisted trace analysis, see [Nsight/Codex guide](docs/en/NSIGHT_CODEX_GUIDE.md) ([Korean](docs/ko/NSIGHT_CODEX_GUIDE.md)). Use CLI statistics and SQLite to find candidates, then validate their timeline context in the GUI.

## Independent inference performance track

A separate [inference performance track](docs/en/INFERENCE_PERFORMANCE_TRACK.md) ([Korean](docs/ko/INFERENCE_PERFORMANCE_TRACK.md)) complements training: representative workloads and SLOs → operator roofline → Nsight diagnosis → engine/kernel changes → service A/B → quality/regression → upstream-ready evidence. Start with a supported pretrained control, then verify the custom hybrid path separately. CPU saved-log/roofline tooling is scaffolding, not measured GPU serving. See [local tools](docs/en/INFERENCE_PERFORMANCE_TOOLS.md) ([Korean](docs/ko/INFERENCE_PERFORMANCE_TOOLS.md)).

## NeMo Gym benchmark results

`scripts/gym_eval.py` is currently a **saved official-result importer/comparator**, not a benchmark runner. It checks paired raw/merged protocols and imports existing official scores without calling a model or judge. `configs/eval/gym_pretrain.json` is an unmeasured manifest template, not a ready-to-run scored benchmark. The official Gym environment, real checkpoint export and serving endpoint are separate prerequisites.

See [Gym preparation, official commands and comparison contract](docs/en/GYM_PRETRAIN_EVALUATION.md) ([Korean explanation](docs/ko/GYM_PRETRAIN_EVALUATION.md)). Do not label synthetic importer fixtures as model results. CE/BPB and CORE remain separate diagnostics with their existing protocols.

## Documentation and publication

- [Project plan](docs/NEMOCHAT_PROJECT_PLAN.md)
- [English CPU walkthrough](docs/en/CPU_IMPLEMENTATION.md)
- [Detailed metrics and ClimbMix](docs/en/PRETRAIN_METRICS_AND_CLIMBMIX.md)
- [Data implementation](docs/en/DATA_IMPLEMENTATION.md)
- [Long-context implementation](docs/en/LONG_CONTEXT_IMPLEMENTATION.md)
- [Actual implementation verification](docs/IMPLEMENTATION_VERIFICATION.md)
- [Three-part execution/blog plan](docs/en/EXECUTION_AND_BLOG_SERIES_PLAN.md)

Blog packages remain **[1–4], [5–6], [7]**. Part I is not complete merely because CPU fixtures pass. Public uploads, Git commits and model releases are separate actions; none are implied by local code generation.

## Attribution and rights

Metric semantics are adapted from [karpathy/nanochat](https://github.com/karpathy/nanochat) (MIT); see `THIRD_PARTY_NOTICES.md`. Original ClimbMix is CC BY-NC 4.0; community MIT metadata does not erase upstream terms. Dataset and trained-weight redistribution require separate review. This repository has no blanket license declaration for all artifacts at this stage.
