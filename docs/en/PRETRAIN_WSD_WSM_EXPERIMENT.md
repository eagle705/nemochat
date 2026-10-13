# Pretraining: WSD training and stable-phase WSM evaluation

## Added performance engineering scope

Follow [the Nsight Systems / optional Triton lab](PRETRAIN_PERFORMANCE_LAB.md). Preserve WSD/WSM and reference defaults; validate forward/backward, native update throughput/memory and resume before adoption. GPU implementation and measurements remain pending.

Status: implementation in progress; no NemoChat GPU training or benchmark improvement measured.

## User-selected method

Use the official Megatron-LM PR 5114 weighted checkpoint merge, not a decay-branch average. Keep the WSD training trajectory intact. At evaluation milestones inside the stable phase, compare the raw checkpoint against an offline weighted merge of recent checkpoints ending at that milestone. The merged output does not replace the optimizer-bearing training checkpoint.

Sources inspected:
- Megatron-LM PR 5114: https://github.com/NVIDIA/Megatron-LM/pull/5114 ; inspected PR head `ca4826e9555032b8710dfb4a4e5693c46c1613dd`. The GitHub API reports this PR merged.
- Bridge scheduler documentation: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/8e2fdb88de2e9d842f28d00120df1af30acffa8a/docs/training/optimizer-scheduler.md
- Bridge configuration: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/8e2fdb88de2e9d842f28d00120df1af30acffa8a/src/megatron/bridge/training/config.py
- NeMo Gym benchmarks: https://github.com/NVIDIA-NeMo/Gym/tree/804c12a7d56a151a98ae150a352350e3c856ba47/benchmarks

## Production scheduler contract (source verified, not GPU exercised)

Bridge exposes `lr_decay_style="WSD"`, `lr_warmup_iters`, `lr_decay_iters`, `lr_wsd_decay_iters`, and `lr_wsd_decay_style`. Define the full schedule horizon explicitly and keep it unchanged across rental sessions and resumes. Set the final annealing duration explicitly: the inspected Bridge resolution defaults an unspecified WSD annealing duration to the entire decay horizon, which is not the desired long stable plateau. Do not mix sample-based and iteration-based fields. Base and minimum learning rate belong to the optimizer configuration; confirm exact field placement in the pinned execution environment before launch.

The final decay remains a training-policy decision. Stable-phase WSM evaluation does not require a dedicated decay run at every milestone and is not itself an LR scheduler.

## Paired evaluation contract

At each eligible stable milestone T:
1. Preserve the raw checkpoint and its training state.
2. Select only checkpoints from the same run/model/layout in a historical window ending at T. Never use future checkpoints.
3. Run the pinned upstream weighted merge with `minus-sqrt`; optionally run its `linear` style as an explicitly labeled ablation.
4. Retain the resolved input iterations, actual consumed-token counters, weights, source revision, dtype, extra-state source and output provenance.
5. Independently load/validate merged weights before model evaluation. A successful save or upstream dry-run alone does not establish downstream model compatibility.
6. Evaluate raw and merged models with identical benchmark revision, datasets, prompts, decoding limits, sampling parameters and scorer configuration. Report each benchmark separately and paired deltas, including regressions. Preserve predictions and failures.
7. Keep held-out CE/BPB as separate diagnostic measures. Do not equate a CPU fixture, tiny HF control or import test with a NemoChat capability benchmark.

The upstream merge tool selects iteration ranges, not token windows. With fixed batch size and sequence length, token-window conversion can be derived from verified counters; variable lengths, packing, skips or long-context schedules require actual recorded counters rather than a nominal multiplication. Choose window length and benchmark cadence after throughput and storage are measured, not by copying Nemotron's absolute token budgets.

## Checkpoint and resume boundaries

Official merging targets compatible `torch_dist` layouts, accumulates floating model tensors in fp32 on CPU, and copies rather than averages Transformer Engine extra state. Reject incompatible keys/shapes/layouts/dtypes. Model-only merged outputs do not contain a correctly merged optimizer or RNG trajectory. Continue the original training run from raw checkpoints. Loading a merged model for a separate fresh training branch requires explicit optimizer/RNG reset and separate provenance.

Treat every merge input as trusted: metadata-driven execution does not make checkpoint deserialization safe for arbitrary files. Pin the implementation and use a new non-overwriting output path.

## Gym scope

Use actual official `benchmarks/` configs and scorers. Benchmark existence is not evidence of base-model suitability: inspect chat-template, instruction-following, tool-use and judge dependencies. Never silently relabel a custom base-model prompt protocol as the official benchmark protocol. Record any adaptation and report it separately. Do not spend judge API credits or download unbounded datasets without an explicit budget.

## Evidence boundary

CPU WSD control, synthetic DCP merge tests, and comparison/importer tests can validate mechanics. Actual NemoChat checkpoint export, hybrid serving support, Gym execution against the model, and raw-versus-merged score improvement require separate real execution. Upstream Nemotron gains are cited upstream results, not predictions or measurements for this project.
