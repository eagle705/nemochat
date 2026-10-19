# Future work: teacher-assisted pretraining with Megatron Bridge

**Historical proposal below. Superseded by the user's 2026-10-17 execution decision:** use the completed post-decay BASE for a new LM+KD midtraining stage with the exact Nemotron 3.5 Super VL **Base** teacher. Current contract, source, native YAML, isolated patches and execution evidence: [`midtraining/`](../../midtraining/README.md). The old non-authorization wording records the earlier documentation-only request; it does not revoke the later explicit download/pilot authorization. Completed BASE remains LM-only.

**Original proposal status: planned, not implemented or launched for NemoChat.** Adding this item does not authorize another training run, new downloads, or changes to the active SFT pipeline. The completed 10B-token BASE pretraining used LM Loss without knowledge distillation.

## Proposed experiment

- Connect the NemoChat 10BA1B hybrid student and an explicitly selected, frozen teacher through Bridge's `convert_to_distillation_provider()` and native `distill()` entry point.
- Start with logits Knowledge Distillation (KD) plus next-token LM Loss. Explicitly set `skip_lm_loss=false`; the documented default skips LM Loss. Freeze the exact weighting and temperature after inspecting the installed Model Optimizer implementation.
- Initially disable intermediate-layer distillation: NemoChat hidden width is 1024 while the proposed Super teacher has width 4096. Do not assume direct representation matching is compatible; projection-based matching would be a separate experiment.
- Decide explicitly whether this is a new from-scratch run or continued pretraining from a saved BASE checkpoint. Do not relabel the existing LM-only run as distilled.

## Required gates and comparison

1. Pin teacher/checkpoint/tokenizer identities and confirm vocabulary size and token-ID semantics, including the separate BASE/post-training tokenizer aliases.
2. Verify HybridModelProvider integration, loss masks and per-token normalization, compatible teacher/student TP/EP/CP groups, teacher freezing and student-only saves in the actual installed runtime.
3. Execute a bounded real GPU update/save/reload pilot. Measure finite Loss/Grad Norm, peak memory, throughput and additional teacher-forward cost. Provider acceptance alone is not operational support.
4. Compare LM-only and LM+KD under matched initialization, data, token budget and evaluation protocol; report wall-clock/compute cost separately. Use heldout CE/PPL and matched downstream benchmarks, not teacher agreement alone.
5. Keep current SFT, original checkpoints and existing waiver policy unchanged. A production KD run requires a separately approved experiment contract and budget.

## Capability evidence, not execution evidence

Read-only inspection on 2026-10-17 found Megatron Bridge 0.6.1 and NVIDIA Model Optimizer 0.46.0rc1 in the current preparation container. Installed `models/distillation_provider.py` accepts GPTModelProvider and HybridModelProvider; `training/distill.py` calls `pretrain(config, forward_step_modelopt)` and marks the API experimental. NemoChat-specific distillation has not been exercised.

Official reference: https://docs.nvidia.com/nemo/megatron-bridge/latest/modelopt/distillation.html

[한국어](../ko/PRETRAIN_DISTILLATION_FUTURE_WORK.md)
