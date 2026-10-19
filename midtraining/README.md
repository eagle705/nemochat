# Midtraining: KD and continued pretraining only

- `data/`: corpus preparation/recovery, offline-KD preparation/materialization and pinned teacher download.
- `configs/`: pilot/calibration/longrun/corpus YAML, including `offline_kd/` cache-corpus recipes.
- `cache/directio/`: optional native-cache direct-I/O gates, identity binding and transport integration.
- `training/`: native KD/longrun/pipeline and `offline_kd/` teacher-cache/native variants.
- `evaluation/`: offline/auto/GPU ablation comparisons.
- `publication/`: KD exports/publication only.
- `patches/`: scoped Bridge/Core compatibility seam and preserved upstream licenses/evidence.
- `docs/`: corpus, production, recovery and verification reports, including the `offline_kd/` guide.
- [Historical variants index](archive/README.md).

**SFT lives exclusively in [../sft](../sft/README.md).** No SFT alias/tree remains here. The historical guide below records its original pilot state, not current GPU qualification. See [stage migration](../docs/en/STAGE_LAYOUT.md).

---

# BASE → LM + Knowledge Distillation → SFT

## Selected starting point — user decision, 2026-10-17

```text
Completed pretraining W → S → D (BASE, update 19,074)
    → model-only warmstart + NEW midtraining WSD schedule (LM + Base-teacher KD)
    → accepted student-only save/reload
    → separately frozen SFT handoff
```

This explicitly selects **post-decay BASE continued pretraining**, not continuation of the old stable phase and not an exact full-state pretraining resume. Preserve the old BASE and its original tokenizer assets. Reset optimizer moments, scheduler/sample cursor and RNG for this experiment. The pilot uses native Bridge warmup/post-update scheduling without the historical pretraining runner's per-step LR rewrite: `lr_used` is recorded, and zero-LR optimizer calls are counted separately from positive-LR learning updates. The four-call pilot is an operational gate, not a production token budget or an LR tuning result.

Reference LR: old peak `1e-4`, old final `1e-5`; pilot peak `1e-5`, minimum `1e-6`, warmup 1 / WSD horizon 4 / final decay 1 iteration. A longer midtraining duration, corpus blend and LR horizon remain to be frozen after measured pilot evidence. No return to an old stable checkpoint or restart of the completed pretraining is selected.

## Current execution contract

This is a new, explicitly authorized midtraining experiment, **not a relabeling of completed LM-only pretraining**. The existing-ClimbMix native LM+KD control has now completed update/save/teacher-free reload gates; see [executed verification](docs/VERIFICATION.md). The new Specialized corpus [preparation](docs/CORPUS.md) has passed real native writer/readback checks: 10,000,610 training tokens and 406,814 validation tokens, zero train/heldout token-document overlap and fresh zero counts for IDs18–29 across all eight files. This is calibration capacity, not a day-long pack. The [24h lifecycle](docs/LONGRUN.md) now has real pinned-image configuration acceptance; duration-control/save/reload and full-batch calibration are separate execution gates. Production training and quality improvement are not yet established.

- Teacher: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`, immutable revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`. Never substitute the Instruct model or Nemotron 3.0 weights.
- Download: 63 BF16 safetensors shards, **248,913,061,312 bytes** excluding auxiliary files. Original Hub weights remain untouched. No downloaded custom Python is executed.
- Student: actual NemoChat BASE (pretraining update 19,074), not the previous SFT391 checkpoint. Native source: `results/pretraining/nemochat_baby/training/runs/baby-main-01/checkpoint/iter_0019074`; validated HF conversion: `checkpoints/exports/nemochat_baby_base_hf_bf16_02`.
- Objective: next-token LM Loss plus full-logits `KL(teacher || student)`, temperature 1, `skip_lm_loss=false`, no intermediate activation pairs. Student width 1024 and teacher width 4096 remain distinct.
- Installed ModelOpt uses detached loss-magnitude balancing: for positive losses, `LM + scale * KL * stopgrad(LM/KL)`. A scale of 0.5 is **not** a raw 0.5 KL mixture; total value can track 1.5×LM while gradients still depend on teacher predictions. No automatic temperature-squared multiplier is assumed.
- Initial scope: bounded native pilot, 4 updates, sequence 4096, global batch 8, microbatch 1, BF16, TP1/PP1/EP8/CP1, fresh optimizer/RNG/scheduler.
- LR reference: actual pretraining peak `1e-4`, final `1e-5`. Conservative pilot peak `1e-5`, not an untested return to `1e-4`; native WSD warmup/stable/decay boundaries must be recorded. This is a pilot choice, not an optimized LR claim.
- Data: existing ClimbMix indexed train/validation prefixes under `datasets/climbmix/nemochat_baby/native-02/`. Training data was already consumed during BASE pretraining: this is deliberate across-stage replay, not new unique data or evidence of quality improvement.
- Pilot tracking is disabled. No checkpoint publication or remote upload is authorized by this experiment.

## Why a text-backbone conversion is necessary

Bridge 0.6.1 in the pinned image recognizes a Nano Omni architecture, not this exact Super Omni architecture. Renaming its outer architecture would not fix missing latent-projection mappings. `patches/teacher_text_view.py` therefore materializes **only** `language_model.backbone.*` and `language_model.lm_head.*`, strips the `language_model.` prefix, and emits a stock `NemotronHForCausalLM` config. Original tensor payloads are copied unchanged, without loading Hub code. Vision, projectors and unused MTP heads are intentionally excluded. This is a text-only KD experiment, not visual distillation.

The pinned runtime's Transformers 5.12.1 has native `NemotronHConfig`, whose `layers_block_type` exposes the legacy `hybrid_override_pattern` property required by Bridge. Actual loading, teacher fidelity, memory, update/save and reload are separate native gates.

## Stage ordering and isolation

The previous `nemochat-nemotron-48h-pipeline-03.service` automatic SFT release was intentionally paused on 2026-10-17; independent CPU SFT preparation was retained. Receipt: `results/midtraining/super35-kd-01/sft-auto-release-paused.json`.

Do not restart the old SFT controller after download completion: it hardcodes the previous SFT391 source and cannot consume the new midtraining checkpoint. A new frozen SFT source contract is required only after accepted midtraining save/reload evidence. This folder does not authorize an unbounded production run, paid services, remote upload, or changing other active jobs.

## Evidence and acceptance

Artifacts are under `results/midtraining/super35-kd-01/`; original downloaded weights are under `checkpoints/teachers/nemotron35-super-base-bf16-4ba52e8f/` and the text view is under `checkpoints/teachers/nemotron35-super-base/4ba52e8ffa79e9693936c1a819ac89f551ceeef6-text/`; training checkpoints use `checkpoints/midtraining/super35-kd-pilot-01/`.

Accept separately:
1. Pinned complete download with exact lengths and authoritative teacher weight digests.
2. Text-backbone conversion and tokenizer/token-ID compatibility on real input IDs.
3. Native provider/config construction with exact frozen teacher weights, distinct architecture dimensions, and no random teacher fallback.
4. All-rank finite LM/KD losses, finite positive Grad Norm, actual optimizer updates, teacher frozen/no gradients, student-only save and clean Engine exit.
5. Native saved student reload and heldout LM evaluation; only then may an SFT handoff be prepared.

Student checkpoint integrity remains `not_checked_explicit_user_waiver`; no new SHA/CRC/direct-read audit of existing student weights is introduced. Newly acquired teacher downloads have their own authoritative digest verification and are not covered by that historical student waiver.

Passing fixture tests is not native teacher loading or GPU training. A running downloader is not a completed download, and a completed pilot is not a long-run quality improvement.

## Corpus transition selected after the operational pilot

The user accepted Specialized-v1.1 **Code-Concepts + Unconditional-Algorithmic + Formal-Logic**, with ClimbMix replay, for the next midtraining corpus. Economics/Multiple-Choice remain excluded pending separate review; Agentic/Instruction-Following/BBQ remain SFT-only. The data implementation owns a separate pinned, bounded preparation recipe and heldout/benchmark exclusions. Source membership approval does not imply an unbounded whole-corpus download or a production token horizon. The existing four-call ClimbMix control is retained to isolate native KD/import defects; it is not silently relabeled as the new corpus pilot. Switch only through a new frozen runner-consumed data contract after preparation acceptance.

## Native execution and evidence locations

- Compact frozen stage YAML: `midtraining/configs/pilot.yaml`.
- SDK-generated full native YAML from the accepted config-only probe: `results/midtraining/super35-kd-01/config-native-03/native-config.yaml`. Runtime segments generate their own actual consumed native YAML and independent `teacher-provider.json`; teacher/KD fields omitted by native provider serialization are not lost.
- Parent-owned sequence: `midtraining/training/run_pipeline.py`, with a frozen ten-source allowlist at `results/midtraining/super35-kd-01/pipeline-01/reviewed-sources.json`. It verifies the complete download receipt, materializes the teacher text view, takes the cooperative eight-GPU lock and checks live occupancy/recovery, runs the four-call native pilot, then teacher-free native reload/evaluation. It never auto-starts SFT or retries a failed native writer.
- Supervisor: `nemochat-midtraining-base35-pipeline-01.service`; exact state/log/Engine receipts are under `results/midtraining/super35-kd-01/pipeline-01/`. Segment all-rank evidence is under `pilot-01/` and `reload-01/`. `NATIVE_PILOT_STARTING` means launch/setup, not proven optimizer progress.
- Pilot saves model-only student DCP (`save_optim=false`, `save_rng=false`). This is a next-stage warmstart artifact, **not a resumable long-run optimizer checkpoint**. Reload is separately verified with no teacher and no optimizer updates.
- Focused verification command: `.venv/bin/python -m pytest -q tests/test_midtraining_download.py tests/test_midtraining_teacher_view.py tests/test_midtraining_native.py tests/test_midtraining_tokenizer_gate.py tests/test_midtraining_shape_patch.py tests/test_midtraining_pipeline.py`.

## Native failure and isolated diagnostic generation

The first pilot (`pilot-01`, Engine `a2fbb3fe1280a7c4d6faa748cb6d970afb702b8532bfff156979c059d807f1ec`) exited1 without OOM; all eight ranks recorded zero **accepted** updates and no save. Teacher freeze audits passed and native execution reached forward/backward. Its combined failure message did not retain the original numeric subcause; do not infer that it failed merely because Loss/Grad Norm was large. There is no magnitude ceiling: large finite values alone are not rejected.

Metadata inspection found a separate concrete import defect: the student's official HF serialization uses singular `backbone.embedding.weight`, while the installed native mapping expects plural `backbone.embeddings.weight`. `patches/student_embedding_import.py` fixes only the bound student import registry, retains native AutoMapping/loader execution, pins source versions, checks all source tensor names, and records actual native embedding transform calls. It does not edit or re-audit original weight payloads. This is a confirmed import compatibility defect, not yet a proven explanation of the original numeric failure.

Generation02 preserves failed generation01 source/proofs and adds opt-in `patches/kd_finite_probe.py` forward observations plus explicit raw Loss/Grad Norm/skip receipts, without altering outputs, losses, clipping or acceptance criteria. Diagnostic timings are not throughput evidence. Exact native import-plan metadata preflight passed with all6289 student source keys mapped and no weight loading. Current diagnostic unit is `nemochat-midtraining-base35-pipeline-02.service`, state/log/Engine receipts in `pipeline-02/`, rank receipts in `pilot-02/` and potential `reload-02/`; the logical fresh checkpoint destination remains `checkpoints/midtraining/super35-kd-pilot-01` because generation01 saved none. Current source allowlist has12 executed files. Read actual current receipts before reporting model restoration, accepted updates, saved state or reload.

[한국어](README.ko.md)
