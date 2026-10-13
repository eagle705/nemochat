# NemoChat: Three-Part Execution and Blog Publication Plan

Status: planning document. No blog publication, release tag, public upload, training result, or benchmark figure has been produced by this planning step.

## 1. Agreed execution and publication units

| Part | Stages | End artifact | Narrative |
|---|---|---|---|
| Part I | [1] Data construction, [2] pretraining, [3] mid-training, [4] long-context extension | Evaluated long-context **base** checkpoint after LC-2 | Build a small Nemotron-inspired hybrid Latent MoE from data to a long-context base model |
| Part II | [5] SFT, [6] RLVR | Evaluated instruction/reasoning checkpoint after RLVR | Turn the base model into a usable assistant and improve verifiable task performance |
| Part III | [7] MOPD | Evaluated Super-distilled checkpoint and final serving/reproducibility bundle | Distill Super into the student using same-node vLLM scoring, then report the final reference |

These are three sequential work/publication packages, not seven independent blog posts. Within a package, keep stage-specific technical documentation and runs. Publish the package blog after its evidence gate, not after a fixed rental week.

**Language:** canonical documentation, example code/comments/docstrings, figures, and primary blog drafts are English. Korean explanatory companions are maintained separately under `docs/ko/` and the proposed `blog/ko/`. Do not mix bilingual explanatory prose into code comments.

**Scope:** use tutorial-budget runs for the first complete series. Extended runs can become explicit follow-up revisions/addenda. Never replace a tutorial run's reported data or performance with an extended run without identifying the new checkpoint/config.

## 2. Common handoff contract

Every part produces an immutable, versioned artifact manifest containing:

- source code commit and local patch state, environment/container digest and SDK versions;
- model config, tokenizer files/hashes and chat/base templates;
- input checkpoint and output checkpoint IDs/checksums;
- actual stage tokens: accepted unique, consumed, loss-bearing; RL rollout/scoring/update tokens separately;
- dataset revisions, source/license manifest, filter/dedup reports and split lineage;
- evaluation-pack version, prompts/templates, decoding settings, metric definitions, raw results and aggregate tables;
- actual GPU topology, active training time, evaluation/profile time, checkpoint/load overhead and measured useful throughput;
- plot-generation inputs/scripts and figure metadata;
- failures, support boundaries, custom integrations, and known limitations;
- public artifact links **only after upload/read-back verification**; until then use an explicitly planned location.

Keep minimal recovery checkpoints plus milestone exports rather than every intermediate training checkpoint. Persist the bundle before the rental ends. Model-only exports do not substitute for resumable optimizer/scheduler/RNG/data-cursor state.

Proposed stable local labels are `part1-base-lc`, `part2-rlvr`, and `part3-mopd`; these are planning names, not existing Git tags or uploaded model IDs. Public release naming and repository hosts are decided at implementation time.

## 3. Part I — Data to Long-Context Base Model ([1][2][3][4])

### Suggested English title

**Building NemoChat, Part I: From Curated Data to a Long-Context Hybrid Latent MoE**

### Execution order

1. Before GPU rental: source/license audit, streamed quota-limited acquisition fixtures, data schema/split/provenance, English/Korean documentation skeleton.
2. Architecture correctness and approved **A/B-only** pilot, including model census, export round-trip and Nsight profiling.
3. [1] Curator filtering/dedup/contamination checks and DataDesigner validated synthetic pilot; build selected local training artifacts, not full-corpus mirrors.
4. [2] Pretraining with actual token quotas, WSD and checkpoint/resume. Compare raw against PR #5114 minus-sqrt stable-window WSM at eligible milestones using suitable pinned official Gym tasks; retain CE/PPL/BPB and CORE as separate diagnostics. Target-model training/official benchmark execution remains untested.
5. [3] Domain-rich mid-training with short-task/general-data retention measurements.
6. [4] **64K long-only LC-1 -> 64K/4K alternating LC-2**. Intermediate lengths are resource/correctness checks; 128K/256K remain separately reported stretch experiments.
7. Run final Part I evaluation and verify artifact handoff.

### Required evidence/figures

- Model architecture diagram and actual total/active parameter census, with active-count convention.
- A/B comparison: useful throughput, peak memory, heldout loss and expert load distribution, plus Nsight interpretation. No unexecuted C run.
- Curation funnel: input/accepted/drop counts and tokens, selected subset/cache/storage footprint, unique vs consumed tokens.
- Pre/mid LM CE and **validation BPB** by source/domain against **consumed tokens** and useful training wall time; nanochat-compatible CORE with task raw/centered scores, pinned quick/full packs and actual inference figures.
- ClimbMix quota/provenance/original noncommercial conditions; see [detailed metrics/data specification](PRETRAIN_METRICS_AND_CLIMBMIX.md).
- LC schedule figure: long vs short update/token exposure and data-blend definitions.
- Actual NIAH heatmaps: pre-LC, post-LC-1, post-LC-2 on the same fixture pack and color scale.
- Short math/code retention plus long QA/RULER-style results; show whether LC-2 actually mitigates regression.
- Measured rental-time and storage table; checkpoints/export/evaluation overhead disclosed.

### Blog outline

1. Reference-tutorial objective and B300 week-long rental constraints.
2. Nano-pattern hybrid plus Super-inspired latent MoE: what is copied and what is changed.
3. Data selection without full-corpus downloads and reproducible token quotas.
4. Curator/DataDesigner responsibilities, validators and license/provenance.
5. A/B GEMM/communication trade-offs and measured choice.
6. Pretraining and mid-training decisions/results.
7. Report-inspired long-only then long/short CPT: token/step distinction.
8. NIAH figures, broader long-context tests and short-task retention.
9. Failure cases, costs, public artifacts and what Part II consumes.

### Exit/handoff gate

- Output is explicitly a **base** model, not yet a fully instruction-tuned assistant.
- Finite/stable training and save/resume/export checks pass; failed tests are disclosed, not hidden by publication.
- Actual maximum validated context and unsupported lengths are reported; config-only support is not counted.
- Data/model redistribution review is complete for each public artifact.
- Frozen evaluation pack, LC-2 checkpoint and tokenizer manifest become Part II's inputs.
- If 64K fails, publish a clearly labeled investigation/partial-result article rather than claiming the complete Part I target passed. It is not a passed execution gate.

## 4. Part II — SFT and RLVR ([5][6])

### Suggested English title

**Building NemoChat, Part II: Instruction Tuning and Verifiable Reinforcement Learning**

### Execution order

1. Load the immutable Part I checkpoint/tokenizer/data/evaluation contract and confirm inference parity.
2. [5] SFT: English chat/template policy, assistant-only loss masking, validated math/code/structured-output examples and some long-context retention tasks.
3. Save/evaluate the **post-SFT checkpoint before RLVR**; this intermediate is essential for causal comparison.
4. [6] Verify math/code/schema reward implementations and reward-hacking/sandbox fixtures.
5. Run small GRPO/RLVR pilot, then the approved rollout/update caps.
6. Evaluate standard vLLM rollout first; managed Dynamo rollout/refit/cache invalidation is a separately validated integration/control comparison where compatible.
7. Evaluate post-RLVR against both base and SFT checkpoints, including NIAH/long-task retention.

### Required evidence/figures

- SFT mixture/length histogram, assistant loss-bearing vs processed tokens, mask/truncation tests.
- Base -> SFT -> RLVR comparison using appropriate, disclosed prompt templates. Template changes are not presented as model-only effects.
- Reward curve alongside **heldout true correctness**, not reward alone.
- Reward-hacking cases, verifier pass/fail fixtures, sandbox timeout/security handling.
- Rollout length, entropy/KL/clipping/zero-variance group rate and generation/update costs as applicable.
- Training-generation token/logprob parity and weight synchronization evidence.
- NIAH post-SFT/post-RLVR heatmaps plus short/long task retention.
- Dynamo vs standard vLLM comparison only when actually run with matched settings; separate SDK integration correctness from performance gains.

### Blog outline

1. Why base pretraining is not assistant alignment.
2. SFT data/template/masking and small-model reasoning length.
3. RLVR rewards and the distinction between executable verification and LLM judging.
4. Student generation/training topology and logprob parity.
5. GRPO/RLVR experiment results and reward-hacking failures.
6. Managed Dynamo integration boundaries and measured behavior.
7. Short-task improvement vs long-context retention, costs and limitations.
8. Release/checkpoint handoff to Part III.

### Exit/handoff gate

- Verifier correctness/sandbox and weight-sync tests pass.
- Both SFT and RLVR checkpoints/results are preserved as selected milestone exports, not only the final result.
- Improvement is reported only if heldout measurements support it; no improvement is an acceptable honest tutorial outcome.
- Long-context regression is reported and corrected or explicitly documented.
- The final RLVR checkpoint is frozen as the primary MOPD starting point. If a post-SFT branch is chosen instead, record that explicit branch rather than silently changing lineage.

## 5. Part III — Super Teacher MOPD ([7])

### Suggested English title

**Building NemoChat, Part III: On-Policy Distillation from Nemotron 3 Super on One Node**

### Execution order

1. Validate frozen Part II student checkpoint and tokenizer compatibility with the fixed Super teacher.
2. Self-distillation/scoring gold cases: causal token shift, masks, BOS/EOS, raw student probabilities and teacher prefix alignment.
3. Implement/validate the **same-node vLLM teacher-scoring adapter and controller**. Prefer time-multiplexed correctness before trying a persistent 4/4 GPU split.
4. Score **student-generated tokens**, not teacher-generated completions mislabeled as MOPD.
5. Run small updates, then the approved generated-token budget; separately count teacher scoring and update work.
6. Compare pre-MOPD and post-MOPD student on the same heldout tasks and NIAH pack. Keep Super as a clearly labeled reference, not a target guaranteed to be matched.
7. Verify final Dynamo serving/export/cache correctness and package the entire series' artifact matrix.

### Required evidence/figures

- Sequence diagram: student rollout -> teacher forced-prefix scoring -> student update.
- Native NeMo RL reference vs custom same-node vLLM extension support-boundary table.
- GPU placement/time-multiplex timeline and measured load/offload/scoring/update costs.
- Self-distillation logprob gap and token-alignment gold-case results.
- Teacher-student gap plus heldout task scores: gap decrease alone is not success.
- Pre/post MOPD math/code/instruction/long-context results and NIAH retention figures.
- Verified final serving TTFT/inter-token latency/throughput/memory under disclosed conditions.
- End-to-end source/checkpoint/dataset/evaluation lineage and public model/data cards.

### Blog outline

1. Why distill after SFT/RLVR, and why Super is the first teacher.
2. MOPD vs offline teacher-answer SFT; single-teacher start within a multi-teacher framework.
3. Sampled-token objective and tokenizer/prefix/logprob correctness.
4. Why native multi-node placement does not translate directly to our one-node setup.
5. vLLM scoring adapter, controller, GPU placement and overhead.
6. Results, failed approaches, retention and limitations.
7. Final Dynamo serving demo and a reproducible clean-start walkthrough.
8. Whole-series costs, public artifacts and future 128K/256K/multi-teacher work.

### Exit/publication gate

- Actual scoring and on-policy snapshot/controller tests pass before claiming MOPD.
- Full native async-MOPD support is not claimed for this custom extension.
- Offline teacher-answer SFT may be documented as fallback, but cannot pass the MOPD completion gate.
- Quantized teacher experiments, if any, are labeled separately from BF16 scoring.
- Checkpoint export/serving, artifact licensing and public upload verification complete before public download links are advertised.
- If teacher scoring or resources block execution, publish an explicit engineering/blocker report rather than fabricated training results.

## 6. Proposed publication files (not created blog posts yet)

```text
blog/
  en/
    part-01-data-to-long-context.md
    part-02-sft-and-rlvr.md
    part-03-super-mopd.md
  ko/
    part-01-data-to-long-context.md
    part-02-sft-and-rlvr.md
    part-03-super-mopd.md
  assets/
    part-01/
    part-02/
    part-03/
  manifests/
    part-01-artifacts.json
    part-02-artifacts.json
    part-03-artifacts.json
```

Technical documentation under `docs/` is the recipe/reference; blogs explain decisions and real results. Link the blog to pinned recipe/config/evaluation artifacts rather than maintaining diverging copied commands. Do not create result-bearing drafts now using guessed curves, scores, screenshots or links.

## 7. Writing and publication workflow

For each part:
1. During runs, capture a concise experiment diary: hypothesis, config/input IDs, real output, change, reason and failure mode. No secrets/private raw data.
2. At the package boundary, freeze the relevant artifacts/metrics and regenerate figures from raw result files.
3. Check consistency: data quotas, checkpoint IDs, stage names, SDK versions, figure captions and measured costs agree.
4. Draft the English article; prepare the separate Korean explanation from the same evidence bundle.
5. Technical review: claims vs evidence, native/custom boundaries, licensing/privacy and reproduction commands.
6. Run a clean-start or representative reproducibility test and disclose any unavailable resources.
7. Publish/upload only after explicit user instruction for the destination; verify remote artifacts, then insert their real links.
8. Later extended experiments append a versioned addendum or new release; do not rewrite historical results without a correction note.

A package may span multiple one-week rentals. Keep task completion and blog publication status separate. Final Dynamo serving belongs in Part III's wrap-up, while the managed RLVR Dynamo integration belongs in Part II; do not invent a fourth required blog part.
