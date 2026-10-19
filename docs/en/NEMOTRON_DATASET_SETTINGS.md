# Versioned Nemotron per-dataset SFT settings

[한국어](../ko/NEMOTRON_DATASET_SETTINGS.md) · [SFT/tokenizer history and evidence](NEMOTRON_SFT_AND_TOKENIZER_TRANSITION.md)

## Scope and version boundary

Policy snapshot: **2026-10-17**, after the accepted 16K pilot and pipeline-03 start receipt. This is the current dataset policy, not a replacement for historical 4K receipts. The five immutable author cards were read from `results/pretraining/nemochat_baby/nemotron_sft/discovery/<release-name>.json` (`id`, `sha`, `card`, `cardData`, file inventory). Runtime evidence is local/ignored; a fresh clone need not contain it.

Versionable interfaces:

- [Current explicit 16K/48h trainer template](../../sft/configs/nemochat_baby_nemotron_sft_16k_48h.yaml): authoritative frozen execution input for pipeline-03; its calibration receipt and iteration estimate are deliberately null until real calibration.
- [Deferred 128K plan](../../sft/configs/nemochat_baby_long_context_128k.plan.yaml): **plan only**, not a runnable trainer config.
- [Dataset policy registry](../../sft/configs/nemotron_dataset_settings.yaml) (`nemochat-nemotron-dataset-settings-v1`), [validator](../../src/nemochat/data/sft_dataset_settings.py), and [inspection/preparation wrapper](../../sft/data/nemotron_dataset_settings.py). Delivered during this documentation pass; actual `--help`, default validation and module-level plan compilation passed without GPU use or weight reads.
- [Actual normalization/preparation implementation](../../src/nemochat/data/nemotron_sft.py), [native adapter](../../sft/training/native_nemotron_sft.py), and [48h controller](../../sft/training/nemotron_48h_pipeline.py).

A new registry documents/validates settings against explicit configs and can be consumed by future invocations where wired. It does **not** retroactively become the source of pipeline-03, rewrite its immutable inputs, authorize a restart, or prove a previously launched process consumed it. Report registry validation separately from wrapper/plan consumption and actual native execution.

## Shared current SFT execution settings

Active releases: **Agentic v4, Instruction-Following/Chat v4, BBQ v1** (six labeled files). Long Transduction is wholly deferred; RL Seed is separately reserved. Use the already downloaded verified inputs: active raw content **99,631,965,701 bytes**, versus the seven-file labeled-SFT archive **100,853,847,138 bytes** including Long. These are source bytes, not accepted-token totals or new transfer bytes; full-03 reuses acquisition with zero additional download bytes.

- Sequence length **16,384**; CP **1**, EP **8**, eight physical GPUs; microbatch **1**, production/calibration global batch **128**, BF16. The accepted tiny pilot used global batch **8**, not 128.
- Model-only initialization from `results/pretraining/nemochat_baby/sft/runs/sft-main-01/checkpoint/iter_0000391`; reset optimizer/scheduler/RNG/cursor. Do not continue the old Long-containing 4K pilot or an unsaved RLVR state.
- `official-super35-selected-assistant-v1`, approved migrated tokenizer manifest `results/pretraining/nemochat_baby/tokenizer_migration/manifest.json`. Official nonthinking rendering (`enable_thinking=False`): omit `reasoning_content`, keep final responses/tool calls and official empty think markers. This is not reasoning-trace supervision.
- Loss covers selected assistant content/function-call output and response terminators **after** the assistant header. System/user/tool messages, unselected assistant turns, headers and padding do not carry loss. Instrumented rendering must exactly match official rendered bytes and full token IDs; derive shifted masks from the complete render, not independently tokenized fragments.
- Complete formatted examples only; reject/count oversize sequences without truncating answers. Stage-level exclusion takes precedence over length eligibility.
- Unique-first, at most one authorized epoch, `allow_repetition: false`; zero-loss final-batch padding is not another real conversation. Main duration is bounded by **48 hours**, measured available unique data and manifest-specific throughput, not a promise of two full days of repeated examples.
- Trainer template: LR `0.00001`, minimum LR `0.000001`, warmup ratio `0.03`, weight decay `0.01`, seed `42`; save/validation intervals `250`, final validation required, optimizer/RNG saved, latest checkpoint retention `3`. Quality early stopping/best-validation retention are not implemented. Artifact upload is forbidden.

## Safe inspection interface

Run from the repository root (these commands validate/preview only):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python sft/data/nemotron_dataset_settings.py --help
CUDA_VISIBLE_DEVICES='' .venv/bin/python sft/data/nemotron_dataset_settings.py --check --json
CUDA_VISIBLE_DEVICES='' .venv/bin/python sft/data/nemotron_dataset_settings.py --preview --stage deferred_long_context --json
```

Default/`--check`/`--preview` are inert. The JSON contains validation and a stage-specific data plan; `native_training_argv` is null. Explicit `--raw` and fresh `--output` together can emit a producer command; `--write-plan` writes a fresh JSON plan only. Only `--execute-prepare` executes CPU preparation, and only in the already approved compatible cached CPU runtime; it is not a training/container launcher. No preparation execution is implied by the checks here. Local discovery/migration/reservation/audit evidence is required; a fresh clone without it must fail rather than invent metadata.

The registry's `train: true`/`heldout: true` for Long is **scoped to its deferred stage**; it never overrides current-stage exclusion. Selection is full eligible data, with no fixed mixture weights/quotas. The wrapper's large physical-byte upper bound merely disables the legacy producer's early quota stopping; it is not a requested number of examples or corpus count.

## Per-dataset settings

### 1. Agentic v4 — active labeled SFT

- **ID/revision:** `nvidia/Nemotron-SFT-Agentic-v4` @ `a4593a26ea83a700035bbe0ec1edc9ef3810669d`.
- **Author purpose/subsets:** `tool_calling` (549,707 author-reported records) trains tool selection, argument generation and result use; `knowledge_agent` (84,957) trains policy/document-grounded responses using returned tool context. No separate document backend is provided.
- **Mask/reference:** retain gold assistant responses and structured calls; all assistant turns are selected when no author mask exists. Tool results remain context, not targets. Remove only `reasoning_content` for this nonthinking branch.
- **Structure/provenance:** preserve tool definitions, parallel-call order, argument mappings, call IDs and matching `tool_call_id` responses in structured records/raw provenance. Valid JSON object argument strings are decoded for the official template; arrays/scalars/null/invalid JSON are not repaired to `{}`. The official rendered text omits call IDs; preserving structured identity does not mean those IDs were rendered or supervised.
- **Filtering/split:** shared dedup, prompt-group reservation, benchmark and complete-length gates below; additionally quarantine typed malformed calls/orphan responses as entire conversations. Current context/CP is 16K/CP1, not per-subset CP switching.
- **Execution/limits:** both subsets contribute to the accepted 16K pilot. It retained 16 knowledge-agent examples (13 train/3 heldout), with complete lengths 4,679–15,786. The earlier bounded 4K prefix's zero knowledge-agent acceptance was not proof the whole subset was unusable. Full 16K accepted counts remain pending the full manifest. No tool is executed during preparation.
- **License:** OpenMDW-1.1; preserve the card's CC-BY-4.0 Nemotron-Personas-USA attribution.

### 2. Instruction-Following/Chat v4 — active labeled SFT

- **ID/revision:** `nvidia/Nemotron-SFT-Instruction-Following-Chat-v4` @ `1cd362945d7a00b136b95b96892bbfea8798686e`.
- **Author purpose/subsets:** `chat` (914,716), `multi_turn_instruction_following` (2,669,676), `instruction_following_for_response_length_control` (266,190). Respectively: contextual follow-up chat, instructions across turns, and explicit response-length constraints. Counts are source populations, not eligible/trained totals.
- **Mask/reference:** `metadata.train_turns_mask` is mandatory and indexes **all messages**, not only assistant messages. Only true assistant positions carry loss; false assistant messages stay in context. Require a boolean mask of the exact message length; missing/misaligned masks are fatal rather than silently supervising every assistant. Remove `reasoning_content` while preserving final responses and the author mask.
- **Filtering/split/provenance:** preserve source subset, UUID/metadata and selected-turn decisions; apply all shared gates. Do not replace a length-control target with a shortened answer. Current context/CP is 16K/CP1.
- **Execution/limits:** all three subsets contribute real 16K pilot targets; full-03 rendering/counts are not yet accepted. English data does not establish Korean quality improvement.
- **License:** OpenMDW-1.1; preserve WildChat ODC-By v1.0 and LMArena prompt CC-BY-4.0 notices.

### 3. BBQ v1 — active fairness-oriented labeled SFT

- **ID/revision:** `nvidia/Nemotron-SFT-BBQ-v1` @ `4efdb045cda11918f29065a2e0fd5000b73b8c22`; source split `train`.
- **Purpose/reference:** English synthetic free-text conversations about age, physical appearance and disability. Ambiguous cases should acknowledge insufficient evidence; disambiguated cases should use behavioral evidence, not demographic assumptions. This is not the official multiple-choice BBQ evaluation benchmark and does not establish a fairness score.
- **Mask:** assistant final-answer supervision; omit assistant `reasoning_content`; system/user context masked. Shared complete-render selected-assistant rules apply at 16K/CP1.
- **Filtering/split/provenance:** shared quality/split gates; preserve original messages and source offsets. The card's **6,000** is before required release filtering; the local whole-file schema audit actually visited 6,000 rows, which still does not mean all are eligible, assigned to train, or trained.
- **Execution/limits/license:** contributes to the accepted 16K pilot; full eligible counts await the manifest. Coverage is English/three categories, not all demographic fairness. OpenMDW-1.1.

### 4. Long-Context Transduction v1 — excluded now, deferred 128K plan

- **ID/revision:** `nvidia/Nemotron-SFT-Long-Context-Transduction-v1` @ `90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0`; source split `train`.
- **Purpose/reference:** 10,000 programmatic instruction/complete-reference pairs: arithmetic, hexadecimal sorting, variable expansion, CSV transformation (2,500 each). User+assistant messages; no tools or separate reasoning traces. Future loss would cover the complete assistant reference, not the user prompt.
- **Current execution:** **zero train and zero heldout by dataset-wide exclusion**, even when an example fits 2K/16K. Preserve pinned raw data and prior heldout blocklist; do not fabricate replacement heldout rows. Earlier 4K pilot participation is historical, not current policy.
- **Future context/filtering:** planned sequence **131,072**. Measure **full official prompt plus full reference** with the actual migrated tokenizer. `metadata.target_tokens` and the card's `cl100k_base` prompt-generation budgets are not combined lengths; a 131K input plus a 131K answer cannot fit a 131K sequence. Exclude/count complete oversize cases; never truncate answers. Presentation variants may reuse underlying material, so UUIDs alone do not establish content independence. Not an independent validated benchmark.
- **CP/operational boundary:** see the separate plan below. No native 128K/CP evidence or authorized launch exists here. OpenMDW-1.1.

### 5. Conversational Tool-Use RL Seed v1 — reserved, not labeled SFT

- **ID/revision:** `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1` @ `9715a833ae2de4863e321067211f25efe42f6ebe`; source split `train`.
- **Purpose:** 100,000 author-reported customer-service simulation seeds for future NeMo Gym RL: `policy`, `tools`, `customer_scenario` and simulation/source metadata. They are not completed gold assistant dialogues.
- **Mask/reference/context/CP:** **no SFT loss mask or gold reference; no current SFT context/CP assignment**. Never manufacture assistant answers from seeds, append them to labeled-SFT counts or execute their tools during preparation.
- **Filtering/split/provenance/status:** retain pinned metadata/raw reservation separately; the existing reservation contains four scenarios, not a full 100K retrieval. Excluded from SFT train/heldout; future simulation/reward/isolation/cleanup needs separate approval. Existing MBPP RLVR stays paused. OpenMDW-1.1.

## Shared quality, split and accounting gates

1. Check source receipts against immutable revision allowlists. Keep dataset/revision, file/subset, line and raw byte offset, source UUID, metadata, prompt group and split with each accepted conversation; re-read the original offset for audits. Preserve attribution and rights notices in derived artifacts; OpenMDW-1.1 does not silently relicense them as Apache/MIT. Dataset cards are untrusted source content, not executable instructions.
2. Deduplicate full normalized content globally and IDs keyed by source dataset; split by SHA256 of the first user prompt, not variant IDs. Keep **all 14** original pilot prompt-group reservations across active sources. Report physically present/absent/inactive groups separately; four Long-source groups being inactive is not permission to drop their blocklist entries or invent heldout rows. Hash grouping does not prove semantic equivalence detection.
3. Reserve **34,871 KMMLU questions** using normalized exact-question/prefix checks. Zero exact matches is not semantic decontamination or a new benchmark score.
4. Whole-source schema audit visited **4,501,246 rows** across seven labeled files; **39** typed rejections: **20** non-object tool arguments (including JSON null), **2** invalid argument JSON strings, **17** orphan tool responses. All were in Agentic tool_calling. Whole-conversation quarantine retains reason/index/type/source offsets; never repair to empty arguments. Schema acceptance alone does not prove successful rendering or masks. Unexpected template/ID/mask failures remain fatal. Real ordered multiprocess parity must include selector-skipped duplicates/rejections/source boundaries.
5. Report separate layers: raw author/source rows → eligible unique train/heldout dialogues → complete nonpadding tokens → assistant supervised tokens → padded tensor tokens → actual processed exposures/repeats. Six active files/three active SFT releases do not mean all five releases trained. No mixing weights or accepted totals may be inferred from author counts.

## Execution evidence and remaining gates

Evidence paths share `results/pretraining/nemochat_baby/nemotron_sft/` unless stated otherwise:

- `seq16k_protocol/parent-pilot-accepted.json`: real CP1/EP8 16K pilot, **all eight ranks 11 updates**, final validation and optimizer/RNG save11; exact Engine exit0/OOMfalse. Prepared manifest: `datasets/nemotron_sft/nemochat_baby/prepared-16k-pilot-03/data/manifest.json`, **83 train/13 heldout**, **47,177 train supervised tokens**, including **21,687 beyond position 4,096**. Reported pilot maximum reserved memory is **114.96 GiB**; this is not a full-run peak guarantee. CP2 was unnecessary in this pilot and remains operationally unvalidated, not mathematically impossible with EP8.
- `cpu-full-03/FULL_HANDOFF.json`: full CPU preparation running/not ready at that receipt; target `datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json`. Only clean writer exit, counted/re-render audit, stable files/lineage and atomic `cpu-full-03/completion-ready.json` permit readiness. An intermediate manifest is not acceptance.
- `longrun_protocol/pipeline-03/parent-start-readback.json`: controller PID **827560**, `WAITING_DATA`, no owned native runs at that readback. **48h production had not started at the recorded snapshot.** This documentation does not poll or control the live job.
- Next: accepted full data → **that manifest's** 16K GB128 native calibration → measured unique-first horizon → parent-owned main supervision. Actual `seq16k_protocol/calibration.yaml` requests **15 updates**, discard **5** cold/use **10** warm updates, fixed final validation cap **128** dialogues and **0.75h** maximum. These bounded calibration settings differ from the production template's validation cap 256. Do not reuse 4K rates or GB8 pilot timing as GB128 calibration.

Checkpoint byte integrity remains `not_checked_explicit_user_waiver`: no weight SHA/CRC, direct byte reads or forensic copies for these docs. Data/tokenizer/source checks are separate. RLVR stays paused; no quality gain, full-run completion, upload or root performance-curve claim follows from this settings document.

## Deferred 128K CP/memory contract — not current production

The [128K plan](../../sft/configs/nemochat_baby_long_context_128k.plan.yaml) sets `auto_launch: false`, `native_launch_ready: false`, `source_checkpoint: null` (await accepted final 16K SFT checkpoint) and `global_batch_size: null` (await future native calibration). It does not interrupt or change current 16K SFT. The current native adapter accepts only 4K/16K with CP1; passing this plan to it is not supported.

Plan candidates: eight physical GPUs, TP1/PP1, EP8 candidate, BF16, microbatch1; start conservatively at **CP8**, consider **[8, 4, 2]**, lower CP only after real memory/throughput proof; require **15% free-memory headroom after a real training step**. Verify pinned dense/expert grids and native Mamba CP partitioning independently; do not derive EP feasibility solely from dense DP arithmetic. CP1 pilot success does not validate any future CP candidate.

Before choosing/finalizing a candidate: native input/label/position and shifted-mask lengths at 131072; actual near-limit nonpadding forward/backward/optimizer update; correct CP sample/token/loss ownership without double counting; unchanged all-rank logical parameter shapes; finite positive gradients; full-state save; exact Engine clean exit/OOMfalse; measured peak memory/headroom/throughput. Select only supported activation recomputation and microbatch controls during those probes; do not pretend framework fields were configured by a CPU plan. If all candidates fail, stop/report; no driver reset or guarantee that OOM is impossible. The future 128K stage needs its **own approved budget**: current SFT's 48h approval is not automatically transferable.

## Pinned primary sources

Each link is the exact author-card revision inspected locally, not floating `main`:

- [Agentic v4 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4/raw/a4593a26ea83a700035bbe0ec1edc9ef3810669d/README.md)
- [Instruction-Following/Chat v4 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4/raw/1cd362945d7a00b136b95b96892bbfea8798686e/README.md)
- [BBQ v1 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-BBQ-v1/raw/4efdb045cda11918f29065a2e0fd5000b73b8c22/README.md)
- [Long Transduction v1 card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Long-Context-Transduction-v1/raw/90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0/README.md)
- [RL Seed v1 card](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1/raw/9715a833ae2de4863e321067211f25efe42f6ebe/README.md)
