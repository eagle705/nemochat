# NVIDIA dataset SFT, Super3.5 tokenizer transition, and private BASE publication

[한국어](../ko/NEMOTRON_SFT_AND_TOKENIZER_TRANSITION.md) · [Repository](../../README.md) · [Versioned per-dataset settings](NEMOTRON_DATASET_SETTINGS.md)

Current dataset policy, masks, filtering, provenance, 16K execution settings and the separately deferred 128K CP plan are collected in the [per-dataset settings reference](NEMOTRON_DATASET_SETTINGS.md). Historical snapshots below remain intact.

## Decision and evidence boundary

Snapshot: **2026-10-17 10:16 KST**. This document records the user's decision to pause MBPP RLVR, prepare additional SFT from the five specified NVIDIA releases, migrate only reserved tokenizer names, and publish the **pretraining BASE**, privately. These are complementary operations, not permission to resume RLVR or publish an SFT checkpoint as BASE.

**Executed**, **prepared**, and **pending** are different states. Runtime evidence under `results/` is local/ignored and may not exist in a fresh clone. Paths below are repository-root-relative; they are provenance locators, not claims that checkpoints or datasets are versioned. The older README implementation table describes the earlier CPU-first baseline; use the stage-specific evidence here for this run, without treating the rest of the roadmap as executed.

| Work item | Observed state at this snapshot | Remaining gate |
|---|---|---|
| Previous Smol Korean/English SFT | **Executed:** 391 native updates and final saves on all eight ranks | This is not the new NVIDIA-data SFT |
| Post-SFT evaluation | **Executed:** KMMLU 0-shot, 21/100 on the requested subset; one real Korean generation repeated itself | Matched-protocol evaluation for any new candidate |
| MBPP RLVR | **Pause requested; observer reports controller inactive.** 59 judged model rollouts, all reward 0; zero common all-rank optimizer updates | Exact Engine stop and full owned-sandbox inventory/cleanup receipt; no automatic restart |
| Five NVIDIA source releases | **Prepared discovery:** five pinned cards, metadata and file inventories inspected; remote pinned cards matched local copies | Bounded retrieval, exact selection/mask/token counts and native data integration |
| Super3.5 tokenizer | **Source inspected:** twelve reserved-name replacements; implementation being prepared | Full old-corpus/old-SFT ID scan and migration manifest, then compatible-runtime tokenization checks |
| New NVIDIA-data native SFT | **Pending:** no new pilot or training completion established here | Real short native pilot before the bounded main run |
| Private HF BASE publication | **Requested/pending:** `eagle705/nemochat-10BA1B-BF16-base` | Actual serialization/export, upload, immutable revision and remote private/card/inventory read-back |

### Existing execution proof and negative results

- Previous SFT: `results/pretraining/nemochat_baby/sft/runs/sft-main-01/verification.json` records 391 updates per rank and save steps 250/391. `results/pretraining/nemochat_baby/SFT_COMPLETION_REPORT.md` records completion at **2026-10-17 02:45:06 KST**, 50,000 dialogues from 25,000 paired KO/EN topics, 1,000 heldout dialogues and 34,535,226 assistant loss-bearing targets. Final heldout CE **2.5068976879119873**, PPL **12.26681547947325** are assistant-target SFT metrics, not KMMLU or directly comparable pretraining loss.
- Later evidence supersedes the completion report's historical “post-SFT KMMLU has not been run” statement: `results/pretraining/nemochat_baby/post_sft_eval/engines/native-0shot-100-01/verification.json` accepts **21/100, 0-shot**, 45 subjects/400 logical choice scores, with all eight ranks restored/frozen and native exit 0. Earlier **28/100** used **PRETRAIN, 4-shot**. Both checkpoint and prompting changed; this does **not** establish an SFT gain or causal regression.
- The same evaluation receipt records a real greedy Korean response to a request for three rainy-day items. It repeated `이 경우`, reached the 128-token cap, and was truncated rather than ending normally. This is a negative instruction-following example, not a good answer or a population-level estimate.
- `results/pretraining/nemochat_baby/post_sft_handoff/rlvr-observer-latest.json` at **2026-10-17 10:07:34 KST** reports the controller `inactive`, 59 judged rollouts and zero common all-eight accepted updates. Direct aggregation of **59** `rlvr/runs/rlvr-main-02/reward_responses/*.json` files found **59 zero rewards, zero positive rewards**. The earlier 09:42 progress file had only 40 rollouts and is stale. Judge receipts are actual model-generated MBPP attempts, not successful learning updates; no RLVR checkpoint is established.
- RLVR Engine identity: `ab1d48467964e1bb70f2d0cdc19d2f35cd7aa1c44c36416898aae8e96152e3f6`. A controller becoming inactive or a service port closing does not prove every owned guest was removed. Preserve stop, exit and exact guest/container inventory evidence separately; this document does not certify full cleanup.

Checkpoint byte integrity remains **`not_checked_explicit_user_waiver`**. Do not restart weight SHA/CRC audits, direct weight reads or forensic checkpoint copies to “complete” this documentation. Tokenizer/data/source checks are separate and permitted. Native restore/save receipts are not byte-integrity or HF loading-compatibility certificates.

## The five user-selected NVIDIA releases

All five pinned cards identify English text data and OpenMDW-1.1 terms. They are NVIDIA dataset releases, not invented stand-ins. Counts below are **author-reported source counts**, not downloaded, eligible, deduplicated or trained counts.[1][2][3]
The RL seed and BBQ-specific qualifications follow separately.[4][5]

| Release | Pinned revision | Author-reported population | Local use |
|---|---|---|---|
| `nvidia/Nemotron-SFT-Agentic-v4` | `a4593a26ea83a700035bbe0ec1edc9ef3810669d` | tool_calling 549,707; knowledge_agent 84,957 | Labeled SFT conversations; keep tools and call/result relationships.[1] |
| `nvidia/Nemotron-SFT-Instruction-Following-Chat-v4` | `1cd362945d7a00b136b95b96892bbfea8798686e` | length control 266,190; chat 914,716; multi-turn IF 2,669,676 | Labeled SFT; mandatory author turn mask.[2] |
| `nvidia/Nemotron-SFT-Long-Context-Transduction-v1` | `90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0` | 10,000 complete instruction/answer pairs | Only examples whose complete formatted input **and** target fit the present context.[3] |
| `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1` | `9715a833ae2de4863e321067211f25efe42f6ebe` | 100,000 simulation seeds | Reserve for later agentic Gym RL; **not gold assistant-answer SFT**.[4] |
| `nvidia/Nemotron-SFT-BBQ-v1` | `4efdb045cda11918f29065a2e0fd5000b73b8c22` | 6,000 **original examples before required release filtering** | Evidence-grounded fairness SFT; count actual released/accepted rows separately.[5] |

Pinned author-card links are in **Sources**, not floating `main` links. Discovery files are `results/pretraining/nemochat_baby/nemotron_sft/discovery/<release-name>.json`; each contains `id`, `sha`, `card`, `cardData` and the source file inventory. All five pinned remote README bodies were read back and exactly matched their stored `card` strings during this documentation pass.

### Preserve author semantics, not just text

1. **Agentic v4:** `tool_calling` and `knowledge_agent` are two SFT subsets. Retain message ordering, tool definitions, assistant tool calls, call identifiers and associated tool-result context. Knowledge-agent examples are conversation traces; a standalone document corpus/backend is not supplied. The card permits removing `reasoning_content` for non-reasoning training while preserving responses and tool calls.[1]
2. **Instruction-Following/Chat v4:** use `metadata.train_turns_mask` to decide which assistant turns carry loss; keep unselected messages as context. Removing `reasoning_content` must not discard the author mask or silently supervise every assistant turn. A final-response-only/non-thinking branch is not permission to ignore this field.[2]
3. **Long-context transduction:** complete rule-generated references cover arithmetic, hexadecimal sorting, variable expansion and CSV transformations. The 2,048–131,072 prompt budgets use `cl100k_base` generation targets; they are not measured Super3.5 lengths or input-plus-output budgets. Measure the full rendered sequence with the actual tokenizer, reject oversize examples with reason/count, and never truncate an answer silently. This source does not grant the present **4,096-token** native model 131K capability or an independent benchmark result.[3]
4. **RL seed:** records provide `policy`, `tools`, `customer_scenario` and simulation metadata for NeMo Gym conversations. They do not supply gold completed assistant dialogues. Do not turn a policy/scenario into a fabricated assistant answer, add these 100,000 seeds to the labeled-SFT count, or run their tools during data preparation. Later RL requires separately authorized simulation, rewards, isolation and cleanup.[4]
5. **BBQ SFT:** ambiguous contexts should acknowledge insufficient evidence; disambiguated contexts should use behavioral evidence rather than unsupported demographic assumptions. This synthetic free-text training set is informed by BBQ fairness design, **not a reused official multiple-choice BBQ benchmark**. Preserve the distinction between the card's original 6,000 count and the actual released/accepted rows.[5]

Keep attribution/rights metadata with every source and export. The cards' OpenMDW-1.1 declarations do not make every derived component Apache/MIT: Agentic cites CC-BY-4.0 persona grounding; IF/Chat cites WildChat ODC-By and LMArena CC-BY-4.0 prompts. Preserve those notices and review applicable redistribution terms separately.[1][2]
All five sources are English; Korean instruction-following improvement is a hypothesis to evaluate, not a consequence of their names.[1][2][3]

## Bounded additional-SFT plan — not an executed result

Use the **four labeled SFT releases** above; reserve the fifth RL-seed release. Do not silently return to Smol KoreanTalk or add ClimbMix because they are already cached. The proposed first increment is roughly **50K–100K eligible conversations**, subject to actual measured availability, source quotas and a measured **24-hour** budget. No eligible count, exact mix, expected improvement or throughput is established at this snapshot.

- Bound JSONL/prefix acquisition to the explicit source download budget (**at most 8 GB initially**); do not silently download the complete large source collection. Prefix/selected samples are not representative full-release surveys.
- Preserve pinned source row provenance, subset, author masks, tool linkage, split grouping, exclusion reasons, accepted examples, formatted token lengths and assistant-target counts. Deduplicate/split before forming training/heldout artifacts; do not claim semantic benchmark decontamination from exact-match checks alone.
- Use the actual approved Super3.5 tokenizer/template in a compatible runtime. Verify assistant-only token loss masks, selected-turn masking, response terminators, tool syntax and non-thinking behavior against real selected examples, not just synthetic fixtures.
- Continue model-only from **SFT391**, with a new SFT optimizer/scheduler/RNG/data cursor and unchanged native normalization/router behavior. Preserve the original PRETRAIN and SFT391 artifacts. Do not resume pretraining optimizer state or train from an unsaved RLVR candidate.
- Gate the main run on a real short native pilot (planned 16/4-update checks), all-rank load/update/save receipts, finite positive gradients, selected-target accounting and heldout/generation checks. A preparation test or CLI help success is not this gate.
- Current native context is **4K**. Retain full fitting long-transduction examples and explicitly count oversize exclusions; broader long-context adaptation remains separate pending work. Report if a requested source contributes zero accepted examples rather than claiming all five trained.

## Tokenizer migration: rename reserved IDs, never resize or permute weights

Old tokenizer files are preserved at `checkpoints/tokenizers/2dc98e2afe4face0e4ce40972a915c45368bd34a/`. Vocabulary size is **131,072**. The requested target is **tokenizer files only** from `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`; the 10:16 inspection resolved revision `1f8953f7f0935358e3605150d22bc15df062061c`. This is not a request to download its weights or replace the Baby architecture.[6]
The subsequently delivered migration config pins **`fc067b34f6b514014d2f8bddba855d2e4bfdfba6`**, not that floating-HEAD discovery revision. Its pinned tokenizer was also read directly and has the same twelve names below; use the **config/manifest pin** for reproduction rather than substituting latest HEAD.[7]

| Existing ID | Old spelling | Super3.5 spelling |
|---|---|---|
| 18 | `<SPECIAL_18>` | `<image>` |
| 19 | `<SPECIAL_19>` | `<img>` |
| 20 | `<SPECIAL_20>` | `</img>` |
| 21 | `<SPECIAL_21>` | `<quad>` |
| 22 | `<SPECIAL_22>` | `</quad>` |
| 23 | `<SPECIAL_23>` | `<ref>` |
| 24 | `<SPECIAL_24>` | `</ref>` |
| 25 | `<SPECIAL_25>` | `<box>` |
| 26 | `<SPECIAL_26>` | `</box>` |
| 27 | `<SPECIAL_27>` | `<so_embedding>` |
| 28 | `<SPECIAL_28>` | `<so_start>` |
| 29 | `<SPECIAL_29>` | `<so_end>` |

These twelve new spellings are read directly from the pinned target tokenizer.[6] Migration acceptance requires:

1. Compare the complete tokenizer backend: only the approved names at exactly IDs 18–29 may change; all other text/control IDs, vocabulary size, BPE merges, normalizer, pre-tokenizer and decoder remain unchanged. Reject any extra difference rather than silently repairing it.
2. **Before activation**, scan the actual packed pretraining corpus and previous prepared SFT token IDs for every ID 18–29. Record full scanned paths, token counts, per-ID occurrences and explicit omissions. A fixture or selected sample is not the requested full 10B-corpus scan; findings are pending until its manifest exists.
3. Preserve immutable old-tokenizer/checkpoint lineage and write a separate migrated tokenizer artifact. No embedding/output vocabulary growth, row reorder, reinitialization or weight rewrite is authorized by this rename.
4. Prove chat/template and text-ID parity on representative old/new text and actual training examples. Report literal strings affected by reserved-name changes and preserve correct response terminators.

An image-like token name **does not add visual inputs, a vision encoder, or multimodal capability**. Nor does a zero occurrence in input/target IDs establish that a corresponding output row was “never trained”: softmax gradients can affect output rows without positive target occurrences. Corpus scanning and unchanged row alignment answer narrower compatibility questions.

Host Transformers **4.57.6** cannot be assumed to load the new `TokenizersBackend` through `AutoTokenizer`. Use the already available native NeMo runtime with Transformers **5.12.1** for actual tokenization; do not upgrade the shared host environment or enable arbitrary remote tokenizer code as a shortcut. Record the actual runtime in migration/preparation evidence.

Expected migration proof location (not yet produced at this snapshot): `results/pretraining/nemochat_baby/tokenizer_migration/manifest.json`.

## Private HF publication: PRETRAIN BASE is a separate lineage

Destination: **`eagle705/nemochat-10BA1B-BF16-base`**, explicitly **private**. Source: `results/pretraining/nemochat_baby/training/runs/baby-main-01/checkpoint/iter_0019074`, **not SFT391**, not RLVR, and not the new tokenizer-transition/SFT branch.

The BASE package must retain the **original tokenizer/reserved names** to describe its actual pretraining lineage; migrating posttraining aliases is a separate variant. Preserve the raw native resumable checkpoint locally and distinguish a BF16 model-only publication from optimizer/RNG-bearing native training state.

The model card must state architecture/configuration and actual serialization, pretraining source/step and original tokenizer pin, non-instruction-tuned BASE limitations, applicable source attribution/rights, evaluation protocols/caveats, and the explicit byte-audit waiver. Do not label a native DCP archive “HF Transformers safetensors” or promise `AutoModel.from_pretrained` support unless the real exporter/loader path proves it. A Hub repository name ending in BF16 is not format or compatibility evidence.

Publication is complete only after exact-target remote read-back confirms **private visibility**, immutable commit revision, card content and the expected uploaded inventory. A local export, repository creation, successful upload request, or another worker's intention is not that receipt. The documentation worker did not upload any weights; publication remains pending in this snapshot. See [existing checkpoint packaging boundaries](HF_CHECKPOINT_PIPELINE.md).

## Reproduction interfaces and remaining receipts

Run from the repository root. The agreed code-worker entrypoints are listed below, **not asserted implemented or executed at this snapshot**:

| Intended entrypoint | Purpose | Companion contract |
|---|---|---|
| `sft/data/prepare_tokenizer.py` | Source pin, backend compatibility, old-ID scan and migration artifact | `sft/configs/tokenizer_super35.yaml`; `src/nemochat/tokenization/posttraining.py` |
| `sft/data/nemotron_sft.py` | Bounded source preparation and selected-turn/token accounting | `sft/configs/nemochat_baby_nemotron_sft.yaml`; `src/nemochat/data/nemotron_sft.py` |
| `pretraining/publication/publish_hf_base.py` | Separate private PRETRAIN BASE packaging/publication | `src/nemochat/hub/base_checkpoint.py` |

At 10:16 KST the three script files/configs were not yet available, so no `--help` success or subcommand/flag spelling is claimed. The tokenizer module was being written; its presence alone did not establish full CLI readiness. Use this non-destructive availability/help check before constructing any operation; it deliberately performs **no data preparation, GPU training or upload**:

```bash
for entry in \
  sft/data/prepare_tokenizer.py \
  sft/data/nemotron_sft.py \
  pretraining/publication/publish_hf_base.py
do
  if test -f "$entry"; then
    .venv/bin/python "$entry" --help
  else
    printf 'PREPARING: %s\n' "$entry"
  fi
done
```

**Readiness update, 2026-10-17 10:21 KST:** `sft/data/prepare_tokenizer.py` and `sft/configs/tokenizer_super35.yaml` are now present. The documentation worker actually ran `.venv/bin/python sft/data/prepare_tokenizer.py --help` successfully (exit 0). It exposes positional stages `download`, `audit`, `verify-runtime`, `publish`, plus `--config`; `publish` here writes the **local tokenizer migration manifest**, not an HF weight upload. The source prescribes that order. `verify-runtime` must run in the compatible native SDK CPU runtime, not the incompatible host environment. None of those four stages was executed by this documentation worker; the migration manifest was still absent at the readiness check. The SFT and BASE publication CLIs were still pending, so their flags remain unspecified.

Do not infer preparation/training/upload options from filenames. Once the code worker delivers a CLI, capture its real `--help`, use only supported flags, and append the exact accepted commands and receipts here. Native-launch commands additionally require the compatible SDK and parent's GPU authorization/gates; none are invented in this document.

Before promoting any pending row to executed, retain: tokenizer migration/full-ID scan manifest; pinned acquisition and prepared-data manifest with actual per-source counts/rejections/targets; source/mask/template gate; actual native pilot then main receipts; matched-protocol quality diagnostics; explicit RLVR stop/owned-resource cleanup evidence; private BASE remote publication receipt. Documentation and CPU checks do not substitute for those results.

## Execution update — after 2026-10-17 10:46 KST

This section supersedes the preparation snapshots above, without erasing their historical status.

- **Private BASE published:** `eagle705/nemochat-10BA1B-BF16-base`, immutable revision `94422e79dd31ede40e52a9940367115b690d5048`. Parent reran the publication CLI's `verify --revision` action and confirmed private visibility, all 17 payload files / 19,620,927,468 bytes, and exact model-card readback. Hub-generated `.gitattributes` is accounted separately.
- **Actual format:** model-only native Megatron/PyTorch DCP, excluding optimizer/scheduler/RNG/cursor. This is not a Transformers conversion; full exported-model native reload/forward remains unexercised and is disclosed in the card. BASE preserves the original tokenizer, not the post-training migration.
- **Tokenizer transition accepted:** `tokenizer_migration/manifest.json` passed. All 131072 IDs compared; exactly 18–29 renamed and the remaining backend structure identical. Full ClimbMix train/validation/calibration plus existing SFT train/heldout scan covered 10,227,216,313 stored IDs with zero occurrences of each reserved ID. Original weights/tokenizer remain unchanged. The compatible pinned CPU runtime exercised the production loader and official nonthinking rendering.
- **Real pilot corpus accepted:** 82 training / 14 heldout conversations, with 46,900 / 11,706 supervised tokens. All four SFT datasets contribute training targets; RL Seed stays separately reserved. All 96 raw examples were reread and re-rendered to verify official bytes/token IDs/shifted masks. Sixteen complete Long Transduction examples fit 2,831–3,506 tokens. All 344 sampled knowledge-agent examples exceeded 4K and were excluded, not truncated.
- **Native pilot launched:** `nemochat-baby-nemotron-pilot-01.service`, Engine `eab7fdb86e7c48945ae5a01d9a1175411527870d4f0f3ef0785442742e3f256f`, SFT391 model-only warmstart, EP8/GBS8, 11 planned updates. Launch does not establish learning or save completion; acceptance requires final `runs/nemotron-pilot-01/verification.json` and all8 receipts.
- **Main remains pending:** bounded CPU main acquisition/preparation targets 50K–100K eligible conversations and preserves all 14 pilot heldout groups. Do not cycle the tiny pilot and call it main training. RLVR remains paused.
- Parent reran the focused tokenizer/data/publication tests: **24 passed**. Existing full-suite scratch-filesystem `renameat2 EINVAL` failures were not hidden or claimed fixed.

Exact executed commands and receipt paths are in the matching Korean update and `results/pretraining/nemochat_baby/{hf_base_publication/publication_receipt.json,tokenizer_migration/manifest.json,nemotron_sft/README.md,nemotron_sft/pilot-01-controller.log}`. Existing run names/outputs must not be reused. Checkpoint byte-audit waiver remains unchanged.

### Pilot accepted — ended 2026-10-17 10:53 KST

Parent reran the native verifier and freshly inspected the exact Engine: **all8 11/11 updates, optimizer/RNG final save11, entrypoint0, Engine exit0/OOMfalse**. Actual checkpoint: `results/pretraining/nemochat_baby/nemotron_sft/runs/nemotron-pilot-01/checkpoint/iter_0000011`. Exact W&B run `nemochat-baby-nemotron-sft-pilot-01` was freshly read as `finished`/step11. Final Training Loss0.7940755486488342; fixed-window Validation CE1.7139155864715576. This proves execution on an82-dialogue pilot, not full additional SFT or generalization improvement. CPU main data preparation remains active; no main GPU run yet. Evidence: `parent-pilot-accepted.json` and `parent-pilot-wandb-readback.json`. Main starts independently from SFT391 instead of cycling or continuing tiny-pilot exposures. No new checkpoint byte audit/full reload is claimed.

### 48-hour/full-data approval and GB128 calibration update

The user explicitly approved approximately48h SFT and all seven labeled files from the four SFT repositories: **100,853,847,138 bytes**. This supersedes the earlier8GB/24h subset policy. RL seeds remain reserved without fabricated answers. Parent verified actual `nemochat-nemotron-full-acquire-01.service` and `nemochat-nemotron-full-prepare-01.service` execution. Full-source digest/length checks,24-worker CPU preparation and counted mask audit gate full-manifest acceptance.

GB128 calibration01 performed20 updates/save/heldout but failed exit1 because adapter-forced and SDK automatic final evaluations exhausted the validation iterator. Preserve that failure; the scheduling fix followed RED→GREEN regressions. Fresh calibration02 was independently accepted: **all8 20 updates/save20/final868-dialogue heldout, Engine exit0/OOMfalse**. Mean15-warm-update time12.352091671002563 seconds; heldout CE2.424147129058838. This is operational/throughput evidence, not quality improvement. Its receipt is bound to the82,557-row manifest and cannot be silently reused for a different full manifest.

**Production48h training has not started.** Full CPU preparation and manifest-specific calibration must pass before freezing actual unique dialogues/exposures/horizon and starting independent supervision. Quality-based early stopping/best-validation retention are not claimed implemented. Evidence: `longrun_protocol/{calibration-01-failure-and-fix.json,parent-calibration-02-accepted.json}` and `cpu-full-01/`.

### Latest user policy: 16K, Long-Context Transduction excluded

Current sequence length is16384. **The entire Nemotron-SFT-Long-Context-Transduction-v1 dataset is excluded from both train and heldout**, including short examples that fit16K; preserve raw inputs for a separately authorized future stage. Agentic/Instruction-Following-Chat/BBQ remain active SFT datasets, with RL Seed reserved separately. Preserve all14 prior prompt-group blocklist entries without fabricating heldout rows from excluded sources. User-cancelled4K generation02 must not restart.

Parent independently accepted the real16K CP1/EP8 pilot: **all8 11 updates/final validation/optimizer-RNG save11/Engine exit0, no OOM**. There are83 training/13 heldout dialogues and47177 supervised targets, including21687 beyond position4096, establishing real longer-context use rather than metadata-only change. CP1 fit; CP2 remains disabled and operationally untested. The pinned SDK has a separate expert grid and Mamba training CP support; do not infer EP limits solely from dense DP degree. GB8 pilot timing is not GB128 calibration.

Full16K CPU preparation runs under fresh cpu-full-03/prepared-full-03 paths. Parent reviewed/froze12 source-only inputs, started `nemochat-nemotron-48h-pipeline-03.service`, and read back its exactPID/argv/WAITING_DATA. It gates full readiness→manifest-specific16K GB128 calibration→measured48h budget and independent main. **The48h production GPU run has not started yet.** Evidence: `seq16k_protocol/parent-pilot-accepted.json`, `longrun_protocol/pipeline-03/parent-start-readback.json`, and `cpu-full-03/FULL_HANDOFF.json`. Previous4K token-budget projections are superseded.

## Sources

[1] https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4/raw/a4593a26ea83a700035bbe0ec1edc9ef3810669d/README.md — Nemotron-SFT-Agentic-v4 (NVIDIA pinned dataset card)
[2] https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4/raw/1cd362945d7a00b136b95b96892bbfea8798686e/README.md — Nemotron-SFT-Instruction-Following-Chat-v4 (NVIDIA pinned dataset card)
[3] https://huggingface.co/datasets/nvidia/Nemotron-SFT-Long-Context-Transduction-v1/raw/90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0/README.md — Nemotron-SFT-Long-Context-Transduction-v1 (NVIDIA pinned dataset card)
[4] https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1/raw/9715a833ae2de4863e321067211f25efe42f6ebe/README.md — Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1 (NVIDIA pinned dataset card)
[5] https://huggingface.co/datasets/nvidia/Nemotron-SFT-BBQ-v1/raw/4efdb045cda11918f29065a2e0fd5000b73b8c22/README.md — Nemotron-SFT-BBQ-v1 (NVIDIA pinned dataset card)
[6] https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/raw/1f8953f7f0935358e3605150d22bc15df062061c/tokenizer.json — NVIDIA Super3.5 tokenizer JSON (pinned; tokenizer only)
[7] https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/resolve/fc067b34f6b514014d2f8bddba855d2e4bfdfba6/tokenizer.json — NVIDIA Super3.5 tokenizer JSON (implementation-config pin)
