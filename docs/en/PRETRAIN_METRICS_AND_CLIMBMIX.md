# Pretraining Metrics and ClimbMix Adoption

Status: historical source-verified evaluation/data design. CPU HF evaluation, bounded acquisition and tiny-model NIAH have since been exercised; target-model GPU/distributed evaluation remains untested. See [execution evidence](../IMPLEMENTATION_VERIFICATION.md). The project is a noncommercial educational tutorial. Metric examples below are arithmetic illustrations, not measured model scores.

## 1. Verified upstream references

nanochat snapshot: `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.

- README: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/README.md
- BPB: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/loss_eval.py
- CORE task scoring: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/core_eval.py
- CORE aggregation/config loading: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/base_eval.py
- Raw token bytes: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/scripts/tok_train.py
- Data/download split: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/dataset.py
- Provenance/repackaging: https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/dev/repackage_data_reference.py
- DCLM report: https://arxiv.org/abs/2406.11794

Read the actual evaluator rather than using the term CORE for any arbitrary benchmark average. nanochat's README emphasizes validation BPB and CORE and records a switch from FineWeb-Edu to ClimbMix. Its README's GPT-2 reference score is not an automatic target or a directly comparable claim for a different evaluator configuration.

The actual nanochat eval bundle was read in memory to verify `core.yaml` and baseline metadata; no inference ran. Bundle URL: https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip . Bytes: 26060410. SHA256: `90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200`. Freeze this hash in the implementation manifest; the URL is mutable. Review benchmark licenses before republishing bundle contents.

## 2. Evaluation dashboard

The selected pretraining experiment uses one WSD trajectory and compares milestone raw weights with PR #5114 minus-sqrt stable-phase WSM using suitable pinned official NeMo Gym tasks. CE/PPL/BPB and existing CORE remain separate diagnostics, not substitutes for the paired Gym protocol. See [experiment contract](PRETRAIN_WSD_WSM_EXPERIMENT.md). Actual target-model training and official benchmark inference remain untested.

| Metric | Meaning | Direction | Role |
|---|---|---|---|
| Heldout LM CE | Mean negative log probability per valid target token | Lower | Training/optimization diagnostic |
| Heldout PPL | Exponential of token CE | Lower | Same-tokenizer LM diagnostic |
| Validation BPB | Model surprisal in bits divided by represented text bytes | Lower | LM-quality diagnostic inspired by nanochat |
| CORE | Equal-task mean of baseline-centered task accuracies | Higher | Separate base-model ICL capability diagnostic inspired by nanochat |
| Task/domain breakdowns | Individual task accuracy and source/domain BPB | Contextual | Detect regressions hidden by aggregates |
| Fixed-prompt samples | Actual completions under frozen decoding settings | Qualitative | Debugging/illustration, not a benchmark |
| NIAH and long QA | Position/length-specific recall and broader long-context ability | Higher | Stage [4] extension/retention diagnostics, not CORE replacements |
| Math/code execution scores | Heldout answer correctness and executable tests | Higher | Targeted capability; not implied by CORE |

## 3. CE and perplexity

For valid next-token targets M, token CE in nats is:

`CE = sum_{i in M} -ln p(x_i | x_<i) / |M|`.

`PPL = exp(CE)`.

Predict each target using only its preceding context. Exclude padding, ignored/truncated labels and loss-masked targets consistently. Keep plain LM CE separate from MoE load-balancing/z-loss or other training regularizers. A training objective containing auxiliary losses is not a valid LM perplexity numerator.

Interpretation: CE measures how surprising the true text tokens are; PPL is the effective uncertainty scale, not percentage accuracy. An illustrative CE of 2 nats/token corresponds to PPL 7.389056. These are not NemoChat results.

PPL depends on tokenizer segmentation, corpus, masks and evaluation context. Different tokenizers can split the same text differently, so token PPL is not a fair cross-tokenizer scoreboard. Report it with tokenizer and evaluation-pack IDs.

## 4. Bits per byte (BPB)

nanochat computes:

`BPB = sum_i NLL_i * eligible_i / (ln(2) * sum_i bytes_i * eligible_i)`.

`bytes_i` is the **raw decoded byte length** represented by the target token; zero for special tokens. `eligible_i` also excludes ignored/padded/masked targets. The numerator uses token NLL in nats; division by ln(2) converts nats to bits. Sum losses and bytes globally, then divide once. Do not average batch/rank BPBs: small-byte batches would receive excessive weight.

Interpretation: fewer bits per byte means the model assigns higher probability to the text relative to its byte content. It is compression-inspired normalization, not the actual size of a produced compressed file. It reduces tokenization-unit/vocabulary-size dependence but does not eliminate corpus, normalization, BOS/context or segmentation effects.

Arithmetic example: 1000 nats over 2000 eligible bytes yields 0.72134752 BPB. This is a worked example, not inference. Fixed heldout raw text is essential for fair comparisons; equal token counts with different tokenizers need not cover equal text.

### Super tokenizer porting requirements

Keep the planned Super tokenizer for later distillation. Do not retrain a nanochat tokenizer merely to use its metrics. Inspect the pinned tokenizer backend and implement exact raw-byte reconstruction for its token representations, including byte fallback and multibyte splits. Do **not** use `len(tokenizer.decode([id]).encode('utf-8'))` without verification: isolated byte tokens may decode as replacement characters; space normalization or cleanup can also change lengths.

Tests: ASCII, whitespace, Unicode/Hangul, partial UTF-8 bytes, punctuation, code, BOS/EOS/added tokens, ignored labels and padding. On known round-trip fixtures, summed ordinary-token bytes must match represented raw UTF-8 bytes under the declared normalization. If exact accounting cannot be established, block BPB reporting rather than silently calling an approximation nanochat-compatible BPB.

Special IDs may still be model context but contribute neither numerator nor byte denominator. Record coverage/mask policy. Eval must restore training state and disable gradient/dropout.

### Megatron distributed requirements

Obtain per-target next-token NLL from vocab-parallel cross entropy without gathering full long-sequence vocabulary logits. TP/EP ranks can contain replicated quantities; CP partitions tokens, DP partitions samples. Sum each unique target exactly once over the appropriate groups, not a naive world-wide all-reduce copied from nanochat's replicated-model DDP evaluator. Use wide accumulators where appropriate and test single-rank vs TP/CP/EP results.

## 5. CORE: base-model capability beyond loss

For task t, let a_t be measured accuracy and r_t the baseline fraction from bundle metadata:

`centered_t = (a_t - r_t) / (1 - r_t)`.

`CORE = mean_t(centered_t)`.

nanochat's CSV stores baselines as percentages, so convert them to fractions first. A 60% score with 25% baseline gives 0.46666667 centered score. A 25% score gives zero; perfect accuracy gives one. Below-baseline scores are negative and must not be clamped away. Some supplied baselines are dataset-specific (e.g. BoolQ 62%), not simply 1/number_of_options.

CORE weights each **task equally**, not each example. Large HellaSwag does not automatically outweigh small COPA. Report raw and centered task scores plus sample counts and uncertainty alongside the aggregate. CORE 0.3 does not mean 30% raw accuracy or 30% of GPT-2's ability.

### Actual task scoring semantics

- Multiple-choice: render few-shot/continuation prompts, score each candidate's continuation, choose lowest **mean per-token NLL**. Replacing this with summed log likelihood, answer-letter generation or byte-normalized likelihood changes the metric.
- Schema: score a common continuation under different context choices and choose minimum mean NLL.
- Language-modeling tasks: check that argmax next-token predictions match **all reference continuation tokens under teacher forcing**. Earlier correct answer tokens are supplied when predicting later ones. This is not free-running answer generation, conventional SQuAD F1, or sampled pass@1.
- Retain the pinned delimiters, BOS handling, few-shot settings and deterministic sampling. Tokenize context+answer jointly and verify target boundaries; standalone prefix tokenization can fail due to token merging.
- Record context truncation and answer preservation; do not silently give base and long checkpoints different effective context and call the change model-only.

The pinned bundle has **22 task entries**:

`hellaswag_zeroshot`, `jeopardy`, `bigbench_qa_wikidata`, `arc_easy`, `arc_challenge`, `copa`, `commonsense_qa`, `piqa`, `openbook_qa`, `lambada_openai`, `hellaswag`, `winograd`, `winogrande`, `bigbench_dyck_languages`, `agi_eval_lsat_ar`, `bigbench_cs_algorithms`, `bigbench_operators`, `bigbench_repeat_copy_logic`, `squad`, `coqa`, `boolq`, `bigbench_language_identification`.

Zero/few-shot HellaSwag are separate entries in the upstream aggregate. The metadata CSV contains additional tasks not selected by `core.yaml`; don't assume all metadata rows belong to CORE. GSM8K, MMLU and HumanEval are **not** entries in this pinned 22-task CORE configuration. Keep their separate evaluation rather than claiming CORE measures all math/code/knowledge.

nanochat's core source includes a SQuAD parity TODO (31% vs 37% reference). Treat our result as **nanochat-compatible CORE**, not unqualified official DCLM parity. Upstream code reuse needs attribution/license review.

### Sampling and cost

Upstream training defaults observed: BPB every 250 steps; CORE every 2000 steps with up to 500 examples/task. Upstream base_eval defaults to 40*524288 tokens per BPB split and no CORE per-task cap. These are source facts, not copied NemoChat budget defaults.

Proposed NemoChat policy: frequent frozen-pack BPB at consumed-token intervals calibrated to evaluation overhead; CORE-quick capped at 100 examples/task at sparse milestones, a 500/example intermediate only when useful, and full pinned pack at Part I handoffs if budget permits. Label capped scores `core_quick`/`core_500`; do not present them as full CORE. Freeze item IDs/prompt IDs and compare paired outputs, reporting sample counts. Restrict repeated tuning on final test tasks.

## 6. ClimbMix: adoption for the educational track

**Decision:** prefer ClimbMix for the 60% general-text pretraining component in this noncommercial educational project. Tutorial quota stays 1.8B, Extended 18B **Super-tokenizer consumed tokens**. Keep code 20%, math/science 10%, long natural text 10% initially, subject to source rights and domain diagnostics. FineWeb-Edu remains fallback and independent heldout control. This is a design-level source substitution, not completed acquisition or a guarantee of higher performance.

Evidence: official card describes English, 400B-token mixture; topic grouping into 1000 groups, advertisement/educational classifiers and weighted quality-filtered mixing. nanochat README records better performance after replacing FineWeb-Edu, but that observation is not proof for our hybrid model, tokenizer or budget. A/B architecture pilot must use the **same frozen data subset**; don't confound architecture with data changes or add a full factorial experiment without approval.

### Provenance and license discrepancy

| Source | Snapshot | Format / advertised license |
|---|---|---|
| `nvidia/Nemotron-ClimbMix` | `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b` | GPT-2-tokenized Parquet; CC BY-NC 4.0, research/development usage |
| `karpathy/climbmix-400b-shuffle` | `915333b4f8b8684f39aeaafea600fea6f43fb703` | Shuffled, detokenized text Parquet; card says MIT |
| `OptimalScale/ClimbMix` | Pin at acquisition if used | Community detokenized text; card says Apache-2.0 |

nanochat repackaging source explicitly identifies NVIDIA as origin, GPT-2 detokenization and seed-42 shuffle. It creates a text-only column, so original URL/document boundaries/source metadata may be unavailable in the repackaging. Preserve revision/shard/row/content hashes and document this provenance limitation; do not claim per-original-document lineage that has been lost.

Community MIT/Apache card metadata is not proof that the underlying NVIDIA dataset's NC/attribution conditions disappear. Noncommercial education makes the official dataset a suitable candidate rather than a blanket blocker; retain original attribution, NC terms and transformations. Review dataset redistribution and trained-weight license separately; code can have its own license. This is not legal advice and does not assert that all model weights automatically inherit CC BY-NC.

For the first tutorial use the convenient text shuffle only after provenance/terms review, with upstream NC conditions explicitly recorded. If derivative permission/metadata remains unresolved, select bounded official tokenized shards and detokenize them locally instead. No commercial/permissive-data claims are made. If use later changes to commercial, reopen the data/rights decision.

### Storage-sensitive acquisition

Do not run the repackaging reference script: it loads/shuffles the whole dataset and can upload; it is historical provenance, not our acquisition path. Do not invoke nanochat's downloader with no quota: its `num_files=-1` runtime selects all training shards despite the CLI help saying disable.

The shuffle repo has 6543 Parquet shards (`00000` through `06542`) according to pinned nanochat code. Reserve `shard_06542.parquet` for validation as upstream does; don't include it in training. Exclude reserved heldout IDs/near duplicates before packing. Use seeded selected shards or bounded row-group reading, limited cache and accepted-token ledger; stop at the actual Super-tokenizer quota after dedup/filtering. Retokenize text with Super; neither GPT-2 IDs nor the advertised 400B counts are our training IDs/counts.

ClimbMix has already been filtered and may contain assembled tokenized segments rather than pristine long documents. Audit duplication, tokenization round trips, language/length/domain and document boundary semantics before additional filtering. Avoid severe generic quality refiltering that destroys the intended mixture. ClimbMix is **not** the sole source for 64K/256K natural long documents.

Validation: fixed ClimbMix heldout for in-distribution BPB; independent FineWeb-Edu/prose, code and math heldouts for cross-domain robustness. Always compare datasets on the same evaluation raw text, not ClimbMix-validation vs FineWeb-validation. Benchmark-contamination scans remain required; already-filtered does not mean benchmark-clean.

## 7. Integration tasks and blog evidence

Historical proposed integration names (not all implemented):
- Actual CPU HF modules are `src/nemochat/eval/{token_bytes,metrics,hf_eval,core}.py` with CLI `scripts/base_eval.py`; tests, bounded data manifests and actual tiny-model predictions are implemented. The original proposed `bpb/core_adapter/base_eval` module names below are not existing APIs. MCore and distributed adapters remain planned.
- `configs/eval/{base_quick,base_full}.yaml`, pinned task/eval bundle and source manifest.
- Tests: NLL/byte sums, special/masked targets, raw UTF-8, distributed parity, teacher-forced answer shift, continuation boundary, mean-vs-sum choice scoring, baseline centering and fixture identity.
- CPU fixtures test formulas/control logic; actual tiny-HF inference has also run. Target-model/backend/distributed parity still requires its own execution.
- Dataset manifest: source/revision/shards, last-shard exclusion, quota ledger, retokenization and source rights.
- Raw task predictions/likelihoods, numerator/denominator/counts; JSON/CSV reports; figures from actual inference only.

Part I blog: BPB vs consumed tokens and useful training wall time, per-domain BPB, CORE/per-task panels, A/B expert/profile evidence, ClimbMix adoption/provenance and NIAH. FLOP-axis graphs are optional only with a documented hybrid/MoE FLOP estimator, not dense-model peak-FLOP substitution. Part II/III reuse the frozen base diagnostics for retention alongside their stage-specific metrics.
