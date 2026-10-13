# NemoChat: Long/Short Alternating CPT and NIAH Evaluation Plan

Status: historical source-verified design. Tiny-HF CPU inference and measured NIAH heatmaps have since been exercised (six requests, 0/6 matches); see [execution evidence](../IMPLEMENTATION_VERIFICATION.md). NemoChat GPU long-context training/inference and target-model heatmaps remain untested.

## 1. What the Nemotron 3 Super report actually says

Source: *Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning*, section 2.6, PDF page 16:
https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf

| Property | Reported setting |
|---|---|
| Learning rate | Constant 4.5e-6 |
| Global batch size | 16 |
| Parallelism | CP64, TP2, EP64 on GB200 |
| LC blend | 20% long-document QA, 80% downscaled Phase 2 data |
| First LC stage | 1,048,576-token sequences, 34B tokens |
| Second LC stage | Alternating 1M and 4K sequences, 17B tokens |
| Motivation for second stage | Mitigate minor regression on math-related benchmarks |

The report does **not** specify, in this paragraph, the alternation frequency, long/short token ratio, separate second-stage batch sizes, or implementation details. Do not assert that it used exactly one long update followed by one short update with equal token exposure.

Section 2.7 says the base-model long-context evaluation uses **RULER with 100 samples per task**. A standalone vanilla NIAH heatmap is an additional NemoChat diagnostic, not a claim of reproducing the report's entire evaluation suite.

The previous NemoChat 8K/16K/32K/64K training-budget table was a proposed curriculum, not the Super schedule. Replace it with the report-inspired two-stage design below. Keep intermediate lengths for correctness and resource checks rather than claiming they are reported Super training stages.

## 2. Single-node adaptation

Goal: keep the reported **long-only extension -> long/short recovery** pattern, while changing the long length, budget, batch sizes, and topology to suit a B300 8-GPU student.

### Main schedule

| Stage | Sequence lengths | Tutorial budget | Extended budget | Purpose |
|---|---|---:|---:|---|
| LC-0 | 8K, 16K, 32K, 64K smoke | Small capped checks, separately logged | Same | CP/masking/state/throughput correctness; not a full curriculum |
| LC-1 | 64K long-only | 67M consumed tokens | 670M | Acquire target-length capability |
| LC-2 | Alternate 64K and 4K optimizer updates | 33M | 330M | Retain long ability while recovering short math/code capability |

The approximately 2:1 LC-1/LC-2 budget split is inspired by 34B/17B, not a demonstrated scaling law. Existing long-context budgets remain 100M and 1B, excluding explicitly reported smoke overhead. Budget caps are approximate at optimizer-update granularity; record actual counts.

Start with a constant LR candidate of 4.5e-6 to match the report's convention, but validate student loss/grad stability and short-task regression. The same LR is not guaranteed optimal for a randomly initialized smaller derivative. Preserve optimizer state across LC-1/LC-2 rather than resetting it between lengths.

For the long stream, start with **20% source-grounded long-document QA and 80% selected downscaled pre/mid replay** by consumed tokens. The replay must preserve valid document boundaries and Mamba reset semantics; independent documents are not automatically a coherent single 64K context. Use source spans and verified QA answers, not unverifiable long teacher prose. Public data substitutes are not the original undisclosed QA dataset.

For the short stream, favor 4K math/code/general replay from the student's pre/mid distributions. This short-stream emphasis is a NemoChat design choice, not a reported Super mixture. Maintain separate dataset-mixture and length-mixture ledgers.

After 64K passes quality gates, optionally run **128K long-only -> 128K/4K recovery**, then **256K long-only -> 256K/4K recovery**. Each is a separately budgeted and resource-gated experiment. A 1M inference smoke is not equivalent to 1M CPT; 1M training is not promised on one node.

### Why the original topology cannot simply be copied

CP64/TP2 already requires a topology larger than eight GPUs. CP and EP are not independent factors that must always be multiplied together: they use different process-group mappings. Inspect Megatron's actual group construction rather than multiplying CP64*TP2*EP64 to infer the original node count.

For the student, investigate TP1 with CP2/4/8 and compatible EP groups. These are **candidate configurations**, not executed configurations. Mamba CP requires the local head/group divisibility checks in the selected implementation. Our 48 heads/8 groups are plausible candidates for these CP factors but passing arithmetic alone does not validate attention/MoE groups, kernels, or optimizer sharding.

At CP8, a 64K sequence has 8K sequence tokens per CP rank, whereas a 1M sequence has 128K per rank. This is only a partition-size comparison, not a full memory estimate: attention attends to the global context and communication/activation costs remain.

Training CP support does not establish inference CP support. The inspected Mamba mixer contains restrictions on CP inference decode/dynamic prefill; vLLM serving and training CP are separate validation paths.

## 3. Alternation policy: steps are not tokens

For equal global sequence batch sizes, one long optimizer step followed by one short step gives:

| Long length | Short length | Long share of consumed tokens |
|---:|---:|---:|
| 64K | 4K | 94.12% |
| 256K | 4K | 98.46% |
| 1M | 4K | 99.61% |

Thus equal step counts do not mean half the tokens are short replay.

**NemoChat baseline:** during LC-2 use equal useful global token budgets per long/short optimizer update, then alternate update types. With one global long sequence, the equivalent short sequence counts are 16 at 64K, 64 at 256K, and 256 at 1M. These are arithmetic examples: actual global batch sizes must satisfy DP/microbatch constraints, and should be scaled together if the minimum long global batch exceeds one.

Different microbatch counts/gradient accumulation may be needed for each stream. Normalize gradients by the intended loss-bearing token objective and do not give a small short microbatch the same weight as a large long microbatch accidentally. Equal useful tokens does not imply equal wall-clock or compute.

Start with a 50:50 long/short **token** share during LC-2. Adjust only after short-task retention and long-context performance are measured. This ratio is our design choice, not documented in Super section 2.6.

## 4. Implementation boundary and correctness tests

The inspected Bridge GPT dataset is configured by a fixed `seq_length`; the existence of Megatron variable-length infrastructure does not prove a ready-made alternating CPT launcher.

Preferred extension:
1. Build two revision-pinned, frozen local mmap streams: long and 4K.
2. An update-level controller chooses one stream and broadcasts the schedule choice across all ranks.
3. Use one length per optimizer update, initially, not a mixed padded 64K tensor containing short examples.
4. Keep topology constant while alternating so process groups/optimizer ownership do not change every step.
5. Adjust microbatch count to the target token batch, preserve optimizer/scheduler/RNG, and log counters separately by stream.
6. Disable CUDA graph capture for the first correctness run. Later investigate two shape-specific graph caches rather than reusing a graph captured at the wrong shape.
7. Save controller position, length-stream choice, both data cursors and quota counters in checkpoints. Test resume at a long/short boundary and mid-rental recovery.

Fallback: alternate bounded long and short run segments with checkpoint resumes using a common compatible topology. This is slower and is coarse-grained alternation, which must be labeled as such. Never silently reset Adam or the LR schedule each segment.

Test targets, not files implemented yet:
- `tests/test_lc_schedule.py`: deterministic choice and cross-rank agreement.
- `tests/test_lc_token_accounting.py`: unequal accumulation and correct useful-token normalization.
- `tests/test_lc_resume.py`: resume schedule/cursors/optimizer exactly.
- `tests/test_packed_mamba_reset.py`: no unintended state leakage across independent examples.
- `tests/test_lc_cp_parity.py`: short gold sequence CP1 vs candidate CP, forward/backward within dtype tolerance.

## 5. NIAH evaluation specification

### 5.1 Reproducible fixture generation

Use an English key/value needle with a freshly generated random value, e.g. a unique record ID and secret value. The question must reveal the key but not the answer. Avoid one famous fixed needle whose answer could have been memorized.

Haystacks come from license-approved heldout text/code with recorded revisions, or a clearly labeled procedural-noise control. Use distinct needles and distractor content across seeds. Repeated filler is a smoke fixture, not the only final benchmark.

Compute context and insertion depth using the **actual student tokenizer**. Count the entire serialized prompt, including instructions, special tokens, query and chat/base template. Reserve answer tokens within the engine's model-length limit; prevent truncation. Validate the needle survives final serialization exactly once for the single-needle task.

Depth is defined relative to the haystack region, not the final query. Record both requested and actual token offsets/depth after sentence-boundary alignment. 100% means the end of the haystack before the query, never after the query.

### 5.2 Evaluation grids

- Coarse development smoke: lengths 4K/32K/64K, depths 0/25/50/75/100%, three seeds per cell: **45 requests/checkpoint**.
- Main 64K grid: lengths 4K/8K/16K/32K/64K, depths 0,10,...,100%, ten seeds per cell: **550 requests/checkpoint**.
- Extended grid adds 128K and 256K: **770 requests/checkpoint** with the same depth/seed counts.
- Ten trials per cell is an economical diagnostic, not a high-confidence estimate. Add repetitions to critical cells for final reporting and report sample counts/confidence intervals.

Separate development fixtures from the final frozen evaluation pack. Use the same final fixture pack across compared checkpoints. Evaluate beyond a checkpoint's configured maximum only as explicitly labeled extrapolation after backend correctness checks; unsupported lengths are N/A, not zero accuracy.

Greedy decoding baseline, a small fixed output cap suitable for a short value, and a stable answer-only prompt. For base checkpoints use a completion-style prompt/answer prefix rather than assuming SFT chat compliance. Do not compare base and chat templates as if prompt differences were model-only differences.

Primary metric: normalized exact match on the extracted value. Normalize only explicitly documented formatting/whitespace; preserve case and punctuation when they are part of the generated value. Use no LLM judge for this primary retrieval score. Track malformed answer, incorrect answer, timeout, OOM, truncation, and API errors separately.

### 5.3 Checkpoints

Run the coarse pack periodically by consumed-token/evaluation-time budget, not every training step. Run the main pack at:
1. pre-LC checkpoint;
2. post-LC-1;
3. post-LC-2;
4. post-SFT;
5. post-RLVR/MOPD where resources allow.

During RL/MOPD evaluation, unload the teacher when possible so it does not compete with the NIAH student inference. Long evaluation cost is dominated by input prefill, not the tiny generated answer; record evaluation GPU-hours/tokens/latency separately.

### 5.4 Figure deliverables

Required generated outputs from **actual model predictions**, never illustrative fabricated scores:

- `results/niah/<checkpoint>/predictions.jsonl`: per-request input hash, source/seed, length/depth, response, gold answer, score/status, timing and model/backend metadata.
- `results/niah/<checkpoint>/cells.csv`: successes/trials, error counts, exact-match rate and confidence interval per cell.
- `figures/niah/<checkpoint>_heatmap.png` and `.svg`: x=context length, y=needle depth, fixed 0..1 accuracy color scale across checkpoints, cell n and score annotations; missing/unsupported cells shaded distinctly.
- `figures/niah/lc_checkpoint_comparison.png`: aligned pre-LC/post-LC-1/post-LC-2 heatmaps.
- `figures/niah/accuracy_vs_length.png`: depth-aggregated accuracy with uncertainty and disclosed weighting.
- `figures/niah/long_vs_short_retention.png`: NIAH performance plus separate short math/code retention panels across checkpoints. Do not place different metrics on an unlabeled shared axis.

Each figure includes checkpoint ID, decoding/template settings, tokenizer/backend revision, evaluation pack hash and trial counts. Save an English caption/limitations and a separate Korean explanatory caption. Derive all plots solely from raw results, so a reviewer can regenerate them without access to training logs.

Planned modules:
- `src/nemochat/eval/niah_generate.py`
- `src/nemochat/eval/niah_run.py`
- `src/nemochat/eval/niah_score.py`
- `src/nemochat/eval/niah_plot.py`
- `configs/eval/niah_{smoke,64k,256k}.yaml`
- `tests/test_niah_fixture.py`, `tests/test_niah_scoring.py`, `tests/test_niah_plot_inputs.py`

CI may test plotting on explicitly labeled deterministic unit-test fixtures. Such test figures are not measured NemoChat benchmark results. Actual tiny-HF control checkpoints have been queried and measured smoke figures exist; they are not NemoChat long-context capability results. Target-model figures require target-model execution.

## 6. Gates beyond NIAH

Single-needle recall does not prove long-document reasoning or agentic coding. RULER explicitly shows that models can do well on vanilla NIAH yet degrade on more demanding tasks.

Use NIAH as a diagnostic alongside multi-needle/distractor retrieval, RULER multi-hop/aggregation, source-grounded long QA, and cross-file coding with executable tests. Track 4K math/code and general heldout NLL through LC-1/LC-2 so the purported retention benefit is measured rather than assumed.

Before blaming long-context training, ensure the model can solve the corresponding 4K task. Weak from-scratch checkpoints may fail NIAH from insufficient instruction/retrieval capability; a low long score alone does not isolate a context-length failure.

## 7. Evidence and support boundaries

- Super report, section 2.6/page 16 and section 2.7: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf
- Bridge Super recipe includes a CP2/32K configuration, not this proposed 64K/4K controller: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_super.py
- Bridge fixed dataset length / packing infrastructure: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/config.py
- Mamba CP head/group divisibility: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py
- Separate inference CP restrictions: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_mixer.py
- RULER task coverage and vanilla-NIAH limitation: https://github.com/NVIDIA/RULER

Pin compatible versions/commits before implementation. Current-source support is evidence for feasibility, not an end-to-end B300 validation. GPU execution is still pending.
