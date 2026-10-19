# Frozen A/B/C evaluation handoff

## Decision contract

- A: original BASE (`checkpoints/exports/nemochat_baby_base_hf_bf16_02`).
- B: **ordinary LM-only, coefficient 1.0**, 1000 accepted updates.
- C: deployed **0.5 LM + 0.5 KD**, 1000 accepted updates.
- Parent must validate matching accepted batch IDs/order, initialization, optimizer/LR, precision and training settings before interpreting B/C. This measures practical recipe value, **not matched-LM-coefficient causal isolation of the KD term**. No third 0.5-LM control is requested or launched.
- BASE→B/C measures total mid-training recipe effects; B→C measures the deployed KD recipe versus ordinary LM training. No quality conclusion follows from training Loss alone.

## Frozen protocol

Canonical artifact: `results/midtraining/offline2b-ablation-eval-01/suite.json`.
Canonical-content SHA256 (excluding its own digest field):
`07609a0cf53e4a63d555a3e50938adfed1cb05471d02b37fb93461e7879af19e`.
Frozen before BASE smoke; exclusive-create writers refuse existing output paths.

| Task | Selection | Prompting | Primary readout |
|---|---|---|---|
| MMLU subset | 10 subjects × first 20 test rows = **200 total** | 5-shot, explicit first five dev rows in published order | summed continuation log-probability, Accuracy |
| ARC-Challenge | first 100 test rows | zero-shot | Accuracy and character-normalized Accuracy (`acc_norm` primary) |
| HellaSwag | first 100 validation rows | zero-shot, official context/ending cleanup | Accuracy and character-normalized Accuracy (`acc_norm` primary) |

MMLU subjects: abstract_algebra, college_computer_science, college_mathematics, formal_logic, high_school_physics, high_school_chemistry, high_school_biology, high_school_world_history, professional_medicine, business_ethics.

This is a **nonstandard, deterministic first-N diagnostic subset**, not full MMLU, not a random representative sample and not a leaderboard result. `suite.samples` contains explicit per-task IDs: MMLU 0–19, ARC/HellaSwag 0–99. Never use `--limit 200` across MMLU subjects (that applies per task). Row indices, native IDs when available, original rows/hashes, gold answers, all five dev examples, exact rendered prompts, each continuation, context IDs and continuation IDs are frozen. No chat template, BOS insertion, generation, answer extraction, padding or truncation. Original BASE tokenizer remains unchanged. Each evaluated model must reproduce frozen vocabulary and encoded benchmark inputs.

## Exact source/data pins

Official EleutherAI lm-evaluation-harness **v0.4.8**, commit
`6d2abda4fd171e68a8789330c4149e37c1ca0bda`.
Fetched primary source: `https://github.com/EleutherAI/lm-evaluation-harness` (commit-bound raw files saved under `research/`). Entry-point reference: `https://lm-evaluation-harness.readthedocs.io/api/entry_points/`.
Actual image-installed fork is `nvidia_lm_eval 25.11.1`, reporting `lm_eval 0.4.8`; installed API/HellaSwag Python hashes and task config hashes are frozen in suite.json. Upstream task YAML, all ten MMLU subject configs, task/model/sampler/scorer source and API metadata snapshots are saved and hashed in `contract.json`. The live ReadTheDocs entry-point fetch returned HTTP 403; the pinned repository has no `entry_points` documentation path. Behavior is grounded in the fetched pinned evaluator/task implementation and actual installed signature/import, not an invented documentation excerpt.

- `cais/mmlu`: `c30699e8356da336a370243923dbaf21066bb9fe`.
- `allenai/ai2_arc`: `210d026faf9955653af8916fad021475a3f00453`.
- `Rowan/hellaswag`: `218ec52e09a7e7462a5400043bb9a69a41d06b76`.

HellaSwag's historical YAML dataset alias `hellaswag` is resolved explicitly to author repository `Rowan/hellaswag`, pinned parquet; no downloaded dataset Python executes. All HF caches are task-local under `/raid`, never `/home`. Metadata snapshots preserve available license/provenance declarations; this handoff is local evaluation, not redistribution approval or proof of clean benchmark contamination.

## Explicit backend deviation and parity evidence

`import lm_eval; from lm_eval import evaluator` succeeds in the pinned runtime. Full **HFLM backend** import fails through optional `peft → transformer_engine` because a deliberately no-GPU container has no `libcuda.so.1`. No installed package was edited, no GPU device was mounted, and no response was fabricated.

`eval.py` is explicitly a **source-derived custom scoring equivalent**, not an official harness run. Preparation compared prompts, request tuples, gold indexing and metric readout against the actual installed ConfigurableTask for **all 400 real selected rows**. Four passing tests additionally execute the downloaded official v0.4.8 `fewshot_context`, `construct_requests`, `process_results` and `_encode_pair` methods on every real frozen row/choice. This separates NVIDIA-fork differences from selected-mode behavior. Unit tensor NLL/shift tests are labeled fixtures, separate from the real model smoke.

Full-vocabulary float32 cross entropy is computed on score-bearing positions of actual BF16 CPU logits. Choice scores sum continuation token log-probabilities without length division; `acc_norm` divides by original choice **character count**, matching official task logic. Save both raw full-vocabulary gold sequence log-probability and choice-conditional gold log-probability; the latter is not LM Loss.

## Fixed heldout LM Loss

Use existing reserved validation corpus from
`datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json`:
code_concepts, algorithmic, formal_logic, climbmix_replay.

**64 windows = 16/source × 1024 target tokens = 65,536 loss-bearing positions**. Each frozen window has 1025 IDs, predicts IDs 1…1024 from preceding context, with the next window sharing exactly one boundary context ID. Per-source flat ranges are `[i*1024, i*1024+1025)`; document-local ranges and token-byte SHA256 are saved. Targets do not repeat. Windows concatenate reserved documents with their existing EOS markers; no reset between documents, no EOS removal and no padding masks. Report pooled token-weighted mean full-vocabulary NLL/PPL and retain per-window/per-source rows. Source balancing is equal here, not training-mixture weighting.

All **eight entire validation bin/idx files** were read through O_DIRECT into owned immutable bytes and matched manifest SHA256. Native index magic/version/int32 sizes/pointers were parsed and checked against full bin lengths; window IDs were copied, not retained as temporary mmap views. The dataset reader-alive hazard is avoided by not creating native reader views at all. All source validation IDs are in vocabulary and IDs 18…29 are absent. Frozen suite hashes are checked before scoring, and inference uses its authenticated in-memory data rather than rereading the faulting source files.

Manifest's historical `train_heldout_token_document_overlap == 0` and reserved-split lineage are retained. This is **inherited exclusion evidence**, not a new exhaustive train-corpus scan or proof of absent benchmark/raw-text contamination. The host read fault is not claimed repaired. Native changing step250/500/750/1000 validation windows are not a paired quality baseline. Model-payload byte integrity stays `not_checked_explicit_user_waiver`; data-byte authentication was **not waived**.

## CPU-only runnable commands

Run from `/raid/yoon/nemochat`. The launcher uses the Engine socket because host Docker CLI is absent. It binds the repository read-only and only its owned results directory writable; runtime UID/GID avoids NFS root-squash, writable HOME/compiler/HF caches remain in that directory. It supplies **no DeviceRequests**, empty CUDA_VISIBLE_DEVICES, NVIDIA_VISIBLE_DEVICES=void, four CPU threads/4 CPUs and **128 GiB memory maximum**. Network is disabled except explicitly authorized preparation.

Pinned image:
`nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf`.
Observed Torch 2.13 prerelease; Transformers 5.12.1; Datasets 4.8.5.

```bash
# Existing freeze must NOT be regenerated/overwritten.
python midtraining/evaluation/ablation_eval/launch.py \
  --name ablation-eval-tests02 --seconds 600 \
  python midtraining/evaluation/ablation_eval/test_suite.py

# Parent-owned post-B release only: fresh per-arm result path/name.
# This invokes the COMPLETE 400-question + 64-window suite (no --smoke).
python midtraining/evaluation/ablation_eval/launch.py \
  --name ablation-eval-A-full01 --seconds 21600 \
  python midtraining/evaluation/ablation_eval/eval.py score \
  --model checkpoints/exports/nemochat_baby_base_hf_bf16_02 \
  --suite results/midtraining/offline2b-ablation-eval-01/suite.json \
  --out results/midtraining/offline2b-ablation-eval-01/A-full01
```

For C, substitute the accepted
`checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01` export and fresh C name/output. For B, parent must supply its actual completed/accepted HF export path; none is guessed here. Model load rejects missing/unexpected/mismatched keys and uses official Transformers `trust_remote_code=False`, `attn_implementation=eager`, BF16 CPU eval/inference mode. Gradients remain absent and parameter versions unchanged. CPU full-suite runtime is unmeasured; the six-hour bound is an operational cap, not a promised ETA. Parent must supervise the foreground launcher with its own bounded execution scope if the tool timeout is shorter. No expensive full scoring was started here.

```bash
python midtraining/evaluation/ablation_eval/compare.py \
  --left results/midtraining/offline2b-ablation-eval-01/B-full01/report.json \
  --right results/midtraining/offline2b-ablation-eval-01/C-full01/report.json \
  --suite results/midtraining/offline2b-ablation-eval-01/suite.json \
  --out results/midtraining/offline2b-ablation-eval-01/B-vs-C.json
```

`compare.py` rejects partial/smoke coverage by default, requires identical suite hashes/IDs/golds, saves paired question/window deltas and 5000 paired bootstraps (seed 1729). MMLU resamples within subject; heldout resamples windows within source. Intervals describe selected-input sampling variation, not training-seed uncertainty or benchmark contamination. A self-comparison of the real smoke report exercises the comparator, with `--allow-smoke`, explicitly labeled selfcheck-only.

Preparation entry point, for a separately approved *new* suite generation only:
`eval.py prepare --suite NEW_PATH`; use the same launcher with `--network`. Do not replace the existing frozen suite after looking at results.

## Executed acceptance and remaining gates

- Frozen all 400 real benchmark prompts/choice IDs; all selected-row official request/gold/metric parity passed.
- Four tests passed in 23.192 seconds; real dataset rows plus separate tensor fixture.
- Actual BASE CPU BF16 smoke: **two questions / eight finite choice scores**, one heldout window / **1024 targets**, clean Engine exit 0, no CUDA available. Both diagnostic questions were incorrect; this two-item result establishes runtime only.
- Actual heldout NLL **2.3266710285244585**, PPL **10.243783446046432**; total scorer elapsed **97.54535006301012 s**. Evidence: `base-smoke01/report.json`, launcher `.log` and `.inspect.json`.
- Remaining: parent's B/C checkpoint/export acceptance, matched accepted-batch/settings audit, complete A/B/C scoring and paired comparison. Nothing here establishes KD benefit yet. No GPU allocation, training, other child's artifact edit/wait, publication or root-ledger write occurred.
