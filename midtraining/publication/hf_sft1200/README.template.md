---
library_name: transformers
pipeline_tag: text-generation
base_model: eagle705/nemochat-10BA1B-BF16-mid-super3.5-distill
base_model_relation: finetune
language:
- en
- ko
datasets:
- nvidia/Nemotron-SFT-Instruction-Following-Chat-v4
- nvidia/Nemotron-SFT-Agentic-v4
- nvidia/Nemotron-SFT-BBQ-v1
tags:
- nemochat
- supervised-fine-tuning
- bf16
- mamba
- mixture-of-experts
- research
license: other
license_name: upstream-constraints-no-additional-grant
license_link: https://huggingface.co/eagle705/nemochat-10BA1B-BF16-SFT-1200/blob/main/USE_CONSTRAINTS.md
---

# NemoChat — actual FINAL SFT1248 (repository name SFT-1200)

**Private, noncommercial educational research checkpoint. The requested repository name ends in `SFT-1200`, but the weights are the actual FINAL native `iter_0001248`: 1,248 cumulative SFT optimizer updates, not step 1,200.** No rounding of the actual milestone is implied. This is a model-only HF BF16 safetensors release, not a native training/resume bundle. [한국어 모델카드](README.ko.md).

**Instruction-following is unreliable.** All six new, once-only English/Korean chat generations reached the 128-token cap without EOS. They include repetition, incorrect Korean responses, ignored formatting constraints and invalid JSON. Finite execution does not establish usefulness, safety or production readiness. The `en`/`ko` metadata describes intended/tested input languages, not proven language quality.

## Exact lineage and what is included

1. BASE: completed native pretraining step 19,074.
2. C: LM+KD mid-training, actual step 1,000 from BASE (524,288,000 processed positions; 999 positive-LR calls and one initial zero-LR call). Exact teacher: **`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`**, revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`, the 88-layer **Base**, not Instruct, text-backbone view. Its cached objective was `0.5 × LM + 0.5 × Ghost-tail KL`, globally normalized T=1, Top32 FP16 log probabilities plus residual tail; no online teacher. See the [inherited C model-card provenance](training/inherited_midtraining_modelcard.md) for data, formula and configuration. Its historical pending comparisons and old runtime limits describe that earlier release, not this checkpoint.
3. D: SFT updates 1–905, initialized from **C model weights only**, with fresh SFT optimizer/scheduler/RNG/input cursor.
4. E, **this release**: native **full-state resume at 905**, followed by **343 updates, 906–1248**. Optimizer, scheduler, RNG and cursor were restored; there was **no optimizer reset at 905**. The restored input cursor was 115,840 conversations. The final native model+optimizer+RNG save completed cleanly; the source training Engine exited 0 at **October 19, 2026, 14:39:16 KST**. Only the model is published.

Failed calibrations 03/04 and discarded runtime05 calibration updates are **not** part of the 1,248-update accepted trajectory or its weights. No averaging, SFT905 weight substitution, additional training, RL/RLVR or WSM merge was performed for this publication. **SFT uses selected-assistant LM loss plus native router auxiliary behavior, with no KD teacher in its SFT loop.**

## Architecture and training settings

| Property | Actual value |
|---|---|
| Official HF class | `NemotronHForCausalLM`, Transformers **5.12.1**, `trust_remote_code=False` |
| Learned parameters | **9,790,967,416**; approximately 964,054,648 active under the inherited dense-plus-top-k **weight-count convention, not FLOPs** |
| Hybrid topology | 52 serialized Mamba/attention/latent-MoE blocks; hidden 1024, latent 512 |
| Experts | 128 routed / top 6; routed width 3072, shared expert width 1024 |
| Attention | 16 heads / 2 KV groups / head dimension 128 |
| Vocabulary / embeddings | 131,072 / untied; no resizing or row permutation |
| Precision | BF16 with **23 FP32 router correction buffers** |
| Training sequence length | **16,384**; not a demonstrated usable context length |
| Native runtime | Megatron Bridge 0.6.1; native distributed AdamW/DDP |
| Parallelism / batch | EP8, TP1, PP1, CP1; global batch 128, micro batch 1; eight B300 GPUs |
| LR / min LR / weight decay / seed | `1e-5` / `1e-6` / `0.01` / `42` |
| Frozen sampler/scheduler horizon | **34,135 updates**; early time caps did **not** rebase this horizon |
| Final observed optimizer-boundary LR | `9.99900171673013e-6`; post-scheduler record `9.998992703267726e-6` |

For updates 1–905 the callback LR and SDK printed post-step LR are retained as distinct streams; **no reconciled actual optimizer-boundary LR is claimed**. For 906–1248, actual optimizer-step parameter-group and post-scheduler records were explicitly logged. A future/post-scheduler LR is not the current applied LR. See [hyperparameters](training/hyperparameters.json) and [all 1,248 observations](training/observed-metrics.csv).

## SFT corpus, supervision and accounting

The frozen prepared pool contained **4,369,280 training conversations**, **3,219,012,753 assistant targets** and **11,383,390,631 nonpadding tokens**. The separate heldout pool contained **45,090 conversations / 33,153,239 assistant targets**. These are **pool counts**, not amounts all consumed by this checkpoint.

| Source / pinned revision | Prepared train conversations |
|---|---:|
| NVIDIA Instruction-Following-Chat-v4, `1cd362945d7a00b136b95b96892bbfea8798686e` | 3,800,207 |
| ↳ chat / response-length control / multi-turn | 895,553 / 263,314 / 2,641,340 |
| NVIDIA Agentic-v4, `a4593a26ea83a700035bbe0ec1edc9ef3810669d` | 563,131 |
| ↳ tool_calling / knowledge_agent | 511,106 / 52,025 |
| NVIDIA BBQ-v1, `4efdb045cda11918f29065a2e0fd5000b73b8c22` | 5,942 |

The exact dataset identifiers are in the YAML and [data provenance](training/data_provenance.json). These prepared-pool counts are **not measured source proportions among consumed examples**. Long-Context-Transduction was excluded and the reserved RL seed was **not used in SFT**. Whole oversized conversations were dropped, not truncated. No real training conversations were repeated. Selected assistant messages only are supervised: user/system/tool content and headers are masked. Reasoning content is omitted in nonthinking mode, while the official empty think markers are preserved. Full source files were pinned; malformed source-schema conversations were quarantined, not repaired. Existing reservations and exact-match checks are not exhaustive contamination clearance.

**Actually consumed across SFT1–1248:**
- **159,744 unique conversations** (`1248 × 128`), without cycling/repetition.
- **117,409,886 assistant-target exposures**, summed from accepted native logs.
- **2,617,245,696 processed positions INCLUDING padding** (`1248 × 128 × 16384`). These are **not supervised tokens**, unique text tokens or completion of the prepared corpus.

### Tokenizer and chat protocol

Exact migrated Super3.5 tokenizer assets at revision **`fc067b34f6b514014d2f8bddba855d2e4bfdfba6`**, including `chat_template.jinja`, are preserved byte-for-byte from the accepted export. Reserved-alias migration did not change embedding rows or nonrenamed IDs. [Upstream tokenizer README](TOKENIZER_UPSTREAM_README.md) is retained as attribution, **not** this model's card or evidence that its tokenizer source model's capabilities were inherited.

Use the official `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` path. The inspected tokenizer EOS/pad is **`<|im_end|>`, ID 11**. In this export `generation_config.json` already has EOS/pad ID 11; the example explicitly sets the inspected IDs anyway (an explicit override, not an unreported different stop rule). The tokenizer's large `model_max_length` is **not** a validated context claim.

## Training observations

![Accepted combined SFT training trajectory](graphs/sft1248-training-summary.png)

Raw native Training LM Loss and a **trailing 16-update mean**, pre-clip Grad Norm and distinct observed LR streams are shown. The dotted step905 boundary is full-state resume, not fresh initialization. Eight ranks were checked for a contiguous, matching accepted 1–1248 trajectory. Calibration attempts are excluded. First16/last16 mean Training LM Loss: **2.964211 / 2.045009**; this is descriptive, not a cross-stage objective comparison.

Native validation used the same seeded, source-stratified **fixed 128-dialogue window**, not all 45,090 heldout conversations: recorded Validation LM Loss at steps **250: 2.299372**, **500: 2.132026**, **750: 2.028384**, **1000: 1.954848**. **No final-1248 validation point exists and none is invented.** This validation is distinct from the raw pretraining-retention NLL below. [Raw observations](training/validation-observations.json).

## Frozen five-arm benchmark — bounded descriptive evidence

![Five-arm question benchmark](graphs/benchmark-five-arms-SFT1248.png)
![Separate pretraining-retention NLL and PPL](graphs/heldout-five-arms-SFT1248.png)

| Arm | MMLU acc /200 | ARC-Challenge acc_norm /100 | HellaSwag acc_norm /100 | Retention NLL /65,536 targets | PPL |
|---|---:|---:|---:|---:|---:|
| A BASE19074 | 24.0% | 33.0% | 52.0% | 2.192594311851152 | 8.958423939014796 |
| C LM+KD1000 | 22.5% | 35.0% | 52.0% | 1.278155480928284 | 3.590011769576160 |
| B LM-only1000 | 20.0% | 32.0% | 52.0% | 1.288966996663131 | 3.629035812717152 |
| D SFT905 | 22.0% | 33.0% | 48.0% | 1.373007705764395 | 3.947204890471098 |
| **E FINAL SFT1248** | **22.5%** | **35.0%** | **48.0%** | **1.399376922740647** | **4.052674050965295** |

Only E was newly evaluated for the latest milestone; A/C/B/D results were reused unchanged, not rerun. E versus D changes are +0.5 / +2.0 / 0.0 percentage points, while retention NLL and PPL **worsen**. This is **not evidence of general improvement, a controlled KD ablation or causality**.

400 questions / 1,600 choices: MMLU ten subjects ×20 test questions, 5-shot; ARC100 test and HellaSwag100 validation, 0-shot. Raw prompts, **no chat template**, full-vocabulary likelihood; MMLU acc, character-normalized ARC/HellaSwag acc_norm. The evaluator derives methods from official task sources; **official harness backend was not executed**. Retention uses 64 fixed pretraining-heldout windows /65,536 targets, not SFT validation. The suite and all choices were frozen; reserved-alias compatibility checked every frozen encoding and the absence of renamed IDs from retention windows. Initial pre-load comparison-tokenizer/receipt-permission failures were retained; the successful E scoring ran once. These are small subsets, **not full benchmark or leaderboard scores**. Contamination, statistical significance, chat quality and tool execution safety are not established. [English report](evaluation/FINAL_REPORT.md), [Korean report](evaluation/FINAL_REPORT.ko.md), [CSV](evaluation/scores.csv), [JSON](evaluation/summary.json), [full E report](evaluation/E-report.json).

## New actual English/Korean generations from FINAL1248

The **same six predefined prompts** as the earlier diagnostic were frozen **before** loading this actual1248 export, then executed **once each**, with no retries, best-of selection, prompt rewriting or expected-answer substitution. Greedy `do_sample=False`, `num_beams=1`, seed42, `max_new_tokens=128`, official nonthinking chat template, BF16/eager, one B300, offline pinned Transformers5.12.1. All loaded tensors and generated scores were finite, DynamicCache was returned, and loading-info error lists were empty. **All six hit the 128-token cap; EOS reached 0/6.** This is a manual failure diagnostic, not a quality benchmark.

The following four representative **verbatim prefix excerpts** deliberately retain failures. Each excerpt is shortened for display only; full unedited continuations for **all six**, prompt/generated token IDs, settings, timing and cache/EOS receipts are in [examples/examples.json](examples/examples.json), with [frozen prompts](examples/prompts-predeclared.json) and [pre-load generation plan](examples/generation-plan-before-load.json). No old905 generated text was reused.

@@EXAMPLES@@

## Runnable loading and generation (private repository)

Authenticate with an authorized Hugging Face login first; never embed credentials in this snippet. Tested runtime: official Transformers **5.12.1**, Torch **2.13.0a0+8145d630e8.nv26.06**, Python3.12, B300 GPU, BF16/eager. About20GB of weights plus runtime/cache memory is required. Other runtimes/backends are not implied supported.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
repo = "eagle705/nemochat-10BA1B-BF16-SFT-1200"
# Set this to the immutable published commit from the Hub history for reproducibility.
revision = "main"
tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=False)
model = AutoModelForCausalLM.from_pretrained(
    repo, revision=revision, dtype=torch.bfloat16,
    trust_remote_code=False, attn_implementation="eager",
    device_map={"": "cuda:0"},
).eval()
messages = [{"role": "user", "content": "하늘은 왜 파란가요? 한 문장으로 답하세요."}]
inputs = tok.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    enable_thinking=False, return_dict=True, return_tensors="pt",
).to(model.device)
assert tok.eos_token == "<|im_end|>" and tok.eos_token_id == 11
cfg = GenerationConfig.from_pretrained(repo, revision=revision)
cfg.update(do_sample=False, num_beams=1, max_new_tokens=128, use_cache=True,
           eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
torch.manual_seed(42)
with torch.inference_mode():
    output = model.generate(**inputs, generation_config=cfg)
print(tok.decode(output[0, inputs["input_ids"].shape[-1]:],
                 skip_special_tokens=True, clean_up_tokenization_spaces=False))
# This diagnostic can return incorrect/repetitive text and hit the cap, as recorded above.
```

No remote model code is required; text-only AutoTokenizer/AutoModel is tested, not an inherited multimodal AutoProcessor. Optional Mamba fast kernels are disabled (`use_mamba_kernels=False`, native scan lower bound preserved as `time_step_min=0`). Export uses reviewed pinned Bridge-registry-derived global-DCP adapters, **not** Bridge's ordinary export CLI: 619 native model records →6,289 serialized records →447 official runtime records, including grouped experts; no randomized missing-key fallback. Full official GPU reload, finite forward, cached generation and bounded likelihood scoring were exercised.

## Verification boundaries and use constraints

- **Native/HF numerical logit parity: not run.** Finite export/reload/benchmark is not parity proof.
- **Source checkpoint payload SHA/CRC forensic reads/copies: not checked, explicit user waiver.** Necessary Hub protocol hashing is allowed but does not change this source-audit status. Remote verification uses exact revision/inventory and small-asset byte equality, **not weight redownload**.
- No native optimizer/RNG/DCP files, raw corpus rows, teacher weights/cache, credentials or local caches are distributed. The sanitized [publication manifest](publication_manifest.json) binds the actual step1248 source and accepted native save to this model-only export.
- No full-context quality test, safety/red-team evaluation, production/service guarantee, native-kernel parity, vLLM/TensorRT-LLM compatibility, multilingual proficiency or reliable tool-use claim is made. English and especially Korean generation failures are visible, not hidden.
- This private publication is **noncommercial educational/research use only**, with **no additional learned-weight license grant**. Preserve upstream NVIDIA dataset/model/tokenizer notices, inherited ClimbMix **CC-BY-NC4.0** restrictions, selected Specialized-v1.1 **CC-BY4.0** attribution, and applicable **OpenMDW1.1** terms for the pinned teacher/SFT data. The SFT manifest records OpenMDW1.1 for all selected SFT sources. A repack's MIT/Apache code/data label does not relicense weights or waive upstream restrictions. Read [USE_CONSTRAINTS.md](USE_CONSTRAINTS.md); this is not legal authorization or a guarantee of trained-weight rights.
