# Official NeMo Gym evaluation of pretraining checkpoints

For new CLI runs, use repository-root paths and fresh artifacts; see [runtime layout](RUNTIME_LAYOUT.md). Inputs belong in `datasets/`, checkpoints in `checkpoints/`, reports/traces/plots in `results/`, and text/W&B logs in `logs/`. Create required parent directories, not final no-overwrite output directories. Checked-in fixtures/configs and historical evidence paths remain unchanged.

## What is implemented (and what is not)

`rl/evaluation/gym_eval.py` is a **result importer and paired comparison**, not a new benchmark runner or scorer. Run the official Gym evaluator separately, then import its numeric metrics and original JSONL rollout artifacts. It makes no model requests, purchases no judge calls, downloads no datasets, and does not convert failed inference into incorrect answers. Existing CE/BPB and CORE diagnostics remain unchanged: use `scripts/base_eval.py` alongside this evaluation, not in place of it.

The source pin is **NVIDIA-NeMo/Gym**, commit `804c12a7d56a151a98ae150a352350e3c856ba47`. `NVIDIA/NeMo-Gym` is not the authoritative repository. Immutable sources:

- [official CLI execution and aggregation](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/nemo_gym/cli/eval.py)
- [rollout collection, identities and official aggregation](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/nemo_gym/rollout_collection.py)
- [RULER pretraining protocol](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/benchmarks/ruler/PRETRAIN.md)
- [RULER pretraining config](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/benchmarks/ruler/config_pretrain.yaml)
- [pretraining completion serving profile](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/responses_api_models/local_vllm_model/configs/pretrain_text_completion.yaml)
- [RULER verifier](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/resources_servers/ruler/app.py)
- [MMLU config and boxed prompt](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/benchmarks/mmlu/config.yaml), [prompt](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/benchmarks/prompts/eval/aai/mcq-4choices-boxed.yaml)
- [GSM8K config](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/benchmarks/gsm8k/config.yaml), [math resource config](https://github.com/NVIDIA-NeMo/Gym/blob/804c12a7d56a151a98ae150a352350e3c856ba47/resources_servers/math_with_judge/configs/math_with_judge.yaml)

## Available tasks and base-model suitability

The pinned `benchmarks/` tree includes `ruler`, `ruler2`, `mmlu`, `mmlu-redux`, `mmlu_pro`, `mmmlu`, `global-piqa`, `gsm8k`, `hendrycks_math`, `math-500`, `gpqa`, `aime24/25/26`, `ifeval`, `human_eval`, `mbpp`, `livecodebench`, translation, long-context, multimodal and agentic tasks. These are **not interchangeable with classic base-LM likelihood benchmarks**. The source tree does not contain `benchmarks/hellaswag/config.yaml` or `benchmarks/arc_challenge/config.yaml` at this pin (both exact-path reads returned 404). Do not silently substitute other tasks.

| Candidate | Official protocol/scorer evidence | Pretraining interpretation |
|---|---|---|
| `ruler/config_pretrain` | Native vLLM `/v1/completions`; no chat template; shared RULER verifier | Best source-verified base/pretraining path. Long context must fit the actual checkpoint, not an arbitrary advertised length. |
| `mmlu` | Generated boxed letter, strict single-letter boxed MCQA grading, one repeat | Usable generation diagnostic if the model can follow that format; **not** option-likelihood MMLU. Missing boxed answers are official model-format failures, not evidence that knowledge is absent. |
| `gsm8k` | Generic math prompt, math-with-judge resource; inherited `should_use_judge: false` | Mathematical generation diagnostic. The resource name alone does not imply paid judging. Preserve actual resolved settings; never enable a judge implicitly. |
| `ifeval` | Instruction-following benchmark resource | Primarily instruction-following, confounded for an unaligned base checkpoint. Not a default pretraining measure. |
| Code, agent, multimodal, long-context chat tasks | Task-specific environments, tools, sandboxes or model capabilities | Inspect each config and runtime first. No claim that all work with NemoChat. |

RULER's verifier matches reference substrings case-insensitively: QA takes any reference match; other subsets average matches over all references. Keep that official verifier. `default` preparation concatenates answer prefix directly; `base` inserts a newline. The prepared rows set per-family output limits: NIAH 128, VT 30, CWE 120, FWE 50, QA 32. These modes are distinct protocols.

## Official execution contract

Use a **separate reviewed, installed, pinned Gym checkout**. This tutorial does not install Gym or run its preparation scripts. Preparation can install dependencies, create virtual environments, download data, and run subprocesses. The bundled completion profile sets `trust_remote_code: true`; inspect checkpoint code and explicitly set it false when supported. NemoChat checkpoint files must first be exported to a tokenizer/model format actually supported by the selected vLLM runtime. A training checkpoint or successful HF CPU forward pass is not proof of vLLM architecture support.

The following are source-verified command templates, **not commands exercised here**. Run from the pinned Gym checkout after installation and hardware validation. Replace paths and use a real supported context length (4096 here is illustrative, not a validated checkpoint limit).

```bash
NEMOCHAT_ROOT=/absolute/path/to/nemochat  # NemoChat root, not the Gym CWD
git checkout 804c12a7d56a151a98ae150a352350e3c856ba47
gym eval prepare --benchmark ruler/config_pretrain \
  "+prepare_script_args.model='$NEMOCHAT_ROOT/checkpoints/exports/step_100/raw'" \
  '+prepare_script_args.length=4096' \
  '+prepare_script_args.data_format=default'

gym eval run --benchmark ruler/config_pretrain \
  --model "$NEMOCHAT_ROOT/checkpoints/exports/step_100/raw" \
  --output "$NEMOCHAT_ROOT/results/gym/step_100/raw/ruler.jsonl" \
  --split benchmark --temperature 0 --top-p 1 \
  ++reuse_existing_data_preparation=true \
  ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.max_model_len=4096 \
  ++policy_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.trust_remote_code=false
```

Repeat against the merged export at the **same training step**, with its own fresh output path and identical prepared dataset. Reuse one tokenizer and one byte-identical input file; do not independently regenerate randomized RULER inputs for each model. Save the collated/materialized input JSONL actually used by collection, not merely an earlier preparation file if processing changed it. The importer requires zero-based contiguous task indices and a complete repeat grid for a single task artifact.

For MMLU, the pinned README documents `gym eval prepare --benchmark mmlu`, `gym env start --model-type vllm_model --benchmark mmlu`, and `gym eval run --no-serve --agent mmlu_mcqa_simple_agent --input benchmarks/mmlu/data/mmlu_benchmark.jsonl --output results/mmlu/rollouts.jsonl --prompt-config benchmarks/prompts/eval/aai/mcq-4choices-boxed.yaml`. That upstream command uses Gym-relative benchmark inputs; for a NemoChat run replace only its output with `--output "$NEMOCHAT_ROOT/results/gym/mmlu/rollouts.jsonl"`. Retain the materialized prepared input in `datasets/gym/<run>/` with provenance without altering upstream preparation contracts. This assumes a working configured model backend. Do not apply a chat template to a base model without declaring the change. The existing `long_context/niah.py` client uses plain `/v1/completions` (or `client.completions.create`) and its own exact-match scorer; `scripts/serve_cpu.py` is a loopback HF smoke server capped at 64 generated tokens. Neither is an official Gym model-server/agent/environment integration, and their scores are not imported as official Gym results. A generic OpenAI chat client is not automatically compatible with Gym's Responses API model-server/agent/environment chain. Native base text completion is source-verified specifically for the RULER pretraining profile, not universally for every task.

Official aggregation calls each environment/resource server's `/aggregate_metrics`. The saved aggregate artifact is a list of entries with `agent_ref`, `agent_metrics`, `key_metrics`, `group_level_metrics`, and `repeat_level_metrics`. Dataset preparation `*_benchmark_metrics.json` files describe dataset statistics and are **not model evaluation scores**. The pinned path helper maps `ruler.jsonl` to `ruler_aggregate_metrics.json`. Locate the aggregation artifact in the actual Gym command output. If `mean/reward` exists for your entry, its JSON pointer is `/0/agent_metrics/mean~1reward`; inspect the artifact first rather than assuming entry order or metric availability.

## Manifest and importer CLI

`rl/configs/gym_pretrain.json` deliberately has no runs: it must fail instead of inventing results. Copy it to the run artifact directory and supply one record for **each step × task × raw/merged variant**. Paths are relative to the manifest, or absolute. Example schema (placeholders, NOT results):

```json
{
  "checkpoint_step": 100,
  "variant": "raw",
  "task": "ruler_pretrain",
  "benchmark_config": "benchmarks/ruler/config_pretrain.yaml",
  "checkpoint": "raw/model.safetensors",
  "tokenizer": "shared/tokenizer.json",
  "input_jsonl": "shared/actual_inputs.jsonl",
  "rollouts": "raw/ruler.jsonl",
  "metrics": "raw/official_aggregation.json",
  "metric_pointer": "/0/agent_metrics/mean~1reward",
  "protocol": "shared/protocol.json",
  "protocol_files": ["shared/prompt.yaml", "shared/scorer.py", "shared/serving.yaml"],
  "sample_count": 100,
  "num_repeats": 1
}
```

Duplicate for `merged`, changing only variant-specific weights, rollouts and metrics. `checkpoint` must be a real file: for sharded exports provide a canonical weight inventory file recording every shard hash; for tokenizer identity similarly inventory tokenizer.json, special-token settings and tokenizer_config.json. Inventory contents and completeness are user responsibility. Save the actual resolved prompt/scorer/serving snapshots in `protocol_files`, in identical order for both arms. Separate model path from shared serving settings so protocol snapshots remain comparable.

`protocol.json` must contain `generation`, `context_length`, `prompt_mode`, `scorer`, `scope`, `runtime`, and `wsm`. Record all generation settings (seed policy, temperature, top-p, output limits, stop policy), actual context/BOS/truncation behavior, prompt formatting, official scorer settings, full versus capped dataset scope, runtime versions and precision. `wsm` is the shared merge policy: checkpoint window, constituent weight hashes, normalized coefficients and causal availability cutoff. Use the same policy record for raw and merged, even though only merged weights apply it. Protocol is hashed byte-for-byte; key formatting differences also reject a pair. A capped input is a partial task result; never label it a full leaderboard evaluation.

```bash
PYTHONPATH=src .venv/bin/python rl/evaluation/gym_eval.py \
  --manifest results/gym/my-run/manifest.json \
  --gym-root /absolute/pinned/Gym \
  --output results/gym/my-run/gym_comparison.json
```

Use one task and one agent per official aggregation artifact; multi-agent files are rejected rather than selecting a potentially unrelated entry. The stdlib-only importer checks source pin, clean tracked Gym checkout, config existence, paired task coverage, checkpoint identity across tasks at each step/variant, input counts, complete rollout identities, finite official rewards, mask/failure fields, numeric selected metrics and matching protocol/tokenizer/input/config fingerprints. It refuses overwriting output. It emits imported official scores, per-task merged-minus-raw deltas, file paths and SHA-256 evidence. It does **not** recompute official metrics or combine tasks with incompatible scales. Completeness checks intentionally reject partial/masked/error runs rather than quietly intersecting surviving examples.

### Trust and verification boundary

Imported metrics and checkpoint/WSM provenance are **user-attested**, not cryptographically linked to the serving process. A clean source checkout and artifact hashes do not prove which model produced saved responses. Preserve official launch logs, resolved configs, export inventories, environment versions and all failure sidecars. The importer cannot detect fabricated upstream artifacts, unused protocol settings, incomplete shard inventories or mislabeled task identity. It does not verify arbitrary snapshot files against upstream source, and untracked Python files in a checkout are not covered by the tracked-source cleanliness check. No scorer is executed by this script.

`--allow-fixtures` accepts only an explicitly `fixture: true` synthetic manifest and labels its output `synthetic fixture only`; this tests plumbing, never capability. Never use fixture output as an actual benchmark table.

## Actual verification in this environment

Pinned source/config/CLI inspection was performed by narrow GitHub reads. `nemo_gym` and `gym` executable are absent in the default interpreter environment; no model endpoint, vLLM runtime, supported NemoChat export or GPU execution was validated for this integration. No official task inference, upstream preparation, judge call or dataset download was performed. `.venv/bin/python -m pytest -q tests/test_gym_benchmarks.py` returned **21 passed**. The CLI also rejected the empty real template without `--gym-root` (exit 2) and created no score artifact. Only clearly synthetic importer tests and CLI failure/success paths were exercised. Real pretraining benchmark scores therefore remain **unmeasured**. Run CE/BPB diagnostics separately and retain both evidence levels.
