# KD step1000 CPU-only HF export

Accepted local export: `checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01`.
Final receipt: `results/midtraining/hf-kd1000-conversion-01/validation_result.json`.

This is the actual final KD native student, not original BASE weights. The original converter, BASE assets/weights and training checkpoint remain unchanged. Existing BASE mapping code, model config and tokenizer are read-only schema/assets inputs; no BASE weight payload is loaded. No Hub publication or card edits are performed here.

## Commands already executed

Run from `/raid/yoon/nemochat`:

```sh
python midtraining/publication/export_kd1000/engine_run.py nemochat-kd1000-schema-01 python midtraining/publication/export_kd1000/convert_kd1000.py --schema-only
python midtraining/publication/export_kd1000/engine_run.py nemochat-kd1000-export-01 python midtraining/publication/export_kd1000/convert_kd1000.py
python midtraining/publication/export_kd1000/engine_run.py nemochat-kd1000-validation-02 python midtraining/publication/export_kd1000/validate_hf.py
```

The first full validation also passed; its receipt was preserved as `validation_initial.json` before a second full reload generated the final source-provenance-bearing `validation_result.json`. Conversion refuses nonempty outputs; validation refuses an existing result. These commands are evidence, not instructions to overwrite accepted artifacts.

## Reuse for same-architecture LM-only B

Set `B_SOURCE`, `B_OUTPUT`, `B_REPORTS`, and `B_STEP` to the exact future source, fresh absolute output/report directories and actual step. This exporter intentionally retains the original hybrid architecture/tokenizer contract, not arbitrary model conversion.

```sh
mkdir -p "$B_OUTPUT" "$B_REPORTS"
python midtraining/publication/export_kd1000/engine_run.py --export-dir "$B_OUTPUT" --receipt-dir "$B_REPORTS" nemochat-b-schema-01 python midtraining/publication/export_kd1000/convert_kd1000.py --source "$B_SOURCE" --output "$B_OUTPUT" --report-dir "$B_REPORTS" --source-step "$B_STEP" --stage midtraining_lm_only --model-prefix '' --schema-only
python midtraining/publication/export_kd1000/engine_run.py --export-dir "$B_OUTPUT" --receipt-dir "$B_REPORTS" nemochat-b-export-01 python midtraining/publication/export_kd1000/convert_kd1000.py --source "$B_SOURCE" --output "$B_OUTPUT" --report-dir "$B_REPORTS" --source-step "$B_STEP" --stage midtraining_lm_only --model-prefix ''
python midtraining/publication/export_kd1000/engine_run.py --export-dir "$B_OUTPUT" --receipt-dir "$B_REPORTS" nemochat-b-validation-01 python midtraining/publication/export_kd1000/validate_hf.py --model "$B_OUTPUT" --result "$B_REPORTS/validation_result.json" --export-result "$B_REPORTS/export_result.json"
```

Inspect B metadata first. KD uses empty model prefix: `embedding.*`, `decoder.*`, `output_layer.*`. If B has a wrapper, pass the exact inspected prefix via `--model-prefix` in both schema/export commands; do not guess or silently strip prefixes. A `module.student.` wrapper is fixture-tested, not observed in this KD source. Only explicit model roots and training-only `optimizer.`, `rng_state/`, `common_state/` namespaces are accepted; model backend BytesStorage records are excluded. Unknown roots (including an unclassified teacher namespace), missing/excess tensors and shape mismatches are fatal. There are no random fills.

The no-GPU runner interface is:

```text
engine_run.py [--export-dir ABS_DIR] [--receipt-dir ABS_DIR] UNIQUE_CONTAINER_NAME COMMAND [ARG ...]
```

It runs the exact pinned `nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf` image, 4 CPU / 128 GiB RAM, no network/device requests, empty CUDA visibility, NVIDIA visibility void, 1200-second inner timeout with kill-after30. Repository is read-only except exact output/report mounts. Runner waits foreground, saves Engine create/state/log receipts, and stops its owned container on interruption. Names must be fresh; completed containers remain stopped for audit. New safetensors are user1001/group1001 mode0640. Source read faults and nonfinite values are fatal; no SHA/CRC byte audit or source copy is performed.

## Evidence boundary

619 native model tensors map exhaustively to 447 official HF runtime records. Official Transformers reverses its checkpoint conversion when saving: the serialized index therefore has 6,289 records (`backbone.*` and individual expert matrices), while reload restores `model.*` and stacked experts. `serialized_to_runtime_keys.json` exhaustively relates every saved key to runtime/native provenance. `final_acceptance.json` records all 18 files (12 safetensors shards), sizes, ownership and exact stopped Engine states. All source/reloaded tensors are finite; all 23 router correction buffers remain FP32, other source tensors BF16. Original tokenizer files match byte-for-byte. Official Transformers5.12.1 / Bridge0.6.1 reload through `AutoModelForCausalLM.from_pretrained(local_files_only=True, trust_remote_code=False)` has empty missing/unexpected/mismatched/error sets. English and Korean forward logits are finite and each prompt generates four tokens with finite scores and a real `DynamicCache`.

This proves CPU loading/execution and caching, not bilingual quality, native/HF logit parity, benchmark improvement, native resume, or waived source byte integrity. Korean completion was `대한민국의 수도는 100`; do not describe this gate as correct bilingual answering. Native-kernel parity was not exercised and no GPU was allocated.
