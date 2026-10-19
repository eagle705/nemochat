# Isolated compatibility adaptations

Prefer the pinned Megatron Bridge implementations and native `ConfigContainer` YAML. Do **not** edit `/opt/Megatron-Bridge`, Megatron-Core, ModelOpt, Transformer Engine or Transformers in place. Native optimizer, WSD scheduler, masked LM/KD loss, distributed execution and checkpointing remain upstream-owned.

## Active adaptation: `teacher_text_view.py`

- Kind: **offline checkpoint-layout adapter**, not an SDK monkeypatch.
- Reason: Bridge 0.6.1 does not register exact Super Omni `NemotronH_Omni_Reasoning_V3`; its Nano Omni mapping does not provide this teacher's Super latent projections. The native text `NemotronHBridge` does support latent MoE.
- Input: verified exact Base revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`.
- Output: a separate stock `NemotronHForCausalLM` bundle; only backbone + LM head tensor bytes copied unchanged; `language_model.` prefix stripped. Vision/projector and unused MTP explicitly excluded; original files never modified.
- Test: `tests/test_midtraining_teacher_view.py` checks real safetensors reload, tensor equality and exclusion policy using a tiny fixture. This does not establish target teacher numerical parity.
- Retirement condition: a pinned Bridge release natively supports the exact Super Omni architecture, all latent mappings, and text-only teacher loading, with accepted operational/fidelity tests.

## Active runtime patch: `distillation_provider_shape_guard.py`

Native config-only execution reproduced a concrete bug: `DistillationProvider.__setattr__` mirrors student finalization assignments into the teacher, changing teacher `num_layers` from 88 to 52. The adapter preserves independent architecture fields while retaining the native setter for shared runtime settings. It is process-local, pins upstream source SHA256 `21b1e634f64c67ed23e602e79dd932df62f9b7db6416dd06cffdf3eb237247b5`, refuses an unreviewed SDK version, and records both upstream and adapter digests. Installed SDK files remain unchanged. Regression: `tests/test_midtraining_shape_patch.py` executes the exact native setter body, reproduces the mutation, and verifies the fix. Real config attempts 01/02 failed with the mutation; config03 passed with teacher width4096 / student width1024 and independent layer counts. Weight loading/update evidence remains separate. Retire when the pinned native setter protects unequal architecture fields and the regression passes without this adapter.

## Active tokenizer gate: `tokenizer_alias_gate.py`

Exact Base teacher and original BASE tokenizer differ only in reserved names at IDs18–29. The gate requires the teacher backend to equal the previously accepted Super3.5 backend, reuses its complete zero-occurrence ClimbMix train/validation audit, and compares the full normalized backend. It explicitly reports reserved semantics as a migration, **not full semantic equality**; no weight permutation or original tokenizer modification occurs. Any unrelated backend or ID change fails closed. This is not a new corpus scan. Test: `tests/test_midtraining_tokenizer_gate.py`.

## Native student import alias: `student_embedding_import.py`

The executed student's official Transformers serialization has singular `backbone.embedding.weight`; this installed NemotronHBridge expects plural `backbone.embeddings.weight` and can skip an absent mapping with initialization disabled. Bind the alias repair only to the student's native HF import hook, rebuild native AutoMapping registry, pin both bridge and loader source hashes, require all6289 source names to map, and record that the native embedding transform actually ran before accepting training. Teacher plural keys remain native. No original weight files are edited, no alternate loader or random fallback is used, and no extra student payload audit is introduced. Regression: `tests/test_midtraining_embedding_import.py`; actual SDK import-plan probe exit0 at `import-plan-02/receipt.json` (metadata only, no weight restoration). Retire once upstream natively recognizes the serialization and a full restore gate passes without this adapter.

## Opt-in observation: `kd_finite_probe.py`

Bounded diagnostic forward hooks record student/teacher embedding, layer and logits nonfinite counts without replacing outputs or changing gradients/loss. They synchronize CUDA and therefore perturb timing; never promote diagnostic timings into performance data. Raw loss/norm/skip receipts remain separate from accepted updates. Regression `tests/test_midtraining_diagnostic.py` verifies original output identity and unchanged simple gradients, including recording rather than repairing NaNs. Disable after identifying the original numeric failure; never use clipping or a higher failure threshold to hide nonfinite values.

## Native dataset observation: `native_dataset_observer.py`

The long-run runner isolates its process-local Bridge GPT registry wrapper here. The original native dataset provider/builder still creates all blends, shuffled document indices and samples. This adapter observes actual per-source coverage/one-epoch training indices and reserves one extra native startup-evaluation window; it restores the original registry on exit. It does not implement a replacement dataset or optimizer. The runner checks the pinned Bridge0.6.1 image and the parent source allowlist hashes this file. Regressions: `tests/test_midtraining_dataset_observer.py` and the native registry delegation/restoration test in `tests/test_midtraining_longrun.py`. Retire when native callbacks expose built indices and startup-evaluation budget reservation without registry wrapping. CPU tests alone are not native GPU evidence.

## Patch inventory

There are currently **no upstream source `.patch` files applied**. If one becomes necessary, place it here with upstream file/revision, rationale, minimal diff, application/check commands, regression test, measured execution evidence, and removal criteria. Record its SHA256 with the run's source allowlist. Never silently fall back to an unpatched alternative algorithm or overwrite image packages.

[한국어](README.ko.md)
