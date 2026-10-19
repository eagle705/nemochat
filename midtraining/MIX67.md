# Approved 67/10/3/20 native LM+KD recovery mix

[Korean companion](MIX67.ko.md)

`longrun24h.mix67.yaml` is a separate production configuration. The legacy
`calibration.yaml` and `longrun24h.yaml` remain 60/10/10/20. Source order is
Code / Algorithmic / Formal Logic / ClimbMix replay. Only those two exact
blends are accepted; this is not an arbitrary blend API.

The new initial budget is **3,200 optimizer calls**, WSD warmup **32**, final
decay **320**, global batch 128, sequence 4096, BF16, EP8/TP1/PP1/CP1. The exact
display name remains `nemochat baby midtraining super35 24h`. The student is
original BASE model-only warmstart, with optimizer/RNG reset; the pinned Base
teacher is frozen. Native Bridge training, loss, optimizer, scheduler, save and
teacher-free reload paths are unchanged. No offline KD, TopK or vLLM change.

## Timing evidence and parent approval

The accepted old-profile receipt is
`results/midtraining/longrun/super35-newdata-calibration-01/controller/accepted.json`.
It measured 16 calls / 15 positive-LR updates, warm updates 9..16 mean
24.29540627438837 seconds, using **60/10/10/20**, not the new mix. Its seed manifest
SHA256 is `52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308`.
The 3,200-call budget is an explicitly provisional same-shape old-profile
projection after the user-approved mix change, not fresh new-mix throughput.
Actual new-main warm-step measurements must come from that main run's native
update receipts; no extra GPU calibration is introduced by this patch. A budget
exit short of 24h is still not a successful 24h duration claim.

The parent generates a fresh `--calibration-compatibility` JSON only after actual
new-corpus and heldout-exclusion audits. Exact required dictionary (replace the
production digest with the actual audited manifest SHA256; no extra fields):

```json
{
  "status": "parent_accepted_approved_mix_change_profile_projection",
  "calibration_manifest_sha256": "52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308",
  "production_manifest_sha256": "<actual audited recovery manifest SHA256>",
  "old_weights": [0.6, 0.1, 0.1, 0.2],
  "new_weights": [0.67, 0.1, 0.03, 0.2],
  "same_provider_tokenizer_sequence_batch_topology": true,
  "heldout_exclusions_preserved": true,
  "user_approved_mix_change": true,
  "projection_only": true,
  "fresh_new_mix_benchmark": false
}
```

`parent_accepted_same_recipe_expansion` remains the legacy branch; it cannot
approve the changed mix. Existing accepted calibration and all-eight-rank
saved/reloaded duration-control gates remain mandatory. The consumed projection
records the compatibility dictionary and explicitly records old timing weights,
new production weights, and pending actual main-run warm measurements.

## Exact shared APIs

In `longrun_native.py`:

- `WEIGHTS`: unchanged legacy tuple; `MIX67_WEIGHTS`: approved new tuple.
- `recognized_weights(value) -> tuple`: exact list/tuple or complete source-key
  map; rejects any other blend or map keys.
- `manifest_weights(manifest) -> tuple`: schema-1 manifests require a source-key
  weights map. The only missing-weights fallback is the legacy in-memory adapter
  fixture lacking both schema/version and status. Production disk input never
  uses that fallback.
- `dataset_weights(c, manifest=None) -> tuple`: validates configured blend and,
  when supplied, exact manifest/config agreement.
- `capacity_gate(manifest, updates, batch, sequence, cursor_samples=0)`: unchanged
  signature; derives weights from manifest. Keeps native 0.5% builder surplus,
  rounding reserve, shifted token and per-source heldout requirements.
- `load_prepared_manifest(path) -> (manifest, sha256)`: both blends accepted only
  under existing schema/source-order checks and SHA-bound companion success.
  Explicit failed companion/result status is rejected rather than rewritten.

In `launch_longrun.py`:

- `validate_calibration_compatibility(accepted, manifest_hash, weights,
  compatibility) -> dict | None`: exact legacy or approved-projection dictionary.
  The new branch binds the known measured old-profile seed and a different new
  manifest digest. `execute` invokes it before capacity/Engine/tracking actions.
- `SOURCES` adds the new YAML, retaining all legacy entries (15 paths). Parent
  review generation must be freshly frozen after both owned patches finish.

Capacity, builder, observer and rank-local runtime use the same actual weights.
Every built source still must have exactly one native epoch; actual first global
batch must contain all four sources, including Formal Logic at 3%. All-rank
finite losses, positive gradients, frozen teacher, final native full-state save
and teacher-free reload acceptance remain intact. Fresh actual `.bin`/`.idx`
SHA readback and reserved-ID 18..29 zero counts remain mandatory despite the
separate explicitly waived student checkpoint byte audit.

## CPU validation only

Strict red/green slices were exercised in `tests/test_midtraining_mix67.py`.
Focused CPU controls cover both blends, source capacity, config/manifest
mismatch, companion failures, native configuration/observer plumbing, one-epoch
rejection, exact compatibility rejection, and launcher branch integration.
Synthetic builder/index fixtures are not real native/GPU results.

```bash
.venv/bin/python -m pytest -q tests/test_midtraining_mix67.py tests/test_midtraining_longrun.py tests/test_midtraining_dataset_observer.py tests/test_midtraining_production_corpus.py tests/test_midtraining_production_pipeline.py
.venv/bin/python midtraining/launch_longrun.py --config midtraining/longrun24h.mix67.yaml --run-id super35-mix67-01 --preview
.venv/bin/python midtraining/longrun_native.py --config midtraining/longrun24h.mix67.yaml --preview
```

These are inert previews and CPU tests, not training. This patch creates no
Engine container, service, download, tracking run, checkpoint or compatibility
receipt. Parent-owned production orchestration, fresh corpus acceptance and
source review remain separate release gates.
