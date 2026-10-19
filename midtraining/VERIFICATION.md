# Native LM+KD control — execution verification

## Accepted scope (2026-10-17, parent readback 19:12 KST)

This is the **existing-ClimbMix operational control**, not the newly selected Specialized-v1.1 corpus experiment and not long-run midtraining or a quality-improvement comparison.

- Student: original post-decay BASE HF conversion `nemochat_baby_base_hf_bf16_02`, lineage BASE native update19,074. Model-only warmstart with optimizer/scheduler/RNG reset.
- Teacher: frozen text backbone of exact `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`, revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`. No Instruct/3.0 substitution; no downloaded modeling Python.
- Native path: Bridge `pretrain(cfg, forward_step_modelopt, callbacks=[Audit()])`, native ModelOpt LM+logits KD, native optimizer/WSD/checkpoint. BF16, TP1/PP1/EP8/CP1, sequence4096, global batch8, microbatch1.
- All8 ranks completed **4 native optimizer calls, including 1 zero-LR call and 3 positive-LR learning updates**. All recorded LM/KD/Total Loss and Grad Norm values finite; all teacher freeze/no-gradient checks passed.
- Actual `lr_used` on every rank: `[0, 1e-5, 1e-5, 1e-5]`. Native configured minimum1e-6 was reached by the post-final-step schedule, **not used by a learning update in this four-call control**. This does not exercise an extended decay-learning window.
- Processed next-token positions:131072. The train prefix was already consumed by BASE; across-stage replay is explicit.
- Final recorded Training LM Loss2.632596492767334, raw logits KD Loss1.0962437391281128, Total Loss3.9488942623138428, Grad Norm1.368605613708496. Total uses native detached magnitude balancing; it is not raw `LM+.5*KL`.
- Grad Norms across the four calls: `[1.333315134048462, 1.3891193866729736, 1.3760486841201782, 1.368605613708496]`. Large finite values were accepted; configured native clip_grad1.0 was unchanged. No magnitude ceiling, finite-loss repair or softened nonfinite acceptance was introduced.
- Every rank recorded876 forward activation observations and zero nonfinite events. These opt-in synchronizing diagnostics perturb timing; **no throughput/performance result** is claimed.

## Saved state and teacher-free reload

- Native student-only save: `checkpoints/midtraining/super35-kd-pilot-01/iter_0000004`.
- All8 ranks recorded save4. Native checkpoint metadata contains6683 records, **zero teacher-named records**; only metadata was inspected, not an additional payload integrity audit.
- Separate native reload built **no teacher**, performed **zero optimizer updates**, and all8 ranks completed finite heldout LM evaluation.
- Fixed-window heldout LM Loss before reload2.3860015869140625; after reload2.385974884033203. This establishes operational reload/evaluation, not bitwise equivalence or a matched quality gain.
- Pilot saves no optimizer/RNG state; it is a model-only next-stage warmstart artifact, not a resumable production optimizer checkpoint.
- Checkpoint integrity remains `not_checked_explicit_user_waiver`. No new full student checkpoint SHA/CRC audit was introduced.

## Exact Engine and parent proof

- Pilot02 Engine `906d2c5e1824579c76f9fe3f825994c4bd94c1f83a08fb85c15d8a5a13d7d662`, exit0/OOMKilledfalse; finished2026-10-17T10:08:57.799661073Z.
- Reload02 Engine `639d42ced1bc0e901bd7b1e9a105460456698f45c71235b0538a69b944a65c32`, exit0/OOMKilledfalse; finished2026-10-17T10:11:50.6992097Z.
- Parent independently re-ran `accept_native` against both exact Engine receipts and all-rank output files after supervisor completion. User service ended success; GPU compute-process query empty at readback.
- Proof root: `results/midtraining/super35-kd-01/pipeline-02/`: `parent-accepted.json`, `checkpoint-metadata-student-only.json`, `pilot-02-accepted.json`, `reload-02-accepted.json`, exact create/state/log receipts and frozen12-source allowlist. Rank records are under sibling `pilot-02/` and `reload-02/`.

## Retained failure and repair boundary

Pilot01 exited1/OOMfalse with0 accepted updates and no saves. Its original aggregate numeric failure message did not retain the failing metric; that numeric subcause cannot be reconstructed honestly.

A concrete metadata/import incompatibility was found: official student serialization uses singular `backbone.embedding.weight`, whereas native NemotronHBridge expects plural `backbone.embeddings.weight`. Native import disables random initialization and can skip absent source mappings. The isolated, source-version-pinned `patches/student_embedding_import.py` corrects only the bound student native registry and retains native loading/mapping. All6289 source tensor names mapped in a real SDK metadata preflight; all8 ranks subsequently recorded one actual native embedding transform. The successful retry supports this repair's operational validity, without proving the exact original numeric mechanism.

The earlier unequal-depth teacher finalization defect is independently isolated in `patches/distillation_provider_shape_guard.py`; original failed config01/02 receipts remain intact. No installed SDK files or original BASE weights were edited. Failed generation01's ten executed sources were preserved under `pipeline-01/source-snapshot/` before modifications.

## Remaining gates

- New Specialized-v1.1 + ClimbMix data recipe/preparation and a **separate newly frozen corpus pilot**.
- Production token budget, LR horizon, warmup/decay duration and sustained throughput/memory calibration.
- Numerical teacher HF-vs-native forward parity and matched LM-only vs LM+KD quality comparison.
- New downstream SFT source contract. Automatic SFT release remains disabled.

[한국어](VERIFICATION.ko.md)
