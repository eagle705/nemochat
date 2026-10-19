# Bounded real cached-KD GPU smoke

This is a task-local operational overfit runner, not a production recipe. It uses the exact existing Base teacher and BASE-19074 student specified in `control.py`, with the original checkpoint byte-audit waiver. It never downloads weights, modifies the installed SDK, starts the main pipeline, or substitutes random models.

## Execution contract

- Physical GPU0 only, under the cooperative `gpu-all8.lock`; recheck occupancy before each allocation.
- Pinned existing NeMo image, no network, UID1001/GID985, repository/model inputs read-only. Only this task's output and owned scratch are writable.
- EP/TP/PP/CP/DP1; BF16; sequence512, MB1/GB1; eight actual corpus windows (two per source), repeated in the same order four times. Four separate producer-heldout windows are evaluated separately.
- Exactly32 native Bridge optimizer calls, LR1e-5, warmup0, native clipping1.0. No private model optimizer/backward loop and no additional epochs to select a better loss.
- Teacher: plain AutoBridge provider, no DDP or optimizer, frozen parameters, dropout0, native saver hooks under `no_grad`. The teacher process exits completely before student allocation.
- Student: scratch-applied opt-in Bridge cached forward factory, real native cache writer/codec/reader, native `pretrain`, optimizer, scheduler and checkpoint save/reload.

The reported primary loss is `.5*LM + .5*top32 ghost-tail KL` at temperature1, not ModelOpt detached magnitude balancing or full-vocabulary KL. The BASE provider's native router auxiliary policy is retained (`seq_aux_loss`, coefficient1e-4, expert bias enabled). That auxiliary term is not included in the reported primary `total loss` scalar.

Fixed pre/post evaluation uses native model `eval/no_grad` CE on identical inputs and masks. It is **LM-only**, including the four disjoint heldout windows; it must not be called heldout cached-KD validation. Epoch means report raw LM/KD/primary-total in identical sample order.

## Runtime compatibility

`optimizer_compat.py` only partitions native TE `multi_tensor_adam` tensor lists into bounded kernel launches. It does not replace Adam or implement the training loop. The pinned TE's fixed native tensor-handle pool cannot accept the EP1 student's large parameter groups in one call. Before model loading, a real-CUDA control compares parameters and moments against unmodified native Adam and exercises6000 scalar parameters. These test-fixture optimizer calls are not student learning updates.

The exact teacher initially encountered allocator fragmentation; `PYTORCH_ALLOC_CONF=expandable_segments:True` allowed the unchanged241GB BF16 teacher to load. Earlier failures remain in the results `failures/` tree. Original checkpoint payloads are not hashed.

## Artifacts and acceptance

Results: `results/midtraining/offline-kd-smoke-01/`.

- `request.json`, `student-retry-request.json`, `*-engine-created/state/final.json`: exact Engine IDs, scope and exits.
- `windows.npz`, `data.json`, `tokenizer.json`: actual selected data, identity binding and fresh reserved-ID check.
- `teacher-complete.json`, `cache/*.tar`: authentic32-iteration teacher cache and byte counts.
- `updates.jsonl`: every completed model update's raw LM/KD/total, preclip Grad Norm, actual LR, skip flag and timing.
- `initial-fixed-lm.json`, `final-fixed-lm.json`: per-window LM before/after.
- `checkpoint/`, `checkpoint-cache-binding.json`, `checkpoint-save.json`: native model/optimizer/RNG state and cache cursor.
- `reload-restored.json`, `reload-fixed-lm.json`, `reload-complete.json`: operational reload with zero additional model updates.
- `objective-note.json`, `teacher-output-correction.json`: semantic boundaries. Native saver hooks disable teacher CE; the initial raw log's `teacher_lm_loss=0` is a writer placeholder, not measured teacher loss.
- `executed-source*/`: source snapshots for executed attempts; `supervisor-complete.json` plus all exact clean Engine exits are required before claiming completion.

The durable task-owned user-systemd supervisor and deadline guard survive subagent exit. `deadline.py` verifies exact IDs, ownership labels and GPU scope before stopping a task container. Do not launch this fixed-output supervisor again over existing artifacts; a new experiment needs a new reviewed output identity. The explicit `--reuse-teacher` mode requires the accepted clean teacher receipt and restarts only the student from the original BASE weights after a failed student attempt.

Focused controls: `.venv/bin/python -m pytest -q tests/test_offline_kd_gpu_smoke.py`. Passing these controls alone is not GPU training evidence; inspect the actual native receipts above.
