---
name: nemochat-observability-validation
description: Validate Lens traces without inventing GPU evidence.
version: 0.1.0
author: Joosung Yoon, Hermes Agent
platforms: [linux]
---

# NemoChat observability verification

## When to Use
- Lens training/rollout traces, Jaeger ingestion/rendering, or semantic source evidence in the Nsight bridge.
- This does not approve native GPU, NeMo RL/Gym/vLLM, Nsight loading or causal production bottlenecks.

## Prerequisites
Read [observability tracks](../../docs/en/OBSERVABILITY_TRACKS.md) and the relevant tutorial/plugin README. Use isolated tutorial/plugin environments without changing the main environment. Collector startup needs authorization; check service readiness instead of assuming it is running. Use only credential-free approved loopback endpoints. Rollout requires an already-cached pinned model.

## How to Run
Through the agent's terminal tool, from repository root with existing local Jaeger:

```bash
tutorials/nemo_lens/.venv/bin/python tutorials/nemo_lens/verify_jaeger.py
tutorials/nemo_lens/.venv/bin/python tutorials/nemo_lens/rollout/run.py --scenario baseline --mode otlp --output results/observability/skill-baseline
tutorials/nemo_lens/.venv/bin/python tutorials/nemo_lens/rollout/diagnose.py --spans results/observability/skill-baseline/spans.jsonl --output results/observability/skill-baseline/diagnosis.json
tutorials/nemo_lens/.venv/bin/python tutorials/nemo_lens/rollout/verify.py --result results/observability/skill-baseline/result.json
```

## Procedure
1. Select execution scope and fresh output path. Completion: actual model/control and injected delays explicitly identified; no ambient proxy/auth/redirect export escape.
2. Execute and inspect local spans plus success/failure artifacts. Completion: exporter completeness and exit status verified; `result.json` alone is not ingestion evidence.
3. Validate Jaeger receipt against every local span ID, identity, hierarchy, attributes and bounded timestamp precision. Completion: verifier succeeds on real ingestion; bool IDs, missing spans and malformed timestamps fail closed.
4. If UI evidence is requested, inspect `scripts/verify_lens_ui.py --help`, pass the actual trace ID/service and render with an existing compatible browser environment. Completion: correct trace and operation labels visibly rendered without blocked network attempts or page errors. API results are not screenshot evidence.
5. Diagnose generation/caller queue/tool/reward separately. Server time nested in client time is not additive; concurrent trajectories are not batch wall time. Completion: label observed candidates, injected delay and unknown gaps, not unproven causality.
6. For bridge source evidence, retain real file snapshots/hashes and exact IDs. Kernel names/time proximity are not source attribution. Completion: mismatches/unknowns explicit; absent CUDA/Nsight fails closed rather than fabricated capture.
7. Record exact artifacts and scope in the ledger; preserve historical measured values. No implicit container shutdown, cloud telemetry, commit or publication.

## Pitfalls
- CPU block duration is not GPU kernel time. Caller semaphore queue is not vLLM prefill/decode queue.
- Float64 tags may decode as integral JSON numbers; semantic types matter. Do not weaken strict bool/int identity guards.
- Scripted CPU rollout is not a learned tool-use policy or an RL optimizer.
- Tiny CPU overhead measurements are not production overhead estimates.
- Preserve stock trainers for dashboard integration: consume existing logs and actually supported native Lens exports, never introduce dashboard-specific callbacks or monkeypatches merely to normalize fields. An optional canonical file adapter is not a deployed producer or native instrumentation.
- Decode exported Lens Console/OTLP JSON resource and span attributes through an explicit typed allowlist; preserve the installed semantic keys such as `nv.dl.run.uuid` rather than guessed spelling. Native span names do not establish native execution scope.
- Preserve OTLP trace/span/parent identity for both standard hex IDs and strictly validated protobuf base64 bytes. Subtract integer Unix nanoseconds before converting to duration units; float conversion first loses small intervals at epoch-scale magnitudes.
- Exercise public Lens setup/registry/span/export interfaces in a fresh local, network-blocked subprocess and separately verify the actual trainer runtime. A host-only SDK probe is not native training integration or a benchmark.
- Rank optimization investigation from selected measured updates/phases, not startup-heavy whole-capture aggregates. Distinguish CUDA summed duration, clipped same-process/device activity union and host duration; none alone proves a whole-node critical path. Confirm communication peer readiness before treating NCCL kernel duration as pure transfer cost.
- Separate exact saved host-operation/phase code from captured kernel Python frames. Verify file/snippet hashes, source identity and inclusive line bounds; count actual exported callchains before promising caller attribution. Do not manufacture source lines, and do not display a trailing split newline as an extra line beyond the verified end range.
- Explain collective variants precisely: NCCL SendRecv is peer send/receive and may underlie higher-level operations; do not label every NCCL kernel a collective or infer MoE dispatch solely from its name. Keep unknown/vendor-opaque symbols explicit.
- Keep exact raw kernel names as the primary identifiers in dashboard lists, charts and detail headings. Put human explanations beside them in a separate column/annotation, never substitute friendly aliases for symbols. Graph labels may be visually shortened, but preserve the full symbol in hover/accessibility and the source/detail view.
- Inspect NCCL generator specialization and device dispatch before equating a raw RING_LL entry name with executed work protocol; canonical kernel names can dispatch different function IDs. Pin reference versions and keep unknown vendor token meanings explicit, separating inferred tile/CTA concepts from observed launch geometry.
- Build iteration chronology from exact process-correlated kernel/API timestamps, not duration aggregates. Keep GPU-start, CPU-enqueue and cross-stream dependency notions separate; retain tiny events in a paginated table, deduplicate CPU API spans shared by multiple kernels, and label omitted memcpy/OS/other-process coverage. Cache offline analysis but measure repetitive snapshot size as well as collection cost.

## Verification
Run through the terminal tool:

```bash
tutorials/nemo_lens/.venv/bin/python -m pytest --import-mode=importlib -q tutorials/nemo_lens/rollout tutorials/nemo_lens/test_tutorial.py tutorials/nemo_lens/test_exporter_privacy.py
plugins/nemo_lens_nsight_bridge/.venv/bin/python -m pytest -q plugins/nemo_lens_nsight_bridge/tests
```

Report skipped/missing optional environments separately; don't install or use paid infrastructure without permission.
