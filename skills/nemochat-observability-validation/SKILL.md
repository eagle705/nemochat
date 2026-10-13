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

## Verification
Run through the terminal tool:

```bash
tutorials/nemo_lens/.venv/bin/python -m pytest --import-mode=importlib -q tutorials/nemo_lens/rollout tutorials/nemo_lens/test_tutorial.py tutorials/nemo_lens/test_exporter_privacy.py
plugins/nemo_lens_nsight_bridge/.venv/bin/python -m pytest -q plugins/nemo_lens_nsight_bridge/tests
```

Report skipped/missing optional environments separately; don't install or use paid infrastructure without permission.
