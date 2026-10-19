# NemoChat training dashboard

A separate, read-only monitoring process for local training artifacts and host telemetry. It does not import PyTorch, initialize CUDA, change the optimizer, start a training job, or turn on profiling. [한국어](README.ko.md).

## Start

From the repository root:

```bash
python3 dashboard/server.py --root . --config dashboard/config.json --port 8787 --interval 5 --reload-config
```

Open `http://127.0.0.1:8787`. For a remote training node, use an SSH tunnel from your workstation rather than opening a public port:

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@TRAINING_NODE
```

The runtime uses Python's standard library and local JavaScript/CSS; no CDN, W&B authentication or extra training-environment package is required. The browser receives cached snapshots over Server-Sent Events (SSE), a one-way persistent HTTP connection suited to monitoring. Polling is the fallback. A browser connection does not create another hardware collector.

## What the monitor means

- Model name, precision and parameter census are read from available run/config evidence. Missing values are unknown, not guessed from a model name.
- Processed tokens, loss-bearing targets and source-prefix reads remain separate. Corpus preparation counts are not trained tokens. Distributed counters must use all-rank evidence, not blindly multiply rank zero.
- Loss, LR and gradient history refer to the selected run and its documented scope. A rank-local loss is not automatically a globally reduced loss.
- A completed four-update pilot is a completed **run**, not completion of pretraining. Stages without selected observations are explicitly unobserved.
- Recorded update time and container wall time are different quantities. Runs can overlap; their durations are not summed into a fictional total project wall time. Throughput from diagnostic/instrumented runs is not steady-state performance.
- Benchmarks are shown only if there are actual compatible evidence records; no placeholder benchmark scores.
- Host CPU/GPU samples describe the node **now**, not necessarily the resources owned by the selected historical run. The dashboard does not stop or control jobs.
- Freshness, file errors and collection duration are visible. A fresh file alone does not establish that training is running.

## Data sources and stock-trainer policy

See [data sources and aggregation semantics](DATA_SOURCES.md) ([한국어](DATA_SOURCES.ko.md)). The monitor processes existing Bridge/native logs and native Lens/OTel exports; it does not require dashboard-specific trainer callbacks or a new trainer log format. Canonical v1 is optional derived/export ingestion, validated read-only:

```bash
python3 dashboard/telemetry.py validate --manifest PATH/run.json --metrics PATH/metrics.jsonl
```

Profile panels include top-kernel CUDA duration and separate HOST NVTX phase graphs, plus Lens/bridge host-operation graphs when available. Hover, keyboard focus or tap shows stage, duration, raw identity, scope and a conservative name-based reference explanation. Unknown source/phase mappings remain unknown. CUDA kernel sums and host spans never share a misleading wall-time interpretation.

The [semantic kernel/source inspector](KERNEL_INSPECTOR.md) ([한국어](KERNEL_INSPECTOR.ko.md)) adds measured-update/phase selection, human operator names, verified numbered saved host code and prioritized investigation targets. Offline analysis is imported through `nsight_analysis`; native host snippets also require explicit `nsight_native_source` and `nsight_source_provenance` paths. No heavy SQL analysis runs per browser. Exact inner native Python caller frames remain unavailable in the current capture. [Executed 171-test/source/UI verification](INSPECTOR_VERIFICATION.md). The [kernel-name/token and iteration timeline extension](KERNEL_NAMES_AND_TIMELINE.md) ([한국어](KERNEL_NAMES_AND_TIMELINE.ko.md)) adds confidence-tagged NCCL/NVJET references and recorded GPU-stream/CPU-API lanes, window/pan and chronological pagination; related232tests/32subtests passed. Runtime dispatch caveats and unknown proprietary suffixes remain explicit.

## Add or change sources

`dashboard/config.json` supports individual approved runs plus bounded `run_roots` enrollment. New supported directories under an explicitly approved root/prefix-stage rule are picked up each collection without editing the individual run list; no recursive data/checkpoint scan. See [live stage correction](LIVE_STAGE_FIX.md) ([한국어](LIVE_STAGE_FIX.ko.md)) for verified MID/SFT/RLVR counters, rollout-only status, English metrics and historical-vs-live capture UX.

`dashboard/config.json` is an explicit allowlist of repository-relative run and profiling paths. Keep it credential-free. Consult `CONTRACT.md` and the collector implementation for the accepted source schema. Restart the dashboard after changing configuration unless the active server explicitly documents hot reload. Existing trainers that append supported rank update logs are observed without modifying their training loop. Unsupported trainer formats need a small adapter; the dashboard is not an automatic arbitrary-log parser.

Keep source paths within the repository. Never add `.env`, container environment dumps, raw prompts or checkpoint payloads. The HTTP API does not accept arbitrary filesystem paths. Artifact readers are bounded and cache unchanged files; the monitor never recursively reads model checkpoints.

## Lens, Nsight and the existing plugin

The dashboard reads exported artifacts; it is not the Nsight GUI, a replacement profiler, or a new native Nsight GUI plugin.

- **NeMo Lens:** reads local exported spans. Their duration is host/OTel span duration, not GPU kernel execution time. Span attributes are allowlisted; prompt/tool contents are not displayed by default.
- **Nsight Systems:** reads a pre-exported SQLite capture and selected-range analysis summary. Raw kernel names, counts and aggregate durations are retained. Whole-capture aggregates include initialization unless filtered by actual scope. Concurrent kernel durations can overlap; sums are not elapsed wall time or whole-device utilization.
- **Lens–Nsight bridge:** reads existing operation annotations and source identities. These identify explicitly wrapped host operations. Kernel-to-source attribution still needs actual launch correlation/backtrace/compiler evidence; kernel names or nearby timestamps are insufficient.

The default Lens/bridge source is a separate educational sparse-MoE control, not Lens instrumentation inside the selected native Bridge training run. The default Nsight native capture is rank-zero scoped; other ranks' completion receipts do not establish whole-node trace coverage. Both limitations must remain visible.

For a new profile, follow the existing `tutorials/nemo_lens/` and `plugins/nemo_lens_nsight_bridge/` workflows in a separate, explicitly authorized diagnostic run. The dashboard never attaches a profiler to a live trainer or starts a collector. Load `.nsys-rep` in Nsight for deeper timeline analysis. Keep unprofiled performance measurements separate.

## Performance and privacy

The default collection interval is five seconds; use a longer interval on a busy node. The backend owns one cached collector and limits file reads, histories and SQLite work. Profiling aggregations are cached by source file metadata. `nvidia-smi` is a bounded read-only query, not a CUDA training context.

This architecture avoids synchronous trainer callbacks but is not a claim of zero overhead: the dashboard shares CPU, storage and NVML/driver resources with training. Check visible collection time and compare unprofiled training with/without monitoring before asserting an overhead percentage. Do not infer an overhead bound from one idle-node sample.

Use loopback and SSH tunneling. There is no Internet-facing authentication, TLS or multi-tenant isolation. Do not publish the service or sensitive run/source metadata. Only server-owned static assets and sanitized snapshot endpoints are exposed.

## Verification

```bash
.venv/bin/python -m pytest -q dashboard/tests tests/test_source_evidence.py
python3 -m compileall -q dashboard
node --check dashboard/static/app.js
git diff --check
```

Synthetic test fixtures validate contracts and error branches only; they are not training/model-quality evidence. Parent live-browser and real-artifact validation are recorded in `VERIFICATION.md`; the current timing-graph/provenance/Lens-interface extension and 106-test rerun are in `UPDATE_VERIFICATION.md`. No commit/push or remote telemetry is implied.
