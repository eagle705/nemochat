# Dashboard data sources and evidence boundaries

[한국어](DATA_SOURCES.ko.md). The dashboard is an artifact reader, not a trainer and not a remote W&B client. Files are selected through `dashboard/config.json`; unsupported files are not automatically discovered or interpreted.

## Data flow

```
Existing Bridge/native training logs -> bounded format adapter -> run snapshot
Current /proc + bounded nvidia-smi -> host snapshot
Exported Lens/bridge JSONL + Nsight SQLite/summary -> scoped profile snapshot
                                            |
                         one separate collector + cache
                                            |
                           SSE / snapshot polling -> browser
```

A browser receives cached, sanitized JSON, not arbitrary files, `.env`, checkpoints, raw prompts, tool payloads or container environment dumps. Fresh host data does not mean a historical selected run is currently running. A file or successful export is not proof of native whole-stage completion.

## Existing source map

### Actual Bridge pilot adapter (`format: bridge`)

- `rank-N-updates.jsonl`: explicit per-update optimizer audit records. Reads update, local_loss, lr, grad_norm, local processed/loss-bearing/source-read counters, seconds and writer rank. Equal-numbered updates are matched across the configured rank set. Local token deltas are summed only when required ranks are present. Local losses use an **unweighted rank mean**, not automatic proof of a target-token-weighted globally reduced loss; the current pilot has equal local target counts.
- `verified-summary.json`: separately audited global token totals, optimizer update count, producer outcome and training-container duration. This can confirm an exact bounded run, not all of pretraining.
- `requested-config.json`: recorded model/provider, precision, dimensions, train_iters/global_batch_size/sequence length. A requested config is not an actual numel count.
- `data-verification.json`: selected data preparation/format facts; prepared tokens are not trained tokens.
- Optional `model_census`: explicitly configured prior parameter census. Its source and prior-run boundary must remain visible. Active parameters are unavailable unless actual supporting metadata exists.

The default Bridge run uses `results/pretrain_followup/bridge-fresh-wsd4-01/`. Rank logs, verification summaries and historical source/config receipts are different evidence levels; aggregation does not make them identical.

### Native E2E receipts (`format: native`)

- `rank-N-phases.jsonl` / `rank-N.json` / `result.json`: actual optimizer completion events and per-rank histories/cumulative counters, diagnostics and result statuses.
- `resolved_config.json`: recorded run plan/data contract and schedule horizon.
- Explicit `terminal_receipt`: final container/producer outcome. A failed producer remains failed even if a separately validated snapshot is recoverable.
- Fixed small heldout CE entries are diagnostic evaluations, not broad official benchmark quality. Warmup and selected measured updates stay distinct; instrumented timings are not unprofiled throughput evidence.

Resumed cumulative counters are not summed again across runs. Missing ranks, bounded/truncated history and incomplete artifacts are surfaced, not filled in with zeros or guessed totals.

### Current host telemetry

- `/proc/stat`: CPU utilization from successive samples, not cumulative percentage from one reading.
- `/proc/cpuinfo`, `/proc/meminfo`: current CPU identity/count and host memory facts.
- Timeout-bounded `nvidia-smi`: current device identity, utilization, memory, temperature and power. This is node telemetry, not exclusive attribution to the selected historical trainer.

No torch import or CUDA context is initialized. The collection interval and measured collection cost are visible; shared CPU/storage/NVML overhead still exists and needs actual workload on/off comparison to quantify training interference.

### Profile imports

- **Nsight SQLite**: `CUPTI_ACTIVITY_KIND_KERNEL` plus `StringIds` preserves raw kernel names and aggregates count, cumulative milliseconds, mean microseconds and duration share. The whole capture includes startup/warmup; overlapping kernels are not additive wall time. Read-only queries are bounded and cached by file/WAL metadata.
- **Producer analysis summary**: selected rank/update/phase CPU NVTX durations and associated kernel counts. These are imported producer results; selected launch correlation and producer provenance are not recomputed/authenticated by this viewer. Host NVTX time, summed CUDA kernel time and activity union are different measurements.
- **Lens Console JSONL / OTLP JSON**: exported host/OTel span identities and durations, preserving explicit resource run UUID/rank and span hierarchy through allowlisted semantic attributes. Pretty JSON and resourceSpans/scopeSpans batches are accepted. This is a local exported-data reader, not a new OTLP network receiver or native trainer instrumentation. It is not CUDA kernel timing.
- **Bridge operations JSONL**: explicit operation title/role/explanation and source identifiers. Host annotations and successful NVTX emission are not verified kernel/source attribution. Default control files are educational sparse-MoE evidence, independently imported rather than assigned to native training stages.

The browser timing visualization must show the selected training stage, source, units and scope. A whole-capture kernel cannot be assigned to forward/backward merely by its symbol or nearby timing. Kernel glossary text is **name-based reference information**, not authenticated source attribution, shape or a causal bottleneck.

## Unified logs: compatibility strategy

**Selected policy: keep upstream trainers stock.** Prefer read-only processing of their existing logs and native NeMo Lens/OTel exports. Do not add a dashboard-specific loss callback, monkeypatch, trainer loop wrapper or mandatory JSONL writer. The canonical schema is an optional dashboard-side normalized/derived interchange contract, not a requirement imposed on the trainer. Native Lens activation is permitted only through actually supported launcher/config interfaces after installed-SDK compatibility and telemetry/export scope checks; environment variables alone cannot instrument a consumer with no existing call sites.

Existing Bridge/native formats remain supported unchanged. New integrations can use the versioned canonical contract described by `telemetry.schema.json` and validated by the standalone `telemetry.py` CLI. The canonical adapter is opt-in; adding support does not silently install callbacks into any trainer.

Required semantics are more important than a filename:

- Stable `run_id`, explicit stage, schema version, step and timestamp.
- Explicit recording rank/world size and one designated writer; do not assume the valid native logger is rank zero.
- Explicit global versus rank-local metric scope. Logging a local loss does not make it globally reduced.
- Canonical token counters are **global cumulative** and processed/loss-bearing/source-read remain separate. Resume continues the cumulative trajectory; restarts or independent branches need new run identities.
- Null means unavailable, zero means measured zero. Booleans/NaN/Infinity are not metric numbers.
- Terminal producer outcome is distinct from checkpoint integrity, evaluation outcome and stage-completion claims.
- Metrics/metadata are allowlisted; do not serialize the full trainer configuration/environment, prompts or model state.

The running standard-library validator and regression tests define accepted behavior. Consult its `--help` and exact schema before wiring a producer. Nsight `.nsys-rep`/SQLite and Lens exports retain their native evidence formats; log standardization is not a conversion that destroys trace correlation or replaces a profiler.

## Reference glossary boundaries

Generic meanings were checked against primary documentation:

- NVIDIA NCCL collective operations: `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html`
- NVIDIA cuBLAS GEMM: `https://docs.nvidia.com/cuda/cublas/`
- PyTorch tensor fill: `https://docs.pytorch.org/docs/2.8/generated/torch.Tensor.fill_.html`

These references explain operation families, not proof that a captured symbol came from a particular model layer or Python call. Bridge explanations are taken from the actual local operation records and remain scoped to that educational control.
