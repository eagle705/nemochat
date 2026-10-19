# Dashboard snapshot contract

Standalone read-only dashboard. No changes to trainers, dependencies or existing artifacts. Python stdlib backend, local HTML/CSS/JS frontend, SSE with polling fallback. No CUDA context, torch import, collector startup, remote W&B fetch, profiler launch or checkpoint loading. Default loopback: 127.0.0.1:8787. A single cached collector runs independently of browser clients, configurable 5s interval (minimum 2s), bounded reads/history. Explicit config file lists run directories and optional profile/Lens/bridge paths. HTTP never accepts arbitrary filesystem paths.

`GET /api/snapshot` and SSE `event: snapshot` have the same JSON object:

```
{
  "schema_version": 1,
  "generated_at": "ISO UTC",
  "collection": {"interval_seconds": 5, "duration_ms": 0, "warnings": []},
  "host": {"hostname": "...", "cpu": {"model": "...", "logical_count": 1, "utilization_percent": null}, "memory": {"total_bytes": 0, "used_bytes": 0}, "gpus": [{"index": 0, "name": "...", "uuid": "...", "utilization_percent": null, "memory_used_mb": 0, "memory_total_mb": 0, "temperature_c": null, "power_w": null}], "sampled_at": "...", "warnings": []},
  "runs": [{"id": "safe identifier", "name": "...", "stage": "pretrain", "status": "completed|failed|stale|observed|running|unknown", "source": "relative path", "updated_at": "...", "model": {"name": "...", "parameters_total": null, "parameters_active": null, "precision": "..."}, "data": {"name": "...", "kind": "...", "scope": "..."}, "tokens": {"processed": null, "loss_bearing": null, "source_read": null, "target": null}, "progress": {"step": null, "total_steps": null}, "timing": {"elapsed_seconds": null, "tokens_per_second": null, "eta_seconds": null, "scope": "..."}, "metrics": {"loss": null, "learning_rate": null, "grad_norm": null}, "history": [{"step": 1, "loss": null, "learning_rate": null, "grad_norm": null, "processed_tokens": null, "loss_bearing_tokens": null, "elapsed_seconds": null}], "benchmarks": [{"name": "...", "value": 0, "scope": "..."}], "warnings": []}],
  "stages": [{"id": "pretrain", "name": "Pretraining", "status": "observed|not_observed", "run_ids": []}],
  "insights": [{"severity": "info|warning|error", "title": "...", "detail": "...", "run_id": null}],
  "profiling": {"lens": {"available": false, "source": null, "spans": [], "warnings": []}, "nsight": {"available": false, "source": null, "kernels": [], "summary": {}, "warnings": []}, "bridge": {"available": false, "source": null, "operations": [], "warnings": []}}
}
```

## Stage-specific profiling and optimization

Profiles are partitioned by training stage, never implicitly shared across stages. Canonical stage IDs: `pretrain`, `mid`, `sft`, `long`, `rlvr`, `mopd`. Normalize `long_context` as an alias where existing sources require it. The UI provides an independent profile-stage selector, defaulting to the selected run stage where possible.

Config supports `profiling.stages` mapping each canonical stage to its own `{lens_spans, nsight_sqlite, nsight_summary, bridge_operations}` paths. Legacy top-level profiling paths are assigned to `pretrain` only. `snapshot.profiling.stages[stage_id]` contains the same `{lens, nsight, bridge}` schema plus an `optimizations` list. Top-level legacy fields may remain for backwards compatibility. Every unobserved stage must explicitly contain unavailable/empty records; missing capture is not missing training proof and is not a fabricated profile.

Optimization record schema: `{stage, severity, title, evidence, hypothesis, next_experiment, confidence: 'observed|candidate', source}`. Display observations, candidate improvements and required validation separately. Recommendations must identify evidence scope (rank, selected updates, instrumented capture) and recommend one bounded correctness + unprofiled A/B experiment. Kernel-name matches alone cannot establish causal bottlenecks or source attribution. Generic stage checklists may be shown only as unmeasured investigation guidance, never performance findings. No profiler launch/attach or training changes are triggered by the UI.

Missing facts are null/unknown, never demo data. Completed pilot runs are not the full stage completion. No sum of overlapping runs into total wall time. History must not double count ranks or resumed cumulative counters. `running` requires credible fresh activity, otherwise use observed/stale; terminal receipts override freshness.

## Stock-trainer and data-source contract

Keep upstream trainers stock. Primary inputs are their existing logs and actually supported native NeMo Lens/OTel exports; no dashboard-specific loss callbacks, monkeypatches, training-loop wrappers or mandatory producer changes. Canonical `run.json`/`metrics.jsonl` is an optional derived interchange format, not a trainer requirement. `telemetry.schema.json` plus the read-only `telemetry.py validate` CLI define v1 acceptance. Canonical counters are global cumulative, written by one explicitly identified logging rank (not assumed rank zero); rank-local metric scope remains labeled. Existing `bridge` and `native` adapters stay supported.

Each run includes `provenance: {adapter, schema_version, sources:[{path, kind, available, updated_at, truncated}], metrics_scope, counter_scope}`. Source paths are allowlisted repository-relative file identities, not a browser file-read capability. Legacy loss aggregation is an unweighted mean of rank-local values, not an asserted globally target-weighted reduction. Current host telemetry remains separate from selected-run history and independent captures.

Profile graphs preserve separate axes/units/scopes for CUDA summed kernel duration, producer-reported HOST NVTX phases, Lens host spans and bridge host operations. Hover/focus/tap shows stage, raw name, source, rank/update/phase where supported, count/durations and evidence limitations. Symbol glossary explanations are name-based reference only, never source/shape/bottleneck inference. Unknown kernel-to-phase mappings remain unknown. OTLP/Lens resource/span identities must be preserved without synthesizing missing instrumentation.

Backend ownership: `dashboard/server.py`, `dashboard/collectors.py`, `dashboard/config.json`, backend tests in `dashboard/tests/test_backend.py` and `dashboard/tests/test_collectors.py`. Expose `collect_snapshot(root: Path, config: dict) -> dict` in collectors; server caches this and adds profiling through `collect_profiling(root, config)` from profiler module. CLI `python dashboard/server.py --root . --config dashboard/config.json --port 8787 --interval 5`.

Profiler ownership: `dashboard/profiling.py`, `dashboard/tests/test_profiling.py`. Expose `collect_profiling(root: Path, config: dict) -> dict` returning lens/nsight/bridge structure above. Config uses `profiling: {lens_spans: relative_path|null, nsight_sqlite: relative_path|null, nsight_summary: relative_path|null, bridge_operations: relative_path|null}`. Cache SQLite aggregation by stat, bounded read-only queries. Kernel records `{name, count, total_ms, average_us, percent}` preserving raw names; no inferred source attribution. Lens span records `{name, duration_ms, trace_id, span_id, parent_span_id, attributes}` with allowlisted attributes only; host span time never GPU time. Bridge operations expose role/explanation/source identifiers, not arbitrary source file contents. SQLite selected-process scope and potential overlaps explicit.

Frontend ownership: `dashboard/static/index.html`, `dashboard/static/style.css`, `dashboard/static/app.js`, `dashboard/tests/test_frontend.py`. No CDN/network dependencies; responsive dense dark monitor with green/amber semantic accents. Runs selector, KPIs, loss/LR/token charts, stages, data/model facts, hardware, benchmarks, Lens/Nsight/bridge panel, collection cost/staleness/provenance. Support SSE reconnect + polling fallback + pause display. DOM textContent for untrusted data. Nulls shown as unavailable. All user-facing primary explanation in Korean, code English.
