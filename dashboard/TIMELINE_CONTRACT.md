# Raw-symbol decoder and selected-iteration timeline extension

Keep stock trainer, raw capture/source files and immutable old outputs unchanged. No GPU workload/recapture/attach/collector. Parent owns frontend/config/docs; backend child owns capture_analysis.py, inspector.py and tests/test_capture_analysis.py; symbol child owns kernel_symbols.py and tests/test_kernel_symbols.py only. Use strict TDD and current tests. Raw names remain primary, explanations alongside, never rename symbols.

## Timeline v1

Optional `update.timeline` for selected measured updates only (not duplicated per phase). Recompute from actual SQLite with the existing same-process runtime correlation and ambiguity refusal. GPU timestamps and API timestamps are distinct. Do not manufacture a cross-stream total dependency order. Optional missing stream/context/API-name columns produce null, not fake stream0 or guessed launch names.

```
{
 "schema_version":1,"available":true,
 "origin":"selected_host_nvtx_start",
 "host_duration_ns":123,"span_duration_ns":456,
 "order":"gpu_start_ns,row_id", "total_events":123,
 "returned_events":123,"truncated":false,
 "names":["exact raw GPU symbols"],"api_names":["recorded runtime API names"],
 "threads":["decimal globalTid strings"],
 "phases":[{"id":"forward","start_ns":0,"end_ns":100}],
 "fields":["row_id","name_index","gpu_start_ns","gpu_end_ns","device_id","context_id","stream_id","correlation_id","api_start_ns","api_end_ns","thread_index","phase_index","api_name_index"],
 "events":[[1,0,10,20,0,1,7,42,1,2,0,0,0]]
}
```

Times are integer nanoseconds relative to selected host NVTX start; numeric offsets/IDs must be safe in JavaScript (<2^53). globalTid stays an exact decimal string. `phase_index` uses complete *CPU API* containment in one recorded phase; null means no/ambiguous phase. `api_name_index`, context_id, stream_id may be null when genuinely absent. Raw kernel GPU intervals remain untrimmed; span_duration_ns includes kernels that extend past host scope. Do not claim all GPU operations, dependencies, memcpy, OS events or an Nsight Systems replacement: this is correlated kernel/API + selected host phase coverage.

Sort events stably by GPU start then SQLite kernel row id. Maximum16384events/update,4096kernel names,512API names,256threads,16phases; overall artifact still fits existing bounded JSON reader. Limit/truncation totals explicit and internally consistent; never discard fast kernels because bars are tiny. Read-only bounded SQL, use real stream/context columns after schema discovery, never assume columns from synthetic fixtures. Preserve old artifact compatibility (timeline absent = explicitly unavailable).

Importer type/bounds/index/order/timing/count checks, allowlisted projection only, no raw JSON or caller assertion passthrough. Cache with existing artifact/capture fingerprints; no heavy SQL/live poll. Parent UI uses canvas for full events, bounded paginated chronological table, GPU device/context/stream lanes and CPU-thread API/host phase lanes, exact timestamp hover/click detail and zoom/window. Distinguish enqueue order vs measured GPU-start order and overlapping streams; missing lanes/phase/source explicit. Source selection may link to the existing aggregate inspector but must not invent launch frames.

## Symbol decoder

`decode_symbol(name:str)->dict` pure bounded local parsing, no network/runtime work:
```
{"family":"NCCL|nvjet|other","evidence":"mixed_symbol_reference",
 "parts":[{"token":"RING","label":"...","meaning":"...","confidence":"documented|inferred|unknown","sources":["official primary-source URL"]}],
 "warnings":["Not Python caller / input tensor shape proof"]}
```

Keep token strings literally unchanged (HTML entities may only be interpreted on a separate parsing copy). NCCL algorithm/protocol and capacity facts require primary source verification. NVJET internal opaque tokens must be marked unknown when public documentation does not establish exact meanings; do not make up tss/64x9/2x2/h/badd/NTT expansions or infer M/N/K/dtype from symbol alone. Numeric tuning tokens are not automatically actual input tensor sizes. Local installed library/schema corroboration is allowed, but distinguish observed launch metadata from convention guesses. Source URLs are fixed allowlisted references, not artifact-provided navigation or remote lookup per render.
