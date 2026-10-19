# Kernel inspector extension contract

Keep stock trainers and current formats unchanged. No GPU workload, recapture, profiler attach, source injection, remote telemetry or collector startup in this task. Native capture has 3,569,543 runtime rows and ZERO non-null callchainId rows, no stack/symbol tables. Precise native kernel-to-Python frame attribution is unavailable in this capture; do not invent it. Host-scope source snippets can be verified separately, but are not kernel stack frames. Existing educational Bridge operation IDs can verify saved host source with the original plugin snapshot/hash/identity rules.

Ownership:
- Analysis/backend agent: `capture_analysis.py`, `inspector.py`, `tests/test_capture_analysis.py`, `profiling.py`. Computes offline selected NVTX/runtime/kernel correlation and imports a sanitized inspector artifact. No source helper/frontend/config/server edits.
- Source agent: `source_evidence.py`, `tests/test_source_evidence.py`. Public `verified_bridge_source(root:Path, operations_path:Path, record:dict)->dict`, `verified_native_phase_source(root:Path, source_path:Path, expected_sha256:str, phase:str)->dict`. No profiling/analysis/frontend/config edits.
- Frontend agent: static/app.js/style.css/index.html, tests/test_inspector_frontend.py (may update own test_frontend.py). No backend/config/docs edits.
- Parent: config/server path validation, docs, actual execution/restart and browser screenshots.

Offline CLI (source paths repository-relative and bounded/read-only; no overwrite output):
`python dashboard/capture_analysis.py --root . --sqlite results/.../capture.sqlite --analysis results/.../analysis-parent-rank0.json --native-source results/.../native_profile.py --source-provenance results/.../source-provenance.json --output results/dashboard_inspector/<fresh>/inspector.json`

Input analysis selects actual ranks/updates/phases with exact NVTX names; recompute them from SQLite. CUDA launch association uses correlationId + globalPid == (globalTid & 0xFFFFFFFFFF000000), complete API containment across all SAME-PROCESS threads, strict duplicate/ambiguity refusal. Preserve worker-thread launches; no assumption of parent thread alone. Use installed report rule source referenced by existing analyzer. GPU duration sum may overlap; per-device/process interval unions are scope-limited, not utilization. Separate update vs phase counts and no parent/child sums. Exclude startup/warmup by selecting actual measured updates 3 and 4; do not rank optimization using whole-capture startup-heavy Fill/AllGather stats. Queries bounded by progress handler / row ceilings; pure read-only SQLite. Offline JSON cached in live server, never run expensive analysis per browser or every poll.

Output `inspector.json` and snapshot `profiling.<stage>.nsight.inspector` (where <stage> means stages[stage], top-level mirrors current legacy selection) schema:
```
{
 "schema_version":1,"available":true,"source":"relative inspector path",
 "scope":"rank0 selected updates only, instrumented independent native diagnostic",
 "evidence_level":"process_nvtx_runtime_correlation",
 "sqlite":{"path":"...","sha256":"...","size":123,"mtime_ns":123},
 "callstacks":{"runtime_rows":3569543,"runtime_rows_with_callchain":0,"python_source_attribution":false},
 "updates":[{"rank":0,"update":3,"name":"native/rank_0/measured/update_3","host_ms":2541.88,"kernel_count":7025,"kernels":[KERNEL],"phases":[{"id":"forward","name":"native/rank_0/update_3/phase/forward","host_ms":2143.62,"kernel_count":2433,"summed_kernel_ms":0,"activity_union_ms":null,"source":SOURCE,"kernels":[KERNEL]}]}],
 "candidates":[{"priority":1,"title":"...","rank":0,"update":3,"phase":"forward","category":"...","evidence":"actual selected metrics","interpretation":"candidate not critical-path certainty","action":"one bounded experiment","confidence":"candidate","source_confidence":"host_scope_verified|unavailable"}],
 "warnings":[]
}
KERNEL = {"name":"raw","category":"...","meaning":"Korean reference description","meaning_evidence":"symbol_reference|unclassified","count":123,"total_ms":1,"average_us":1,"percent":1,"worker_launches":1,"source_attribution":"unavailable","source":null}
SOURCE = {"status":"host_scope_verified|snapshot_verified|unavailable|rejected","kind":"host_scope|explicit_bridge_operation","path":"repo/source identifier","function":"...","start_line":1,"end_line":5,"file_sha256":"...","snippet_sha256":"...","source_id":"...or null","snippet":"exact saved lines","warning":"Not a native kernel Python stack frame"}
```

`profiling` config per group gains optional `nsight_analysis` path. Imported artifact should be bounded, typed/allowlisted, record caps (<=8updates,<=16phases/update,<=64kernels per scope,<=12candidates); compare configured SQLite identity/fingerprint safely and cache full digest by unchanged stat. Source snippets have strict bounded lines/bytes (e.g.60lines/12KiB), verified full-file hash and snippet hash; not arbitrary HTTP file reads. Imported assertion of source verification is not authenticated unless source helper validates it against explicit local configured evidence. Fail closed if mismatch. Display available and rejected separately.

`profiling.bridge.operations[]` can gain `source_evidence: SOURCE` from configured operation's saved source reference, verified using existing plugin scheme. Hash verification does not authenticate malicious producer rewrites and never establishes kernel launch/source correlation.

Frontend adds selected update/phase controls within Nsight panel, defaults to selected measured scope when inspector available, keeps whole-capture aggregate as explicit alternate diagnostic view. Shows exact raw symbols as primary identities with semantic/category/token reference annotations alongside; counts/time/rank/phase/coverage and short what-it-does explanations; kernel row/bar selection drives a source/evidence inspector. If native kernel source unavailable, shows verified host scope nearby but labels different confidence; no generic reference snippets masquerading as captured Python frames. Source view exact code only with line numbers and immutable hash. Bridge UI displays validated existing operation source separately. Ranked investigation cards say observed target vs hypothesis, necessary next experiment, no invented speedup or whole-node bottleneck certainty. Null/missing source is explicit. Preserve SSE/source/stage isolation and safe textContent; support keyboard/tap/mobile. Existing 106-test baseline and backward-compatible absence of inspector.
