# Nsight Systems analysis with Codex

## Scope

Use CLI statistics and a SQLite export for reproducible, read-only agent analysis; use the GUI to validate timeline context. This guide belongs to P1/P3 of the [pretraining performance lab](PRETRAIN_PERFORMANCE_LAB.md). No trace has been collected or analyzed on the current host, and no repository analyzer is implemented by this guide.

## 1. Collect a useful trace

On an authorized GPU host, check the installed version and help before selecting flags:

```bash
nsys --version
nsys profile --help
nsys stats --help
nsys stats --help-reports
nsys export --help
```

Instrument the actual native training path with NVTX step/module ranges; capture a short steady-state window after compilation/autotuning/warmup. Include CUDA activity and NVTX; collect CPU sampling/runtime activity when needed and permitted. Record the exact command, enabled collectors, missing collectors/permissions, environment, source revision and rank/device mapping. Missing CPU sampling is missing evidence, not zero overhead. Choose capture-window and launcher options from the installed help: this document is not a complete native training launch recipe.

Keep distinct run/rank output names. First inspect a rank for readability, but capture relevant ranks before distributed claims. Preserve the original `.nsys-rep`; do not overwrite traces. Check trace completeness and whether capture starts/ends inside an event or step. Trace/NVTX text can contain paths or sensitive workload information; review before sharing outside the authorized environment.

## 2. Start with summary reports

In a new output directory containing the copied report, illustrative commands are:

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum --format csv baseline.nsys-rep
nsys export -t sqlite baseline.nsys-rep
```

Confirm report names and syntax against installed help. The first command writes formatted statistics to the console; save stdout to a new file if needed. The second normally creates `baseline.sqlite`; check the actual generated path. Stats may automatically export SQLite as well: inspect existing artifacts rather than force overwriting them. No command here was executed against a real report on this host.

Summary reports identify kernel counts, durations, synchronization APIs and NVTX costs, but do not establish launch starvation or the critical path. Kernel/API percentages are relative to their own report totals, not necessarily the training wall time.

## 3. SQLite analysis contract

Codex can analyze exported data using Python `sqlite3` without running training again. Open an absolute path with a properly escaped SQLite URI and `mode=ro`; do not silently create a new empty database for a missing filename.

Discover tables through `sqlite_master` and columns through schema inspection before constructing queries. Inspect metadata and available time/process/device/stream/correlation fields; schema is version-dependent. Resolve names through the actual string tables if required. Record schema, timestamp units and origin rather than assuming them.

- Select complete steady-state NVTX steps with explicit process/rank/thread scope. NVTX CPU range duration is not GPU duration. Nested or overlapping ranges require a declared attribution rule; avoid double counting.
- Extract per-step counts and kernel/API duration distributions, then bounded candidate windows with identifiers.
- Link launch APIs to GPU activities using actual correlation fields plus process/context scope where available. A correlation ID alone must not be treated as globally unique. Missing events or truncated capture can break the join.
- A kernel gap on one stream is not device-wide idle: inspect the interval union of all captured relevant compute/copy activity. Gaps in captured activity are not proof that untraced work was absent. Report scope explicitly.
- Do not sum overlapping kernel durations as wall time. Distinguish CPU API duration, queued launch-to-start interval, GPU execution duration and idle-gap candidate. A long queued interval may indicate dependencies or GPU backlog, not slow CPU submission.
- Check synchronization, transfer/data stalls, NCCL and rank imbalance as alternatives. CPU sampling/backtraces and source/NVTX mapping may be needed to substantiate a host-side cause; a CUDA kernel name alone is not a Python call site.
- Avoid time-filtered/partial exports that drop related events unless completeness is independently verified. Preserve bounded query outputs instead of dumping a giant database into agent context.

## 4. Ready-to-use Codex task

```text
Read AGENTS.md, NemoChatAgent.md and docs/en/PRETRAIN_PERFORMANCE_LAB.md.
Analyze the supplied baseline.sqlite read-only; do not run training,
modify source, install tools or start paid work.

First inspect schema, metadata, timestamp units and collector coverage.
If no complete steady-state steps or necessary events exist, report the gap.
Analyze the user-selected NVTX steps/ranks, not initialization or JIT.
Summarize CUDA APIs, kernels, counts and duration distributions.
Find repeated short-kernel sequences and launch/idle-gap candidates.
Correlate APIs and GPU activities using process/context-aware identifiers.
Inspect other streams/devices/ranks before attributing a gap to the CPU.
Separate launch starvation, sync, data stalls and communication/dependencies.
Map candidates to NVTX and source where evidence permits; do not invent mappings.

For each candidate include rank/device/stream, timestamp origin and units,
start/end, NVTX range, API/kernel identifiers, observation, hypothesis,
alternative explanations, confidence and a precise GUI confirmation task.
Propose Triton fusion only where the sequence and semantics support it.
Keep measurements separate from causal hypotheses.
Produce bounded CSV/JSON and a Markdown report in a fresh output directory.
Do not claim end-to-end speedup from trace summaries or microbenchmarks.
```

Replace “supplied” and “user-selected” with real paths/window/rank identifiers when dispatching. If no source mapping is collected, mark it unresolved and request narrower NVTX instrumentation instead of guessing.

## 5. GUI confirmation

Open the original report with a compatible Nsight Systems GUI. Locate the candidate using recorded timestamps/time origin, process/rank, stream and NVTX range; inspect CUDA API correlation, other streams, copies and dependencies. Distinguish a host thread waiting for the GPU from the GPU waiting for host submission. Confirm whether the region lies on the update critical path. Add a focused screenshot as supplemental evidence, not the numerical source of truth.

## 6. Baseline versus fusion

Use equivalent capture configuration, workload, steps and scope for before/after traces; match by step/NVTX semantics, not absolute timestamps across independent runs. Compare counts, sequence shape and gap candidates. Capture perturbation can change scheduling, so measure final repeated full-update throughput and memory with profilers disabled. Follow P2/P3 correctness, backward and resume gates before adopting a kernel.

## Proposed analyzer and validation

`scripts/analyze_nsys.py` is a future candidate, not an existing command. Proposed outputs under ignored `outputs/performance/<unique-run>/` are `summary.json`, `kernel_summary.csv`, `candidate_windows.csv` and `analysis.md` with trace identity, command/version, collector coverage and confidence.

If implemented, unit-test read-only/missing-file behavior, unknown schemas, timestamp units, overlapping intervals, multiple streams, correlation-ID reuse, nested NVTX, incomplete capture and missing events using clearly labeled synthetic SQLite fixtures. Then exercise against a real exported trace and independently check sample candidates in the GUI. Synthetic fixture tests do not validate Nsight export or real bottleneck diagnosis.

## Source

[Official Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html): CLI stats/export/analyze and collector/capture documentation. The CLI documentation supports CSV/JSON statistical reports and SQLite export; consult the installed version for exact supported options.
