# Profile graphs, provenance and native Lens data fidelity — executed update

## Verified on 2026-10-13, 20:04 KST

The parent reran the integrated dashboard suite: **106 passed**. Python compilation, Node syntax and `git diff --check` returned zero. Raw commands/output: `logs/dashboard/20261013-200018-update-verification.log`.

The running server was restarted with the completed backend at loopback `http://127.0.0.1:8787`. Independent HTTP read-back retains eight configured runs/eight GPUs and the actual Bridge counters4096/4096/4128. Two new SSE snapshots differ, including the source metadata: `results/dashboard_update/20261013-200018/sse-readback.json`. Twelve actually attempted input files for the selected Bridge run are displayed; these are source/availability/mtime/bounded-read facts, not an assertion that every file is independently authenticated.

## Profile time visualization

- Top12 CUDA kernel cumulative-duration bars, preserving raw kernel names and whole-capture scope. No CUDA sum is interpreted as elapsed wall time.
- Separate producer-reported HOST NVTX phase bars for rank/update data, forward, backward+sync and optimizer. These are not kernel durations.
- Lens host-span and bridge host-operation bars when their own artifacts are available.
- Pointer hover/move, keyboard focus/Enter/Escape and tap inspection. Tooltip includes stage, raw identity, count/durations, source/scope and conservative symbol-reference meaning. Unknown kernel→rank/update/phase/source mappings remain unknown.
- Desktop kernel and host-phase charts use separate side-by-side axes; mobile uses internally scrollable chart panels. Actual desktop/mobile checks showed no document-wide horizontal overflow.
- Parent moved the actual Chrome pointer over the recorded NCCL AllGather bar. Tooltip visibly showed85calls,4726.8429msCUDA cumulative duration,55609.9164usmean and25.5%duration share, plus explicit unverified-source and whole-capture warnings. No model-layer/shape/bottleneck attribution from these names.

Real browser captures:
- `results/dashboard-profile-timing-hover.png`
- `results/dashboard-data-sources.png`

The parent checked profile/run data, source-table expansion, actual tooltip visibility and captured error/unhandled-rejection checks during interactions; no errors were recorded. Desktop1440×1080 and mobile390×844 remained within document width.

## Stock trainers remain unchanged

Existing Bridge/native logs and native NeMo Lens/OTel exports remain primary. No dashboard loss callback, monkeypatch, trainer wrapper, mandatory writer, trainer restart, GPU job, profiler attach or telemetry collector was added. Canonical v1 is an **optional read-only derived/export interchange format**, not an imposed trainer output contract.

`telemetry.schema.json` and `telemetry.py validate` enforce run/stage identity, designated writer, finite numeric/null metrics, global cumulative counters, monotonic records, bounded append-only reads and sanitized rejection. Rank-local loss remains rank-local. Existing adapter loss aggregation is explicitly an unweighted rank-local mean, not a target-weighted global reduction. Optional reward/policy/value/KL/entropy/distillation metrics are contract fields, not newly measured model scores.

Parent CLI validation exercised two synthetic records, exit0, with unchanged input SHA256s. This is contract evidence, not training evidence:
`results/dashboard_update/20261013-200018/canonical-synthetic-fixture/parent-cli-receipt.json`.

## Native NeMo Lens data interfaces

The reader now handles Console JSONL, OTLP ExportTraceServiceRequest JSON with resourceSpans/scopeSpans/spans, pretty JSON and batch JSONL. OTLP JSON data import is **not an OTLP network receiver** and no collector endpoint was started. Standard hex IDs and validated generic protobuf base64 IDs preserve trace/span/parent identity. Nanosecond endpoints are subtracted as integers before unit conversion.

Allowlisted Lens/MCore resource/span fields include explicit `nv.dl.run.uuid`, `nv.dl.rank`, `dl.iteration`, `dl.microbatch_id` and supported Megatron semantics. The viewer does not infer native integration or run identity from operation names. Educational-control warnings apply to actual valid bridge.id annotations, not all Lens records.

A fresh, network-blocked, host-only subprocess executed the installed **Lens0.2.0+4d045c0 / OTel1.43.0** public interfaces: NemoLensConfig, SpanRegistry, setup_telemetry, managed_span, a local exporter, encode_spans and protobuf MessageToDict. The parent separately read back the real export and checked exact trace/span/parent IDs and integer-derived durations. The two validation spans are not native training or benchmark evidence and were not added to the default dashboard run/profile configuration. Existing21educational Console spans remain compatible.

Preserved SDK output and independent receipt:
`results/dashboard_update/20261013-200018/lens-sdk-host-probe/{receipt.json,otlp.json,console.jsonl}` and `parent-readback.json` in its parent directory.

**Still not established:** native Lens activation in the installed Bridge/Core training runtime, complete all-rank coverage, CUDA kernel→source correlation, monitoring slowdown, or later-stage captures. Upstream span-group documentation was inspected separately; that is not validation that this host's exact native SDK supports config-only activation. Instrumentation-free trainers cannot emit missing spans merely by setting environment variables.

## 한국어 요약

- 시간 그래프·실제 호버·입력 파일 출처 표를 브라우저에서 확인했고 전체 대시보드 테스트106개가 통과했습니다.
- CUDA 누적 시간, HOST NVTX 구간, Lens/bridge host 시간을 구분합니다. 커널 뜻은 명칭 기반 참고 설명이며 소스/구간 매핑 미확인은 그대로 표시합니다.
- trainer는 순정 그대로입니다. 기존 로그 가공이 기본이며 canonical JSONL은 선택적 파생/export 입력 계약입니다. 별도 trainer callback/writer를 설치하지 않았습니다.
- Lens의 표준 Console/OTLP JSON 자료를 직접 읽습니다. 실제 SDK의 로컬 export로 ID·계층·시간을 확인했지만 native 학습 안에서 Lens를 활성화·검증한 것은 아닙니다.
- 서버 주소는 기존 localhost8787이며 사용자 PC에서는 SSH 터널로 접속합니다. 상세 데이터 소스·해석은 DATA_SOURCES.ko.md를 참고하세요.
