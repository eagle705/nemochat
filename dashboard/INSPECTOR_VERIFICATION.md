# Semantic kernel/source inspector — execution verification

## Parent result (2026-10-13 21:11 KST)

**171 related tests passed:** 138 dashboard tests + 33 saved-source verification tests. JUnit counts were parsed and reconciled. Node syntax, Python compilation and `git diff --check` exit0. This is focused dashboard/source coverage, not a rerun of the full GPU project suite.

Raw receipts:
- `results/dashboard_inspector/parent-20261013-204627/final-verification.log`
- `results/dashboard_inspector/parent-20261013-204627/final-tests.xml`
- `results/dashboard_inspector/parent-20261013-204627/execution-receipt.json`
- `results/dashboard_inspector/parent-20261013-204627/live-readback.json`

## Real analysis, not illustrative numbers

The parent independently exercised the new offline CLI against the actual exported SQLite and saved selection/source records:

```bash
.venv/bin/python dashboard/capture_analysis.py --root . \
  --sqlite results/e2e/20261013-1539/profile_retry/rank0-01-trace/capture.sqlite \
  --analysis results/e2e/20261013-1539/profile_retry/analysis-parent-rank0.json \
  --native-source results/e2e/20261013-1539/profile_retry/native_profile.py \
  --source-provenance results/e2e/20261013-1539/profile_retry/source-provenance.json \
  --output results/dashboard_inspector/parent-20261013-204627/inspector.json
```

That output is already reserved; the CLI refuses overwrite. Use a fresh output for another execution. It is a read-only CPU analysis, never a profiler launch.

Actual selected rank0 updates3/4 retained7025/6834kernels and3537/3417worker-thread launches. API correlation uses recorded correlationId plus process identity across same-process threads; duplicate/ambiguous reuse fails closed. The whole capture's startup-heavy631489kernel aggregate is not used for selected-update investigation priorities.

Observed leading targets:
- Update3 **forward AllGather**:23calls,1655.210039ms summed CUDA duration,95.4182388100572% of that phase's summed kernel duration.
- Update4 **forward AllGather**:23calls,2189.044144ms,98.02283492659052%.
- Update3 **backward+sync SendRecv**:69calls,250.177988ms,88.25597398761069%.
- Update4 **backward+sync SendRecv**:69calls,212.038401ms,86.70329765500895%.

These identify **communication as the first investigation area in this captured diagnostic**, not proved whole-node critical-path causes. NCCL duration can include peer readiness/waiting; instrumented rank0 statistics do not distinguish pure data-transfer cost, other-rank delay and collection effects. Actions therefore check ranks/groups/buckets/launch dependencies first, confirm the actual caller, and validate any supported overlap/bucket change with correctness/memory/unprofiled full-update A/B. No automatic setting change or claimed speedup.

The capture is an independent manual native MCore diagnostic, not the selected actual Bridge training pilot. Source, scope, instrumented timing and missing all-rank coverage remain visible.

## Source evidence actually verified

The parent read back eight native phase source cards (four phases across two updates), verifying saved file SHA256 `4a477a375757202812bf26f93ba2c68571928957f30025c8c7bb6126109a4325`:
- data:124–129
- forward:130–131 (`model(...)`)
- backward+sync:132–133 (`loss.backward(); finalize_model_grads(...)`)
- optimizer:134–135 (`opt.step()`)

These are **host-scope source snippets, not exact kernel Python frames**. The SQLite still has3569543runtime rows and0non-null callchain IDs. No native kernel is upgraded to verified Python attribution.

All21saved Bridge operation references verify across9distinct source IDs/functions. Example actual saved source: `lens_nsight_bridge/control.py`, `routing`, lines6–7. File SHA256 `aada59cdeece1938dd1699a3d3e870d2bdc3205879af33fc3b291191989174f8`. This is educational explicit host instrumentation, separately unscoped, not native MoE kernel-source proof.

Verification uses descriptor-safe bounded reads, immutable file/snippet hashes, source identity, typed inclusive line bounds and exact stored text. No source execution or arbitrary browser file route. Declared/changed/rejected source cannot masquerade as validated code. Hashes detect drift but do not authenticate a maliciously rewritten producer/manifest pair.

## Live UI and service

Default config now points to the parent inspector artifact plus explicitly allowlisted native source/provenance. Source snippets are regenerated/verified by the importer; artifact snippets are not blindly trusted. SQLite digest is checked/cached and analysis is not rerun per browser or live collection.

The parent restarted the running server at `http://127.0.0.1:8787` and verified real HTTP availability, measured controls, kernel/source selection, rank0/update4/forward facts, exact saved snippets, candidate actions and Bridge source display. Actual Chrome interaction recorded no error events or document-width overflow. The source viewer was corrected after a browser review to avoid displaying an extra trailing line beyond the verified end range. Collectives and peer send/receive use distinct human labels rather than one misleading collective label.

Captured screens:
- `results/dashboard-inspector-native-source.png`
- `results/dashboard-inspector-priorities.png`
- `results/dashboard-inspector-bridge-source.png`

No trainer, source snapshot, checkpoint, optimizer, GPU workload, recapture, collector, SDK dependency or remote export changed. Service remains background/loopback, not public authenticated/autostart deployment.

## Remaining boundary

The current analyzer/importer intentionally supports this saved capture's rank0 measured updates3/4 and known native phase labels. Other capture naming/selection schemes need their own explicit adapter; UI controls do not imply every native trainer is automatically supported.

Exact inner native Python caller frames require a **new approved, backtrace-enabled diagnostic capture and a verified reader for its actual exported stack schema**. Installed help was inspected, but no such capture or source-frame decoder validation is claimed. Autograd workers, compiled/fused operations and graphs may require additional metadata; even a captured host frame does not automatically identify every originating forward source or a causal critical path.

## 한국어 요약

- 관련 테스트171개(대시보드138+소스검증33), 문법/공백 검사 통과. 실제 SQLite를 부모가 다시 분석하고 live API·Chrome 화면을 확인했습니다.
- 초기화 제외 rank0측정 update3/4에서 forward AllGather가 해당 구간 CUDA 누적 시간의95.4–98.0%, backward+sync SendRecv가86.7–88.3%를 차지했습니다. 통신을 먼저 조사할 근거이며 전체 노드 병목/순수 전송 비용/속도 개선을 확정하지 않습니다.
- 실제 파일·행 번호·코드·hash를 표시하며 native8host구간과Bridge21연산/9sourceID를 검증했습니다. native 내부 Python frame은 수집되지 않아 여전히 미확인입니다.
- trainer는 순정 유지, 새 GPU 실행 없이 기존 자료만 사용했습니다. 내부 native 호출 위치까지 확정하려면 승인된 추가 stack capture가 필요합니다.
