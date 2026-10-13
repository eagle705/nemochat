# NeMo Lens Tutorial과 Nsight Bridge

Canonical: [영문 범위·검증 계약](../en/OBSERVABILITY_TRACKS.md).

## 두 폴더의 목적

- [Tutorial](../../tutorials/nemo_lens/README.ko.md): 실제 Lens SDK로 작은 offline 학습을 계측하고 local trace UI에서 확인한다. 설치·실행·UI 탐색과 instrumentation 오버헤드를 함께 배운다.
- [Nsight Bridge](../../plugins/nemo_lens_nsight_bridge/README.ko.md): 별도 웹 dashboard가 아니라 기존 Nsight에서 MoE 커널의 의미와 실제 코드 위치를 모델러가 이해하도록 돕는 선택적 instrumentation/plugin이다.

기존 학습·WSD/WSM·checkpoint/resume 경로와 main dependency 환경을 바꾸지 않는다. GPU 대여·외부 telemetry·공개·commit은 이 작업에 포함되지 않는다.

## Tutorial에서 확인할 것

1. 실제 optimizer update와 SDK span export를 실행한다. 작은 CPU control은 NemoChat target-model 학습·품질 증거가 아니다.
2. 전용 loopback backend에 실제 span이 들어왔는지 API로 확인한다.
3. UI를 브라우저로 열고 service 선택 → 검색 → trace 열기 → step/하위 span 확인을 수행한다. HTTP 응답·API 성공과 화면 렌더링을 구분한다.
4. wrapper 없는 baseline / disabled / enabled local export / enabled OTLP를 같은 입력·seed·thread 수로 비교한다. 독립 subprocess, warmup, 반복·교차 순서, raw timing과 median, 동일 결과 확인, flush/shutdown 시간을 포함한다.
5. **오버헤드 없음은 기본 가정이 아니다.** 짧은 CPU 측정은 실제 GPU·분산 MoE overhead를 대신하지 않는다. 이후 전체 update throughput·memory·rank 수·span 상세도·exporter 실패 영향까지 재측정한다.

Upstream의 예제 compose는 Weave routing이 기본이므로 그대로 복사하면 local UI가 비어 있을 수 있다. 이 tutorial의 local 구성과 upstream production 권고를 혼동하지 않는다.

## Bridge에서 확인할 것

- Router → 선택/정리 → expert compute → combine 등의 실제 코드 범위를 이해하기 쉽게 설명한다. 기존 backend NVTX를 먼저 조사하고 중복 annotation을 피한다.
- 원래 CUDA 커널 이름을 보존한다. CPU scope 시간과 실제 GPU kernel 시간을 구분한다.
- 실제 프로파일링 당시 source file·line·hash로 snippet을 추출한다. 변경된 파일·path traversal·모호한 mapping은 거부한다. LLM이 그럴듯한 코드를 만들면 안 된다.
- 동일 bridge ID를 Lens span과 NVTX에 넣는 것은 연결 수단이다. 그 자체가 커널 알고리즘·정확한 shape·성능 병목을 증명하지 않는다.
- CUDA correlation/backtrace/explicit annotation/compiler metadata 기반 mapping을 구분한다. 시간상 가까운 Python 라인으로 추정 연결하지 않는다.
- Fusion, backward, CUDA Graph replay는 한 커널=한 Python 줄 가정을 깨므로 별도 검증한다.
- 교육용 CPU sparse-MoE는 native fused Nemotron/Megatron EP 통신 구현이 아니다. 가짜 NCCL/dispatch communication 이벤트를 만들지 않는다.

공개 문서에서 확인한 collection/configuration plugin과 **native GUI 설명·코드 패널**은 다른 기능이다. 실제 plugin discovery·capture·GUI 확인은 호환되는 Nsight와 GPU가 필요하다. 설명 패널은 지원 API 또는 제품팀 협업을 확인한 다음 단계다.

## 실제 브라우저 UI 검증

`scripts/verify_lens_ui.py`는 별도 Playwright 환경에서 실제 Jaeger 검색 페이지와 저장된 학습 trace를 열고 span 이름·JavaScript 오류·스크린샷을 확인한다. 새 viewer가 아니며 Nsight native GUI를 검증하는 도구도 아니다.

```bash
python3 -m venv outputs/lens-browser/.venv
outputs/lens-browser/.venv/bin/python -m pip install playwright
outputs/lens-browser/.venv/bin/python -m playwright install chromium
outputs/lens-browser/.venv/bin/python scripts/verify_lens_ui.py \
  --url http://127.0.0.1:16687 --output outputs/lens-ui-check
```

먼저 tutorial backend를 시작하고 실제 training trace를 전송한다. 짧은 trace를 보려면 실제 `result.json`의 `trace_id`를 `--trace-id`로 전달한다. 출력 폴더는 새 경로여야 한다. Browser 다운로드·OS 라이브러리 문제는 blocker로 표시하며 가짜 화면으로 대체하지 않는다. 이 dependency는 main environment와 분리한다.

스크린샷·body text에는 trace 내용이 들어가므로 private ignored output에 유지하고 고객 자료를 자동 업로드하지 않는다. Verifier는 승인한 origin만 허용하며 redirect·외부 요청·service worker·WebSocket을 차단한다. 실제 trace DOM에서 full ID·service·모든 예상 operation을 확인하고 실패하면 성공 receipt를 만들지 않는다.

Browser 회귀 테스트는 opt-in이다. Main dependency를 바꾸지 않고 project pytest를 read-only import한 실제 실행 방식:

```bash
PROJECT_SITE=$(.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
PYTHONPATH="$PROJECT_SITE" NEMOCHAT_LENS_BROWSER_TESTS=1 \
  outputs/lens-browser/.venv/bin/python -m pytest -q \
  tests/test_lens_ui_browser.py tests/test_lens_ui_cli.py
```

Browser 환경 경로를 실제 설치 경로로 바꾼다. 부모는 `/home/ubuntu/runs/nemo-lens-browser/.venv`를 사용했다. 기본 suite에서 browser 테스트를 skip하는 것과 위 별도 실행의 통과를 구분한다.

## 안전과 상태 기록

Local UI는 loopback에만 bind하고 SSH tunnel로 접근한다. 기존 container를 건드리지 않는다. Customer 코드·경로를 외부 모델에 자동 전송하지 않는다. 결과는 fresh ignored output 경로에 저장한다.

상태는 source-verified / CPU-executed / backend-ingestion-verified / GUI-rendered / GPU-executed로 분리한다. CPU 테스트 통과를 GPU plugin/UI 실행 성공으로 표시하지 않는다. 실제 결과와 다음 실행 gate는 각 폴더의 문서 및 [NemoChatAgent.md](../../NemoChatAgent.md)에 기록한다.
