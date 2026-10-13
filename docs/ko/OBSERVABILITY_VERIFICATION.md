# Lens Tutorial / Nsight Bridge — 부모 실행 검증

Canonical: [전체 명령·수치·범위](../en/OBSERVABILITY_VERIFICATION.md).

## 실제 확인한 것

- 기존 파일을 보존하고 main dependency 환경을 변경하지 않았다. 시작 baseline: 765 passed, 2 skipped, 25 subtests.
- Tutorial CPU MLP를 4 update 실행해 실제 local Jaeger로 OTLP 전송했다. root 1 + step 4 + forward/backward/optimizer 각 4 = **17 span**. 개수·run identity·계층 연결 확인.
- 실제 Chromium으로 Jaeger 검색·trace 페이지를 렌더링했고 다섯 operation 이름과 네 step waterfall을 확인했다. JavaScript 오류 없음. API 성공만으로 GUI 성공이라 하지 않았다.
- 실제 screenshot: `outputs/lens-ui-parent-03/trace.png`; 저장소 browser 검증 script의 초기 실행 결과는 `outputs/lens-ui-parent-04/`. 후속 리뷰에서 verifier의 browser origin 제한·모든 operation/trace identity 검증이 부족함을 확인했다. 실제 화면 관찰은 유효하지만 초기 verifier 구현의 승인으로 해석하지 않는다. 별도 수정·실패 경로 검증 전까지 최종 승인은 보류한다.
- Bridge CPU sparse-MoE(8 token, 4 expert, top-2)를 실행해 forward·gradient reference 비교 통과. **21 operation / 21 Lens span**의 bridge ID 집합이 정확히 일치했다.
- 실제 activation 소스 snapshot을 hash 검증 후 추출했다. 모델/LLM이 만들어낸 코드가 아니라 `control.py`의 실제 SwiGLU 함수다.

## 오버헤드 — 0이 아니다

부모 독립 측정은 200 update, warmup 40, 네 mode를 네 번씩 교차 순서로 독립 subprocess 실행했다. 모든 arm의 loss sequence·최종 parameter hash가 일치하고 export 개수 확인도 통과했다.

| Mode | Median training ms / 200 update | Baseline 대비 |
|---|---:|---:|
| Wrapper 없는 baseline | 87.397029 | 기준 |
| Disabled wrapper | 93.068929 | +6.489809% |
| Enabled file | 185.613084 | +112.379169% |
| Enabled local OTLP | 153.911699 | +76.106329% |

이것은 아주 작은 CPU MLP의 세밀한 계측 비용이다. 일반적인 Lens overhead나 실제 GPU MoE overhead로 일반화하면 안 된다. Setup·shutdown·전체 process 시간과 raw duration은 `outputs/lens-parent-overhead-01/benchmark.json`에 별도 기록했다. Worker의 400 update/8반복 측정은 다른 실행이므로 이 표와 섞지 않는다.

## 남은 blocker

- GPU driver가 동작하지 않고 `nsys`가 없다.
- 요청한 NVTX 및 Nsight launch는 exit 2로 실패했고 성공 output 폴더를 만들지 않았다.
- 실제 plugin loading, CUDA kernel→source correlation, fused EP/native Nemotron, Nsight GUI 설명·코드 패널은 **미검증**이다.
- Native 학습 전체 throughput·memory·분산 rank/계측 상세도·exporter 실패 비용 재측정이 필요하다.

## 테스트

- Tutorial 독립 환경: 9 passed.
- Bridge 독립 환경: 32 passed.
- Browser CLI: 7 passed.
- Main suite: **772 passed, 2 skipped, 25 subtests**.
- Main pip check·compileall·diff check 통과. 독립 리뷰는 Tutorial exporter의 `.netrc` 인증·redirect 외부 전송 문제 두 건으로 **FAIL**했다. API 검증기의 redirect 방어와 exporter 전송 방어는 별개였다. 수정 worker를 별도로 실행했으며 수정 후 테스트·재리뷰 전에는 승인하지 않는다. Bridge의 검토된 코드에는 blocking 문제가 없었으나 launcher/NVTX 실패 회귀 테스트 보강이 권고됐다. 초기 Bridge worker timeout으로 누락된 work log·문서도 보완 대상이다.

## 수정 후 부모 재검증

- Tutorial의 local-only Session은 `trust_env=False`, redirect 비활성화와 3xx 거부를 적용했다. 부모 원본+보안 테스트 **15 passed**. 새 실제 전송 `outputs/lens-parent-privacy-fixed-01/`에서 **17 span** 계층·identity 확인. 새 overhead 측정은 아니다.
- Bridge 부모 테스트 **40 passed**. Launcher/NVTX 실패 경로 포함. 수정 후 새 CPU 실행 `outputs/lens-bridge-parent-fixed-01/`의 forward·gradient 비교 통과. 누락 WORK_LOG.md 보완 및 local artifact ignore 추가.
- 두 폴더 독립 재리뷰 **PASS**: blocking security/logic 없음, reviewer도 Tutorial 15 / Bridge 40 테스트 통과. 부모가 검토된 SHA256과 현재 파일 일치를 확인했다. Verdict는 `outputs/observability-reviewed-verdict.json`. 별도 browser 수정도 부모가 실행했다: **Chromium/CLI 38 passed**. Redirect·외부 resource·WebSocket·잘못된 ID·누락 operation·JS 오류를 실패 처리한다. WebSocket callback deadlock을 tight test로 재현하고 handshake 전 동기 close를 제거했으며 목적지 미접촉을 확인했다. 실제 최종 화면 `outputs/lens-ui-parent-hardened-02/`는 full trace ID·service·다섯 exact operation을 확인한 17 span trace다. Main 최종 **783 passed, 22 skipped, 25 subtests**: 20 browser opt-in skip은 별도 실제 실행 통과, 기존 skip 2개. Browser 독립 재리뷰 **PASS**: reviewer Chromium/CLI 38 passed, security/logic blocker 없음. 부모가 세 파일 hash 일치를 확인했다. Proxy 환경·service worker·same-origin WebSocket·popup 추가 테스트는 nonblocking 권고다. GPU/Nsight 미검증과 구분한다.

Jaeger UI는 `127.0.0.1:16687`에만 열려 있고 원격 접속은 tutorial의 SSH tunnel 절차를 따른다. 메모리 저장이므로 container 재시작 시 trace가 사라진다. Commit·push·공개·외부 telemetry·유료 GPU 실행은 하지 않았다.

## 런타임 경로 migration 참고

위 과거 실행 기록의 경로·timestamp·결과는 원래 증거로 보존합니다. 새 실행은 [저장소 루트 런타임 규약](RUNTIME_LAYOUT.md)을 따르고, 이동된 로컬 산출물은 runtime migration manifest의 old → new 매핑으로 찾습니다. Receipt의 remote path/revision/hash와 inventory 상대 경로를 변경하거나 기존 산출물을 덮어쓰지 않습니다.
