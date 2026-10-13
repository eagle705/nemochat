# Rollout 지연 진단: Generation인가, Tool 호출인가?

Canonical: [English](README.md). Native 연결·coverage 한계: [RL/Gym 가이드](../../../docs/ko/RL_ROLLOUT_OBSERVABILITY.md).

이것은 **실제로 실행하는 CPU 관측 실습**이다. 학습된 tool-use agent, NeMo RL/Gym/vLLM 실행, RL optimizer 또는 공식 benchmark가 아니다. 각 trajectory는 실제 LM generation → 실제 loopback HTTP 계산 도구 → 결정적인 정답 검증을 수행한다. Tool과 arguments는 script가 정하며, 생성된 텍스트로 선택하지 않는다.

## 환경

상위 tutorial `.venv`를 재사용하며 script는 dependency를 설치하거나 바꾸지 않는다. 실제 확인: Lens `0.2.0+4d045c0`, OTel `1.43.0`, torch `2.14.1+cpu`, transformers `4.57.6`.

- `sshleifer/tiny-gpt2`, revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`가 사전에 cache에 있어야 한다.
- `local_files_only=True`: cache가 없으면 실패하고 자동 다운로드하지 않는다.
- 4개 trajectory, 각각 실제 새 token 8개, 계측 밖 warmup 1회.
- Caller semaphore가 CPU generation을 한 번에 하나만 허용한다. Tool 요청은 병렬 실행한다.
- Queue span은 이 semaphore 대기이며 vLLM queue·GPU scheduling·prefill/decode가 아니다.

[상위 tutorial](../README.ko.md)의 Jaeger를 먼저 실행한다. 기본 OTLP `127.0.0.1:14318/v1/traces`, UI/API `127.0.0.1:16687`. 실습 script가 container를 시작하거나 변경하지 않는다.

## 실행

저장소 root에서 매번 **새로운 output 경로**를 사용한다. Run 이름은 output basename이지만 join에는 실제 trace/span ID를 사용한다.

```bash
P=tutorials/nemo_lens/.venv/bin/python
LAB=tutorials/nemo_lens/rollout
$P tutorials/nemo_lens/verify_jaeger.py
for scenario in baseline tool-slow tool-straggler; do
  out="results/lens-rollout/my-${scenario}"
  $P "$LAB/run.py" --scenario "$scenario" --mode otlp --output "$out"
  $P "$LAB/diagnose.py" --spans "$out/spans.jsonl" --output "$out/diagnosis.json"
  $P "$LAB/verify.py" --result "$out/result.json"
done
$P "$LAB/test_rollout.py" -v
$P -m pytest --import-mode=importlib -q "$LAB"
```

Collector 없이 local JSON spans만 필요하면 `--mode file`을 사용한다. OTLP mode는 동일한 실제 spans를 collector에도 보낸다. `result.json`은 실행/export 결과, `jaeger_receipt.json`은 API에서 모든 local span ID·service/run·operation count·parent 계층이 일치함을 확인한 별도 증거다. 실패·불완전 trace를 정상 측정값으로 다루지 않는다.

## Jaeger에서 읽는 순서

Service `nemochat-lens-rollout`, operation `demo.rl.batch`를 선택하고 trace를 연다.

```text
demo.rl.batch
  demo.rl.trajectory [trajectory ID 0..3]
    demo.rl.caller_queue
    demo.rl.generation
    demo.rl.tool.client [client]
      demo.rl.tool.server [server, W3C remote parent]
    demo.rl.reward
```

`asyncio.to_thread`로 넘긴 caller context와 HTTP의 W3C traceparent를 연결한다. 같은 이름이라는 이유가 아니라 **동일 trace ID·정확한 client parent ID·trajectory ID**로 server를 확인한다. 두 endpoint가 같은 process라도 실제 HTTP를 수행한다.

- **baseline:** 인위적 server delay 없음. CPU admission/generation을 관찰한다.
- **tool-slow:** 각 handler에 명시적 **250 ms** sleep을 넣는다. 병렬 tool 시간의 합은 batch wall이 아니다.
- **tool-straggler:** 마지막 trajectory의 tool만 **750 ms** 지연시킨다. 마지막 완료 tail을 본다.
- `demo.injected_delay_ms`는 의도한 실험 개입이며 실제 고객 병목을 발견한 값이 아니다.

원격 host라면 `ssh -N -L 16687:127.0.0.1:16687 ubuntu@YOUR_HOST`로 UI만 전달한다. Collector는 in-memory라 재시작하면 live trace가 사라진다. Raw artifacts와 screenshots는 private ignored output에 유지한다.

## 중복 합산하지 않는 진단

`diagnose.py`는 실제 nanosecond timestamps로 계산한다.

1. 각 trajectory의 **서로 순차적인 direct-child stage** 네 개와 unknown gap을 따로 기록한다. Stage가 겹치면 독점 구간이라고 가정하지 않고 거부한다.
2. `tool.server`는 이미 `tool.client`에 포함되므로 따로 표시하되 합계에서 제외한다. Parent+child를 더하지 않는다.
3. Batch wall과 sum of trajectory wall을 구분한다. 후자는 동시 trajectory의 latency/work 합이며 elapsed time/utilization이 아니다.
4. 마지막 완료 trajectory의 가장 큰 stage와, **직전 trajectory 완료 후** tail에 겹치는 stage를 모두 본다. Queue 총량이 크더라도 마지막 tail은 generation일 수 있다.

이는 관찰된 병목 **후보**이지 production 인과 증명이나 전체 dependency-DAG critical path가 아니다. OS/executor scheduling·HTTP·계측 비용이 포함된다. Client−server 차이도 순수 network latency라고 단정하지 않는다.

## 실제 부모 실행 증거 — 재리뷰 이전 snapshot

`outputs/precommit-whole/rollout-{baseline,tool-slow,tool-straggler}/`에서 각 25 spans·모든 trace/span/resource ID·client/server parent를 확인했다. 각각 네 trajectory 모두 8 tokens를 생성하고 scripted tool 검증을 통과했다.

| Scenario | Batch wall ms | Trajectory wall 합 ms | Tail 후보 | 직전 완료 후 tail ms |
|---|---:|---:|---|---:|
| baseline | 91.265 | 234.486 | generation | 19.702 |
| tool-slow | 341.431 | 1232.682 | tool.client | 20.652 |
| tool-straggler | 852.563 | 1006.801 | tool.client | 775.117 |

각 scenario **한 batch**의 관측이며 반복 비교·통계적 유의성·production 개선이 아니다. [실제 기록](executed_results.json)에서 초기 worker 결과, 부모 결과와 source hash를 구분한다. 이후 수정 후 실행은 별도 기록하며 이 값을 최신 코드 성능으로 바꾸지 않는다.

실제 Chromium에서 straggler trace의 ID·service·7 operation·25 spans를 확인했다. Screenshot은 `outputs/precommit-whole/rollout-ui/trace.png`. 이는 Jaeger UI 검증이며 Nsight native GUI가 아니다.

## 안전·한계

- Metrics/logs는 끄고 `.env`를 자동 로드하지 않는다. 부모의 hardened exporter/API helper를 재사용한다.
- Tool은 숫자형 `127.0.0.1`·ephemeral port만 사용한다. Requests `trust_env=False`, nonlocal/credentialed URL 거부, redirect 거부, response cap 적용. API도 proxy/redirect를 거부한다.
- Span attributes에 prompt·생성 텍스트·tool arguments/results를 넣지 않는다. Raw spans에는 코드 위치·run identity가 남으므로 private하게 유지한다.
- `--timeout`은 connection/read socket inactivity timeout이지 전체 generation/request hard deadline이 아니다. Timeout 후 server가 완료해도 실패는 실패다.
- Reward는 script가 정한 도구 계산의 정답 검증이다. 모델 reasoning 품질·tool 선택 능력·정책 학습을 측정하지 않는다.
- 독립 리뷰·실패 경로 보완·최종 commit gate는 [전체 리뷰 보고](../../../docs/ko/WHOLE_REPOSITORY_REVIEW.md)에서 구분한다.

## 최종 cycle-2 검증

독립 재리뷰 PASS(CPU/scripted 소스 범위). 위 pytest 전체 명령은 **23 tests·57 subtests**, 기존 standalone unittest는 8개 normal-path 실습을 실행한다. Exporter 손실·초기화/cleanup·손상 receipt·bool ID·timestamp 실패 경로도 회귀 검증했다.

Pinned OTel 1.43.0 JSON formatter는 원래 ns가 아니라 반올림한 6자리 µs를 저장한다. 저장된 정밀도를 보존하고 Jaeger의 start/duration floor 차이만 검증된 1µs start·2µs duration/end·1µs parent-end 범위로 허용한다. 기록값을 바꾸지 않는다. Float64 tag의 integer JSON 표현은 semantic type으로 확인하며 bool/int identity는 엄격히 구분한다.

최종 세 시나리오는 `outputs/precommit-whole/final-cycle2-{baseline,tool-slow,tool-straggler}/`에서 각각 25-span 정확한 계층·timestamp 검증을 통과했다. 최신 straggler trace `631328f77ad238252a254beb63c04620`를 실제 Chromium에서 7 operation·25 spans·JS error 없음으로 확인했다. Screenshot `outputs/precommit-whole/final-cycle2-ui/trace.png`는 private ignored artifact. 이전 관측값은 별도로 보존하며 native RL/Gym/vLLM/Nsight/GPU 승인이 아니다.

## Runtime layout / 실행 자료 경로

Use repository-root `datasets/`, `checkpoints/`, `results/`, and `logs/`; see [runtime layout](../../../docs/en/RUNTIME_LAYOUT.md) and [한국어](../../../docs/ko/RUNTIME_LAYOUT.md). Relative CLI output paths use the caller working directory. Historical recorded paths/measurements below remain unchanged; prior host artifacts physically migrated, with relative compatibility links and SHA-256 inventory in root `results/runtime-layout-migration.json`. Tutorial benchmark stdout/stderr now go to root `logs/nemo_lens/<output-name>/` (override `--log-output`); use a fresh name for both results and logs.
