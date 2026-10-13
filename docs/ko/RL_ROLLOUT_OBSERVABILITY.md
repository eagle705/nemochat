# RL Rollout 병목 진단 — NeMo Lens

Canonical: [공식 소스·범위·주의사항](../en/RL_ROLLOUT_OBSERVABILITY.md).
실행 실습: [local rollout lab](../../tutorials/nemo_lens/rollout/README.ko.md).

## 무엇을 구분할까?

Rollout이 느리다는 말만으로 GPU 증설을 결정하지 않는다.

- **Generation 대기:** 호출자 admission/semaphore인지 실제 inference engine queue인지 구분.
- **모델 생성:** 실제 응답 생성 시간이 큰가? 출력 token 길이·동시성도 함께 확인.
- **Tool client:** 요청 전체 시간이 큰가? transport·대기·서버 실행이 섞여 있다.
- **Tool server:** 도구 handler 자체가 오래 실행되는가?
- **Connection-pool 대기:** HTTP connection slot을 못 얻는가?
- **Verify/reward:** 검증·judge가 오래 걸리는가?
- **Straggler:** 대부분 끝났지만 마지막 trajectory 하나 때문에 batch가 기다리는가?
- **실패/retry:** 실패한 요청을 빼서 빠르게 보이는 것은 아닌가?

**Concurrent trajectory 시간의 합은 batch wall time이 아니다.** Parent+child span, tool client+그 안의 server span을 모두 더하면 중복 계산이다. 마지막으로 끝난 trajectory의 순차 구간·대기·누락 구간을 보고 후보를 찾는다. 가장 긴 span이 실제 최적화 가능 bottleneck임을 보장하지는 않는다.

## NeMo RL의 기존 기능을 이용하기

실제 조회한 revision은 `NVIDIA-NeMo/RL@67fc59b2ccc6db48dc873eefa86d64506ca38126`다. Rollout/generation/logprob/reward/advantage/policy_update/data_plane/efficiency 그룹이 있고, 짧은 진단에는 아래와 같은 **부분 설정**을 검토할 수 있다:

```yaml
telemetry:
  enabled: true
  span_groups: per_step,per_prompt
```

전체 실행 recipe는 아니다. 설치 버전과 exporter 설정이 별도로 필요하다. Per-prompt는 prompt 수에 비례해 span이 늘어나므로 항상 켜두지 않는다. `job`을 제거하면 trace 크기는 줄 수 있지만 async collector의 context 전달도 바뀔 수 있으므로 무조건 권장하지 않는다.

`rl.gym.run_rollouts`, generation/reward/policy training과 `rl.idle.buffer_starvation`, `rl.idle.refit_bubble`을 실제 경로에 있을 때 확인한다. Sync generate와 async generate_async의 coverage는 다르다.

vLLM 내부 OTLP trace는 RL 바깥 wrapper와 독립적이다. Offline `LLM.generate()`에서는 inspected integration이 request context를 직접 넘기지 못해 native span과 rollout의 parent-child 연결을 보장하지 않는다. Run/resource ID로 연결된 별도 trace일 수 있다. Wrapper 시간만 있는데 queue/prefill/decode를 임의로 나누면 안 된다.

## Gym은 tool call을 어떻게 보여주나?

실제 조회한 revision은 `NVIDIA-NeMo/Gym@8fddfb519e9473c6e51b8e8bf89e02aa7e5fa0c1`다. 짧은 진단에 사용하는 **부분 설정**:

```yaml
telemetry:
  enabled: true
  span_groups: per_rollout
```

이 preset에는 server/http_client/rollout/verify/agent/model_call이 들어간다.

**Gym에는 `tool_call` span group이 없다.** Tool은 `POST /get_weather` 같은 resources-server route로 보인다. HTTP client가 `traceparent`를 보내고 server가 읽어 같은 trace로 연결한다. 둘 중 하나를 끄면 프로세스별로 끊어진 trace가 될 수 있다.

- `gym.rollout`: rollout 전체
- `gym.model.*`: 모델 호출
- Tool route의 SERVER span: 실제 handler
- Outbound HTTP CLIENT span: tool 요청 전체
- `gym.verify`: 평가·검증 호출

RL과 Gym이 같은 프로세스면 Lens setup을 두 번 하지 않는다. RL이 먼저 초기화하면 Gym이 provider를 재사용하며, 별도 Gym 프로세스에는 exporter/identity 설정이 따로 필요하다.

## Connection 대기와 tool 실행을 구분하기

Native `gym.http.connection_pool.queue_duration_ms`는 **aiohttp connection 획득 대기**다. `ok/abandoned` outcome, `total/per_host/unknown` 제약을 구분한다. 즉시 획득한 호출에는 queue histogram sample이 없다.

같은 destination·window의 `gym.http.connection_pool.connect_total`과 같이 봐야 몇 번이나 queue에 걸렸는지 알 수 있다. Retry·redirect·reconnect의 acquisition은 별도 attempt이므로 semantic tool call 수와 같지 않다.

Server active requests·route duration/status와 rollout/verify duration도 함께 확인한다. **`gym.verify.success_rate`는 verifier 호출이 정상 완료됐는지이지, 정답률이 아니다.** Reward·benchmark 품질은 W&B 등의 experiment telemetry와 분리한다.

## Tool timing evidence는 별도 기능

Gym의 structured rollout evidence에는 model-call capture·agent observation·trajectory·tool timing이 있다. Inspected simple agent는 trajectory collection이 켜지면 monotonic duration과 `timing_source="executor"`, invocation/tool-call ID·status를 기록한다.

이것은 OTel span과 다른 artifact 경로다. 명시적 ID로 join하며 시간상 인접하다는 이유로 소유권을 추정하지 않는다. Prompt·tool 입력/출력·conversation이 포함될 수 있으므로 **승인과 private 저장**이 필요하다. 미지원·누락 evidence는 gap으로 남긴다.

## Local 실습과 실제 RL 실행 구분

Local tutorial은 실제 local 모델 호출·HTTP tool·Lens 전송으로 trace를 읽는 법을 배우는 control이다. 의도적 delay는 실습 fixture로 표시한다. 학습된 tool policy·native NeMo RL/Gym·GPU vLLM·실제 고객 workload 병목을 실행했다고 부르지 않는다.

Native 적용 단계에서는 실제 algorithm/backend/version·rank/run/service·sampling/coverage·token usage·failure/retry·cohort window와 전체 RL throughput/품질을 고정해 비교한다. GPU compute가 의심되면 짧은 Nsight capture로 내려가고, OTel만으로 kernel 병목을 판정하지 않는다.

모든 공식 링크는 [영문 source map](../en/RL_ROLLOUT_OBSERVABILITY.md)에 pinned revision으로 정리했다.
