# 독립 Inference Performance 트랙

Canonical: [INFERENCE_PERFORMANCE_TRACK.md](../en/INFERENCE_PERFORMANCE_TRACK.md).

## 목적과 경계

기존 pretraining/WSD/WSM을 유지하면서 **성능 모델 → profiler 근거 → engine/kernel 코드 변경 → serving A/B → quality/regression → upstream 기여**를 별도 트랙으로 진행한다.

첫 코드 범위는 CPU에서 검증 가능한 saved-request 분석과 single-operator roofline 계산이다. Load generator, GPU serving, quality evaluator, Nsight parser를 구현했다는 뜻이 아니다. Synthetic 데이터는 계약 테스트이며 실제 성능이나 production 경험을 대신하지 않는다.

## 모델과 engine 선택

처음에는 pinned engine 하나와 지원되는 pretrained 모델 하나로 진행한다. NemoChat 학습 완료를 기다리지 않는다.

- Pretrained control: serving benchmark/최적화 방법 검증
- NemoChat hybrid: export/native inference/cache-state 지원 gate 별도
- VLM control: 후반 선택 과제로 preprocessing/encoder/decoder를 분리

Model/tokenizer revision, container/source, driver/runtime, precision, hardware/topology, parallelism, graph/cache/batch 설정과 workload/generation 정책을 고정한다. 비밀값·고객 prompt는 저장하지 않고 외부 작업은 승인 범위에서 수행한다.

## I1: Workload와 SLO benchmark

긴 입력/짧은 출력, 짧은 입력/긴 출력, mixed length, shared prefix, sustained arrival/burst, 긴 context/high concurrency를 구분한다. A/B는 request ID·실제 token 수·arrival schedule을 고정한다.

Closed-loop concurrency와 open-loop arrival 실험은 별개다. Target arrival/actual send, client saturation/backpressure, warmup/drain을 기록한다. 서버가 느려져 client도 요청을 덜 보내는 효과를 실제 capacity라고 부르지 않는다.

TTFT/E2E는 client monotonic 시간 기준으로 정의한다. Queue/server/network 분해는 별도 계측이 있어야 한다. **Streaming chunk는 token이 아니다.** 실제 per-token timing이 없으면 ITL은 unavailable로 표시한다. 출력이 없거나 하나뿐이면 별도 처리한다.

Success/error/timeout/cancelled, percentile/sample count, throughput, explicit-window SLO goodput을 기록한다. 실패도 분모에 포함한다. 작은 표본의 P99를 안정적인 production tail이라고 주장하지 않는다. Quality-gated goodput에는 실제 quality 결과가 필요하다.

CPU gate: strict schema/finite/monotonic/count consistency/중복/synthetic opt-in/no overwrite/CLI 테스트. GPU gate: 실제 engine 계측과 대표 부하 재현, 독립 timing 확인.

## I2: Roofline/Speed-of-light

Operator FLOPs, DRAM bytes, effective compute FLOP/s, bandwidth byte/s와 출처를 명시한다. 단순 하한은 `max(FLOPs/compute_rate, bytes/bandwidth_rate)`이며 전체 service runtime 예측은 아니다.

HBM/L2 traffic을 구분하고 precision/shape/cache/launch/sync/communication을 고려한다. Occupancy 자체를 최대화하는 것이 목표는 아니다. Datasheet peak와 실측 ceiling은 별도 입력이다. Ceiling이 없으면 measurement_required, 관측 시간이 모델 하한보다 짧으면 가정 불일치로 처리한다. 계산한 headroom을 실제 speedup으로 주장하지 않는다.

첫 질문은 prefill/decode의 expert token 분포와 작은 GEMM/routing/traffic 차이이며, 지원되는 MoE control에서 익힌 뒤 custom hybrid로 확장한다.

## I3: Profile → 코드 변경

[Nsight/Codex 가이드](NSIGHT_CODEX_GUIDE.md)를 따른다. Systems로 timeline/host/communication, Compute로 후보 kernel, NVTX/PyTorch Profiler로 계측한다.

실제 engine critical path에서 후보 하나를 고르고 독립 재현·optional patch·reference 비교를 수행한다. 작은 activation/routing transform/cast/copy/host tensor 준비/graph blocker가 후보이며 실제 trace 확인이 우선이다.

Output/logits, shape/layout, prefill/decode, cache continuation, graph replay, 지원 quantized path를 검증한다. Inference만이면 backward는 필수가 아니지만 training과 공유하면 검증해야 한다. Profile 조건을 맞춘 뒤 최종 serving 측정은 profiler 없이 한다. TTFT/ITL/P99/goodput 악화와 무개선도 보고한다.

## I4: Scheduling/cache/Graph

지원되는 batch token budget/chunked prefill/admission/cache budget/graph scope를 한 번에 하나씩 변경한다. Tokens/s 최고값 대신 latency/throughput/goodput Pareto를 만든다.

Hybrid에서는 attention KV뿐 아니라 Mamba state, buffers, graph memory도 고려한다. Request state 격리/reset/reuse/eviction, stateless vs cached correctness를 검증한다. 일반 KV 공식만으로 hybrid capacity를 설명하지 않는다.

## I5: Precision과 quality

Weight/activation/KV quantization을 별도로 실험한다. Task quality threshold를 결과 확인 전 정하고 실제 quality와 성능을 함께 비교한다. Logit tolerance만으로 task quality를 보장하지 않는다. Lower bit가 항상 빠르지는 않다.

Speculative decoding은 baseline 이후 draft/verification 비용·acceptance 분포·correctness·workload별 효과를 포함한다.

## I6: 분산 및 VLM 후반 확장

모델이 들어가는 구성에 한해 같은 총 자원/SLO에서 TP와 replicas를 비교한다. Collectives/topology/rank imbalance를 확인한다. 단일 노드 NVLink를 inter-node RDMA 경험으로 확대하지 않는다. PD disaggregation은 실제 request-level KV/state transfer를 확인한 경우만 주장한다.

VLM은 지원 모델로 image resolution/count, preprocessing cache, prompt/output 분포를 고정하고 preprocessing/encoder/decoder를 분해한다.

## I7: Run 기록/regression/upstream

Config/input/source identity, raw requests, window/warmup/repeats, 실패, quality, trace를 보관한다. Git 제외 `outputs/inference_perf/`의 새 경로만 사용한다. 호환되는 workload/resource/quality policy끼리 비교한다. 안정된 runner에서 noise/regression 기준을 사전 정의한다.

첫 engine upstream 기여는 작은 patch·최소 재현·correctness·benchmark evidence로 묶는다. 회사 IP/publication 규정을 확인하고 commit/push/공개 PR은 별도 승인한다.

## 첫 구현 순서

1. CPU saved-request analyzer 및 synthetic 계약 테스트
2. CPU single-operator roofline 계산과 provenance
3. 실제 engine instrumentation/load generator — 대기
4. 실제 병목 개선과 full-serving A/B — 대기
5. Quality/regression 및 upstream change — 대기

실행 방법은 [도구 문서](INFERENCE_PERFORMANCE_TOOLS.md)를 참고한다. Saved-log와 roofline 도구는 준비 단계이지 실제 serving 성능 검증이 아니다.
