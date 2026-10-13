# Inference Performance CPU 도구

Canonical: [INFERENCE_PERFORMANCE_TOOLS.md](../en/INFERENCE_PERFORMANCE_TOOLS.md). [독립 트랙](INFERENCE_PERFORMANCE_TRACK.md)의 준비 단계다. Engine 실행, load generation, GPU 측정, quality 평가를 수행하는 도구가 아니다.

## Saved-request 분석 실행

```bash
python3 scripts/analyze_inference_requests.py --help
mkdir -p outputs/inference_perf/local-smoke
python3 scripts/analyze_inference_requests.py \
  --requests configs/inference_perf/synthetic_requests.jsonl \
  --config configs/inference_perf/synthetic_config.json \
  --output outputs/inference_perf/local-smoke/requests.json \
  --allow-synthetic
```

Fixture는 명시적 synthetic이며 실측 serving 결과가 아니다. Opt-in 없이 거부하고 output은 새 파일만 허용한다. 재실행은 다른 출력 파일을 지정한다.

### JSONL 요청 계약

각 줄에 schema v1 object 하나. 빈 줄/중복 key/중복 request ID/unknown field를 거부한다. 모든 timestamp는 동일 client monotonic seconds다.

- `schema_version=1`, unique `request_id`, boolean `synthetic`
- `clock="client_monotonic_seconds"`
- `status`: success/error/timeout/cancelled
- `arrival_s <= send_s <= completion_s`: finite nonnegative
- `first_token_s`: 출력이 있으면 send~completion 범위, 없으면 null
- `input_tokens`, `output_tokens`: 실패 요청의 부분 출력까지 실제 integer token count
- `token_timing`: individual_tokens 또는 unavailable
- `token_arrivals_s`: 실제 output token당 timestamp 하나 또는 unavailable이면 null. 첫 token은 first_token과 일치하고 순서는 nondecreasing. 출력 0이면 individual timing은 빈 배열.

**Chunk timestamp를 복제해서 token ITL로 만들면 안 된다.** 실제 per-token 계측이 없으면 unavailable로 입력한다. Importer는 producer의 계측을 인증하지 못하고 실패 요청의 누락 usage를 복구하지 못한다.

### Config와 측정 구간

Exact fields: schema_version, workload_id, tokenizer_id, decoding_id, measurement_start_s, measurement_end_s, slo.

Arrival은 `[start,end)` 안에, 모든 completion은 end 이하여야 한다. **완전히 drain된 attempt cohort**를 분석하는 정책이며 warmup이나 in-flight 요청을 조용히 제외하지 않는다. Warmup은 별도 수집하고 drain을 포함한 명시적 구간을 쓴다. 지속 부하의 time-window 집계는 향후 별도 정책이 필요하다.

`slo`의 ttft_s/e2e_s/max_token_gap_s는 threshold 또는 null이며 최소 하나는 설정한다. 모든 설정 조건을 inclusive하게 만족해야 한다. Gap SLO는 request별 max adjacent gap이다. Timing이 없거나 token이 2개 미만이면 unknown으로 goodput에 넣지 않는다.

### 지표 해석

- TTFT: first_token − send
- E2E: completion − send
- Arrival-to-send/completion: 별도 client 측 지연이며 server queue time이라고 단정하지 않는다.
- Percentile: successful-request nearest rank. P99 표본 수/coverage와 작은 표본 경고를 확인한다.
- Token gap: 실제 개별 token 관측, chunk gap이나 server decode time 아님.
- SLO goodput: 모든 조건을 만족한 success 수 / explicit window seconds. 실패/timeout/cancelled은 전체 attempt count에 남는다.

이 report만으로 canonical benchmark record가 완성되지는 않는다. Engine/model/source/hardware/topology/config identity와 quality 결과를 별도로 보관한다. Producer timestamp/workload ID는 실제 환경이나 기록 완전성을 증명하지 않는다.

## Roofline 실행

```bash
python3 scripts/plan_inference_roofline.py --help
python3 scripts/plan_inference_roofline.py \
  --config configs/planning/inference_roofline_synthetic.json \
  --output outputs/inference_perf/local-smoke/roofline.json \
  --allow-synthetic
```

Parent directory가 존재해야 하며 새 output만 허용한다. 예시에는 실측 B300 spec/performance가 없다.

입력: scope=single_operator, operator, positive flops/dram_bytes, provenance(kind=synthetic/source/measurement와 description), optional effective_compute_flops_per_second/effective_bandwidth_bytes_per_second/observed_seconds.

출력: intensity, compute/memory 하한, max 하한, model classification, 관측과 모델의 비교, 경고/한계, exact input-byte SHA256.

Ceiling이 없으면 measurement_required. Observed time이 하한보다 짧으면 가정 불일치로 표시하고 headroom 추론을 억제한다. Compute와 memory 하한이 큰 값 기준 5% 이내면 balanced다. 모델상 utilization/headroom은 hardware counter나 달성 가능한 speedup이 아니다. Cache/overlap/concurrency/shape/precision/launch/sync/communication 한계를 함께 읽는다.

## 테스트

```bash
.venv/bin/python -m pytest -q tests/test_inference_perf.py tests/test_inference_roofline.py
```

CPU 테스트와 synthetic CLI 성공은 local 계약만 검증한다. 실제 serving/SLO/quality/GPU memory/Nsight/성능 회귀 및 upstream 기여는 대기다.
