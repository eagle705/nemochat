# 에이전트와 함께하는 학습 시스템 reference lab

Canonical: [English](../en/TRAINING_SYSTEMS_LAB.md).

## 목적과 경계

에이전트 애플리케이션을 만드는 프로젝트가 아니라 **GPU 학습 시스템을 이해하고 실험을 재현하는 프로젝트**다. 에이전트는 YAML 계약으로 반복 실행을 맡고, 사람은 병목·성능·메모리와 correctness를 판단한다. 기존 hybrid/WSD/WSM 목표는 유지한다.

새 dense Transformer control은 실제 forward/backward/optimizer update를 실행하지만 입력은 생성된 token replay다. 실제 corpus 학습·모델 품질·native hybrid 성능을 증명하지 않는다. CPU 처리량으로 B300 성능을 추정하지 않는다.

## 실습 순서

1. [YAML 계약](AGENT_EXPERIMENT_CONTRACT.md)을 validate/dry-run하고 설정·자원 예산을 고정한다.
2. 연산 shape, FLOPs와 논리적 operand bytes를 먼저 예측한다.
3. Warmup을 제외한 full update를 반복 측정한다. CUDA는 window 경계에서 synchronize하고 phase마다 동기화하지 않는다.
4. GPU timeline에서 launch·data·sync·communication·dependency를 구분한다. CPU range는 GPU 실행 시간이 아니다.
5. Checkpointing 등 하나의 변경과 반증 가능한 가설을 기록한다.
6. Finite loss/gradient는 sanity check일 뿐이다. 채택 전 output/input/parameter gradient 비교, 짧은 학습과 native resume 검증이 추가로 필요하다.
7. [비교 도구](TRAINING_PERFORMANCE_TOOLS.md)로 분포·메모리·실패를 남긴다. Throughput 개선 하나로 기본 설정을 채택하지 않는다. 구현/runtime·timing/model/optimizer descriptor가 양쪽에서 모두 빠졌다는 이유로 호환성을 인정하지 않는다. Descriptor 비교는 원격 실행의 진위를 인증하는 절차는 아니다.

## 버전 관리된 control 설정

- `dense_cpu.yaml`: unprofiled baseline.
- `dense_cpu_checkpointing.yaml`: 같은 workload/protocol에서 checkpointing만 변경하여 비교.
- `dense_cpu_profile.yaml`: timed window와 분리한 단일 update CPU trace.
- `dense_cuda.yaml`: GPU 환경용 template; CPU host에서는 blocked.

모두 `configs/experiments/` 아래에 있다. 작은 실행 경로 검증 설정이며 실제 workload 성능·품질 평가를 대신하지 않는다.

## Linear forward/backward planner

```bash
.venv/bin/python scripts/plan_training_operator.py --config configs/performance/linear_unmeasured.json --output runs/my-linear-estimate.json
```

부모 디렉터리가 있어야 하고 기존 결과를 덮어쓰지 않는다. `Y=X@W`에서 X:[M,K], W:[K,N]이면 forward, dX, dW의 각 GEMM FLOPs는 `2*M*N*K`다. 논리적 operand bytes는 각 연산에 대해 `(M*K+K*N+M*N)*dtype_bytes`로 계산한다. 실제 cache/tiling/DRAM traffic은 다르다.

이 도구는 bias·activation·attention·optimizer·통신·checkpoint recompute를 포함하지 않는다. **논리적 bytes를 DRAM 측정값으로 둔갑시키지 않는다.**

필수 JSON 필드는 schema_version:1, batch_tokens/in_features/out_features(양의 정수, 각각 1,000,000 이하), dtype(float32/bfloat16/float16), provenance(kind/description)다. 중복·알 수 없는 필드·비유한 수·64KiB 초과 입력을 거부한다.

선택적으로 effective_compute_flops_per_second/effective_bandwidth_bytes_per_second를 넣고, measurements의 forward/input_gradient/weight_gradient에 dram_bytes·observed_seconds·description을 제공할 수 있다. 측정의 device/precision/shape/counter 범위를 맞춰야 한다. 값은 사용자 제공이고 도구가 진위를 인증하지 않는다.

실제 traffic 입력 없이는 roofline을 만들지 않는다. Ceiling이 부족하면 measurement_required, 관측 시간이 모델 하한보다 작으면 assumptions_inconsistent다. Synthetic fixture는 --allow-synthetic이 필요하다. 단일 연산 하한을 더해서 전체 step 하한이나 speedup이라고 주장하지 않는다.

## 에이전트 사용 원칙

설정 검증 → dry-run → 새 디렉터리 baseline → 한 가지 변경 → 별도 디렉터리 candidate → 비교와 실험 보고서 순서다. 설정·raw repeat·실패·남은 gate를 보존한다. GPU 임대·업로드·네트워크·기본 설정 채택은 자동 승인되지 않는다. 실행기의 예산 제한은 OS/container sandbox가 아니며 외부 에이전트의 독립 행동까지 통제하지 않는다.
