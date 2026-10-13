# 오프라인 dense 전체 학습 업데이트 성능 도구

새 CLI 실행은 저장소 루트 기준 경로와 새 산출물 이름을 사용합니다. [런타임 디렉터리 규약](RUNTIME_LAYOUT.md)에 따라 입력 데이터는 `datasets/`, checkpoint는 `checkpoints/`, 보고서·trace·plot은 `results/`, 텍스트/W&B 로그는 `logs/`에 둡니다. 필요한 부모 폴더만 만들며 기존 fixture/config와 과거 증거 경로는 보존합니다.

## 구현 범위와 증거

`src/nemochat/training_perf.py`는 교육용 **독립 dense causal Transformer**입니다. 시드로 생성한 토큰에서 실제 cross-entropy, backward, AdamW 업데이트를 실행합니다. NemoChat native hybrid Latent MoE/Bridge/MCore 학습은 여전히 미구현입니다. 실제 코퍼스, tokenizer, 다운로드, 품질 평가, 분산 학습, Triton, checkpoint/resume은 포함하지 않으며 WSD/WSM 기본 정책을 바꾸지 않습니다.

- `CPU-model-executed`: CPU 제어 모델 실제 실행. GPU 성능 증거가 아닙니다.
- `GPU-smoke-executed`: 동일 독립 모델의 단일 CUDA 실행. native hybrid/분산 검증이 아닙니다.
- CUDA 부재 또는 bfloat16 미지원: `blocked`, 종료 코드 **3**, CPU 대체 실행 없음.
- OOM은 종료 코드 **3** (`out-of-memory`), 협조적 시간 초과는 **4** (`time-budget-exceeded`)입니다. 두 경우 측정 status는 failed이며 runner는 각각 blocked/timeout과 정확한 reason을 보존합니다. 나머지 실패는 **1** (`nonfinite-or-no-update`, `execution-error`), 성공은 **0**입니다.
- CPU profiler trace는 로컬에서 실행할 수 있습니다. CUDA 시간/메모리/profiler 검증은 승인된 GPU에서 별도로 수행해야 하며 CPU-only 호스트에서는 미검증입니다.

## 직접 실행

완전한 JSON 예제는 [영문 문서](../en/TRAINING_PERFORMANCE_TOOLS.md)의 `Direct CPU smoke` 절을 사용합니다. CPU smoke는 batch=2, sequence=16, hidden=32, FFN=64, layers=1, vocabulary=128, LR=0.001, warmup=1, measured=3, repeats=2입니다. schema_version=1과 experiment/workload/measurement/resources의 모든 필드를 제공해야 합니다. 알 수 없거나 누락된 키, 정수 대신 bool, nonfinite LR, 잘못된 count는 거부됩니다. JSON/YAML은 [동일한 제한 스키마](AGENT_EXPERIMENT_CONTRACT.md)를 사용합니다. 이름은 영문/숫자로 시작하는 ASCII 영문/숫자/`_.-` 1–64자, seed는 0–2^32−1, vocabulary는 2–65536, LR은 (0,1], max_seconds는 1–3600입니다. CPU는 float32/GPU 예산 0, CUDA는 GPU 예산 1을 요구합니다. 모든 개별/조합 예산 상한을 적용합니다. config JSON은 최대 32 KiB, 비교 JSON은 파일당 최대 8 MiB이며 중복 키와 NaN/Infinity/지수 overflow를 거부합니다. 잘못된 UTF-8/JSON, parser 재귀/거대 정수 오류는 출력 생성 없이 CLI 거부를 반환합니다.

```bash
.venv/bin/python scripts/benchmark_training.py --help
.venv/bin/python scripts/benchmark_training.py \
  --config results/training_perf/my-inputs/dense-control.json --output results/training_perf/my-baseline/measurement
# activation_checkpointing만 true로 바꾼 별도 JSON:
.venv/bin/python scripts/benchmark_training.py \
  --config results/training_perf/my-inputs/dense-checkpoint.json --output results/training_perf/my-candidate/measurement
.venv/bin/python scripts/compare_training_runs.py \
  --baseline results/training_perf/my-baseline/measurement/result.json \
  --candidate results/training_perf/my-candidate/measurement/result.json \
  --output results/training_perf/my-comparison.json
.venv/bin/python -m pytest -q tests/test_training_perf.py
```

매번 새 경로를 사용합니다. 빈 기존 디렉터리도 benchmark는 거부하며 비교 결과는 새 JSON 파일만 생성합니다. runner 호출은 `python scripts/benchmark_training.py --config output/resolved_config.json --output output/measurement`입니다. 직접 실행도 runner와 동일한 전체 제한 검증기를 사용합니다. 산출물은 `resolved_config.json`, `result.json`, 선택적 `trace.json`입니다. 잘못된 입력은 출력 생성 전에 거부합니다. 런타임 실패의 부분 반복 기록은 성공으로 승격하지 않고 비교에서 거부합니다.

## 모델과 측정 규약

토큰/학습 위치 embedding, single-head causal SDPA, pre-LN attention/FFN residual, GELU, 마지막 layernorm, untied head를 명시적으로 구현합니다. dropout은 없습니다. activation checkpointing은 `use_reentrant=False`입니다. hidden_size의 4 배수 제한은 명시적 bounded-control 계약이며 single-head 아키텍처 필수 조건이 아닙니다. `B*4*S*S*L`은 보수적인 4배 안전 상한입니다. CPU parameter/activation은 float32만 허용하고 CUDA는 float32 또는 지원되는 bfloat16을 허용하며 CE와 gradient norm은 float32입니다. FP32 master weight mixed precision이 아니며 AdamW 상태는 parameter dtype을 따르고 GradScaler는 없습니다.

CPU-local Generator가 B×(S+1) 토큰을 생성합니다. 앞 S개 입력과 한 칸 이동한 S개 target으로 업데이트당 **processed B*S, loss-bearing B*S** 토큰을 셉니다. padding/mask/drop은 없습니다. 생성 토큰 재생은 실제 코퍼스 또는 고유 학습 데이터가 아닙니다. batch/parameter SHA256는 진단용으로, 조작된 산출물의 진실성을 보장하지 않습니다.

각 반복은 같은 시드/model/AdamW 초기 상태와 같은 batch 순서를 사용합니다. warmup, batch 생성, model 초기화, fingerprint는 시간/카운터에서 제외합니다. 측정은 data/H2D, zero-grad, forward CE, backward, device-side finite/nonzero gradient 체크 및 optimizer 전체를 포함하고 profiler는 제외합니다. finite loss/gradient/갱신 parameter와 실제 측정 구간 parameter 변화가 성공 조건이며 수학적 동등성/유용한 학습을 증명하지는 않습니다.

CUDA 동기화는 **측정 창 시작/끝**에서만 수행하며 phase/update마다 수행하지 않습니다. device-side 진단 결과는 측정 종료 후 읽습니다. warmup 진단 동기화는 측정 밖입니다. CPU phase 시간에서 GPU kernel 시간을 추론하지 않습니다. CPU thread는 최대 2개로 제한 후 복원하며 caller CPU RNG와 선택한 CUDA device RNG를 보존합니다. 다른 GPU를 seed하지 않으며 multi-GPU 재현성을 주장하지 않습니다. 고정 초기 상태/입력도 모든 CUDA kernel의 bitwise 결정성을 보장하지 않습니다. deterministic/TF32/matmul 설정은 변경하지 않고 기록합니다.

max_seconds는 준비/반복/profile 전체 예산이며 update 경계의 **협조적 검사**입니다. 진행 중 operator를 강제로 중단하지 못합니다. 엄격한 마감은 runner subprocess timeout을 사용합니다. 아주 짧은 smoke 결과의 변동성으로 실용적 속도 개선을 단정하지 않습니다.

## 메모리, 환경, profiler

CUDA peak allocated/reserved를 분리하고 warmup 뒤 allocator peak를 초기화합니다. 반복 사이 미사용 cache는 측정 밖에서 해제합니다. model/AdamW 상태, backward 임시 및 저장 tensor는 peak에 포함하지만 외부 라이브러리 할당/GPU 전체 사용량은 아닙니다. CPU memory는 **null/unavailable**이며 0 또는 GPU 메모리로 표시하지 않습니다.

Python/PyTorch/CUDA runtime/cuDNN, platform, thread, precision flag, 구현 hash, CUDA device 이름/capability/총 메모리/SM 수만 기록하며 환경변수/비밀정보를 덤프하지 않습니다. driver version은 조회하지 않으므로 진지한 GPU 실험에서는 별도로 고정/기록해야 합니다. 동일 metadata만으로 독점 GPU, clock/power/온도, host contention까지 같음을 증명하지 못합니다.

profile=true는 일반 측정 후 별도 fresh/warmup 상태에서 한 업데이트를 기록합니다. CPU와 CUDA 호스트의 경우 CUDA activity, shapes, memory 및 `record_function`의 data/forward/backward/optimizer label을 Chrome trace.json으로 저장합니다. profiler 시간은 throughput에 넣지 않습니다. CPU trace는 GPU gap/critical path/stream overlap 또는 Nsight 진단을 증명하지 않습니다. [사전학습 lab](PRETRAIN_PERFORMANCE_LAB.md), [Nsight 안내](NSIGHT_CODEX_GUIDE.md)를 별도 GPU/native 진단에 사용합니다.

## 분석적 GEMM 추정과 비교

QKV, attention-output, FFN-up/down, head마다 M=B*S, N=출력 폭, K=입력 폭과 **2MNK forward FLOPs**를 기록합니다. element bytes×(MK+KN+MN)은 입력/weight 한 번 읽기와 output 한 번 쓰기의 이상적 **최소 byte 하한**이며 실제 DRAM traffic이 아닙니다. FLOPs/byte는 이 모델의 비율이지 실제 roofline이 아닙니다. forward+backward projection은 forward의 3배로 추정하며 checkpointing은 block projection forward 재계산을 명목상 추가합니다. non-reentrant early-stop 때문에 실제 계산이 다를 수 있습니다. Attention QK/AV, softmax, embedding/norm/GELU, CE, optimizer, finite check는 제외합니다. 실제 traffic과 precision별 hardware compute/bandwidth ceiling 없이 achieved FLOPs/s나 GPU 여유/roofline을 주장하지 않습니다.

비교는 성공 schema-1 산출물, 완전한 finite counter/diagnostic, 정확한 token/update 비율, finite-check 통과를 요구합니다. workload/seed/batch/초기 model, measurement/profile 규약, 구현 hash, 환경/device/증거 범위가 같아야 합니다. **activation_checkpointing만 workload 차이를 허용**하며 experiment 이름/resource 예산은 달라도 됩니다. CPU-vs-GPU, 실패/blocked/부분 반복은 거부합니다.

raw throughput 분포(min/median/mean/max/stdev), median 비율, 순서상 반복 비율, CUDA allocated/reserved 분포/최대 차이를 보고합니다. 반복 비율은 descriptive이며 무작위 paired trial 또는 유의성 검정이 아닙니다. CPU 메모리 trade-off는 알 수 없습니다. 결정은 항상 **experimental-only**입니다. 출력/gradient 동등성의 dtype/shape 범위, checkpoint/resume, native hybrid 통합, GPU/profile 증거는 별도 gate입니다. 작은 CPU 동등성 테스트를 CUDA/native 정확성으로 확대하지 않습니다. 속도만으로 채택하지 않고 회귀/변동/no-improvement도 정직하게 보고합니다.
