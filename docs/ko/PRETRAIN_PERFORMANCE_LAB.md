# Pretraining 성능 실습: Nsight Systems → Triton fusion

새 CLI 실행은 저장소 루트 기준 경로와 새 산출물 이름을 사용합니다. [런타임 디렉터리 규약](RUNTIME_LAYOUT.md)에 따라 입력 데이터는 `datasets/`, checkpoint는 `checkpoints/`, 보고서·trace·plot은 `results/`, 텍스트/W&B 로그는 `logs/`에 둡니다. 필요한 부모 폴더만 만들며 기존 fixture/config와 과거 증거 경로는 보존합니다.

Canonical 명세: [PRETRAIN_PERFORMANCE_LAB.md](../en/PRETRAIN_PERFORMANCE_LAB.md).

## 상태와 목표

사용자가 선택한 pretraining 작업 범위에 포함한다. 아직 kernel 구현이나 GPU 실험을 완료했다는 뜻은 아니다. 현재 호스트는 `nvidia-smi -L`이 driver 통신 오류로 실패하며 `nsys`도 PATH에 없다. CPU 회귀 테스트나 CUDA 테스트 skip을 GPU 검증으로 보고하지 않는다.

목표는 **trace로 병목을 찾고, 작은 fusion을 구현하고, gradient correctness와 전체 학습 성능을 검증하는 과정**을 익히는 것이다. 기존 단일 WSD와 stable-window 공식 WSM 정책은 유지한다.

## 새 실행 가능한 dense control

독립 dense Transformer의 generated-token full-update 측정, YAML 실험 실행기, linear forward/backward 연산량 추정 도구가 추가되었다. [학습 순서](TRAINING_SYSTEMS_LAB.md), [실험 계약](AGENT_EXPERIMENT_CONTRACT.md), [성능 도구](TRAINING_PERFORMANCE_TOOLS.md)를 참조한다. CPU control은 native hybrid 학습·Triton·GPU/Nsight 검증을 대신하지 않는다. 아래의 미구현 상태는 해당 native GPU gate에 대한 기록이다.

## P0: 실제 GPU baseline

- 승인된 GPU와 pinned container/SDK 환경에서 실제 NemoChat forward/backward/update 및 checkpoint reload부터 확보한다. tiny HF CPU control과 구분한다.
- 모델·초기 checkpoint·실제 batch replay·seed·precision·sequence·GBS·MBS/accumulation·병렬 구성·routing/drop·recomputation을 고정한다.
- Warmup/JIT/autotuning과 steady-state를 분리하고 반복 횟수·측정 구간을 미리 정한다.
- 구간 경계에서 GPU 완료를 확인하되 phase마다 synchronize해서 overlap을 깨뜨리지 않는다.
- 실제 processed/loss token 증가량, optimizer update 시간, 학습/운영 포함 tokens/s, rank별 peak allocated/reserved와 GPU 전체 사용량을 기록한다. TP/CP/EP rank로 token을 중복 합산하지 않는다.

## P1: Nsight Systems로 가설 만들기

NVTX로 data/H2D/forward/backward/optimizer/checkpoint와 후보 module을 표시한다. 짧은 steady-state 구간부터 분석하고 분산 주장은 관련 rank 전체로 검증한다.

**Fusion 후보:** 짧은 kernel의 반복, launch 공급 지연, pointwise 연산 사이 중간 tensor 왕복.

**다른 원인:** dataloader stall, `.item()`/CPU copy 동기화, collective나 stream dependency, expert/rank imbalance. CPU가 synchronize에서 기다린 시간이 길다는 사실만으로 CPU 계산 병목이라고 결론내리지 않는다. 다른 stream/rank와 critical path를 함께 본다.

실제 소스 위치와 correlation을 연결하고 SDK에 이미 fused 구현이 있는지 확인한다. 후보가 없다는 결과도 유효하다.

[Codex Nsight 분석 가이드](NSIGHT_CODEX_GUIDE.md)에 CSV/SQLite export, schema-aware 분석, 요청문, GUI 확인과 전후 비교를 정리했다.

## P2: 작은 Triton 실험

첫 학습 후보는 `silu(gate) * up`인 SwiGLU activation이다. **실제 NemoChat 병목으로 확인한 후보는 아니며 GEMM은 포함하지 않는다.**

- Eager PyTorch / 지원되는 `torch.compile` / custom Triton을 비교한다.
- Forward뿐 아니라 backward도 구현한다.
- 실제 shape/stride/dtype, 작은 입력과 tail masking을 다룬다.
- Precision, layout, aliasing/in-place, 지원하지 않는 입력의 fallback 또는 거부를 명시한다.
- 기본 학습 경로는 유지하고 실험 옵션으로만 연결한다. Routing/drop/RNG/모델 수학을 바꾸어 속도를 높이지 않는다.
- CPU에서 CUDA/Triton 없이 import할 수 있어야 한다.

CUDA Graph는 필요할 때 별도 launch-overhead 비교군으로 둔다. Fusion과 같은 방식이 아니며 필수 첫 과제도 아니다.

## P3: 실제 학습 A/B와 채택 gate

- 같은 raw checkpoint와 batch 순서에서 별도 출력 경로로 baseline/fused pilot을 실행한다. WSD horizon/LR/token count와 raw resume state를 유지한다. Pilot branch를 새로운 production WSD/WSM 경로로 취급하지 않는다.
- 사전에 정한 dtype별 tolerance로 output/input gradient/관련 parameter gradient를 비교한다. 오차를 기록하고 통과시키려고 threshold를 늘리지 않는다.
- FP32 reference·실제 dtype·경계 shape·layout·큰 값/작은 값·가능한 고정밀 derivative reference를 검증한다.
- 짧은 학습 loss/gradient, finite 값, checkpoint 저장과 동일 구현 resume를 확인한다. Baseline/fused 수치 동등성과 resume의 exactness 요구는 별개다.
- 재profile로 launch 수/idle gap 가설을 확인한 후 **profiler를 끄고** 전체 update·tokens/s·memory를 반복 측정한다. 최선 값만 보고하지 않는다.
- Backward saved tensor와 임시 메모리도 포함하고 관련 rank의 imbalance/overlap을 확인한다. 단일 GPU 결과를 8 GPU 결과로 확대하지 않는다.
- Correctness 실패나 전체 학습 악화는 채택 거부 또는 experimental 유지 사유다. 연산만 빨라지고 전체 update가 그대로여도 그대로 보고한다. Production 기본값 변경은 별도 검토 후 수행한다.

## 향후 구현 산출물

아래는 **예정 경로**이며 지금 만들어진 실행 도구가 아니다.

- `src/nemochat/kernels/`: reference와 optional Triton forward/backward
- `scripts/benchmark_pretrain_fusion.py`: 독립 benchmark와 JSON
- `tests/test_pretrain_fusion.py`: CPU 계약 테스트 및 CUDA correctness 테스트
- `configs/performance/`: shape/측정 protocol
- Git에서 제외되는 `results/performance/<unique-run>/`: 설정·raw timing/memory·오차·trace·결과 보고서

W&B에서는 `pretrain` stage 아래 implementation/benchmark-role을 구분한다. Microbenchmark/replay 결과를 실제 corpus 학습 throughput과 섞지 않고, profiler 때문에 느려진 값을 budget planner에 넣지 않는다.

## 완료 기준과 증거 수준

CPU에서는 구현 후 reference/입력 계약/optional import를 검증한다. 승인된 CUDA 환경에서 Triton forward/backward, timing/memory, Nsight 가설을 검증한다. Native multi-GPU 학습의 full-update A/B·checkpoint/resume는 별도 gate다.

현재 CUDA/Triton/Nsight 및 실제 학습 gate는 모두 대기 상태다. **이번 변경은 계획·실행 계약 추가이며 성능 개선 완료가 아니다.**

각 실험은 관찰 → 가설 → 변경 → correctness → 연산 성능 → 전체 update 성능 → memory trade-off → 채택/거부 → 남은 한계를 남긴다. 음성 결과도 보존한다.
