# MoE 연산 역할 카드

[영문 원본](ROLES.md). 실제 함수는 [`control.py`](../lens_nsight_bridge/control.py)에
있습니다. 각 operation record에 파일/스니펫 SHA256, 정확한 함수 행, source ID가
저장됩니다. CLI는 이 설명을 코드로 만들어내지 않고 캡처 snapshot을 읽습니다.

| 역할 / 실제 함수 | 의미 / 모델러가 확인할 지점 |
|---|---|
| routing / `routing` | 토큰 `[T,D]`를 expert logits `[T,E]`로 투영. expert 출력 계산과 다릅니다. |
| top-k / `topk` | 토큰별 expert 2개를 선택하고 선택된 logits에 softmax. 전체 expert가 활성인 것은 아닙니다. |
| permutation / `permutation` | T×K 토큰 복사본을 expert별 정렬. token ID와 weight 대응을 보존합니다. |
| dispatch / `dispatch` | expert별 로컬 gather. 빈 batch도 허용. 네트워크/NCCL/EP all-to-all이 아닙니다. |
| expert up / `expert_up` | `[N_e,D] @ [D,2H]` 투영. grouped/fused native GEMM이 아닙니다. |
| activation / `activation` | SwiGLU: `SiLU(gate) * value`. eager 연산, native fusion 아님. |
| expert down / `expert_down` | `[N_e,H] @ [H,D]`로 모델 폭으로 복귀. |
| combine / `combine` | routing weight를 곱하고 원래 token ID로 scatter-add. top-2 기여를 합산. |
| backward / `backward` | loss에서 autograd 진입. 각 gradient 커널의 forward source를 뜻하지 않음. |

T=8, D=16, E=4, K=2, H=24의 랜덤 텐서 교육용 mean-square loss입니다. 학습
품질/Nemotron 성능 측정이 아닙니다. token-major 참조는 routing/top-k weight를
공유하며 dispatch/combine의 출력 및 gradient를 검사합니다. native fused/분산
동등성을 입증하지 않습니다.

증거 수준은 **명시적 host 함수 범위 → 실제 CUDA launch correlation 및 수집된
backtrace/compiler metadata → 커널 source 대응** 순서입니다. 지금 CPU 검증은
첫 번째 수준이며 실제 NVTX 하드웨어도 미실행입니다. eager one-to-many,
fusion many-to-one, backward forward-origin, graph replay 모호성을 남겨 둡니다.
시간 근접/추측한 커널 이름으로 역할을 붙이지 않습니다. 기존 Nsight의 원래 커널
이름을 유지합니다. 네이티브 설명/소스 GUI 위젯은 별도 미완료 게이트입니다.
