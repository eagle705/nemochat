# Pretraining: WSD와 stable-phase WSM 실험 해설

## 성능 실습 범위 추가

[Nsight Systems / optional Triton 실습](PRETRAIN_PERFORMANCE_LAB.md)을 pretraining에 포함한다. WSD/WSM과 기본 경로를 유지하고 forward/backward, 실제 update throughput/memory 및 resume를 검증한다. GPU 구현과 측정은 아직 대기다.

영문 canonical: [PRETRAIN_WSD_WSM_EXPERIMENT.md](../en/PRETRAIN_WSD_WSM_EXPERIMENT.md).

## 목표와 선택한 방식

하나의 WSD 학습 경로를 유지하면서 stable 구간의 평가 시점마다 raw checkpoint와 과거 window를 merge한 모델을 비교한다. 사용자가 선택한 것은 [Megatron-LM PR #5114](https://github.com/NVIDIA/Megatron-LM/pull/5114)의 **minus-sqrt weighted checkpoint merging / WSM**이다. 시점마다 별도 decay branch를 학습한 뒤 평균내는 방식은 이번 선택이 아니다.

WSD는 LR schedule, WSM은 offline model-weight 합성이다. Merge 결과가 원래 optimizer-bearing 학습 checkpoint를 대체하지 않으며 원래 학습은 raw에서 계속한다. 최종 decay의 실행 시점과 길이는 별도 학습 정책이다.

## 평가 시점 T의 작업

1. Raw checkpoint와 optimizer/RNG/scheduler/data cursor를 보존한다.
2. 동일 run/model/layout의 **T까지의 과거 stable checkpoint**만 선택한다. Future checkpoint를 사용하지 않는다.
3. 고정 revision의 공식 merge 도구를 실행한다. 계수·입력 iteration/token counter·dtype·extra-state source·provenance를 기록한다.
4. Merge output을 독립적으로 로드해 값과 형식을 검사한다. Dry-run/save 성공만으로 모델 inference 성공을 주장하지 않는다.
5. 동일 NeMo Gym task revision·입력 sample·prompt/template·tokenizer·decoding·scorer로 raw와 merged를 평가한다.
6. Task별 score와 merged-minus-raw delta를 보고한다. 감소·실패·부분 실행도 숨기지 않는다. 개선을 완료 조건으로 강요하지 않는다.
7. Held-out CE/PPL/BPB와 기존 CORE는 별도 진단으로 유지한다. Synthetic fixture/tiny HF control 점수를 target 모델 능력으로 표현하지 않는다.

## 실제 token window

공식 merge CLI는 iteration 구간을 선택하며 token window를 직접 해석하지 않는다. 실제 cumulative consumed/loss-bearing counter로 구간을 변환하고 요청한 token window와 실제 선택된 span을 함께 기록한다. Packing·가변 길이·loss mask·skipped work가 있으면 nominal batch×context length는 실제 counter를 대체하지 못한다.

Window 길이와 저장·평가 주기는 실제 throughput/storage/budget 측정 후 정한다. Nemotron의 절대 token 규모를 그대로 작은 튜토리얼에 복사하지 않는다.

## Bridge 설정과 runtime

조사한 Bridge scheduler는 `lr_decay_style="WSD"`, `lr_warmup_iters`, `lr_decay_iters`, `lr_wsd_decay_iters`, `lr_wsd_decay_style`를 노출한다. 전체 horizon을 명시해 재개 시 바뀌지 않도록 한다. 조사한 config에서는 WSD decay duration 생략 시 전체 horizon으로 기본값을 잡으므로 원하는 stable plateau를 위해 duration을 명시해야 한다. Sample/iteration 기반 필드를 섞지 않는다.

이것은 pinned source 확인이지 실제 GPU 실행 검증이 아니다. [WSD CPU 구현](WSD_IMPLEMENTATION.md)과 [공식 WSM runtime](WSM_IMPLEMENTATION.md)을 구분한다. CPU HF checkpoint를 공식 Megatron `torch_dist` 입력으로 직접 사용할 수 없다.

## 신뢰와 state

DCP metadata/common/extra state 역직렬화는 신뢰된 입력만 허용한다. 평균 대상은 model floating tensors이고 extra state는 선택한 한 source에서 복사한다. Model-only merge는 유효한 optimizer/RNG trajectory를 합성하지 않는다. 새로운 학습 branch를 merged weights에서 시작하려면 optimizer/RNG reset과 별도 provenance가 필요하다.

## Gym 평가 적합성

공식 `NVIDIA-NeMo/Gym/benchmarks/`의 config와 scorer를 사용한다. Chat template·instruction format·tool·judge 의존성을 먼저 검사하고 base model 평가 해석을 명시한다. 변경한 base prompt protocol을 공식 protocol과 동일하다고 주장하지 않는다. [Gym 구현 해설](GYM_PRETRAIN_EVALUATION.md)을 참고한다.

현재 Gym 코드는 저장된 공식 결과의 **importer/comparator**다. 실제 공식 benchmark runner와 모델 endpoint/export는 별도 prerequisite이며, 실제 NemoChat raw/merged score는 미측정이다.

## 증거 수준

CPU scheduler, tiny-HF exact resume, synthetic DCP 공식 merge, importer fixture 검증은 각각 mechanics를 입증한다. 실제 NemoChat GPU 학습·hybrid export/reload·Gym benchmark inference·WSM 성능 향상은 별도 실제 실행이 필요하다. 최신 증거와 미해결 리뷰 사항은 [검증 기록](../IMPLEMENTATION_VERIFICATION.md)과 [agent log](../../NemoChatAgent.md)를 따른다.
