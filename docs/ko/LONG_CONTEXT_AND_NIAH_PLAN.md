# 장문·단문 교대 CPT와 NIAH 평가 해설

영문 canonical 명세: [Long Context and NIAH Plan](../en/LONG_CONTEXT_AND_NIAH_PLAN.md)

상태: 논문과 SDK 소스 확인을 마친 역사적 **설계 문서**. 이후 tiny-HF CPU NIAH 요청 6개와 실제 0/6 heatmap을 생성했다. [실행 기록](../IMPLEMENTATION_VERIFICATION.md)을 참고한다. 실제 NemoChat GPU 장문 학습·inference·성능 Figure는 미검증이다.

## 1. Super 논문에서 확인한 내용

공식 Technical Report PDF 16쪽 §2.6:

- constant LR: 4.5e-6, global batch size: 16.
- GB200에서 CP64 / TP2 / EP64.
- LC data blend: 장문 document QA 20%, downscaled Phase 2 데이터 80%.
- 먼저 1,048,576-token 장문 CPT를 34B tokens 수행.
- 다음은 1M·4K를 교대 학습하며 17B tokens 수행. 이유는 수학 benchmark의 작은 성능 저하를 완화하기 위해서.

즉 논문 방식은 **긴 context 능력 확보 → 장문·단문 교대 학습으로 짧은 task 능력 회복**이다. 기존의 8K→16K→32K→64K 예산 표는 NemoChat의 제안이었고, Super의 보고된 schedule은 아니었다.

해당 절에는 교대 간격, 장문/단문 token 비율 및 두 번째 단계의 길이별 batch 설정이 명시되지 않았다. 따라서 '논문이 정확히 1:1 step 또는 1:1 tokens였다'고 말할 수 없다.

## 2. 우리 단일 노드에서는 어떻게 바꾸나?

NemoChat은 **64K long-only CPT → 64K/4K alternating CPT**를 기본으로 한다.

| 단계 | Tutorial | Extended |
|---|---:|---:|
| LC-1: 64K 장문 전용 | 67M tokens | 670M |
| LC-2: 64K/4K 교대 | 33M tokens | 330M |

대략 2:1의 단계 예산은 논문의 34B:17B를 참고한 제안이다. 작은 모델에서 최적이라는 뜻은 아니다. 8K/16K/32K는 큰 학습 단계를 별도로 배분하기보다 mask/state/CP/memory correctness를 확인하는 smoke 구간으로 둔다.

장문 stream은 20% grounded document QA + 80% selected pre/mid replay로 시작한다. 단문 stream은 4K 수학·코드·일반 데이터 replay를 강화한다. 이는 공개 가능한 대체 데이터 설계이며 원본 Super QA 데이터의 재현이라고 주장하지 않는다.

64K를 통과하면 같은 패턴으로 **128K long-only → 128K/4K**, 다음 **256K long-only → 256K/4K**를 별도 예산으로 확장한다. 1M 훈련은 기본 목표가 아니며 단일 노드에서 보장하지 않는다.

## 3. '번갈아'의 중요한 함정

장문과 단문의 global sequence batch 수가 같을 때 64K step과 4K step을 1회씩 수행하면, 장문이 전체 tokens의 약 94.12%를 차지한다. 256K/4K에서는 약 98.46%다.

따라서 **step 비율, token 비율, wall-clock 비율을 따로 정의**해야 한다.

우리 LC-2 초기안은 **장문/단문 update당 유효 token batch를 맞춘 뒤 1:1 교대**한다. 장문 64K sequence 1개와 동일 token 수를 맞추려면 단문 4K sequence 16개가 필요하다. 실제 GBS는 DP/microbatch 제약을 만족하도록 함께 늘린다. 이 50:50 token 비율은 우리 설계값이며 Super 논문에서 확인한 비율은 아니다.

## 4. SDK 지원은 어디까지 확인됐나?

- Megatron Core에 Mamba context parallel 구현이 있고, Bridge Super recipe에도 CP2/32K 구성이 있다.
- 하지만 기존 recipe가 64K/4K를 자동 교대하는 launcher를 제공한다고 확인한 것은 아니다.
- 별도 long/short dataloader와 optimizer-update 수준 controller가 필요하다. 모든 rank의 schedule을 동기화하고, 두 data cursor와 optimizer/scheduler/RNG를 함께 resume해야 한다.
- 첫 구현은 같은 CP/TP/EP topology로 교대한다. 길이가 바뀔 때마다 process group을 재구성하지 않는다.
- CP2/4/8은 조사 후보다. Mamba head/group divisibility뿐 아니라 attention/MoE 그룹과 kernel을 실제 실행으로 검증한다.
- CUDA graphs는 처음에는 끄고, 나중에 long/short shape별 capture를 검토한다. 짧은 데이터를 64K로 모두 padding해서 교대를 흉내 내지 않는다.
- Training CP와 inference CP는 다른 지원 경로다. Mamba inference CP에는 별도 제한이 있으므로 serving 가능성을 따로 검사한다.

## 5. NIAH 평가와 Figure

NIAH는 긴 haystack의 특정 위치에 무작위 key/value를 넣고, 끝에서 key를 물어봐 정확한 value를 되찾는지 평가한다.

### 평가 grid

- 빠른 smoke: 4K/32K/64K × depth 0/25/50/75/100% × seed 3개 = 45 requests/checkpoint.
- 64K main: 4K/8K/16K/32K/64K × depth 0,10,...,100% × seed 10개 = 550 requests/checkpoint.
- 256K extended: 128K/256K를 추가하여 770 requests/checkpoint.
- Seed 10개는 경제적인 diagnostic 수준이다. 최종 주요 cell에는 반복 수를 추가하고 confidence interval과 n을 보고한다.

길이와 depth는 실제 학생 tokenizer로 계산한다. Instructions/query/chat special tokens와 생성 여유를 포함해 최대 길이를 넘지 않게 하고, 최종 prompt 안에 needle이 살아 있는지 검사한다. 첫/끝 depth의 정의도 고정한다.

### 산출물

1. 길이 × 삽입 위치 **정답률 heatmap**, PNG/SVG.
2. **Pre-LC / Post-LC-1 / Post-LC-2 heatmap 비교 Figure**.
3. 길이에 따른 평균 정답률과 uncertainty 그래프.
4. 장문 NIAH와 짧은 수학·코드 retention을 별도 panel로 나타내는 Figure.
5. 모든 예측의 `predictions.jsonl`, cell별 성공/시도/error 수의 `cells.csv`.

Heatmap은 모든 checkpoint에서 같은 0~1 color scale을 사용한다. 미지원 길이/OOM/timeout은 정확도 0과 섞지 않고 별도 표시한다. Figure는 **실제 inference 결과만**으로 만들며 tiny-HF control의 실제 heatmap은 존재하지만 NemoChat 장문 성능 결과가 아니다. 실제 target checkpoint 없이 그럴듯한 target 성능 heatmap을 만들지 않는다.

영문 Figure labels/caption을 기본으로 두고 한국어 해설을 따로 제공한다. 모델·tokenizer·backend revision, checkpoint ID, seed/task hash, decoding 설정을 결과에 남겨 재생성할 수 있게 한다.

## 6. NIAH만으로 256K 코딩 능력을 주장하지 않기

Super 논문 §2.7은 long-context 평가에 RULER를 사용하며 task당 100 samples라고 명시한다. Vanilla NIAH만 맞혀도 multi-hop reasoning이나 coding은 실패할 수 있다.

따라서 NIAH에 더해 multi-needle/distractor, RULER-style reasoning/aggregation, grounded QA, 실제 cross-file 수정과 unit tests를 평가한다. LC-1에서 수학 성능이 떨어지고 LC-2에서 회복되는지도 별도 heldout으로 측정한다.

또 4K에서도 retrieval/출력 task를 못 하는 약한 base 모델이라면 긴 길이의 실패를 context 문제로만 해석하지 않는다. Base model과 SFT model의 prompt template 차이도 공개한다.

출처:
- [Super Technical Report, §2.6–2.7, PDF p.16](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf)
- [Megatron Mamba CP](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py)
- [Bridge Super recipe](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_super.py)
- [RULER](https://github.com/NVIDIA/RULER)
