# Pretraining metric 상세 해설과 ClimbMix 채택

영문 canonical: [Pretraining Metrics and ClimbMix](../en/PRETRAIN_METRICS_AND_CLIMBMIX.md)

상태: 실제 nanochat 코드·평가 bundle·dataset card를 확인한 **설계 문서**다. GPU 모델 평가, 데이터 구축 완료 또는 측정 Figure가 아니다. 숫자 예시는 수식 설명용이며 모델 실측값이 아니다.

## 1. 결론

- 선택한 pretraining 실험은 **WSD 단일 학습 경로 + stable 평가 시점의 PR #5114 minus-sqrt WSM**이다. 동일 pinned 공식 NeMo Gym benchmark로 raw/merged를 비교한다.
- nanochat의 **validation BPB와 CORE**는 CE/PPL과 함께 별도 진단 지표로 유지한다. 실제 NemoChat 학습·공식 Gym inference는 미실행이다.
- CE/PPL도 남기되 MoE auxiliary loss와 분리한다.
- CORE의 task별 점수와 domain별 BPB를 함께 보고한다.
- Math/code 실행 평가와 NIAH는 별도 지표로 유지한다.
- **비상업적 교육용**이라는 프로젝트 목적에 맞춰 기존 60% FineWeb-Edu 주 blend를 ClimbMix로 변경한다. 데이터 구축·권리/provenance 확인은 실행 gate로 남는다.
- 나머지 code 20%, math/science 10%, long natural text 10%와 전체 예산은 유지한다. ClimbMix를 code/math/long source의 완전한 대체물로 간주하지 않는다.

## 2. CE: 실제 다음 token을 얼마나 잘 예측하는가

각 token의 실제 확률이 p일 때 `-ln(p)`를 계산하고 유효 target tokens 전체 평균을 낸다.

`CE = Σ(-ln p(target | prefix)) / 유효 target 수`

- 낮을수록 실제 텍스트를 더 높은 확률로 예측했다.
- Padding/ignored/loss-masked tokens는 제외한다.
- 학습이 좋아지는지, source별 분포 이동/과적합이 있는지 점검하기 좋다.
- **MoE load balancing, z-loss 등 regularizer를 더한 total training loss와 별개**다. Total loss를 exp해서 PPL로 쓰면 안 된다.

## 3. PPL: CE를 지수로 표현한 불확실성

`PPL = exp(CE)`

예: CE=2 nats/token이면 PPL≈7.389056. 이는 정답률 7.4%나 vocabulary가 7개라는 뜻이 아니라 예측 불확실성의 스케일이다.

같은 raw text도 tokenizer가 다르면 token 수와 평균 token 길이가 달라진다. 따라서 PPL은 **동일 tokenizer/corpus/context/mask 안에서의 비교**에 사용한다. Super tokenizer와 nanochat tokenizer의 PPL을 그대로 순위 비교하지 않는다.

## 4. BPB: tokenizer 단위 대신 text byte로 정규화

`BPB = 유효 target들의 총 NLL(nats) / (ln(2) × 해당 text bytes 합)`

- 분자의 nats를 ln(2)로 나눠 bits로 변환한다.
- 분모는 ordinary target tokens가 표현하는 **raw bytes**다.
- BOS/EOS 등 special tokens와 padding/ignored targets는 분자·분모에서 모두 제외한다.
- 배치나 rank별 BPB를 평균내지 않는다. 전체 NLL과 bytes를 각각 합한 뒤 한 번 나눈다.

예: 총 NLL 1000 nats, 유효 bytes 2000이면 **BPB≈0.72134752**. 실측이 아닌 산술 예시다.

**해석:** 같은 텍스트를 더 적은 bits/byte로 설명할 정도로 예측력이 좋아졌다. 실제 압축 파일을 생성한 크기라는 뜻은 아니다. 낮을수록 좋다.

BPB는 vocabulary/token 단위 영향을 줄여 주지만 모든 비교 조건을 없애지는 않는다. **동일한 raw heldout 텍스트**, normalization, context/mask/BOS 정책으로 비교해야 한다. Code BPB와 prose BPB는 자연적인 entropy 차이도 있으므로 서로 낮은 쪽이 곧 더 좋은 domain 성능이라는 뜻은 아니다.

### Super tokenizer에서 특히 조심할 점

`decode([token_id]).encode('utf-8')`의 길이를 그대로 쓰면 partial UTF-8 token이 replacement character로 바뀌거나 space cleanup이 개입할 수 있다. 원래 tokenizer backend의 raw-byte 표현을 검사해 byte 길이를 정확히 복원해야 한다.

ASCII/공백/한국어/부분 UTF-8/code/special/ignored/padding fixture로 bytes 합의 round-trip을 검증한다. Super tokenizer는 유지하고 metric 구현을 포팅한다. Metric 때문에 tokenizer를 다시 학습하지 않는다.

Megatron에서는 TP/EP에 복제된 loss를 여러 번 세면 안 된다. CP token partitions와 DP sample partitions를 고려해 **각 target을 딱 한 번** 합산한다. nanochat의 DDP용 global all-reduce 코드를 그대로 복사하지 않는다.

## 5. CORE: loss 감소가 실제 base capability로 이어졌는가

CORE는 여러 ICL tasks의 성능을 **baseline 보정 후 동일 task 가중치로 평균**한다.

`centered_t = (accuracy_t - baseline_t) / (1 - baseline_t)`

`CORE = task별 centered_t의 평균`

예: baseline 25%, accuracy 60%이면 `(0.60-0.25)/(1-0.25)=0.46666667`.

- Baseline 수준이면 0, 완벽하면 1, baseline보다 못하면 음수다. 음수를 0으로 clamp하지 않는다.
- `CORE=0.3`은 raw 정답률 30%가 아니다.
- 데이터 수가 큰 task가 자동으로 더 큰 가중치를 갖지 않는다.
- baseline은 단순 선택지 수 역수가 아니라 bundle metadata를 사용한다. 예를 들어 BoolQ baseline은 62%다.
- Raw task accuracy, centered score, sample 수를 같이 보여야 aggregate가 숨기는 regression을 찾을 수 있다.

### 실제 nanochat 채점 방식

1. **Multiple choice:** 각 선택지 continuation의 평균 token NLL이 가장 낮은 선택지를 고른다. A/B/C 문자 생성, NLL 합, byte-normalized likelihood로 바꾸면 다른 metric이다.
2. **Schema:** 서로 다른 context 아래 공통 continuation의 평균 NLL로 고른다.
3. **Language modeling:** reference continuation의 각 다음 token에서 argmax가 모두 맞는지 확인한다. 앞의 정답 token은 입력으로 주어진다.

특히 3번은 **teacher-forced exact-token match**다. 정답 전체를 자유 생성하는 QA, 일반 SQuAD F1, sampled pass@1과 같지 않다. 이 차이를 블로그에서 설명한다.

Pinned eval bundle은 **22개 task entries**다. HellaSwag zero/few-shot, ARC, COPA, PIQA, OpenBookQA, LAMBADA, Winograd/Winogrande, Jeopardy, BIG-bench tasks, LSAT AR, SQuAD/CoQA/BoolQ 등이 포함된다. 전체 task IDs는 영문 명세에 기록했다.

**이 설정에는 MMLU/GSM8K/HumanEval이 포함되지 않는다.** Metadata CSV에는 더 많은 tasks가 있지만 `core.yaml`에 선택된 항목만 aggregate에 들어간다. Math/code 실행 평가는 별도로 유지해야 한다.

또 nanochat 소스 자체에 SQuAD reference parity TODO가 있다. 따라서 무조건 '공식 DCLM 완전 동등'이 아니라 **nanochat-compatible CORE**로 보고한다.

### 평가 빈도와 샘플링

- BPB: frozen pack으로 consumed-token 기준 주기 평가; 실제 overhead로 주기를 정한다.
- CORE quick: task당 최대 100개 고정 samples, sparse milestones.
- 필요하면 500개/task intermediate.
- Part I 주요 handoff에서는 예산이 허용하는 full pinned pack.

Quick/500/full은 같은 점수인 것처럼 혼용하지 않는다. 작은 sample의 uncertainty와 task별 수를 보고한다. Few-shot/delimiter/BOS/answer boundary/context truncation은 고정하며, tokenizer 변경으로 boundary가 바뀌면 테스트해야 한다.

## 6. ClimbMix가 적절한가?

**비상업적 교육용 reference에는 적절한 주 데이터 후보라서 채택 방향으로 변경했다.** 공식 자료에 따르면 400B-token 영어 데이터이며 topic 1000개 grouping, 광고/교육 품질 classifiers와 가중 mixture를 사용했다. nanochat의 FineWeb-Edu→ClimbMix 개선 기록도 근거가 된다.

하지만 우리 hybrid/latent MoE·Super tokenizer·소규모 예산에서 같은 개선을 보장하지는 않는다. 승인된 A/B 구조 비교에는 **같은 ClimbMix subset**을 써야 한다. 데이터 비교를 동시에 바꾸면 A/B 결과를 해석할 수 없다.

### 출처와 라이선스

- NVIDIA 공식 `nvidia/Nemotron-ClimbMix`: **CC BY-NC 4.0**, GPT-2 tokenized Parquet.
- nanochat `karpathy/climbmix-400b-shuffle`: detokenized/shuffled text Parquet; card 표기는 MIT.
- 커뮤니티 `OptimalScale/ClimbMix`: detokenized text; card 표기는 Apache-2.0.

nanochat repackaging code는 NVIDIA origin, GPT-2 detokenization, seed-42 shuffle을 명시한다. **커뮤니티 표기가 원본 NC·attribution 조건을 없앤다는 근거로 쓰지 않는다.** 교육용이라 사용을 일괄 차단하지 않고, 원본 조건/출처/변환을 유지한다.

정제 dataset 재배포와 weights 공개 조건은 각각 확인한다. 모든 weights가 자동으로 CC BY-NC를 상속한다고 단정하지 않으며, code license도 별도다. 이후 상업용으로 목적이 바뀌면 해당 결정은 다시 검토한다. 법률 자문은 아니다.

### 다운로드·token quota

- Tutorial: general-text **1.8B**, Extended: **18B**, 실제 Super tokenizer consumed tokens 기준.
- 400B 전체를 받지 않는다. Revision/shard 선택과 bounded row-group/cache, accepted-token ledger로 필요한 subset만 확보한다.
- nanochat는 6543 shards 중 마지막 `shard_06542.parquet`를 val로 사용한다. 해당 shard를 train에서 제외하고 cross-split near dedup을 확인한다.
- GPT-2 원본 IDs는 Super에서 쓰지 않는다. Text shuffle을 retokenize하거나 선택한 official token shards만 detokenize→retokenize한다.
- **nanochat downloader를 quota 없이 실행하지 않는다.** `num_files=-1`은 실제로 전체 train shards를 선택한다. Repackaging reference도 전체 corpus 로드/shuffle/upload를 포함하므로 실행용으로 가져오지 않는다.
- Text-only 재포장에는 원 URL/문서 경계가 사라졌을 수 있다. Revision/shard/row/content hash만으로 보존되는 lineage의 한계를 공개한다.
- 이미 필터된 mixture를 과도하게 다시 필터하지 않되 PII/secret/dedup/평가오염 검사는 유지한다.
- ClimbMix가 곧 자연 64K/256K 장문 corpus는 아니다. 별도 long source는 유지한다.

ClimbMix heldout BPB 외에 독립 FineWeb-Edu·code·math heldout을 둔다. 데이터셋을 비교할 때는 같은 heldout raw text로 비교하며 각자 자기 validation 점수를 직접 비교하지 않는다.

## 7. 문서·블로그 Figure

Part I에 추가:
- Validation BPB vs consumed tokens와 실제 유효 training wall time.
- Domain별 BPB 및 CORE aggregate/task별 score panel.
- ClimbMix quota/curation/provenance와 license 설명.
- A/B 성능·expert/profile와 NIAH/retention.

Part II/III에도 동일 frozen base-eval pack을 retention diagnostic으로 유지한다. 모든 Figure는 실제 inference 결과로 생성한다. 이 문서는 역사적 설계다. 이후 CPU HF evaluator/inference, bounded data acquisition과 tiny-model NIAH는 실제 실행했다. [검증 기록](../IMPLEMENTATION_VERIFICATION.md)을 참고하며 NemoChat GPU/분산 evaluator와 공식 Gym benchmark 실행은 별도 미검증 과제다.
