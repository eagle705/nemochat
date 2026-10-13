# 대규모 corpus 계획·checkpoint 보관·실험 모니터링

Canonical: [영문 운영 명세](../en/OPERATIONS_PIPELINE.md).

**상태:** 신규 작업 진행 중. 이전 pre-commit 승인은 이전 snapshot의 검토 범위이며, 이번 추가 기능에 그대로 적용되지 않는다. 이 문서 작성 단계에서 실제 HF 업로드나 W&B online 실행은 하지 않았다.

## 사용자 선택과 작업 순서

- Corpus 전체가 너무 크다고 가정해 **bounded 대표 표본**으로 target tokenizer 기반 token 규모를 추정한다.
- 중간 checkpoint를 HF에 보관하고 다음 B300 임대에서 다운로드·검증·재개한다.
- 각 stage에 W&B run/lineage와 실제 지표를 기록한다.

임대 전 표본 조사·예산 시나리오·필요 subset 준비 → 임대 시작 시 환경·checkpoint 복구 검증 → 학습과 실제 counter 모니터링 → 완료된 update boundary에서 snapshot/upload → 임대 종료 전에 read-back·백업 확인 순서다. Upload를 큐에 넣은 것만으로 checkpoint가 안전하게 보존됐다고 판단하지 않는다.

## Corpus 추정

Dataset 카드의 token 수나 압축 파일 byte 수를 그대로 사용하지 않는다. Source별 known population과 표본 설계를 명시하고, 실제 tokenizer revision·전처리·EOS 정책으로 문서 길이와 tokens를 조사한다.

- 균등 문서 표본과 정확한 문서 population은 평균 기반 추정을 지원한다.
- 임의로 고른 shard나 첫 N개 문서는 전체 corpus의 대표 표본이라고 보장할 수 없다.
- Cluster/shard 표본이면 그 설계를 반영한 불확실성 계산 또는 계산 불가 표시가 필요하다.
- Filtering/dedup/split 전후 scope를 구분한다. 작은 표본의 dedup 비율을 전체로 확정하지 않는다.
- 추정 artifact에 seed, source/shard coverage, population/weight, tokenizer/config hash, 문서 한도·표본 selection을 보존한다.
- 최종 준비한 subset의 exact token 수와 corpus 전체 추정치는 별개다.

## 예산과 ETA

실제 model/context/precision/parallel 구성 pilot의 tokens/s 또는 step time을 사용한다. GBS는 일반적인 고정 batch에서 MBS×실제 DP×accumulation이며 TP/CP 등으로 같은 token을 중복 집계하지 않는다.

Prepared·processed·loss-bearing token과 반복 노출을 구분한다. Training-only throughput + overhead 방식과 end-to-end effective throughput 방식을 혼합해 overhead를 두 번 반영하지 않는다. Node 시간당 가격과 GPU 시간당 가격을 구분하고 주 단위 과금·storage·egress·CPU 비용을 별도로 기록한다. 4K throughput으로 64K를 예측하지 않는다.

## HF artifact 두 종류

**재개용 raw training bundle:** 모델/DCP·optimizer·scheduler·RNG·data cursor·실제 token counter·tokenizer/config·환경/source manifest·파일 inventory를 보존한다. Native Megatron bundle을 model repo에 저장해도 `from_pretrained` 지원 형식이 되는 것은 아니다.

**모델-only export/WSM 평가 artifact:** 검증된 export 형식·tokenizer·origin/merge provenance·inventory를 저장한다. 정상적인 optimizer trajectory를 포함한다고 주장하지 않는다.

Private repo의 stage/run/checkpoint별 path와 immutable commit SHA를 사용한다. 신뢰된 snapshot만 허용하고 새 destination에 다운로드한 후 inventory를 검증한다. Pickle 가능한 training state는 HF 출처만으로 안전해지지 않는다. 업로드 성공은 immutable revision의 read-back 검증으로 확인한다. 로컬 pack/restore 및 mock API 테스트는 실제 원격 왕복과 별도 증거다. Public release는 라이선스·데이터 권리 검토 후 별도 승인한다.

## W&B 각 stage의 기준

- **Data:** 조사 scope, accepted/rejected docs/tokens, EOS, tokenizer/data manifest.
- **Pretraining:** CE, 실제 LR/phase, processed/loss tokens, update time, checkpoint ID와 별도 평가.
- **Mid-training:** source blend/exposure, retention, parent checkpoint와 처리량.
- **Long-context:** 길이별 비용/노출·long/short update와 실제 retrieval 결과. 현재 schedule-only 도구는 planning run이지 학습 run이 아니다.
- **SFT:** assistant loss mask/truncation·실제 loss tokens·generation 평가.
- **RLVR:** prompts/rollouts/update tokens 분리, reward와 실제 verifier correctness, 측정된 KL/entropy 등.
- **MOPD:** teacher scoring/generated/update tokens·teacher/student 실제 비용과 평가.
- **Evaluation/WSM:** raw/merged protocol, fixture/partial scope, errors, merge 입력/계수/output handle.

공통 project/group, 고유 run ID/stage/parent artifact 및 data/tokenizer/config/source revision을 유지한다. 후반 stage는 아직 실행 코드가 없으므로 contract를 준비해도 실행 검증됐다고 표시하지 않는다.

Tracking은 disabled 또는 explicit offline이 기본값이다. Online은 entity/project·credentials·실행 의도가 확인돼야 한다. Secret은 nested config에서도 제거하되 tokenizer/token counter 같은 정상 metadata를 무차별 삭제하지 않는다. Failed run은 실패 status로 종료하고 logging이 학습 RNG를 바꾸지 않는지 검증한다. Offline run ID 공유는 server-side resume 보장이 아니다. 분산 rank-zero 연동은 실제 GPU 검증 전에는 계획으로 표시한다.

큰 weight를 W&B에 중복 업로드하기보다 검증된 HF immutable handle을 lineage로 연결한다.

## 완료 조건

표본/계획 수학 테스트와 실제 tokenizer bounded CLI → local bundle pack/restore·hash 검증 → 실제 W&B offline tiny-HF run 및 exact resume → 독립 리뷰·부모 전체 회귀 검증 순서다. 실제 HF remote round trip과 W&B online은 목적지·권한 확인 후 별도 gate로 남긴다.
