# 대규모 코퍼스 표본 추정과 학습 시간·예산 계산

## 실행

저장소 루트에서 CPU/test 환경의 Python을 사용합니다.

```bash
python scripts/estimate_corpus.py --config configs/planning/corpus_local_smoke.json --output /tmp/corpus-local.json
python scripts/plan_training.py --config configs/planning/training_measured_example.json --output /tmp/training-plan.json
python scripts/plan_training.py --config configs/planning/training_unmeasured.json --output /tmp/unknown-plan.json
python -m pytest -q tests/test_planning.py
```

출력 파일이 있으면 덮어쓰지 않고 실패합니다. 입력 경로는 설정 파일 위치가
아닌 **현재 작업 디렉터리 기준**입니다. 코퍼스를 자동 다운로드하지 않습니다.
실제 토큰화에는 `transformers`, Parquet에는 선택 의존성 `pyarrow`가 필요합니다.
토크나이저도 캐시에서만 읽습니다(`local_files_only=True`, remote code 비활성화).
캐시가 없다고 가짜 토크나이저로 대체하지 않습니다.

실제 추정에는 데이터 준비 정책과 같은
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`, revision
`2dc98e2afe4face0e4ce40972a915c45368bd34a`를 사용합니다.
GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`는
`purpose: smoke`일 때만 허용됩니다. 제공된 smoke도 Super를 사용합니다.
모델 가중치나 GPU 추론은 필요 없습니다. revision과 backend hash는 선언된
토크나이저 식별 정보이지 변경 가능한 로컬 캐시의 진위 증명은 아닙니다.

## 전체 다운로드 없이 표본 만들기

`strata`의 각 항목은 하나의 source/shard 모집단을 명시합니다.
로컬 `.jsonl` 또는 `.parquet`, `population_documents`(알려진 모집단 문서 수),
`sample_documents`(표본 수), `mode`를 지정합니다. 문자열 `text` 필드를
읽으며 `text_field`로 다른 필드를 지정할 수 있습니다.

- **`reservoir`**: 명시적으로 선택한 로컬 샤드 전체를 읽고 Algorithm R로
  문서를 균등 비복원 추출합니다. 토큰화는 표본만 합니다. 읽은 문서 수와
  선언된 모집단 수가 다르면 실패합니다. 필수 `seed`와 source/shard 이름에서
  독립적인 난수 seed를 만들어 strata 순서를 바꿔도 표본이 같습니다.
  전체 코퍼스를 읽는 것은 아니지만 선택한 샤드는 끝까지 읽어야 합니다.
- **`representative`**: 사용자가 외부에서 미리 만든 대표 표본 파일을
  사용합니다. 파일에 정확히 `sample_documents`개 행이 있어야 합니다.
  `population_documents`는 표본이 대표하는 원래 모집단의 문서 수입니다.
  `sampling_provenance`에 모집단 frame, 샤드/문서 선택 설계, 가능하면 포함
  확률 및 모집단 수의 근거를 적습니다. 대표성은 사용자 진술이며 코드는
  검증하지 못합니다. 조건부 점 추정만 제공하고 신뢰구간은 **null**입니다.
  더 긴 파일의 앞부분만 읽고 대표 표본이라고 부르지 않습니다.

영문 문서의 representative 예제 또는 smoke 설정을 복사해 실제 경로와
모집단 수를 넣습니다. 층별 표본 수를 사용자가 정하며 임의의 비례 배분은
하지 않습니다. 중복 source/shard, 중복 파일, 모집단보다 큰 표본은 거부합니다.
필수 상한: `max_files`, `max_input_bytes`, `max_document_bytes`,
`max_sample_documents`, `max_sample_text_bytes`, `max_sample_tokens`.
파일 크기를 미리 확인하고 JSONL 행도 제한된 크기로 읽습니다. 표본 바이트나
토큰 상한을 넘으면 문서를 잘라 추정하지 않고 실패합니다. reservoir 메모리는
표본 문서 수 × 문서 크기 상한으로 제한되며, 최종 표본 바이트 상한은 선택 후
검사합니다. Parquet는 압축 파일 크기보다 메모리가 훨씬 커질 수 있습니다.
Arrow의 페이지 디코딩 때문에 한 행 batch도 엄격한 RSS 상한은 아닙니다.
실행 중 입력 파일을 변경하지 마세요. 동시 교체는 신뢰 경계 밖입니다.

**선택한 몇 개 샤드는 전체 source의 균등 표본이 아닙니다.** 이 도구는 명시된
모집단만 합산합니다. 샤드 개수로 곱해 전체 source로 확대하지 않습니다.
전체 추정이 필요하면 외부에서 설계·모집단을 확립한 대표 표본 manifest가
필요합니다. 문서 수준 신뢰구간으로 샤드 선택 편향을 없앨 수는 없습니다.

## 토큰·바이트와 불확실성

층 h에서 추정 text 토큰 = `N_h × 표본 평균 text 길이`입니다.
EOS 토큰 = `N_h`, 총 토큰 = text + EOS입니다. UTF-8 원문 바이트도
`N_h × 표본 평균 바이트`로 추정합니다.
`add_special_tokens=False`, BOS 없음, 문서당 EOS 1개를 사용합니다.
EOS는 **원문 바이트 0개**입니다. EOS 문자열의 길이를 원문 바이트에 더하지
않습니다. 보고서에는 표본 행/hash, 문서 수, 바이트, text/EOS 수, 층별
min/max/mean/p50/p95/p99, 토크나이저 revision/hash, 상한과 scope가 있습니다.
표본 분위수는 모집단 분위수나 토큰 가중 context 분포가 아닙니다.

고정된 각 샤드 안에서 완전한 reservoir 표본일 때 분산은
`N_h² × (1 - n_h/N_h) × 표본 길이 분산 / n_h`입니다.
층별 분산을 더하며 서로 다른 샤드 문서를 모두 IID라고 가정해 합치지 않습니다.
이는 **선택된 고정 샤드에 조건부인 층화 추정**입니다. 전체 source의 샤드
추출 불확실성까지 처리하는 2단계 cluster 추정은 아닙니다.
전수 표본은 분산 0, 비전수 n=1 또는 representative 항목이 있으면 전체 CI는
null입니다. 정규근사 95% 구간은 작은 표본/긴 꼬리에서 부정확할 수 있습니다.
모집단 수의 오류, 미선택 샤드, 필터·중복 제거 불확실성을 포함하지 않습니다.
작은 smoke의 구간을 강한 통계적 증거로 사용하지 마세요.

`scope`에는 `population`, `shard_selection`, `heldout`, `filter`, `dedup`을
반드시 설명합니다. 추정기는 빈 문자열과 중복을 포함한 **원시 행**을 셉니다.
heldout 제외, 필터, dedup, 토크나이저 빈 결과 제외, quota truncation,
packing을 수행하지 않습니다. 실제 준비 파이프라인은 빈 문서·예약 문서·중복을
제외하므로 결과가 그대로 실제 학습 토큰 수인 것은 아닙니다. 처리 후 모집단
수를 원하면 이미 같은 처리가 끝난 frame의 표본/모집단 수와 처리 근거를
제공하세요. 작은 표본의 중복률을 전체 dedup 효과로 확대하면 안 됩니다.

## 측정 기반 시간·비용 계획

작고 직접적인 함수로 계산하며 GPU 이름에서 성능을 추측하지 않습니다.

- `GBS = data_parallel × micro_batch_size × gradient_accumulation_steps`.
  DP는 TP/PP가 있을 때 전체 GPU 수가 아닌 실제 데이터 병렬 replica 수입니다.
  별도 GBS가 있으면 식과 같아야 합니다. 모델 메모리나 TP/PP/EP 배치는 검증하지
  않습니다.
- 단계마다 `sequence_length`, `target_tokens`, 선택적 `completed_tokens`,
  측정값과 `measurement_provenance`를 넣습니다. 긴 context에는 별도의 측정값이
  필요합니다.
- 실제 `tokens_per_update` + 관측 `step_seconds` 목록, 또는
  `tokens_per_second` 중 하나를 제공합니다. step 목록은 평균 시간을 사용합니다.
  목표와 측정의 nonpadding/EOS 기준이 같아야 합니다. 측정 토큰/update는
  GBS × length 이하여야 합니다. throughput만 주고 토큰/update가 없으면
  완전 packing 가정을 명시하고 GBS × length를 사용합니다.
- `effective_wall`이면 idle/checkpoint/eval이 이미 측정에 포함됩니다.
  utilization은 1, 별도 checkpoint/eval 시간은 0이어야 합니다. 시작 setup은
  측정 rate에서 제외하고 `remaining_setup_seconds`에 한 번만 넣습니다.
- `core_compute`이면 checkpoint/eval과 기타 idle을 제외한 계산 측정입니다.
  기타 idle만 반영하는 `utilization`(0 초과 1 이하)을 명시합니다.
  `checkpoint_seconds`/`checkpoint_every_updates`,
  `evaluation_seconds`/`eval_every_updates`로 별도 비용을 한 번만 더합니다.
  주기로 나누어떨어지는 update(마지막 포함)에 이벤트가 발생합니다.
  남은 작업 기준으로 주기를 다시 세므로 실제 resume cursor와는 다릅니다.
  정확한 resume overhead는 effective-wall 측정 또는 조정된 입력을 사용하세요.

보수/기본/낙관 rate 배율 기본값은 0.8/1/1.2이며 신뢰구간이 아닌 민감도 가정입니다.
배율을 바꿀 때 기본값은 1, 보수 ≤ 기본 ≤ 낙관이어야 합니다.
마지막 update는 올림하며 추가 토큰 용량을 보고합니다.

`price_unit`은 `node` 또는 `gpu`입니다. 시간당 cluster 비용은 단가 × 노드 수,
GPU 단가일 때만 GPU/노드를 추가로 곱합니다. setup/idle/eval/checkpoint 동안도
전체 자원이 과금된다고 가정합니다. `elapsed_seconds`, `spent_cost`(없으면
경과 시간의 비용), `time_limit_seconds`, `budget_limit`으로 남은 제한을 계산합니다.
진행된 토큰은 각 단계의 completed_tokens로 별도 입력합니다. 실제 update 단위로
이진 탐색하여 주기 overhead까지 포함한 용량을 구하고, 앞 단계가 끝나지 않으면
뒤 단계로 건너뛰지 않습니다. 남은 시간/예산, 예상 총/잔여 시간·비용, 제한 충족,
양수/음수 여유분, 처리 가능한 토큰을 보고합니다. setup만으로 제한이 소진되어도
처리량은 0이며 음수가 되거나 성능을 임의로 올리지 않습니다.

선택적 `storage`는 `checkpoint_bytes`, `retain_checkpoints`,
`existing_checkpoints`(기본 1), `eval_artifact_bytes`,
`retain_eval_artifacts`(기본 1)를 사용합니다. 보관량과 쓰는 중 checkpoint 하나를
더한 peak 상한을 계산합니다. 사용자가 지정한 고정 크기 추정이며 모델/optimizer/
resume 상태 크기를 직접 포함해야 합니다. 데이터셋·캐시·복제본·기존 eval 파일,
실제 디스크 여유 공간, object store 요금은 계산하지 않습니다.

**측정값이 없으면 알 수 없다고 보고합니다.** `training_unmeasured.json`의
B300은 이름일 뿐입니다. 세 시나리오 모두 `measurement_required`, 시간·비용·
처리량 null입니다. `training_measured_example.json`은 산술 예시 fixture이며
실제 GPU benchmark가 아닙니다. 실제 의사결정 전 rate와 출처를 교체하세요.

## 실제 검증 범위

캐시의 pinned **Super** 토크나이저로 CLI를 실제 실행했습니다. 전체 선언 모집단은
CC0 수작업 fixture 3개 문서이며 seed 1729로 2개를 reservoir 추출했습니다.
표본은 2·3번 행이며 text 45토큰 + EOS 2토큰, 원문 UTF-8 136바이트입니다.
세 문서의 조건부 추정은 text 67.5 + EOS 3 = **70.5토큰**, 원문 204바이트였습니다.
ClimbMix 전체 크기나 GPU 성능 결과가 아닙니다. 테스트는 seed/균등 표본,
층화 분산/전수, 대표 표본 CI 거부, quota/잘못된 입력, 실제 로컬 Parquet,
시간·비용·overhead·용량, planner CLI를 검증합니다.
