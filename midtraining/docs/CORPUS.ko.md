# 제한된 midtraining corpus 준비

English: [CORPUS.md](CORPUS.md). 실행 recipe: [corpus.yaml](../configs/corpus.yaml).

## 계약과 현재 검증 범위

이 작업의 **10M tokens는 준비용 pilot capacity**이지 장기 학습 예산이 아니다. `production_training_budget`는 의도적으로 `null`이다. 약 24시간 학습 의도가 있어도 이 작은 pack을 반복해서 채우지 않는다. 부모 프로세스가 실제 production batch의 비프로파일링 처리량을 측정한 뒤 별도의 데이터 용량/학습 horizon을 확정해야 한다.

준비 코드는 문서 절단, 자동 cycling, source 대체, GPU 학습 시작, checkpoint weight payload 읽기, 온라인 W&B, 업로드를 하지 않는다. 준비 성공은 native optimizer update나 품질 개선의 증거가 아니다.

`--help`, 인자 없는 실행, `--preview`는 디렉터리 생성·다운로드·Transformers import·tokenizer 로딩을 하지 않는다. `--execute`에는 부모의 source/benchmark 검토 receipt와 승인된 CPU image의 **Transformers 5.12.1**이 필요하다. 현재 호스트의 4.57.6을 업그레이드하지 않는다. 실제 corpus 규모의 다운로드/준비는 부모가 시작할 작업이며, 구현 검증에서는 명시적인 수작업 fixture를 사용했다.

## 고정 source와 실험 가중치

Dataset은 `nvidia/Nemotron-Pretraining-Specialized-v1.1`, revision은 **`13fa979be2e7f7e62913eee0ec5e97c8fd6e24af`**이다. 원본 metadata는 `results/midtraining/super35-kd-01/corpus/hub-metadata.json`에 있다. License metadata는 `cc-by-4.0`이며 각 행의 `license`도 정확히 이 값인 경우만 허용한다. 파생 artifact에 NVIDIA/source attribution과 pinned recipe를 보존한다. Metadata 확인만으로 모든 후속 사용의 법적 허가가 확립되는 것은 아니다.

| Source | 허용하는 유일한 remote 파일 | 정확한 bytes | Training token 목표 | Blend weight |
|---|---|---:|---:|---:|
| Code-Concepts | `Nemotron-Pretraining-Code-Concepts/part_000060.parquet` | 477,960,523 | 6,000,000 | 60% |
| Unconditional-Algorithmic | `Nemotron-Pretraining-Unconditional-Algorithmic/part_000000.parquet` | 208,343,534 | 1,000,000 | 10% |
| Formal-Logic | `Nemotron-Pretraining-Formal-Logic/part_000000.parquet` | 73,074,239 | 1,000,000 | 10% |
| ClimbMix replay | 기존 local native training prefix | 새 remote bytes 없음 | 2,000,000 | 20% |

세 Parquet 파일의 합계는 **759,378,296 bytes**이다. 표 순서대로 SHA256은 다음과 같다.

```text
b3e69c8bb02be54d12e6773a8fde5594af30cd357e539b93326f02cde2ba6559
706bab9781f6541cb9f40b8af61242e45c6e3dabc0d2f9b3171f08571da37ccb
b8b641309dec27e836db6a2003d3045c6ed060c8d6a23e3a93100e1c8a3a450b
```

**60/10/10/20은 이 실험의 설계이며 NVIDIA의 공식 sampling weight가 아니다.** 고정 sequence 길이에서의 token exposure 비율이며 raw row 비율이 아니다. 준비 단계는 각 quota를 넘기는 마지막 **완전한 문서**까지 선택하고 실제 overshoot를 보고한다. Token 수에는 EOS가 포함된다. 정확한 quota를 맞추기 위해 문서를 자르지 않는다. 1M-token 문서 안전 한도를 넘으면 문서 전체를 제외한다. Validation은 source별 별도의 100K-token 목표이며 역시 complete-document overshoot를 허용한다. Train/validation 용량이 부족하면 source 대체나 반복 없이 실패한다.

학습 대상은 `text`뿐이다. Code-Concepts의 `metadata.problem_prompt`는 identity용이며 `generated_solution`을 추가 학습 텍스트로 복제하지 않는다. Formal-Logic에도 여러 문항/정답, MCQ 형식의 텍스트가 들어 있다. 별도 Multiple-Choice config를 제외한다는 것은 모든 객관식 형식을 제외한다는 뜻이 아니다.

**제외:** Economics, 별도 Multiple-Choice subset은 contamination/license 별도 검토 전까지 포함하지 않는다. Agentic, Instruction-Following, BBQ는 SFT용으로 남기며 여기서는 받거나 섞지 않는다.

## Tokenizer와 native 저장 형식

Tokenizer는 `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`의 **`4ba52e8ffa79e9693936c1a819ac89f551ceeef6`** revision이다. 이미 받은 local assets는 `checkpoints/teachers/nemotron35-super-base-bf16-4ba52e8f`에 있다. YAML에 `tokenizer.json`, `tokenizer_config.json` SHA256을 고정했다. Local-only, `trust_remote_code=False`로 읽으며 EOS는 **11**, token ID 범위는 `[0, 131072)`이다. 새 문서는 `add_special_tokens=False`, `truncation=False`로 encode하고 EOS 하나를 붙인다. 전체 next-token LM+native KD용 입력이며 **assistant-only SFT mask가 아니다**.

출력은 native MCore int32 `.bin/.idx`이며 item 하나가 완전한 문서 하나다. 보존된 upstream reader/builder는 `results/pretraining/nemochat_baby/climbmix/corpus_sdk.py`이고 SHA256은 **`add90e4c4ac95df509d89b7e246002daf291eab775038755a01d5e13d5dd9613`**이다. Task-local loader는 기존 corpus 준비와 같은 방식으로 사용하지 않는 cloud/distributed import와 CPU Dataset base만 격리한다. Reader/builder 메서드는 변경하지 않는다. 설치된 SDK 수정이나 별도 training loop는 없다. 게시한 각 prefix는 native reader로 재확인하며 전체 token 범위/EOS, token/document 수와 output SHA256을 검사한다.

## ClimbMix replay 계보와 원래 split 보존

입력은 기존 prefix이다.

- `datasets/climbmix/nemochat_baby/native-02/train/corpus`
- `datasets/climbmix/nemochat_baby/native-02/validation/corpus`

계보는 `karpathy/climbmix-400b-shuffle@915333b4f8b8684f39aeaafea600fea6f43fb703`, upstream은 `nvidia/Nemotron-ClimbMix@5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`이다. BASE가 이미 사용한 source이므로 replay 2M tokens를 **새로운 고유 지식 획득으로 주장하지 않는다**.

기존 `datasets/climbmix/nemochat_baby/control/preparation-status-02.json`을 SHA256으로 고정하고 native 파일 길이도 확인한다. 이전 준비 작업에서 train/validation의 마지막 item은 절단되었다. 새 준비는 그 마지막 item들을 제외하고 완전한 문서라고 잘못 주장하지 않는다. 원본 파일은 수정하지 않는다.

기존 validation을 새 split으로 재분배하지 않는다. 현재 validation quota를 채운 뒤에도 **원래 validation 전체**의 token-document hash를 예약한다. Replay 후보는 target tokenizer로 decode→encode하여 token ID가 정확히 왕복하는지 확인한다. 불일치 시 조용히 재토큰화하지 않고 실패한다. 전체 기존 validation scan을 포함해 replay scan은 최대 20M tokens이며 부족하면 cycling 없이 실패한다.

## Grouped heldout과 benchmark 제외

**받은 세 bounded shard의 모든 행을 quota 선택 전에 끝까지 스캔**한다. 선택된 pool 내에서는 정확한 UTF-8 content hash로 전역 문서 중복을 제거한다.

1. **Code-Concepts:** 정규화한 `metadata.problem_prompt`를 사용한다. UUID, 답안, 함수명 하나, 행 번호를 problem identity로 대체하지 않는다. Prompt가 없으면 schema rejection으로 센다.
2. **Formal-Logic:** 결합된 각 문항의 `A.`/`A)` 보기 이전 stem을 추출하고 `Answer:`를 identity에서 제거한다. 하나라도 같은 정규화 stem을 공유하는 문서들은 connected component로 합친 뒤 split한다. 답안 변형과 다중 문항 연결이 train/validation 양쪽에 갈라지지 않는다. 알 수 없는 형식은 명시적 rejection이다. 거대 component 때문에 train/validation 용량이 부족해질 수 있으며, 이때 component를 임의 분리하지 않고 실패/검토한다.
3. **Unconditional-Algorithmic:** 확인한 metadata schema에는 source question ID가 없다. 정규화된 전체 텍스트 중복만 묶는다. 의미상 같은 문제·번역·paraphrase 분리까지 입증하는 것은 아니다.

정규화는 기존 SFT helper와 같은 NFC + whitespace 제거 + casefold이다. Component의 가장 작은 question hash와 고정 seed를 SHA256하여 group ID를 만든다. `int(group_id, 16) % 100 == 0`이면 heldout이며, 약 1%의 **component**를 예약하는 것이지 tokens의 정확한 1%는 아니다. 완전히 고정된 bounded pool에서 결정적이다. Pool을 확장하면 component가 달라질 수 있으므로 새 검토 generation이 필요하다.

`heldout-reservations.jsonl`은 validation quota 때문에 실제 출력하지 않은 문서를 포함한 **전체 결정적 예약**을 보존한다. `catalog.sqlite`에는 스캔 텍스트, dedup ownership, component ID, split이 남는다. 원래 ClimbMix의 **84,990개** raw content hash 예약도 전체 SHA/count로 확인한다. Specialized 데이터에 같은 문서가 있으면 해당 problem component 전체를 제외한다. 현재 validation window만 남기는 방식으로 예약을 축소하지 않는다.

KMMLU는 기존 **34,871문항 전체** manifest인 `results/pretraining/nemochat_baby/sft/data/raw/kmmlu-reserved-questions.json`을 반드시 사용한다. Revision은 `d61b3f19e552c576bf5960dd24289763edc36a88`, SHA256은 **`61a804e227bd8aedbb4f1818c2559391f122627398e6f80085cd58c2525b5f0b`**이다. 기존 question 정규화를 재사용하고 text 및 Code-Concepts problem prompt에 **정규화된 literal substring containment**를 적용한다. 긴 문서 전체와 질문의 단순 equality만 비교하지 않는다. 일치하면 component 전체를 제외하며 replay도 decoded text를 검사한다. Benchmark 정답을 학습 예제 생성에 사용하지 않는다.

**포괄적인 decontamination 증명이 아니다.** 등록된 literal 질문은 잡지만 의미상 paraphrase, 번역, 동등한 코드 문제, answer-only fragment, 등록하지 않은 benchmark는 보장하지 못한다. 부모는 예정된 모든 평가 benchmark를 조사하고 추가 `{"questions": [...]}` manifest를 `policy.additional_benchmark_manifests`에 등록한 뒤 정확한 SHA/count를 검토해야 한다. 그 뒤에만 `all_intended_benchmarks_registered=true`로 승인한다. KMMLU가 있다는 사실만으로 다른 CORE/evaluation task까지 포함했다고 주장할 수 없다. 미완료/미승인 review는 실행이 거부된다. 최종 native-token audit는 전체 예약과의 train overlap 및 source 간 train token-document 중복이 있으면 실패한다.

## 제한된 획득과 부모 승인

세 개의 pinned HTTPS path만 받는다. 정확한 bytes/SHA256을 확인해야 Parquet scan으로 넘어간다. 중단된 `.part`를 보존하며 재개에는 정확한 `206 Content-Range`가 필요하다. Server가 Range를 무시하면 잘못 append하거나 무제한 처음부터 받지 않고 거부한다. 실행당 source별 request 시도는 최대 3회다. 완성된 cache도 존재 여부만 믿지 않고 다시 검증한다.

공유 acquisition cache에는 single-writer lock과 영속 transfer ledger가 있다. Retry/restart를 합친 payload 예산은 **1,000,000,000 bytes**이며 TLS/HTTP header는 dataset payload 수치에 포함하지 않는다. 필요하면 이미 보존된 cache bytes도 보수적으로 차감한다. Digest가 틀린 partial은 자동 삭제/재다운로드하지 않고 검토용으로 남긴다. 준비 output은 반드시 새 경로여야 한다. 실패하거나 게시된 prefix를 재개/덮어쓰지 않는다. 검증된 raw cache는 재사용하되 새 output generation으로 준비를 다시 시작한다.

코드는 wall time 6시간, output/cache 합계 8,000,000,000 bytes, replay scan 20M tokens, repo `datasets/midtraining/` 쓰기 범위를 제한한다. 부모는 추가로 CPU/memory 제한(예: CPU 8개, RAM 12 GiB), GPU device request 없음, exact image pin을 강제해야 한다. Python 버전 검사만으로 실제 image identity가 증명되지 않으므로 receipt에는 image 자체를 이 프로세스가 attest하지 않았다고 기록한다. 부모의 Engine image/exit receipt를 따로 보존한다. 이 모듈은 container/service를 시작하지 않는다.

승인 runtime image:

```text
nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf
```

원래 receipt의 prefix identity를 유지하려면 repo를 `/raid/yoon/nemochat`라는 같은 절대 경로로 mount한다. Python, Transformers 5.12.1, NumPy, PyArrow, PyYAML이 필요하고 실행 버전을 기록한다. 호스트에 설치하거나 Hub 코드를 실행하지 않는다.

부모가 source 검토 후 작성하는 review JSON에는 다음이 필요하다.

- `approved: true`, `all_intended_benchmarks_registered: true`.
- 정확히 세 `files` key와 실제 SHA256: `midtraining/data/prepare_corpus.py`, `midtraining/configs/corpus.yaml`, `results/pretraining/nemochat_baby/climbmix/corpus_sdk.py`.
- 선언한 모든 benchmark manifest의 `path`, 실제 `sha256`, `count`. KMMLU 항목은 위 path/SHA와 `count: 34871`이어야 한다.

[영문 문서의 JSON template](CORPUS.md)은 의도적으로 `approved: false`이며 **승인 receipt가 아니다**. 검토 없이 값을 true로 바꾸지 않는다. 검토 후 code/YAML이 바뀌면 gate가 실패한다. 실행은 검토한 canonical `midtraining/configs/corpus.yaml`만 소비한다. Docs/tests는 runtime 실행 의존성이 아니다.

## 실행 명령과 receipt

Repo root에서 호스트 안전 명령:

```bash
.venv/bin/python midtraining/data/prepare_corpus.py --help
.venv/bin/python midtraining/data/prepare_corpus.py --preview
.venv/bin/python -m pytest -q tests/test_midtraining_corpus.py
```

**부모가 승인한 CPU container 안에서** 새 output을 대상으로 실행:

```bash
python midtraining/data/prepare_corpus.py --execute \
  --reviewed-sources results/midtraining/super35-kd-01/corpus/reviewed-sources.json \
  --raw-directory datasets/midtraining/specialized-v1.1-13fa979/raw --local-only \
  --output datasets/midtraining/super35-kd-01/pilot-10m-v1
```

`progress.json`에는 timestamp/PID, 실제 acquisition payload bytes, phase, 끝까지 스캔한 files/rows, exclusions/dedup, 현재 선택 token/document 수, elapsed time이 기록된다. 다운로드 bytes와 준비 tokens는 분모가 다르다. 실패는 `failure.json`에 남기며 최종 audit 성공 후에만 `success.json`을 쓴다. 일부 `.bin/.idx`가 있어도 success가 없으면 ready가 아니다. 전체 heldout 예약, SQLite catalog, `bridge-dataset-fragment.yaml`도 생성한다.

Fragment는 native Bridge `GPTDatasetConfig`/MCore blend 표현, 별도 train/validation prefix 목록, 명시적 weights, `seq_length=4096`, `eod_mask_loss=False`를 사용한다. 별도 trainer나 바로 실행할 수 있는 완전한 학습 recipe가 아니다. 정확한 pinned-image API construction과 native blend sample-index 검증은 부모의 통합 gate다. Receipt의 source별 `(prepared_tokens - 1) // 4096`은 보수적 fixed-sequence **capacity 상한**이며 승인된 update 횟수가 아니다. Native dataset도 요청 samples가 많으면 여러 epoch를 만들 수 있다. 부모는 실제 duration/sample 요구량을 확정하고 generated index가 이 pilot을 반복/cycle하지 않는지 확인해야 한다. 실제 native LM+KD update/save/reload는 별도의 training gate이며 10M에서 장기 horizon을 추론하지 않는다.

공유 parent acquisition은 `datasets/midtraining/specialized-v1.1-13fa979/raw`에서 재사용한다. `--raw-directory`와 `--cache`는 같은 인자다. `--local-only`이면 파일이 없을 때 중복 다운로드하지 않고 실패한다. HF cache/home은 전부 `datasets/midtraining/specialized-v1.1-13fa979/.hf-home` 아래에 둔다. 부모의 완료 seed receipt는 `results/midtraining/super35-kd-01/corpus/seed-verified.json`이다. 준비 과정에 필요한 정확한 raw-file gate는 계속 적용한다.

### Native consumer manifest 계약 (schema 1)

기본 경로: `datasets/midtraining/super35-kd-01/pilot-10m-v1/manifest.json`. 함께 생성된 `success.json`의 `phase == "prepared_verified"`와 `success["result"]["manifest_sha256"]`에 일치하는 실제 manifest SHA256을 반드시 확인한다. Manifest만 있고 success가 없으면 producer 완료로 승인하지 않는다.

- `schema_version: 1`, `status: "prepared_verified"`, `sequence_length: 4096`, `cycling: false`, `production_training_budget: null`.
- `source_order`: `["code_concepts", "algorithmic", "formal_logic", "climbmix_replay"]`.
- `weights`: source key별 `0.6/0.1/0.1/0.2`이며 raw row가 아니라 fixed-sequence token exposure 비율이다.
- `sources[source_key]["train" 또는 "validation"]`: 확장자 없는 절대 native `prefix`, `tokens`, `documents`, `dtype: "int32"`, `reader_verified`, `bin_bytes`, `idx_bytes`, `bin_sha256`, `idx_sha256`, `target_tokens`, `overshoot_tokens`, `truncated_documents: 0`.
- `tokenizer`: recipe의 repository/revision/local path/EOS/vocabulary/asset SHA256 객체.
- `capacity_by_source`: source별 보수적 full-sequence capacity. `audit`에는 실제 train/validation token/document 수와 overlap/duplicate 0 검증이 있다.
- `heldout_reservations_sha256`, `companion_success_receipt`, `native_training_gate`로 readiness 연결과 남은 native 통합 범위를 보존한다.

Global batch 128, sequence length 4096에서 16-update calibration은 별도 evaluation을 제외하고 **8,388,608 processed positions**를 사용한다. 20 updates는 10,485,760 positions를 요구하므로 이 10M pilot을 조용히 cycle해서 채우면 안 된다. Calibration만으로 24시간 corpus 예산이 확정되는 것은 아니며 실측 production-batch 처리량으로 추가 source 용량을 검토해야 한다.

## 검증 한계와 남은 gate

RED→GREEN 테스트는 inert CLI, pins/caps, 실제 local HTTP range resume/digest 검사, 전체 shard fixture scan, problem component grouping, 원래 heldout 예약 전파, benchmark containment, 완전한 문서 native writer/readback, quota 부족 실패, 기존 replay split/token 왕복 보존, quota 뒤의 전체 heldout 예약, progress counter, 최종 overlap audit를 검증한다. Parquet/native 통합 fixture는 수작업 텍스트와 명시적인 fixture tokenizer를 쓴다. 실제 corpus/GPU 검증으로 대체하지 않는다.

남은 일은 부모의 source/benchmark/license 검토, exact CPU image/tokenizer 실행, 759 MB 제한 실제 acquisition 및 전체 scan/tokenization, grouped heldout 용량 확인(특히 Formal-Logic), 최종 준비 receipt, native Bridge config/blend/index 검사다. 약 24시간 학습을 위한 데이터 확장과 production 용량은 실측 처리량에 맞춰 따로 확정한다. 이 구현 과정에서는 GPU job, 장기 다운로드, 온라인 tracking, 업로드, commit, installed SDK 수정, 원래 corpus 변경을 하지 않았다.
