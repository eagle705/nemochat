# Offline2B 고유 코퍼스 용량 게이트

English: [CORPUS.md](CORPUS.md).

## 바뀌지 않는 학습 예산

이번 요청은 과거 3,200 update 온라인 KD의 재개가 아니라 **새 offline KD**다.
학습 시작점은 원래 `checkpoints/exports/nemochat_baby_base_hf_bf16_02`이며
smoke overfit 모델이 아니다. 이 생산기는 모델 weight를 열거나 해시하지 않는다.
사용자의 weight 무결성 검사 면제는 tokenizer/data byte 검사에는 적용되지 않는다.

Sequence4096 × Global Batch128 기준 **3,815 update / 2,000,158,720 input positions**다.
이 수치는 고유 준비 토큰이나 loss target 수와 다르다. 순서와 비율은
Code Concepts / Algorithmic / Formal Logic / ClimbMix Replay = **67/10/3/20**으로 고정한다.
Native의 0.5% oversampling, 올림, 2 sample 여유, shifted 추가 토큰을 포함한
source별 최소 준비량은 다음과 같다.

| Source | 필요한 고유 준비 토큰 |
|---|---:|
| Code Concepts | 1,346,818,049 |
| Algorithmic | 201,027,585 |
| Formal Logic | 60,317,697 |
| ClimbMix Replay | 402,042,881 |

반복/cycling, 묵시적 epoch 추가, budget 축소, 비율 변경, source 대체,
새 다운로드, 문서 자르기는 허용하지 않는다. 고유성은 이번 generation 내부의
정확한 token-document hash 기준이며 semantic dedup 또는 BASE 대비 새로운
지식이라는 뜻이 아니다. ClimbMix는 이미 BASE가 소비한 Replay다.

## 실제 과거 실패와 새 Replay 처리

보존된 recovery는 504.4558초 뒤
`ValueError('replay pinned tokenizer roundtrip mismatch')`로 실패했다.
직전 선택량은 Replay 17,435,561 tokens / 27,791 docs다.
실패 JSON SHA는 `8f5de4d23ecd7de41b842fc7aee4a12f0e28d609b5edd368ed20c34d83cfd041`이다.
실패한 전체 manifest를 성공 데이터로 승격하지 않는다.

새 생산기는 pinned tokenizer로 native 문서의 decode/encode를 확인한다.
불일치 문서는 **문서 전체를 제외**하고 명시적인 counter와 제한된 진단 사례를
남긴다. 텍스트 정규화, token ID 수정, 재토큰화로 억지 통과, 검증 생략을 하지 않는다.
범위/EOS 오류나 reserved IDs18..29는 제외 가능한 이상치가 아니라 즉시 실패다.
Roundtrip을 통과한 문서에도 benchmark/full-heldout/중복 검사를 적용한다.
원래 잘린 마지막 train 문서는 제외하고, validation의 잘린 끝은 계속 예약한다.
불일치의 구체적인 원인은 실제 사례의 replacement-character 수와 mismatch
위치로 확인하며, 예외 메시지만 보고 단정하지 않는다.

## 데이터 계보와 보존 범위

Legacy source는 읽기만 한다. 원래 production의 동결된 6개 source와 recovery의
동결된 8개 source, 현재 helper digest를 모두 확인한다. 공식 repository/revision에
고정된 기존 13 raw 파일 6,032,365,862 bytes와 tokenizer asset, benchmark 목록,
과거 전체 scan 계보를 다시 인증한다. 네트워크는 사용하지 않는다.

정상적인 전체 production catalog만 `mode=ro&immutable=1`로 연다.
SHA는 `9d55e9cde8dfef7f61ad8d6b97271563f1245c4a34ce24574f36b6abb36e7eb1`이다.
망가진 pilot SQLite는 읽기/해시/수리/대체하지 않는다.

실패 recovery에서 완성된 native part는 bin/idx SHA, 길이, native reader의
문서/토큰/dtype/document-index를 새로 검사한다. 모든 재사용 문서에
범위/EOS/reserved-ID, heldout/교차-source 중복 검사를 적용한다.
새 SHA 측정값을 과거에 이미 검증된 digest인 것처럼 꾸미지 않는다.
각 source의 zero-skip 과거 selector와 시작/끝/내부 7개 문서 재토큰화 비교로
first-N cursor를 확인한 뒤 나머지 eligible 행 전체를 한 번씩 토큰화한다.

전체 old+new heldout JSONL, 원래 ClimbMix raw heldout 84,990개, 원래 native
Replay validation 전체, 고정된 네 source validation bin/idx를 보존한다.
Validation byte SHA는 pilot seed와 일치해야 한다.

## 실행 및 인계

Repository root에서:

```bash
.venv/bin/python -m pytest -q tests/test_offline2b_corpus.py
.venv/bin/python midtraining/data/offline_kd/prepare.py --preview
python midtraining/data/offline_kd/prepare.py --census \
  --reviewed-sources results/midtraining/offline2b-corpus-01/reviewed-sources-01.json \
  --output results/midtraining/offline2b-corpus-01/census-FRESH
```

실제 census는 이미 로컬에 있는 pinned image, Transformers5.12.1,
UID1001:GID985, `runc`, no-network/no-device/no-GPU로 실행한다.
Repository는 read-only이며 이 worker의 results/scratch만 read-write다.
Engine receipt에 image ID, mount, source review, 64GiB memory/swap,
12 CPU와 container ID를 기록한다. 2시간 watchdog과 Replay scan 2B-token 상한이 있다.

Specialized는 전체 eligible source의 실측치다. Replay는 필요한 토큰에 도달하면
멈추므로 **충분한 용량의 하한**이며 10B 전체의 고유성 검사라고 주장하지 않는다.
Source 하나가 끝날 때마다 `capacity.json`에 결과를 게시해 부족분을 일찍 알린다.
`census-completed.json`은 측정 완료이지 학습 코퍼스 준비 완료가 아니다.
용량 확인 전에는 dataset을 다시 쓰거나 학습 manifest/success를 발행하지 않는다.

`results/midtraining/offline2b-corpus-01/data-contract.json`에 manifest schema1과
native `GPTDatasetConfig` 계약이 있다. 학습 release는 모든 source 최소량,
네 train/네 validation prefix와 파일 digest, 전체 heldout digest, tokenizer 계보,
reserved-ID 0, train-heldout 겹침 0, source간 중복 0, companion success의
manifest SHA 일치, 정확한 CPU Engine 정상 종료를 요구한다.
Native sample-index no-repeat, teacher cache, GPU release는 부모 worker의 별도 게이트다.

첫 번째 실측에서 Code 전체 **2,697,481 docs / 1,293,070,646 tokens**를 확인했다.
고정 native reserve 대비 **53,747,403 eligible Code tokens 부족**이다.
평균 길이 추정이 아니라 전체 source 실측이므로 추가 승인된 Code 데이터 없이는
요청한 2B 학습을 시작할 수 없다.

첫 census는 이어 Algorithmic tail의 reserved IDs18..29를 발견해 정상적으로
Engine exit1로 중단했다(OOM 아님). 이 실패와 source snapshot을 보존했다.
별도의 진단 generation은 해당 문서를 전부 제외하고 occurrence/문서 hash를 기록하며
나머지 source 측정을 계속하지만, **source compatibility gate 자체는 명시적 검토 전까지
blocked로 유지**한다. 진단을 계속하는 것이 alias 변경이나 해당 문서의 학습 허용은 아니다.
기본 선택 경로는 여전히 reserved ID를 만나면 즉시 실패한다.

전체 Specialized 실측 결과:

| Source | Eligible 고유 토큰 | 문서 | 부족량 |
|---|---:|---:|---:|
| Code Concepts | 1,293,070,646 | 2,697,481 | 53,747,403 |
| Algorithmic (forbidden-ID 문서 제외) | 193,495,508 | 179,436 | 7,532,077 |
| Formal Logic | 60,564,276 | 408,328 | 0 |
| ClimbMix Replay (실측 하한) | 402,043,247 | 628,502 | 0 |

최종 CPU census는 정확한 Engine exit0/OOM 없음으로 완료됐다.
Container ID는 `421cb3930a963cfb39279cd7e4148fdd4d2df9a68cff3ac54f183822ab5c5de2`다.
Replay는 628,536 docs / 402,088,071 tokens를 검사하고 roundtrip 불일치 34개 문서를
통째로 제외했다. 과거 실패 지점 native doc27791도 재현했다.
첫 mismatch 위치15, 원래 non-EOS 1,067 tokens → 재인코딩 1,054 tokens이며
Unicode replacement character는 0개다. 이 사례는 손실성 replacement character로
설명되지 않는다. Native ID와 현재 pinned encoder의 canonical ID가 다르며,
이를 맞추려고 원래 ID를 고치지 않았다.

모든 적격 후보에서 IDs18..29=0, 전체 heldout token-document 겹침=0,
source간 token-document 중복=0이다. 전체32,397개 old+new 예약,
원래 ClimbMix raw heldout84,990개, 고정 validation bytes와 tokenizer 계보를 연결했다.
새로 측정한 native bin/idx SHA도 결과에 들어 있다.

최종 capacity report SHA:
`467608152ca9288b561d5111d9f39d33cd26d546ea05d876024755e13f9593db`.
결론은 **blocked_insufficient_unique_source_capacity**다.
이번 worker는 `offline2b-v1` 학습 manifest/success나 teacher cache를 만들지 않았고
production 학습을 release하지 않았다.

메타데이터 후속 수정으로 원래 Replay source의 `upstream_last_document_truncated:true`,
`truncated_documents:1`을 명시했다. 보존된 census02 source receipt의 adapter counter0을
upstream이 잘리지 않았다는 뜻으로 해석하면 안 된다. 실제 선택에서는 이미 마지막 문서를
제외했다. `replay-source-readback-03.json`은 같은 pinned CPU image에서 실제 원본
40,001,077,252-byte bin과 idx를 다시 해시하고 census02의 문서/토큰/SHA와 일치함을
확인한 exit0 증거다. 이전 source snapshot/측정 결과를 덮어쓰지 않았다.
현재 source는 이 계보를 명확히 보존한다. 관련 corpus/recovery/production/tokenizer
집중 테스트 **62개가 통과**했다.

제외된 Algorithmic 7개 문서는 총 4,735 tokens이며, ID19(`<img>`) 10회와
ID18(`<image>`) 1회가 검출됐다. 이 문서들을 학습에 허용하지 않는다.
Pinned 공식 metadata에서 Algorithmic Parquet는 **이미 전부 조사한 1개뿐**이다.
따라서 Code part 하나만 더 받아도 현재 비율/예산을 만족시킬 수 없다.
Algorithmic에도 별도 승인된 source/revision에서 최소 7,532,077 eligible 고유 토큰을
추가하거나, 별도 승인을 받아 recipe를 바꿔야 한다. 어느 쪽도 임의 실행하지 않았다.

실측 결과와 종료 상태는 `results/midtraining/offline2b-corpus-01/`에서 확인한다.
부족하면 필요한 **추가 eligible 데이터 용량을 명시적으로 보고**하고 승인을 받아야 한다.
예산을 줄이거나 반복시켜 2B 학습을 가능하다고 표시하지 않는다.
