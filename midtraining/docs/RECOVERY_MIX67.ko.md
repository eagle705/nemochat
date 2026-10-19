# 67/10/3/20 고유 데이터 빠른 복구

정식 계약 및 전체 JSON 설명: [RECOVERY_MIX67.md](RECOVERY_MIX67.md).

## 범위

`recover_corpus.py`는 기본 실행과 `--preview`에서 쓰기 작업을 하지 않는
로컬 전용 **CPU corpus producer**이다. 서비스, 컨테이너, GPU, 다운로드,
원격 tracking 및 학습을 시작하지 않는다. student/teacher model-weight payload는
읽거나 hash하지 않으며 명시적인 checkpoint-byte 검사 면제를 기록한다.
Tokenizer asset과 데이터 바이트 검증은 면제하지 않는다.

저장소 `/raid/yoon/nemochat`에서:

```bash
.venv/bin/python midtraining/data/recover_corpus.py --help
.venv/bin/python midtraining/data/recover_corpus.py --preview
.venv/bin/python -m pytest -q tests/test_midtraining_recover_corpus.py
```

3,200 updates, global batch128, sequence4096이며
Code67% / Algorithmic10% / Formal3% / ClimbMix replay20%를 고정한다.
각 source의 sample 목표는 `ceil(ceil(3200*128*w)*1.005)+2`이고,
token 목표에는 shifted token 하나를 더한다.

| Source | 목표 tokens | 복사할 기존 tokens | 새 준비 |
|---|---:|---:|---|
| Code | 1,129,705,473 | 1,011,675,137 | 최소118,030,336 추가 |
| Algorithmic | 168,620,033 | 168,620,068 | 기존 published documents 전체 복사 |
| Formal | 50,593,793 | 0 | 정상인 새 catalog에서 encode |
| Replay | 337,231,873 | 0 | 원본 native ClimbMix와 tokenizer roundtrip 검사 |

문서 경계 초과분은 허용하지만 문서 잘라내기, cycling, 암묵적인 epoch,
source 대체, 부족한 quota 축소는 금지한다. 고유성은 이번 generation 내부의
**정확히 같은 token document** 기준이다. 의미적 중복 제거 또는 BASE 대비
새 데이터라는 주장이 아니다. ClimbMix는 이미 소비한 replay로 표시한다.
기존24.295초 calibration은 예전 mix의 측정이다. 새 mix benchmark 또는
24시간 지속 측정으로 표현하지 않는다.

## 1B Code를 다시 tokenize하지 않는 근거

원본은 `production-24h-v1/data/failure.json`의 **Formal quota 실패**이다.
전체 corpus 성공으로 승격하지 않는다. 실패 SHA:
`5bb7bc7c94d16cb67a2f9588db7fa9e6b7162135169dd3240a114a91dab599ac`.

정상인 새 production catalog만 `mode=ro&immutable=1`로 연다.
9,301,573,632 bytes이며 SHA는
`9d55e9cde8dfef7f61ad8d6b97271563f1245c4a34ce24574f36b6abb36e7eb1`이다.
SHA, integrity, 필수 tables를 검사하며 누락/손상/transaction sidecar는 즉시
실패한다. 자동 수리하거나 원본을 수정하지 않는다. **손상된 pilot SQLite는
읽지도 hash하지도 않는다.**

과거 `oversized_documents`, `reserved_token_documents_skipped`,
`duplicate_token_documents_skipped` 세 counter가 모두0이어야 한다.
없는 counter는0이다. 당시 고정 producer SHA:
`5bd5afc188402bd0908364b3c6d00731a48fa50bef5a27953d2c0813960adcb8`.
이 producer는 catalog id 순서의 eligible TRAIN 문서를 처음부터 선택했다.
따라서2,111,817번째 eligible Code TRAIN row가 재사용 prefix의 마지막 row다.
처음/끝/중간의 결정적인7개 표본만 다시 tokenize하여 실제 native document와
비교하고 UTF-8 content SHA도 확인한다. 나머지1B Code는 document 단위로
복사하고 새 native index를 만든다. 복사 문서 전체의 EOS/range/reserved IDs,
과거 selected membership, heldout 및 중복 검사까지 생략하는 것은 아니다.

과거 lineage는 `production-source-snapshot-01`의 **여섯 frozen files**로
검증한다. 현재 수정된 live `longrun_native.py`를 과거 증거로 사용하지 않는다.
현재 dependency SHA는 별도로 새 parent review에 고정한다.
`recovery-origin-parts-verified-01.json`의 Code/Algo SHA는 새로 측정한 값이지,
옛 producer가 원래 기록했던 authoritative SHA라고 꾸미지 않는다.
기존 committed `selected_tokens`는 Code+Algo2,229,682개이며 실패한 Formal
transaction은 rollback되었다.

원본13개 raw file의 전체 scan/grouping 결과를 상속한다. 복구 중 raw records를
다시 scan했다고 주장하지 않는다. 다만13개 공식 raw SHA/bytes와 Parquet
schema/footer/row 총계는 다시 확인한다. 승인된 raw pool은6,032,365,862 bytes다.

## 정확한 parent review 계약

영문 문서의 JSON 표기는 실행 불가능한 placeholder이며 실제 PIN object는
`{"path":"로컬 경로","sha256":"소문자 hex64자"}`다.
다음 top-level keys가 **정확히** 필요하다:

```text
schema_version (=1)
approved (=true는 parent 검토 후에만)
files
origin_failure
origin_catalog
origin_review
origin_parts
origin_snapshot
full_reservations
raw_verification
calibration
```

`expected_code_cursor`만 추가로 허용한다. 생략하면 cursor를 유도하고 표본으로
검증한다. 제공할 때에는 문서 수나 token offset이 아니라 **마지막 catalog id**다.
잘못된 값은 실패한다.

`files`는 다음8개 active 경로를 실제 현재 SHA에 매핑한다:

```text
midtraining/data/production_corpus.py
midtraining/configs/production_corpus.yaml
midtraining/data/prepare_corpus.py
midtraining/configs/corpus.yaml
midtraining/training/longrun_native.py
results/pretraining/nemochat_baby/climbmix/corpus_sdk.py
midtraining/data/recover_corpus.py
midtraining/configs/recovery_mix67.yaml
```

`origin_snapshot`은 앞의6개 dependency 이름을 각 frozen copy의 PIN에 매핑한다.
각 SHA는 `origin_review.files[name]`와 일치해야 한다. 위 경로의 active source와
frozen copy를 혼동하지 않는다. 다른 writer가 작업을 끝낸 후 review를 고정한다.
성공 직전에도 active source SHA를 다시 확인한다.

PIN 대상은 모두 실제 저장소 아래에 있다:

- `origin_failure`, `origin_catalog`, `full_reservations`:
  `datasets/midtraining/super35-kd-01/production-24h-v1/data/`의
  `failure.json`, `catalog.sqlite`, **전체** `heldout-reservations.jsonl`.
- `origin_review`, `origin_parts`, `raw_verification`:
  `results/midtraining/super35-kd-01/corpus/` 아래의
  `production-reviewed-sources-01.json`, `recovery-origin-parts-verified-01.json`,
  `production-raw-verified-01.json`.
- `origin_snapshot`: 같은 corpus 디렉터리의 `production-source-snapshot-01/`.
- `calibration`: 원본 review의 `calibration` object를 그대로 사용한다.
  `results/midtraining/longrun/super35-newdata-calibration-01/controller/accepted.json`,
  SHA `96e2a098793e6ac1ac98ba7f671a1062d58d0fba799d775c38c4a2ed0a984250`.

전체 seed manifest SHA는
`52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308`이다.
전체 seed reservation의 SHA/count와 새 전체 reservation 안의 포함 관계를
검사한다. 일부 validation window만 보존하지 않는다. 기존 benchmark inventory,
question-group 정책,34,871문항 KMMLU, 원본 ClimbMix heldout84,990개 raw hashes,
Base tokenizer asset과 원본 replay receipt 검증을 그대로 재사용한다.

## 실행과 결과물

**기존 승인된 production CPU runtime 내부**에서만 다음을 실행한다.
이 프로그램 자체는 runtime이나 서비스를 시작하지 않는다.

```bash
python midtraining/data/recover_corpus.py --execute \
  --reviewed-sources results/midtraining/super35-kd-01/corpus/RECOVERY-REVIEW.json \
  --output datasets/midtraining/super35-kd-01/recovery-67-v1/data
```

Image는
`nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf`다.
기존 로컬 dependency overlay의 Transformers5.12.1을 사용한다. host upgrade나
새 다운로드는 하지 않는다. parent가 no-network/no-GPU,64GiB CPU-memory cgroup,
image/Engine exit receipt를 보장한다. producer가 이를 독자적으로 증명했다고
기록하지 않는다.

제한은12h SIGALRM, 관측 output64,000,000,000 bytes, Replay scan2B tokens,
문서당1M tokens다. output 크기는 보통5초 간격과 시작/종료 직전에 확인하므로
OS-level hard disk quota가 아니다. blocking 작업 또는 OOM kill은 partials와
progress만 남길 수 있다. 실패 output은 자동으로 재개/삭제하지 않는다.

새 output에는4 train/4 frozen validation native pairs, `selection.sqlite`,
전체 reservation 복사본, `code-cursor-proof.json`, `manifest.json`,
`progress.json`, `success.json` 또는 `failure.json`이 생긴다. 원본 Formal partial,
기존 catalog, 옛 출력은 변경하지 않는다. Code 추가분은 기존 selected hash를
보수적으로 제외한다. Formal 자신의 불완전한 과거 subset은 새 output 중복으로
취급하지 않지만, **새 selected set은 모든 source가 공유**하여 상호 중복을 막는다.
Replay는 upstream clipped tail을 제외하고 roundtrip/benchmark/전체 heldout을 검사한다.

마지막으로8개 output 전체를 새로 검사한다: native 문서/token 수, EOS/range,
bin/idx bytes와 SHA, IDs18..29 모두0, 전체 train/heldout 겹침0,
source 간 token-document 중복0, 새 mix의 native capacity 충족.
Manifest schema1에 `inherited_full_scan_provenance`,
`recovery_did_rescan_raw:false`, 실패 원본을 정직하게 표시한 재사용 provenance,
frozen validation reuse, 완전한 Base tokenizer identity를 기록한다.
`success.phase=prepared_verified` 및 정확한 `result.manifest_sha256`가 필요하고,
성공 반환 전에 native consumer로 다시 읽는다. Native sample-index no-repeat와
실제 GPU 학습은 parent의 별도 gate다.

## 수행한 검증과 남은 작업

RED→GREEN으로 cursor/skip/corruption, 실제 native materialization, quota 부족,
정확한 review schema, watchdog, provenance 검증을 구현했다. Fixture tokenizer는
명시적인 수작업 가역 mapping이며 모델 tokenizer 또는 실제 규모의 증거가 아니다.
보존한 실제 indexed SDK를 이용했다. 관련6개 suite에서91개 test가 통과했고
compile 및 `git diff --check`가 통과했다. 실제 immutable failure와 parent receipt에
대해 frozen6개 history를 읽기 전용으로 검증했다. 이 작업에서는 전체 복구,
서비스, GPU를 시작하지 않았다. 실제 데이터의 최종 capacity는 parent 실행 후에만
확정할 수 있다.
