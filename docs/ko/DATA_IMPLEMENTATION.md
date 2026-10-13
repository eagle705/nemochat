# CPU 우선 ClimbMix 데이터 구현

새 CLI 실행은 저장소 루트 기준 경로와 새 산출물 이름을 사용합니다. [런타임 디렉터리 규약](RUNTIME_LAYOUT.md)에 따라 입력 데이터는 `datasets/`, checkpoint는 `checkpoints/`, 보고서·trace·plot은 `results/`, 텍스트/W&B 로그는 `logs/`에 둡니다. 필요한 부모 폴더만 만들며 기존 fixture/config와 과거 증거 경로는 보존합니다.

## 구현 범위와 권리

`src/nemochat/data/climbmix.py`와 `scripts/prepare_data.py`는 명시적으로 선택한
Parquet shard 또는 로컬 JSONL을 CPU에서 제한된 quota만큼 토큰화한다.
`transformers`, `requests`, `pyarrow`는 필요할 때만 import한다. 전체 snapshot
다운로드, corpus shuffle, 업로드, GPU 학습은 수행하지 않는다.

- 재포장: `karpathy/climbmix-400b-shuffle`, revision
  `915333b4f8b8684f39aeaafea600fea6f43fb703`.
- 원본: `nvidia/Nemotron-ClimbMix`, revision
  `5eaa64b9c0c85b7f56af01d7dffdb0795816b12b`.
- **NVIDIA 출처 및 CC BY-NC 4.0 조건을 유지**한다. 재포장 card의 MIT 표기는
  원본 NC 조건을 없애지 않는다. 본 프로젝트는 비상업적 교육용이다.
  데이터 재배포와 학습 weight 권리는 별도 검토해야 한다.

ClimbMix config의 `source`는 코드의 `SOURCE`와 정확히 일치해야 하며,
`rights_acknowledged: true`가 필요하다. 이 값은 사용자 확인 기록이지 법적 권리
검증이 아니다. 로컬 입력도 출처와 `CC0-1.0` 또는 `CC-BY-NC-4.0`를 명시한다.
`src/nemochat/data/fixtures/smoke.jsonl`은 직접 작성한 CC0-1.0 fixture이다.

## 실행과 토크나이저

```bash
.venv/bin/python -m pytest tests/test_data.py -q
python3 tests/test_data.py  # 외부 패키지 없이 fixture/control test
.venv/bin/python scripts/prepare_data.py \
  --config configs/data/climbmix_smoke.json \
  --output datasets/climbmix/my-smoke-fresh
```

제공된 config는 `shard_00000.parquet` 하나, `max_shards: 1`, quota 256,
다운로드 제한 100,000,000 bytes를 지정한다. 256 tokens만 필요해도 선택한
Parquet 전체 약 92 MB를 다운로드한다. HTTP는 1 MiB block 단위 streaming,
connect/read timeout, Content-Length 및 실제 byte budget 검사를 사용한다.
한 번에 한 shard만 임시 보관하고 64행 batch로 읽는다.

`token_quota`, `max_shards`는 명시적 양의 정수여야 한다.
`00000`~`06542` 범위의 알려진 파일명만 허용한다. train에는
`shard_06542.parquet`를 금지하고 validation은 이 마지막 shard만 허용한다.
선택한 순서 그대로 읽으므로 corpus 대표성이나 무작위 표본을 주장하지 않는다.

기본 tokenizer:
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16`, revision
`2dc98e2afe4face0e4ce40972a915c45368bd34a`.
HF AutoTokenizer는 `trust_remote_code=False`, 모델 weight 없이 로드한다.
GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`는 `purpose: smoke`만
허용하고 training에는 금지한다. `add_special_tokens=False` 후 EOS 하나를
붙인다. revision, EOS ID, 가능한 경우 fast backend JSON SHA256을 저장한다.
이는 모든 원본 tokenizer 파일 hash를 의미하지 않는다. 테스트 tokenizer 주입은
manifest에 별도 표시된다.

로컬 JSONL config 예시(기본 pinned Super 사용):

```json
{
  "mode": "local_jsonl",
  "files": ["src/nemochat/data/fixtures/smoke.jsonl"],
  "local_license": "CC0-1.0",
  "local_attribution": "NemoChat self-authored smoke fixture",
  "rights_acknowledged": true,
  "split": "train",
  "purpose": "smoke",
  "token_quota": 64,
  "max_shards": 1,
  "quota_policy": "prefix",
  "reserved_hashes": {"train": [], "test": [], "validation": []},
  "reserved_manifests": []
}
```

경로는 config 위치가 아니라 **실행 working directory** 기준이다. 각 JSONL
행은 문자열 `text`를 가진 object여야 한다. 잘못된 JSON/문자열 아닌 text는
실패하고 빈 문서·공백 문서는 제외한다. 로컬 파일은 기본 1 GiB/file
`max_input_bytes`, 문서/JSONL 행은 기본 16 MiB `max_document_bytes` 제한이다.
전체 선택 파일 hash는 계산하지만 quota로 중단된 뒤의 행은 검증하지 않는다.

## Quota, 중복 및 split 의미

- `accepted_tokens`는 text tokens + EOS tokens이다. quota를 초과하지 않는다.
- 기본 `prefix`는 남은 quota에 맞춰 token prefix + EOS를 저장한다.
  원본 전체 text hash, 원본 token 수, 실제 수, `prefix_truncated`를 기록한다.
  prefix를 완전한 원문 문서라고 표현하지 않는다. UTF-8 byte token 중간에서
  끝날 수 있으므로 출력은 decoded text가 아니라 token pack이다.
- `complete`는 들어가지 않는 문서를 건너뛴다. 두 방식 모두 underfill 가능하며
  text + EOS를 위해 최소 두 자리가 필요하다. `quota_reached`로 완료 여부를
  보고한다. accepted 문서가 0개면 빈 artifact 대신 오류이다.
- `tokenized_text_tokens`는 필터/중복 제외 후 방문 문서의 quota 적용 전 token
  수이며 accepted나 원본의 광고상 token 수와 다르다.
- `consumed_training_tokens`는 **0**이다. 후속 trainer가 반복·packing·loss mask
  등을 반영한 실제 consumed ledger를 별도로 기록해야 한다.
- `reserved_hashes`에는 train/test/validation 세 목록을 반드시 명시한다.
  다른 split의 exact hash를 제외한다. `reserved_manifests`로 이전 artifact의
  `record_hashes`를 불러올 수 있다. heldout을 먼저 준비한 뒤 train에 manifest를
  전달하거나 모든 split hash를 미리 동결해야 한다. 빈 목록은 미예약 smoke이지
  누수 방지 완료가 아니다.
- SHA256 중복 제거는 방문한 선택 행과 명시된 예약 hash에만 적용한다.
  corpus 전체 dedup, near-duplicate 제거, benchmark contamination 확인 또는
  동시 실행 전역 split registry는 구현하지 않았다. raw UTF-8 기준이며
  whitespace/Unicode 정규화하지 않는다.

## 출력과 실패 처리

새 directory에만 다음 세 파일을 atomic rename으로 공개한다.

1. `tokens.u32`: little-endian uint32 flat token IDs, 문서마다 text + EOS.
   **Megatron indexed `.bin/.idx`가 아니다.** MCore에는 검증된 adapter가 필요하다.
2. `records.json`: source shard/file, 0부터 시작하는 행 번호, 원문 SHA256/byte 수,
   token-element offset/count, prefix 여부, record token bytes SHA256.
   byte offset은 token offset * 4이다.
3. `manifest.json`: config, 출처/권리, tokenizer, 입력 SHA256, 각 counter,
   record hash 목록, token 파일 및 records 파일 SHA256.

재포장 text는 원본 URL/문서 경계를 잃었을 수 있다. shard/row/hash provenance는
원본 NVIDIA 문서 lineage 복구가 아니다. 원문 text는 artifact에 포함하지 않는다.

sibling exclusive lock과 동일 filesystem staging을 사용한다. 기존 output 및
symlink는 덮어쓰지 않는다. 일반적인 입력/HTTP/tokenizer/Parquet 오류는 임시
파일과 cache를 정리하고 오류를 보고한다. 합성 결과로 대체하지 않는다.
SIGKILL/전원 장애는 stale lock/staging을 남길 수 있으므로 실행 process가 없는지
확인 후 수동 정리한다. fsync 기반 전원 장애 내구성이나 lock을 무시하는 외부
writer에 대한 보장은 아니다. HTTP 자동 retry는 하지 않는다.

메모리는 방문 hash 집합과 accepted record metadata에 따라 증가한다. 문서 하나를
한 번에 토큰화하므로 bounded CPU pilot용이며 billion-token out-of-core framework가
아니다. disk/cache budget, scalable ledger, near-duplicate/오염 감사, 검증된 MCore
adapter 및 실제 학습 consumed ledger는 후속 작업이다.

## 실제 검증

fixture/control suite: 15 tests와 16 subtests 통과. stdlib unittest runner도 15 tests 통과. 실제 선택 shard GPT-2 CPU
smoke는 2행에서 원문 323 tokens를 토큰화하고 text 254 + EOS 2 = 256 tokens를
수용했다. record 하나를 prefix로 잘랐고 학습 consumed는 0이다. binary 1,024 bytes의
SHA256은 manifest와 일치했다. 입력은 92,291,918 bytes, SHA256
`054ddbd98abf30d773c54de578fc9d579bafeb6c14e04e97bd36aa90e825bf9b`이다.
pinned Super도 remote code 없이 로드되었고 EOS는 11이었다. 실제 Super 선택 shard
pilot은 원문 317 tokens를 토큰화해 text 254 + EOS 2 = 256을 수용했다. 출력 SHA256은
`ecde7f64fa1a3b32ee925f2a084c587e814cd48e0a8ee58ee87410da614027f7`이다.
실제 Super 로컬 영어/한글 fixture는 중복 하나·빈 문서 하나를 제외하고 text 34 +
EOS 3 = 37 tokens를 수용해 64 quota의 underfill을 보고했다.
이는 데이터 준비/토크나이저 검증이며 GPU 학습·평가·전체 corpus 취득
결과가 아니다. 자세한 계약은 영문 `docs/en/DATA_IMPLEMENTATION.md`와 동일하다.
