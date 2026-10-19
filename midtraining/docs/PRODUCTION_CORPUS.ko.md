# 운영용 코퍼스의 제한적 확장

English / 검토 JSON 전체 스키마: [PRODUCTION_CORPUS.md](PRODUCTION_CORPUS.md).

## 실행 경계

`production_corpus.py`는 **로컬 파일만 읽는 CPU 전처리기**입니다. 다운로드,
서비스, GPU, 모델 가중치 로드, 학습, 원격 추적을 시작하지 않습니다.
기존 `prepare_corpus.py`의 실제 Parquet 스캐너, `Catalog`, 벤치마크 매처,
보존된 MCore indexed SDK, native writer와 최종 토큰 문서 감사를 재사용합니다.
기존 파일과 승인된 pilot 산출물, native 학습 입력은 수정하지 않습니다.
`--help`, 기본 호출, `--preview`는 출력 디렉터리를 만들거나 Transformers를
가져오지 않습니다. 검증한 코퍼스는 24시간 학습이나 품질 개선의 증거가 아닙니다.

저장소 루트 `/raid/yoon/nemochat`에서 안전하게 확인할 명령:

```bash
.venv/bin/python midtraining/data/production_corpus.py --help
.venv/bin/python midtraining/data/production_corpus.py --preview
TMPDIR=/raid/yoon/nemochat/datasets/midtraining .venv/bin/python -m pytest -q \
  tests/test_midtraining_production_corpus.py tests/test_midtraining_corpus.py \
  tests/test_midtraining_longrun.py
```

## 실제 calibration으로 정하는 용량

10M pilot의 고정 quota를 운영 학습 예산으로 사용하지 않습니다. 실제 부모
승인 영수증의 `status=calibration_save_reload_accepted`, `all_ranks=8`, pilot
manifest SHA256
`52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308`가
필수입니다. GB128 / sequence4096 / 비프로파일링 warm update9..16의 평균으로
기존 native helper가 다음 값을 계산합니다.

```text
train_iters = floor(86400 * 0.90 / mean_step_seconds)
samples = train_iters * 128
required_samples = ceil(ceil(samples * source_weight) * 1.005) + 2
required_tokens = required_samples * 4096 + 1
```

10%는 측정한 지속 실행 성능이 아니라 저장/평가 등의 **가정한 여유율**입니다.
가중치는 Code-Concepts / Unconditional-Algorithmic / Formal-Logic / ClimbMix
replay 순서로 .6/.1/.1/.2이며 변경하지 않습니다. 실제 native `capacity_gate`를
통과해야 합니다. 문서는 자르지 않고 마지막 전체 문서만큼 초과할 수 있습니다.
용량이 부족하면 선택한 고유 train 토큰/문서 수와 요구량을 실패 영수증에
남깁니다. 소스 대체, 반복, 부족한 quota의 다른 소스 전가는 없습니다.
고유성은 이번 선택 집합에서 정확히 같은 토큰 문서를 뜻하며 의미적 고유성이나
BASE 대비 새로운 지식은 아닙니다. ClimbMix는 이미 BASE가 소비한 replay입니다.
운영 학습 시작점은 ORIGINAL BASE이며 calibration 가중치가 아닙니다.
이 전처리기는 모델 체크포인트 자체를 선택하지 않습니다.

## 부모 검토와 원본 데이터 제한

**새 파일/의존성의 부모 검토 전에는 실행하지 마십시오.** 이전 원본 획득 승인은
최대1GB, seed 세 파일 합계759,378,296바이트까지였습니다. 이 구현은 전체
저장소 획득을 승인하지 않습니다. 추가 shard가 필요하면 부모가 적격 파일 목록,
공식 digest, 전체 크기 상한을 새로 검토하고 별도 획득 승인을 받아야 합니다.
다운로드는 부모가 별도로 수행하며 이 프로그램은 완료된 로컬 파일만 검증합니다.

이후 사용자가 Code-Concepts `part_000000.parquet`부터 `part_000009.parquet`까지
추가 열 shard, 합계5,272,987,566바이트를 명시적으로 승인했습니다. 기존 seed
세 파일을 포함한 총 원본은6,032,365,862바이트입니다. 다른 파일은 승인하지
않으며 획득은 여전히 부모 작업입니다.

현재 새 YAML의 상한은 원본 allowlist7GB, 전처리12시간, 출력 관측64GB,
replay 스캔2B토큰, 문서당1M토큰입니다. 운영 한도를 바꾸려면 **새 YAML만**
수정하고 부모 검토와 소스 digest를 갱신하십시오. 구현의 절대 상한은24시간,
출력256GB, 원본64GB, replay100B토큰, 문서1M토큰이며 이 값들은 다운로드
승인이 아닙니다. 원본/출력/캐시는 `/raid/yoon/nemochat/datasets/midtraining`
아래로 제한됩니다. 출력은 새 경로여야 하며 실패한 generation을 재개하거나
덮어쓰지 않습니다. HF/XDG 캐시는 새 출력 아래를 사용하며 offline입니다.

부모 검토 JSON의 필수 내용:

- `approved=true`와 검토 완료 후의 `all_intended_benchmarks_registered=true`.
- 정확히 여섯 `files` 경로와 실제 SHA256:
  `midtraining/data/production_corpus.py`, `midtraining/configs/production_corpus.yaml`,
  `midtraining/data/prepare_corpus.py`, `midtraining/configs/corpus.yaml`,
  `midtraining/training/longrun_native.py`,
  `results/pretraining/nemochat_baby/climbmix/corpus_sdk.py`.
- 기존 전체 KMMLU manifest의 경로/SHA/count를 가진 `benchmark_manifests`.
- 공식 `hub-metadata.json` 경로와 SHA인 `metadata`.
- `raw_byte_cap`과 **명시적인** `raw_files` 목록. 각 행은 `key`, 공식 `path`,
  `local_path`, `size`, `sha256`을 포함합니다.
- `calibration`의 실제 승인 영수증 경로와 SHA256.

최상위 필수 필드는 `approved`, `all_intended_benchmarks_registered`, `files`,
`benchmark_manifests`, `metadata`, `raw_byte_cap`, `raw_files`, `calibration`입니다.
정확한 키 집합을 강제하는 것은 여섯 경로의 `files`이며 최상위의 과거 추가 필드를
거부하지는 않습니다. `seed_catalog_sha256`은 폐기된 필드입니다. 요구하거나
읽거나 검증하지 않으며 과거 값을 넣어도 증거가 되지 않습니다. 변경된 producer의
digest를 다시 계산한 뒤 부모 검토를 갱신해야 합니다.

영문 문서의 JSON은 고의로 승인되지 않은 불완전 예시입니다. 그대로 실행할 수
없습니다. **원래 seed 세 shard 모두**와 추가 승인 shard만 목록에 넣으십시오.
variant 전파를 위해 seed 세 파일의 전체 재스캔이 필수입니다. 공식 metadata의
저장소/revision, 파일 이름, 크기, LFS SHA와 로컬 실제 바이트/SHA가 모두 일치해야
합니다. 세 허용 디렉터리의 `part_XXXXXX.parquet`만 가능하며 glob, 자동 확장,
중복 경로, 누락 파일 다운로드는 없습니다.

저장소는 `nvidia/Nemotron-Pretraining-Specialized-v1.1`, revision은
`13fa979be2e7f7e62913eee0ec5e97c8fd6e24af`로 고정입니다. 원래 전체
KMMLU34,871문항과 ClimbMix84,990개 heldout 원문 hash를 기존 검증기로
검사합니다. 이는 등록된 기존 벤치마크 범위의 정규화된 리터럴 부분 문자열
배제이지, 모든 벤치마크나 의미적 오염 제거를 보장하지 않습니다. 벤치마크
인벤토리 확대는 별도 부모 작업이며 기존 YAML은 수정하지 않습니다.

## Heldout 보존과 실제 처리 경로

1. immutable seed manifest SHA와 성공 companion을 검사하고, 기존 reservation
   **전체**의 SHA/count를 확인합니다. 선택한406k validation 토큰만으로 축소하지
   않습니다. 과거 `catalog.sqlite`는 비권위적 캐시이며 운영 전처리기가 열거나
   hash를 계산하거나 복구하거나 복사하지 않습니다.
2. raw 행을 넣기 **전에** 모든 seed heldout content hash와 원래 ClimbMix
   heldout hash를 `Catalog.original_reserved_hashes`에 설정합니다. 허용 파일
   전체를 스캔한 후 원래 질문 component 정책으로 finalize합니다. 과거 heldout과
   연결된 답변 variant와 벤치마크 일치 component 전체를 train에서 제외합니다.
3. 새 `heldout-reservations.jsonl`에 과거 행을 모두 보존하고 새 결정적 heldout을
   추가합니다. 새로 전체 스캔한 `Catalog`에서 **모든** 과거 reservation의
   `content_hash`로 원문을 찾고 UTF-8 원문의 정확한 SHA256을 재계산한 뒤
   validation quota 밖의 과거 heldout까지 토큰 예약합니다. 누락이나 hash 불일치는
   train 선택 전에 즉시 실패하며 답변 variant로 원문을 대체하지 않습니다.
   증거는 공식 raw 크기/SHA256, seed shard 전체 필수 재스캔, 모든 과거 hash의
   정확한 원문 존재 확인이며 과거 캐시의 가독성이 아닙니다.
   새 heldout 전체와 원래 native ClimbMix validation
   전체도 train 선택 전에 토큰 예약합니다.
4. pilot의 네 validation `.bin/.idx` 쌍을 새 출력에 복사하고 digest와 native
   readback을 확인합니다. 이는 **동결된 heldout 풀의 명시적 재사용**이며 train
   반복이 아닙니다. 과거 heldout 때문에 제외된 component에서 새 validation
   quota를 억지로 채우지 않습니다.
5. specialized train을 고정 quota까지 선택한 뒤 원래 ClimbMix train을 선택하며
   알려진 clipped 마지막 항목은 제외합니다. tokenizer decode/encode가 토큰 ID
   단위로 정확히 왕복해야 합니다. 전체 예약 토큰 문서와 선택된 토큰 문서의
   중복을 소스 간에도 건너뛰고 quota를 채웁니다. replay의 benchmark/과거 원문
   heldout hash도 재검사합니다.
6. 모든 native 출력의 문서/토큰 합계, EOS11, vocabulary[0,131072), train/heldout
   교집합과 소스 간 정확 토큰 중복을 재검사합니다. **여덟 .bin 전체**에서 ID18..29를
   새로 스트리밍 검사하고 하나라도 나오면 성공을 막습니다. 실제 .bin/.idx
   SHA와 길이도 검증합니다.

Tokenizer는 기존 YAML의 정확한 Base 자산 digest를 사용합니다.
`local_files_only=True`, `trust_remote_code=False`이며 다운로드한 Python을
실행하지 않습니다. 부모 승인 pinned CPU 이미지의 Transformers5.12.1이 필수이고
호스트 업그레이드는 이 작업 범위가 아닙니다. 프로세스 자체는 이미지 신원을
증명하지 못하므로 부모가 정확한 Engine 이미지/exit 증거와 CPU/RAM/네트워크
제한을 보존해야 합니다.

## 명령과 영수증

실제 calibration 승인 후, 측정된 quota를 안전하게 확인할 수 있습니다.

```bash
.venv/bin/python midtraining/data/production_corpus.py --preview \
  --calibration /ABS/CALIBRATION/controller/accepted.json
```

부모 검토/획득을 완료한 뒤 **부모가 승인한 CPU 컨테이너 내부에서**, 같은 절대
저장소 mount를 유지하고 GPU 및 네트워크 없이 실행합니다.

```bash
python midtraining/data/production_corpus.py --execute \
  --reviewed-sources /ABS/PARENT-PRODUCTION-REVIEW.json \
  --calibration /ABS/CALIBRATION/controller/accepted.json \
  --output datasets/midtraining/super35-kd-01/production-24h-v1
```

이 모듈은 컨테이너를 시작하지 않습니다. 실패 시 다른 새 출력 generation을
선택하십시오. SIGALRM이 blocking 작업의 시간도 제한하며 SIGTERM/SIGINT는
Python이 처리할 수 있으면 실패 영수증을 남깁니다. 일반 progress 호출의 출력
트리 관측/영수증 저장은 5초마다 한 번 이하이며 최초와 성공 직전에는 강제
검사합니다. 이는 **관측 상한**이지 OS의 hard disk quota가 아닙니다.
상한 실패 후에는 마지막 관측 이후의 쓰기(최대 5초 검사 간격과 blocking
batch/문서 처리량) 및 조사용 영수증만큼 남을 수 있으므로
부모가 외부 CPU/RAM/cgroup/disk 한도를 적용하십시오. OOM/강제 종료 시 progress만
남을 수 있으며 성공으로 간주하지 않습니다.

출력은 전체 catalog와 reservation, 네 train/네 동결 validation native 쌍,
manifest 및 timestamp/PID/단계/실제 진행량을 담은 progress/failure/success
영수증입니다. 일부 native 파일의 존재만으로 완료를 판단하지 마십시오.

manifest는 기존 native schema1/소스 순서/가중치를 유지하고 측정된 운영 예산,
실제 capacity gate, 새 reserved-ID 감사와 validation 재사용을 기록합니다.
`success.json`의 phase`prepared_verified`와 실제 manifest SHA가 필수이며
`longrun_native.load_prepared_manifest`로 최종 readback합니다. 실제 native
sample-index 무반복 구성,24시간 학습, 품질 개선은 별도 부모 검증이며 확대된
manifest의 calibration compatibility는 [LONGRUN.md](LONGRUN.md)를 따릅니다.

테스트는 **수작업 Parquet/가역 fixture tokenizer**를 명시하고 실제 보존 native
SDK를 실행합니다. 과거 캐시가 없거나 손상되어도 raw 복원이 가능한지, 과거 원문
누락 및 재계산 SHA 불일치 시 실패하는지, 전체 문서 readback, 과거 reservation/variant 전파, 새 heldout,
validation digest 재사용, quota/capacity/중복/실패/시간 제한을 검사하지만 실제
모델 tokenizer 실행이나 운영 데이터의 충분한 용량, GPU 성능 증거는 아닙니다.
전체 생산 전처리와 희소 소스의 충분한 train 용량은 부모 실행 후에만 확정됩니다.
