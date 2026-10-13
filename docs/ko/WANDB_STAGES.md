# W&B 단계 추적: 구현 범위와 안전한 사용

새 CLI 실행은 저장소 루트 기준 경로와 새 산출물 이름을 사용합니다. [런타임 디렉터리 규약](RUNTIME_LAYOUT.md)에 따라 입력 데이터는 `datasets/`, checkpoint는 `checkpoints/`, 보고서·trace·plot은 `results/`, 텍스트/W&B 로그는 `logs/`에 둡니다. 필요한 부모 폴더만 만들며 기존 fixture/config와 과거 증거 경로는 보존합니다.

## 구현된 범위

W&B는 선택적 관측 도구이며 학습 백엔드가 아니다. 가상환경에서
`python -m pip install -e '.[tracking]'`으로 설치한다. 선언된 SDK 범위는
`wandb>=0.30,<0.31`이며 오프라인 검증 버전은 **0.30.0**이다.

| 단계 / 실행 유형 | 진입점 | 증거의 범위 |
|---|---|---|
| `data` / `stage` | `scripts/prepare_data.py` | 제한된 데이터 준비, 카운터, 데이터셋 메타데이터 |
| `pretrain` / `stage` | `scripts/train_cpu.py` | CPU tiny-HF 제어 실험의 업데이트/저장/재개; NemoChat/Bridge 학습 아님 |
| `pretrain` / `evaluation` | `scripts/base_eval.py` | 실제 CPU HF 추론; 부분/예제 수 제한 CORE 범위 유지 |
| `pretrain` / `evaluation` | `scripts/gym_eval.py` | 저장된 Gym 결과 비교; fixture 명시; 모델/판정 API 호출 없음 |
| `longcontext` / `schedule-plan` | `scripts/lc_schedule.py` | 스케줄/커서 출력만 수행; 모델 학습 없음 |
| `midtrain`, `sft`, `rlvr`, `mopd` | 등록용 단계 이름만 지원 | 해당 학습 진입점 미구현 |

`checkpoint-merge`는 허용된 유형 이름이지 추적되는 병합 진입점의 증거가 아니다.
`configs/tracking/stages.json`은 등록/연동 가능성 설명이며 **실행 원장도,
`--wandb-config` 입력도 아니다**. GPU/분산 학습 및 모든 단계 실행은 입증되지 않았다.

## 활성화, 설정, 실행 식별자

기본 모드는 `disabled`이다. 비활성 모드 또는 `RANK != 0`에서는 SDK를
import하거나 추적 receipt를 쓰지 않는다. 순위 판정은 `RANK`(기본 `0`)만
사용하며 `LOCAL_RANK`나 분산 동기화를 처리하지 않는다.

공통 옵션: `--wandb-mode {disabled,offline,online}`, `--wandb-config`,
`--wandb-id`, `--wandb-entity`, `--wandb-project`, `--wandb-group`,
`--wandb-parent-id`, `--wandb-checkpoint-id`, `--wandb-dir`, `--wandb-resume`.
모드는 CLI가 `NEMOCHAT_WANDB_MODE`보다 우선하며 JSON으로 활성화할 수 없다.
식별자/디렉터리 우선순위는 CLI → 해당 `NEMOCHAT_WANDB_<FIELD>` 환경변수
→ JSON → 래퍼 기본값이다. 환경변수 필드에는 `PARENT_ID`, `CHECKPOINT_ID`,
`DIR`가 포함된다.

`configs/tracking/default.json`을 템플릿으로 사용한다. JSON 옵션은
`run_id`, `entity`, `project`, `group`, `parent_id`, `checkpoint_id`, `directory`로
제한한다. 엄격한 허용 목록이 미지원/예약 키(`mode`, `stage`, `run_type`,
`resume`, `config`, 자격증명 포함)를 안내가 있는 `ValueError`로 거부하며,
제공된 임의 키나 값은 오류에 출력하지 않는다. 값은 비어 있지 않은 문자열이며
선택적 식별자만 null을 허용한다(project/directory는 불가). 단계 레지스트리를
넣지 않는다. JSON으로 활성화/재개할 수 없다. ID 미지정 시 임의 ID가 생성된다.
그룹/부모/체크포인트 ID는 사용자 제공 메타데이터이며 자동 출처 검증이나 복구가 아니다.

이미 준비된 로컬 모델 예시(경로를 실제 경로로 교체):

```bash
mkdir -p logs/wandb
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/train_cpu.py --model checkpoints/pretrained/my-model \
  --texts datasets/my-texts.jsonl --output checkpoints/cpu/my-offline-run \
  --steps 2 --context-length 32 \
  --wandb-mode offline --wandb-config configs/tracking/default.json \
  --wandb-id cpu-example --wandb-dir logs/wandb
```

W&B 오프라인 모드는 모델/데이터 다운로드를 차단하지 않는다. 다운로드 금지 시
로컬 입력과 HF 오프라인 환경변수를 사용한다. 별도 승인 없이 `wandb sync`를
실행하지 않는다. 온라인 모드는 CLI/환경변수의 명시적 활성화와 환경변수
`WANDB_API_KEY`를 요구하며 대화형 로그인을 하지 않는다. 키를 JSON, 소스,
예시 CLI, 실행 이름, 경로, artifact 메타데이터에 넣지 않는다.

## 지표와 artifact

학습은 전역 업데이트, 누적 loss-token 수, LM cross-entropy, 학습률/단계,
이번 호출의 입력/loss-token 수 및 업데이트 시간을 기록한다. 입력 토큰에는
reset prefix가 포함되지만 loss 토큰에는 포함되지 않는다. 처리량은 측정된
업데이트 초당 loss 토큰 수이며 종단간 GPU 처리량이 아니다.
성공 종료의 `wall_seconds_this_invocation`은 이번 호출 시간이며 누적 실행 시간이
아니다. 실패 정리는 선택적인 summary 쓰기를 건너뛴다.
데이터 준비 카운터는 optimizer가 소비한 학습 토큰 수가 아니다.
중첩 scalar는 `/`로 평탄화하고 원시 배열은 로컬 보고서에 보존한다.
평가 증거는 실제 추론, 사용자 확인 기반 결과 가져오기, 합성 fixture를 구분한다.

Artifact는 **메타데이터 전용**이다. 로컬 체크포인트/데이터/보고서 경로,
식별자와 카운터만 기록하며 래퍼는 가중치나 데이터/보고서 파일을 추가하지 않는다.
HF 참조는 40자리 소문자 commit과 저장소 ID를 검증하며 가중치를 업로드하거나
다운로드하지 않는다.

Sanitizer는 설정/지표/artifact와 실행 식별자 메타데이터(receipt 포함)의
인식된 자격증명 **키**를 재귀적으로 가린다. Tokenizer와 토큰 예산 키는
유지한다. **키 기반 필터링이며 인증을 수행하지 않는다**. 일반 비밀정보 스캐너나
URL 검증기가 아니므로 문자열, URL userinfo/인증 query, 경로, 자유 텍스트
식별자 값 및 인식되지 않은 키의 비밀값 보호는 보장하지 않는다. 일반 메타데이터
키에도 자격증명 URL/비밀값을 넣지 않는다. JSON 허용 목록은 임의 옵션의 전달을
막지만 명시적으로 제공한 학습 config의 비밀값을 검사하지 않는다. 온라인 사용 전에 검토한다. Git/코드 수집과 시스템 통계는
비활성화되지만 SDK 진단 로그와 requirements 메타데이터 파일은 생성된다.

## 실패와 RNG 의미

초기화/로그/설정/artifact/종료 SDK 예외는 전파되며 조용히 성공한 비추적 실행으로
전환하지 않는다. 호출자가 로그 실패를 잡더라도 실행 실패 상태는 유지된다.
본문 실패는 종료 코드 1을 사용한다. 로컬 `tracking-<id>-<호출>.json`에는
initializing/running/finished/failed 상태, 단계/유형/계보 및 SDK 디렉터리가 기록된다.
Receipt는 관측 자료이며 트랜잭션 학습 체크포인트가 아니다. 온라인 키 누락은
receipt 생성 전에 거부된다. Receipt I/O가 최초 실패라면 예외가 전파된다.
추가 receipt 쓰기/종료 실패는 기존 초기화/작업/본문 예외를 덮지 않으며 값이 없는
일반 예외 note로 알린다. SDK 초기화 성공 후 초기화가 실패하면 종료 코드 1로
SDK 정리를 시도한다. 실패 상태는 유지되며 실패 경로는 성공 receipt를 쓰지 않는다.
I/O가 계속 불가능하면 receipt가 없거나 오래되거나 불완전할 수 있어 성공 증거가
아니다. SDK 종료와 receipt 쓰기는 원자적 트랜잭션이 아니다. 이미 저장된
체크포인트는 추적 실패 시 롤백되지 않는다.

SDK 호출은 Python random, 이미 import된 NumPy legacy RNG, 이미 import된
**CPU** torch RNG를 저장/복원한다. CUDA RNG, 독립 NumPy Generator,
호출 중 처음 import된 모듈의 RNG는 보호하지 않는다. CPU 증거로 GPU RNG
불변성을 주장하지 않는다.

## 서로 다른 두 종류의 재개

`train_cpu.py --resume CHECKPOINT`는 파일 inventory와 동일한 학습 계약을 검증한
후 optimizer, CPU torch RNG, 전역 step, 데이터 cursor, 누적 loss-token 수를 복원한다.
`--steps`는 추가 업데이트 수이다. WSD는 동일한 전역 스케줄 계약/기간을 유지해야
하며 기존 출력 경로는 거부한다. W&B artifact로 모델을 복구하는 기능이 아니다.

추적 활성화 상태의 체크포인트 재개는 명시적 `--wandb-id`를 요구한다.
`--wandb-resume`도 명시적 ID가 필요하다. 온라인은 `resume="must"`를 전달하지만
오프라인은 SDK 재개 정책을 전달하지 않으며 **동일 ID의 새 오프라인 segment**를
생성한다. 기존 history를 자동으로 다시 열거나 병합하지 않는다. 재개 시 선택한
로컬 디렉터리의 동일 ID receipt는 실행 ID/단계/유형/entity/project가 일치해야
하며 충돌은 SDK 초기화 전에 거부한다. 이는 존재하는 로컬 단계 식별자만 검증하며
체크포인트 내용/원격 history/optimizer 또는 데이터 연속성을 증명하지 않는다.
로컬 history가 없으면 검증된 것으로 간주하거나 만들어내지 않는다. 명시적 ID만으로
연속성이 입증되지 않는다. 래퍼는 체크포인트의 이전 추적 ID와 일치하는지 확인하지
않으므로 부모/체크포인트 식별자를 사용자가 올바르게 유지해야 한다.
모델/optimizer 텐서가 같아도 타이밍 지표는 달라질 수 있다.

## 로컬 검증 증거 — 모든 단계 실행의 증거가 아님

검토 시 `/tmp/nemochat-wandb-smoke` 자료를 확인했다.

- SDK 0.30.0 환경에서 `tests/test_tracking.py`: **19 passed**. 등록/실패/RNG
  단위 테스트는 fake SDK이며 전용 오프라인 테스트만 실제 SDK를 사용한다.
- `review-tests/test_real_offline_sdk0`에 finished receipt와 실제 `.wandb` 파일이
  있다. `strace -f`의 `review-tests.network`에서 AF_INET/AF_INET6 connect 호출은
  **0건**이며 원격 W&B 호출은 관측되지 않았다.
- 기존 `continuous`/`resumed`는 업데이트 2, `first`는 업데이트 1이다.
  연속/재개 최종 체크포인트의 모델 텐서 28개, optimizer 상태, CPU torch RNG가
  모두 정확히 일치했다.
- 기존 데이터 준비, CPU 학습/평가, fixture Gym 비교, schedule-plan receipt는
  finished였고 검토한 해당 네트워크 trace에도 인터넷 connect 호출이 없었다.
  Gym fixture와 스케줄 출력은 학습이 아니다.

환경에서 `unshare -n`이 허용되지 않아 강제 네트워크 namespace 대신 오프라인
설정과 syscall trace를 사용했다. 온라인 테스트는 수행하지 않았다.
수정 검증: `.venv/bin/python -m pytest tests/test_tracking.py -q`는 실제 SDK
0.30.0으로 **46 passed**였다(실제 오프라인 테스트 2개, 나머지는 fixture).
전체 `tests/`는 **358 passed, 1 skipped, 25 subtests passed**였다. 이번 재실행은
syscall trace 없이 수행했으므로 새로운 네트워크 격리 증거는 아니다.
회귀 테스트는 엄격한 옵션 검증, receipt/종료 동시 실패 시 원래 예외 유지,
SDK 초기화 후 실패 정리, 실패 상태 유지 및 로컬 단계/유형/entity/project 연속성을
검증한다. 실제 SDK 오프라인 테스트는 로그/artifact 메타데이터, 동일 ID의 별도
재개 segment, 본문 실패를 검증하며 GPU나 미구현 단계 학습은 수행하지 않는다.
남은 한계는 CUDA/분산 RNG 및 rank 검증, 임의 비밀값, 사용할 수 없는 receipt 저장소,
체크포인트와 추적 식별자의 일치 검증이다. GPU RNG/모든 단계 실행을 주장하지 않는다.
