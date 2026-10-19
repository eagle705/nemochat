# Production CPU → native main 연속 실행

[English / 정확한 전체 실행 명령](PRODUCTION_RUN.md)

현재 generation02는 `systemd-run --user`에서 `User=`를 명시하지 않고 현재 user manager의 권한을 상속합니다. generation01은 supplementary group 변경 권한 문제로 Python 실행 전 `216/GROUP` 실패했으며 원본 source snapshot과 journal을 보존했습니다. unit의 `User` 값이 비어 있어도 실제 `/proc` UID·정확한 MainPID·전체 argv 검사는 유지합니다. 이 controller 수정으로 CPU corpus writer나 GPU 학습을 재시작하지 않았습니다.

`run_production.py`는 CPU 데이터 준비 종료 후 승인된 native main을 연결하는
실행 가능한 controller다. controller 실행 자체는 학습 시작/완료 증거가 아니다.
인수 없는 실행, `--preview`, `--help`는 파일 생성, Engine 요청, 서비스 생성,
GPU 작업, 다운로드, W&B 등록 없이 정보만 출력한다. 부모가 코드 검토와 별도
source freeze를 완료한 후 독립 user-systemd 서비스를 시작한다.

## 고정 범위와 실제 gate

- 기존 CPU producer ID:
  `9e0861343af8efa265c622623ff78714109132a1cd48cc59c4a2731ec456bab5`,
  이름 `nemochat-midtraining-production-prepare-01`.
  승인된 identity/payload receipt 전체 SHA256이 controller에 고정되어 있다.
  이 prerequisite을 새로 시작하거나 재시도·중지하지 않는다.
- exact ID에 대해 Engine `POST /containers/<ID>/wait?condition=not-running`를
  **한 번 blocking 호출**한다. 2026년 10월 17일 11:31:59 UTC 시작 기준
  남은 12시간 + 300초 grace 안에서 기다린다. artifact polling은 없다.
  호출 전후 exact ID/name/`nemochat.*` labels/`Config.Image`/`Config.Cmd`를
  확인한다. 상속된 NVIDIA labels는 허용한다. wait status 0, 실제 exit 0,
  `Running=false`, `OOMKilled=false`가 모두 필요하다.
- 실제 production manifest:
  `datasets/midtraining/super35-kd-01/production-24h-v1/data/manifest.json`.
  native loader로 companion success receipt의 manifest SHA를 확인한다.
- seed manifest SHA
  `52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308`을
  고정한다. 기존 heldout 예약 **전체**의 source/content hash/group ID가 새
  예약에 포함되어야 하며, 네 validation `.bin/.idx` digest와 크기도 유지한다.
  현재 validation에 사용되지 않은 예약도 train으로 풀지 않는다.
- native per-source capacity gate를 **3200 updates / GB128 / sequence4096**,
  weights `[0.6, 0.1, 0.1, 0.2]`로 다시 적용한다. cycling은 금지다.
  train/validation 전체 여덟 `.bin`을 새로 streaming scan하여 token ID
  18–29 count가 모두 0임을 확인하고 `.bin/.idx` digest·크기도 확인한다.
  이전 corpus의 audit를 새 corpus 증거로 재사용하지 않는다.
- calibration consumed config의 frozen SHA를 확인하고 production config와
  original BASE, exact Base teacher, precision/parallel, sequence/batch,
  optimizer/DDP/KD/RNG, source weights, checkpoint 정책을 비교한다.
  stage/name, 데이터 경로, logger, duration/watchdog, WSD horizon,
  validation 주기, save 경로 변경만 허용한다. tokenizer 전체 identity도 비교한다.
  실제 비교와 heldout gate 통과 후에만 기존 launcher가 요구하는 정확한
  `parent_accepted_same_recipe_expansion` dictionary를 생성한다.
- 이는 **동일 recipe 데이터 확장에 대한 명시적 부모 승인 projection**이다.
  새 manifest의 독립 calibration 또는 24시간 sustained 학습 입증이 아니다.
  기존 native warm step 평균은 24.29540627438837초이며, 10% overhead는 가정이다.
- main ID는 `super35-longrun-01`, online destination은 `nvidia/nemochat`이다.
  시작점은 **original BASE model-only reset**이며 calibration 학습 가중치가 아니다.
  기존 student checkpoint byte-integrity 명시적 waiver를 유지한다. 추가 checkpoint
  hash 검사나 다운로드는 없다. private key는 기존 launcher/native 경로에 남고,
  controller argv/config/log에 포함하지 않는다.

## 서로 다른 source allowlist

controller startup review는 launcher `SOURCES` 14개와 producer `REVIEW_FILES`
6개, 새 `midtraining/training/run_production.py`의 정확한 합집합이다(중복 제거).
`{relative_path: sha256}` JSON이며 기존 review와 섞지 않는다.

기존 native launcher에는 계속
`results/midtraining/super35-kd-01/longrun-source-review-02.json`만 전달한다.
producer의 기존 six-file review도 변경하지 않는다. 부모는 네 파일 구현이 끝난
후 직접 검토하고 startup review를 생성해야 한다. 이 handoff 자체는 review
artifact를 만들지 않는다. 정확한 freeze 명령은 영문 문서의
**Two independent source reviews**에 있다. 기존 freeze를 덮어쓰지 않는다.

source는 CPU wait 전후, 데이터/main release 전, subprocess 실행 직전에
재확인한다. prerequisite receipt도 시작 시 digest를 고정하고 release 전에
비교한다. 누락/추가/중복 JSON key/변경은 실패 처리한다. armed 상태에서
reviewed source를 수정하면 안 된다.

## 부모 시작과 runtime bound

영문 문서 **Exact parent launch**의 전체 `systemd-run` 명령을 부모가 검토하고
실행한다. `--preview`에도 완전한 `supervisor_argv`와 `main_argv`가 출력된다.

```bash
cd /raid/yoon/nemochat
.venv/bin/python -m pytest -q tests/test_midtraining_production_pipeline.py \
  tests/test_midtraining_longrun.py tests/test_midtraining_production_corpus.py
.venv/bin/python midtraining/training/run_production.py --preview
```

고정 unit은 `nemochat-midtraining-production-02.service`, supervisor runtime은
**140400초(39시간)**다. CPU 최대 12시간과 main subprocess 전체 **93600초
(26시간)**, orchestration 여유를 위한 supervisor bound이며 39시간 학습이 아니다.
데이터 검증 후 main 전체 예산이 부족하면 GPU release를 차단한다. native
학습 목표는 그대로 86400초이고 WSD horizon은 3200/32 warmup/320 final decay다.

서비스는 `KillMode=control-group`, `Restart=no`, exact main cleanup용
`ExecStopPost`를 사용한다. `NEMOCHAT_PRODUCTION_SUPERVISED=1`만 설정한다고
실행되지 않는다. exact unit MainPID, UID, 실제 interpreter/script/전체 argv가
일치해야 한다. `-u` 인수를 추가하지 말고 문서의 buffering 환경변수를 사용한다.
명령 생성 함수는 실제 UID를 사용하며 영문 전체 예시는 승인된 UID 1001이다.

생성된 controller output, main output/checkpoint/audit 경로가 이미 있으면
암묵적으로 resume/retry하지 않는다. 이전 증거를 삭제해 재실행하지 않는다.
동일 main을 별도로 직접 실행하지 않는다.

## readback와 정직한 완료 판정

부모는 시작 직후 unit MainPID/argv/phase를 직접 read back한다:

```bash
systemctl --user show nemochat-midtraining-production-02.service \
  --property=MainPID,User,ActiveState,SubState,ExecStart,ExecStopPost,RuntimeMaxUSec,KillMode
.venv/bin/python -c 'import json,pathlib; print(json.dumps(json.loads(pathlib.Path("results/midtraining/longrun/super35-production-02/controller/state.json").read_text()),indent=2))'
```

fsync된 phase 이력 및 atomic state는
`results/midtraining/longrun/super35-production-02/controller/`에 남는다:
`WAITING_CORPUS` → `VALIDATING_DATA` → `NATIVE_MAIN_CONTROLLER_RUNNING` →
`FINISHED`, 실패 시 `FAILED`.

`NATIVE_MAIN_CONTROLLER_RUNNING`도 optimizer update 증거는 아니다.
실제 all-rank native update receipt를 확인하고, main 시작 후 부모가 정확한
remote W&B run name/update를 독립 read-only 검증해야 한다. controller는 보조
run을 생성하지 않으며 remote tracking 검증 완료를 주장하지 않는다.

SIGTERM/INT는 실패를 기록한다. systemd control-group이 launcher에 stop을
전파하고 `ExecStopPost --cleanup-owned`는 기존 exact ID/name/task-label 검증을
거쳐 이 main의 config/train/reload container만 정리한다. main release 기록이
없으면 cleanup은 아무 작업도 하지 않는다. CPU prerequisite/다른 서비스/
foreign job을 중지하지 않고 GPU reset, 자동 retry/resume, SFT도 없다.

`FINISHED`에는 별도 launcher의 실제 `production_save_reload_accepted`, 동일
manifest, all eight ranks가 필요하다. `training_24h_accepted`는 원본 boolean을
그대로 보존한다. 3200 iterations가 native 86400초보다 먼저 끝나면 **false**이며
24시간 성공으로 바꾸지 않는다. CPU synthetic fixture와 실제 tiny child-process
테스트는 GPU 실행, full corpus capacity, 품질, 24시간 안정성 증거가 아니다.
실제 production capacity는 producer와 후속 실제 audit가 끝난 뒤에만 확정된다.
