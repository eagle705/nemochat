# Codex로 Nsight Systems 분석하기

새 CLI 실행은 저장소 루트 기준 경로와 새 산출물 이름을 사용합니다. [런타임 디렉터리 규약](RUNTIME_LAYOUT.md)에 따라 입력 데이터는 `datasets/`, checkpoint는 `checkpoints/`, 보고서·trace·plot은 `results/`, 텍스트/W&B 로그는 `logs/`에 둡니다. 필요한 부모 폴더만 만들며 기존 fixture/config와 과거 증거 경로는 보존합니다.

영문 canonical: [NSIGHT_CODEX_GUIDE.md](../en/NSIGHT_CODEX_GUIDE.md). [Pretraining 성능 실습](PRETRAIN_PERFORMANCE_LAB.md)의 P1/P3에 해당한다.

## 권장 방식

**Codex는 CSV/SQLite에서 후보를 찾고, 사람은 GUI에서 timeline/overlap/의존성을 확인한다.** GUI 자동화나 스크린샷만으로 수치를 추정하는 것보다 반복 가능한 분석이 된다. 현재 실제 trace 수집/분석이나 repo 분석 script 구현은 미완료다.

## 1. Trace 수집 준비

승인된 GPU 서버에서 installed version과 옵션을 먼저 확인한다.

```bash
nsys --version
nsys profile --help
nsys stats --help
nsys stats --help-reports
nsys export --help
```

실제 native 학습에 step/module NVTX를 추가하고 JIT/autotuning/warmup 이후 짧은 steady-state 구간을 수집한다. CUDA/NVTX를 포함하고 CPU 원인 분석에 필요하면 허용된 CPU sampling/runtime tracing을 켠다. 정확한 command, collector, 권한 제한, SDK/source revision, rank/device mapping을 기록한다. 수집되지 않은 이벤트는 비용이 0이라는 뜻이 아니다.

Profile window와 launcher 옵션은 설치된 help에 맞춘다. 이 문서는 완성된 GPU 학습 launch recipe가 아니다. Run/rank별 새 출력 경로를 사용하고 `.nsys-rep` 원본을 보존한다. 분산 주장은 관련 rank를 모두 확인한다. Capture가 step/event 중간에서 잘렸는지 점검한다. Trace에는 경로/NVTX 문구 등 민감한 정보가 포함될 수 있으므로 외부 공유 전 검토한다.

## 2. Summary와 SQLite export

새 `results/performance/my-run/`에 report를 복사한 뒤 저장소 루트에서 사용하는 예시다. 현재 호스트에서 실제 report로 실행한 명령은 아니다.

```bash
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum --format csv results/performance/my-run/baseline.nsys-rep
nsys export -t sqlite --output results/performance/my-run/baseline.sqlite results/performance/my-run/baseline.nsys-rep
```

Report 이름과 flags를 실제 help에서 확인한다. 첫 명령은 통계를 console에 출력하므로 필요하면 새 파일에 저장한다. 두 번째는 일반적으로 `baseline.sqlite`를 생성하나 실제 경로를 확인한다. Stats가 SQLite를 자동 생성할 수도 있으므로 기존 파일을 무조건 덮어쓰지 않는다.

Summary는 후보 탐색용이다. Kernel/API percent는 해당 report의 합계 기준이지 학습 wall time 비중이라고 단정할 수 없다. Summary만으로 launch starvation을 확정하지 않는다.

## 3. Codex SQLite 분석 규칙

Python `sqlite3`의 properly escaped absolute URI와 `mode=ro`로 연다. 잘못된 파일명으로 빈 DB를 자동 생성하지 않는다. `sqlite_master` 및 실제 column을 먼저 조회한다. Version에 따른 schema 차이, string table, timestamp 단위와 origin을 확인한다.

- 완전한 steady-state NVTX step을 process/rank/thread 범위와 함께 선택한다. CPU NVTX 시간이 곧 GPU 시간은 아니다.
- Nested/overlapping NVTX attribution 규칙을 정하고 중복 집계하지 않는다.
- Correlation ID에 process/context 범위를 고려한다. ID 하나가 전역적으로 유일하다고 가정하지 않는다.
- 특정 stream의 gap을 GPU 전체 idle로 부르지 않는다. 관련 compute/copy interval union을 확인하고 수집 범위를 명시한다.
- 겹치는 kernel 시간을 합산해서 wall time으로 해석하지 않는다.
- CPU API 실행 시간, launch-to-start queue 시간, kernel 시간, idle 후보를 구분한다. Queue 대기는 dependency/backlog 때문일 수도 있다.
- Sync, data/H2D, NCCL, rank imbalance를 대안 원인으로 조사한다. Kernel 이름만으로 Python 소스 위치를 만들어내지 않는다.
- 부분 export가 관련 이벤트를 누락하면 correlation이 잘못될 수 있다. 누락/잘린 capture는 한계로 기록한다.
- 큰 DB 전체를 context에 넣지 말고 SQL/Python으로 좁힌 결과만 전달한다.

## 4. Codex 요청문

```text
AGENTS.md, NemoChatAgent.md와 영문 performance lab/NSIGHT_CODEX_GUIDE를 읽어라.
지정한 baseline.sqlite를 read-only로 분석하라.
학습 실행, 소스 수정, 도구 설치, 유료 작업은 하지 말라.

Schema, collector coverage, timestamp 단위/origin부터 확인하라.
지정된 NVTX steady-state step과 rank에서 API/kernel 분포와
작은 kernel sequence, launch/idle-gap 후보를 찾는다.
Process/context-aware correlation으로 API와 GPU activity를 연결한다.
다른 stream/rank를 보고 launch starvation, sync, data,
communication/dependency 원인을 구분한다.

후보마다 rank/device/stream, timestamp와 단위/origin, 구간,
NVTX, API/kernel 식별자, 관찰, 가설, 대안 설명, confidence,
소스 위치(확인된 경우), GUI에서 확인할 항목을 적는다.
증거가 없으면 미확정으로 표시하고 mapping을 추측하지 않는다.
새 output directory에 bounded CSV/JSON/Markdown을 작성한다.
Trace/microbenchmark만으로 전체 학습 speedup을 주장하지 않는다.
```

실제 dispatch 때 파일 경로·분석 구간·rank를 넣는다. Source mapping이 없으면 좁은 NVTX 계측을 추가할 다음 실험을 제안한다.

## 5. GUI 확인과 전후 비교

원본 `.nsys-rep`를 호환되는 GUI로 열고 Codex가 기록한 timestamp origin/단위·rank·stream·NVTX로 후보를 찾는다. 다른 stream, copy, API correlation, 의존성을 함께 확인한다. CPU가 GPU 완료를 기다리는지 GPU가 CPU 제출을 기다리는지 구분하고 update critical path에 있는지 본다. 스크린샷은 보조 자료다.

Baseline/fused capture 조건과 workload를 맞추고 별도 실행의 절대 timestamp가 아닌 step/NVTX 의미로 비교한다. 최종 update throughput/memory는 **profiler 없이** 반복 측정한다. Fusion 채택은 forward/backward correctness 및 checkpoint/resume gate를 통과한 뒤 결정한다.

## 향후 분석 도구

`scripts/analyze_nsys.py`는 예정 후보이며 현재 실행 가능한 도구가 아니다. `results/performance/<unique-run>/`에 summary JSON, kernel/candidate CSV, analysis Markdown과 trace/version/수집 범위를 보관하는 구성을 권장한다.

구현 시 synthetic SQLite로 read-only/없는 파일/unknown schema/단위/overlap/multi-stream/ID reuse/nested NVTX/불완전 capture/이벤트 누락을 테스트한다. 이후 실제 export와 GUI 후보 확인을 별도로 검증한다. Synthetic 테스트를 실제 Nsight 분석 검증으로 보고하지 않는다.

출처: [Nsight Systems 공식 가이드](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).
