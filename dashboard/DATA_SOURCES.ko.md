# 대시보드는 어떤 데이터를 읽는가?

[English](DATA_SOURCES.md). 이 대시보드는 **기존 로컬 로그·결과를 읽는 모니터**입니다. 학습을 수행하거나 W&B 서버에서 원격 history를 자동 조회하지 않습니다. 읽을 실행·프로파일은 `dashboard/config.json`에서 명시적으로 선택합니다.

## 데이터 흐름

```
Bridge/native 학습 로그 → 포맷별 어댑터 → 실행 메트릭
현재 /proc + nvidia-smi → CPU/RAM/GPU 상태
Lens/bridge JSONL + Nsight SQLite/분석 JSON → 프로파일
                           ↓
                별도 수집 프로세스 + 공유 캐시
                           ↓
                   SSE / 주기 조회 → 화면
```

브라우저가 받는 것은 정제된 snapshot입니다. `.env`, 원문 프롬프트, 체크포인트, 임의 파일, 컨테이너 환경 덤프는 공개하지 않습니다. 현재 노드 상태와 과거 실행 기록, 별도 프로파일 캡처를 하나의 실행인 것처럼 섞지 않습니다.

## 현재 읽는 자료

### 1. 실제 Bridge 실행

기본 경로는 `results/pretrain_followup/bridge-fresh-wsd4-01/`입니다.

- **`rank-N-updates.jsonl`**: 스텝별 optimizer audit 로그. loss/LR/grad norm, rank-local 토큰 delta, 스텝 시간을 읽습니다. 같은 스텝을 rank끼리 맞추고 필요한 rank 기록이 있는 경우에만 토큰을 집계합니다.
- **`verified-summary.json`**: 검증된 글로벌 토큰 수, 업데이트 수, 실행 결과와 컨테이너 전체 시간입니다.
- **`requested-config.json`**: 모델/provider·BF16·차원·batch/sequence·스텝 목표입니다. 설정값만으로 실제 파라미터 수를 세었다고 하지 않습니다.
- **`data-verification.json`**: 선택한 데이터 준비 범위입니다. 준비 토큰과 처리 토큰은 별개입니다.
- **선택적 `model_census`**: 이전 native 실행에서 실제로 센 파라미터 수의 근거입니다. 현재 실행에서 새로 측정한 값이 아니라는 경고와 경로를 표시합니다.

현재 어댑터의 local loss 평균은 **rank별 산술평균**입니다. 일반적으로 각 rank의 target 수가 다르면 token-weighted 글로벌 loss와 다를 수 있습니다. 현재 pilot은 동일한 local target 수를 사용하지만, 이 집계 방식을 모든 trainer에 자동 적용하면 안 됩니다.

### 2. Native E2E 실행

- **`rank-N-phases.jsonl` / `rank-N.json` / `result.json`**: 실제 optimizer 완료 기록, 실행 이력, 누적 토큰·진단 값·결과입니다.
- **`resolved_config.json`**: 데이터 계약, 실행 계획과 schedule horizon입니다.
- **명시적 terminal receipt**: producer/container의 최종 상태입니다. producer가 실패했지만 checkpoint snapshot이 별도로 유효한 경우를 분리합니다.
- **heldout CE**: 고정된 작은 heldout 진단입니다. 전체 공식 벤치마크 성능과 같지 않습니다.

재개 실행의 누적 토큰을 다른 실행과 다시 더하지 않습니다. rank 누락, 부분 쓰기, 제한된 이력은 경고하며 값이 없다고 0으로 채우지 않습니다.

### 3. 현재 CPU/GPU

`/proc/stat`, `/proc/cpuinfo`, `/proc/meminfo` 및 시간 제한이 있는 `nvidia-smi` 조회입니다. CPU 사용률은 연속 샘플 간 변화로 계산합니다. GPU 사용량은 **현재 노드** 정보이며, 선택한 과거 실행의 전용 자원이라는 뜻이 아닙니다.

PyTorch import/CUDA context는 생성하지 않습니다. 수집 주기·비용은 화면에 표시하지만 학습 slowdown 비율은 별도 on/off 비교로 측정해야 합니다.

### 4. 프로파일

- **Nsight SQLite:** 실제 커널 테이블과 이름 테이블에서 원래 커널 이름·횟수·누적 ms·평균 µs를 집계합니다. 전체 capture에는 초기화·warmup이 들어 있고, 병렬 커널 시간 합은 wall time이 아닙니다.
- **선택 구간 분석 JSON:** producer가 보고한 rank/update별 NVTX 구간 시간과 연결된 커널 수입니다. 대시보드가 launch correlation을 다시 계산하거나 producer provenance를 인증하지는 않습니다.
- **Lens Console JSONL / OTLP JSON:** host/OTel span의 이름·ID·계층·시간 및 명시적인 resource run UUID/rank를 읽습니다. pretty JSON과 resourceSpans/scopeSpans batch도 지원합니다. 이는 로컬 export 자료 reader이며 새 OTLP 수신 서버나 native trainer instrumentation을 뜻하지 않습니다. CUDA 커널 시간이 아닙니다.
- **Bridge JSONL:** 실제 annotation의 역할·설명·source 식별자입니다. 명시적 host 연산 설명이며, GPU kernel→source 검증 완료라는 뜻이 아닙니다.

그래프는 **커널 CUDA 누적 시간 / host NVTX 구간 시간 / Lens host 시간**을 구분합니다. 호버에서 단계·단위·출처·집계 범위를 표시하며, 커널이 Forward/Backward 어디에 속하는지 근거가 없으면 미확인으로 표시합니다. 커널 이름 기반 뜻풀이는 참고 설명이고 shape·소스·병목을 확정하지 않습니다.

## 공통 로그 포맷

**선택한 원칙은 upstream trainer를 최대한 순정으로 유지하는 것입니다.** 기존 로그와 native NeMo Lens/OTel export를 읽고 대시보드 쪽에서 정규화합니다. 대시보드 전용 loss callback, monkeypatch, 학습 루프 wrapper, 필수 JSONL writer를 trainer에 넣지 않습니다. canonical schema는 선택적인 대시보드 내부/파생 결과 교환 계약이며 trainer 출력 요구사항이 아닙니다.

Lens는 설치된 SDK가 실제로 제공하는 launcher/config 인터페이스를 확인한 뒤 기존 instrumentation을 활용합니다. 환경변수를 지정하는 것만으로 instrumentation이 없는 trainer에 span이 생기지는 않습니다. 지원·export 범위를 확인하기 전 자동 활성화하거나 collector를 띄우지 않습니다.

기존 파일은 깨뜨리지 않고 어댑터로 읽습니다. 새 producer에는 `telemetry.schema.json`과 `telemetry.py` 검증 CLI의 **versioned canonical 형식**을 선택적으로 사용할 수 있도록 합니다. 어댑터 추가가 기존 trainer의 출력 방식 변경이나 callback 설치를 의미하지 않습니다.

핵심 규칙:

- `schema_version`, `run_id`, stage, step, timestamp를 명시합니다.
- 기록 rank/world size와 **지정된 writer 하나**를 정합니다. 실제 SDK logger가 rank 0이라는 가정은 하지 않습니다.
- loss가 글로벌 reduction인지 rank-local인지 기록합니다.
- canonical 토큰은 **글로벌 누적값**이며 processed/loss-bearing/source-read를 구분합니다. 재개 시 누적값을 이어가고 독립 실행·분기는 run ID를 바꿉니다.
- 미수집은 `null`, 관측된 0은 숫자 `0`입니다. bool·NaN·Infinity는 메트릭 숫자로 인정하지 않습니다.
- producer 성공/실패, checkpoint 무결성, evaluation 결과, 전체 stage 완료는 구분합니다.
- 환경 전체·프롬프트·모델 state를 덤프하지 않고 허용된 필드만 출력합니다.

정확한 허용 필드는 실행 가능한 validator와 schema가 기준입니다. producer에 연결하기 전에 CLI `--help`와 계약을 확인하세요. Nsight/Lens는 상관관계 정보를 보존하기 위해 원래 export 형식을 유지합니다. 모든 profiler 파일을 일반 학습 JSONL로 억지 변환하지 않습니다.

## 설명 근거

일반 연산 의미는 NVIDIA NCCL/cuBLAS 및 PyTorch 공식 문서를 확인했습니다. 구체적인 자료는 영문 문서의 reference 항목에 있습니다. 이 문서는 실제 캡처에서 특정 layer/Python call의 출처를 입증하는 것이 아닙니다. Bridge 설명은 별도 교육용 control의 실제 연산 기록 범위로 제한합니다.
