# NemoChat 학습 대시보드

학습 프로세스와 분리된 **읽기 전용 웹 모니터**입니다. 기존 학습 로그·결과와 현재 호스트 상태를 한 화면에서 확인합니다. PyTorch import/CUDA 초기화, 학습 실행, 옵티마이저 변경, 자동 프로파일링은 하지 않습니다. [English](README.md).

## 실행과 접속

저장소 루트에서 실행합니다.

```bash
python3 dashboard/server.py --root . --config dashboard/config.json --port 8787 --interval 5 --reload-config
```

브라우저에서 `http://127.0.0.1:8787`에 접속하세요. 원격 학습 서버라면 **사용자 PC에서 SSH 터널**을 엽니다.

```bash
ssh -N -L 8787:127.0.0.1:8787 USER@TRAINING_NODE
```

Python 표준 라이브러리와 로컬 JS/CSS만 사용하므로 학습 환경에 패키지를 추가하거나 W&B 키를 입력할 필요가 없습니다. 기본 갱신은 **SSE(지속 HTTP 연결을 통한 단방향 서버 푸시)**이며, 실패하면 주기 조회로 전환합니다. 브라우저를 여러 개 열어도 하드웨어 수집 프로세스를 복제하지 않습니다.

## 표시 정보와 해석

- 선택한 실행의 모델 이름·정밀도·파라미터 정보, 데이터 종류와 실제 관측 범위.
- 처리 토큰, loss 계산 대상 토큰, source-prefix 읽기 수를 **분리**합니다. 데이터 준비 토큰은 학습 토큰이 아닙니다. rank 0 수치를 근거 없이 GPU 수만큼 곱하지 않습니다.
- loss/LR/gradient norm 이력과 실행 진행률. rank-local loss와 글로벌 평균을 혼동하지 않습니다.
- 스테이지별 관측 여부. 4-update pilot 완료는 해당 실행의 완료이며 **전체 pretraining 완료가 아닙니다**.
- 실행별 시간과 근거가 있는 처리량/잔여 시간. update 시간과 컨테이너 전체 시간을 구분하며, 겹치는 실행 시간을 더해 총학습 시간을 꾸며내지 않습니다.
- 실제 기록이 있는 벤치마크만 표시합니다. 없는 값은 미관측이며 임의 점수가 없습니다.
- **현재 노드 전체** CPU/RAM/GPU 사용량·메모리·온도·전력. 과거에 선택한 실행의 전용 리소스라는 뜻은 아닙니다.
- 최신성, 소스 경로, 수집 소요 시간, 오류와 분석 범위 경고.

누락된 사실은 미관측으로 표시합니다. 파일이 최근에 수정됐다는 이유만으로 학습 중이라고 단정하지 않습니다. 이 대시보드는 학습을 시작하거나 중단하는 관리 콘솔이 아닙니다.

## 데이터 출처와 순정 trainer 원칙

[입력 파일·집계 방식 설명](DATA_SOURCES.ko.md)을 참고하세요. 기존 Bridge/native 로그 및 native Lens/OTel export를 읽고 대시보드에서 가공합니다. trainer 전용 callback이나 새로운 출력 포맷을 강제하지 않습니다. Canonical v1은 선택적인 파생/export 결과의 입력 계약이며 읽기 전용으로 검증할 수 있습니다.

```bash
python3 dashboard/telemetry.py validate --manifest PATH/run.json --metrics PATH/metrics.jsonl
```

프로파일에는 상위 커널 CUDA 누적 시간과 별도의 HOST NVTX 구간 그래프, 자료가 있으면 Lens/bridge host 연산 그래프를 표시합니다. 호버·키보드 포커스·탭에서 단계·시간·원래 이름·출처·참고 설명을 확인합니다. 소스/구간 매핑이 없는 것은 미확인으로 두며 CUDA 시간과 host 시간을 혼동하지 않습니다.

[커널·소스 인스펙터](KERNEL_INSPECTOR.ko.md)는 측정 update/구간 선택, 쉬운 연산 이름, 행 번호와 hash로 검증된 저장 host 코드, 우선 조사 항목을 제공합니다. `nsight_analysis`로 별도 분석 결과를 읽으며 native host 코드에는 `nsight_native_source`·`nsight_source_provenance`를 명시합니다. 매 브라우저 연결마다 무거운 SQL 분석을 하지 않습니다. 현재 캡처의 정확한 native 내부 Python 호출 frame은 여전히 미수집입니다. [실제 테스트171개·소스·UI 검증](INSPECTOR_VERIFICATION.md). 이어서 [원시 이름 토큰 해석·iteration 타임라인](KERNEL_NAMES_AND_TIMELINE.ko.md)을 추가했습니다. NCCL/NVJET의 확인·추정·미확인, 실제 GPU stream/CPU API 구간, 시간 창·이동·시간순100행 페이지를 제공합니다. 관련232tests/32subtests 통과. NCCL 실제 디스패치와 비공개 suffix의 한계도 표시합니다.

## 실행 추가

`dashboard/config.json`에 개별 실행 또는 승인된 `run_roots`를 등록합니다. 루트·prefix→stage 규칙에 맞는 지원 실행 폴더가 추가되면 매 수집에서 자동 등록하며 dataset/checkpoint 재귀 탐색은 하지 않습니다. [live단계 반영 수정](LIVE_STAGE_FIX.ko.md)에 MID/SFT/RLVR 실제 값·rollout_only·영어 metric·historical/live 구분 검증을 기록했습니다.

`dashboard/config.json`의 저장소 상대 경로 목록에 관측할 실행을 추가합니다. 허용된 형식은 `CONTRACT.md`와 `collectors.py`를 참고하세요. 서버가 hot reload를 명시적으로 지원하지 않으면 설정 변경 후 재시작하세요.

지원되는 rank-update 로그를 기록하는 기존 실행은 학습 루프 변경 없이 계속 읽습니다. 새로운 trainer 포맷은 별도 작은 어댑터가 필요하며 임의 로그를 자동으로 해석하는 시스템은 아닙니다. `.env`, 컨테이너 환경 덤프, 원문 프롬프트, 체크포인트 payload를 소스로 지정하지 마세요. HTTP 요청으로 임의 파일을 읽는 기능은 없습니다.

## Lens / Nsight / 기존 plugin

대시보드는 기존 결과를 **읽어 보여주는 연결 계층**입니다. Nsight GUI나 GPU profiler를 대체하지 않으며 native Nsight GUI plugin을 구현했다고 주장하지 않습니다.

- **NeMo Lens:** 로컬 export span의 계층·시간·ID를 표시합니다. 이는 host/OTel span 시간이며 GPU 커널 시간이 아닙니다. attribute는 허용 목록으로 제한합니다.
- **Nsight Systems:** export된 SQLite와 실제 선택 구간 분석을 읽어 커널 원래 이름·횟수·누적 시간을 보여줍니다. 전체 capture 통계에는 초기화가 포함됩니다. 커널 시간은 서로 겹칠 수 있어 합계를 wall time이나 GPU utilization으로 해석하지 않습니다.
- **Lens–Nsight bridge:** 기존 operation annotation의 설명·source 식별자를 표시합니다. 단순 커널 이름이나 시간 근접성만으로 kernel→source 매핑을 확정하지 않습니다.

기본 Lens/bridge 자료는 별도의 교육용 sparse-MoE control입니다. 선택한 native Bridge 학습 안에 Lens가 이미 통합됐다는 뜻이 아닙니다. 기본 native Nsight 자료는 rank 0 범위이며 다른 rank 완료 기록은 전체 노드 trace를 의미하지 않습니다.

새 capture가 필요하면 기존 `tutorials/nemo_lens/` 및 `plugins/nemo_lens_nsight_bridge/` 절차로 **승인된 별도 진단 실행**을 수행하세요. 대시보드는 학습 중인 프로세스에 profiler를 붙이거나 collector를 자동 실행하지 않습니다. 깊은 타임라인 분석은 `.nsys-rep`를 Nsight에서 여세요. 프로파일링된 시간과 비프로파일링 처리량 측정은 분리해야 합니다.

## 학습 간섭과 보안

기본 5초 수집이며 부하가 큰 서버에서는 간격을 늘릴 수 있습니다. 수집기 하나가 캐시를 공유하고 파일 읽기·이력 길이·SQLite 쿼리를 제한합니다. 변경 없는 profile 집계는 재사용하며 `nvidia-smi`는 시간 제한이 있는 읽기 전용 조회입니다. 체크포인트 전체를 재귀적으로 스캔하거나 로드하지 않습니다.

이 구조가 학습 callback에서 동기 네트워크 요청하는 것보다 분리되어 있더라도 **오버헤드 0이라는 보장은 아닙니다**. CPU·스토리지·NVML 자원은 공유합니다. 표시되는 수집 비용을 확인하고 실제 비프로파일링 학습의 모니터 켜기/끄기 비교로 판단하세요. idle-node 한 번의 수집 시간은 학습 오버헤드 상한이 아닙니다.

인증/TLS/멀티테넌트 격리가 있는 공개 서비스가 아니므로 loopback과 SSH 터널을 사용하세요. 서버를 외부에 공개하거나 민감한 실행/source 정보를 무단 공유하지 마세요.

## 검증

```bash
.venv/bin/python -m pytest -q dashboard/tests tests/test_source_evidence.py
python3 -m compileall -q dashboard
node --check dashboard/static/app.js
git diff --check
```

합성 fixture는 입력 계약과 오류 처리를 검증하며 실제 학습·품질 성능 근거가 아닙니다. 실제 파일과 브라우저 통합 검증 결과는 `VERIFICATION.md`, 시간 그래프·출처·Lens 인터페이스 확장과 테스트 106개 재실행 결과는 `UPDATE_VERIFICATION.md`에 기록했습니다. commit/push 및 원격 telemetry는 수행하지 않습니다.
