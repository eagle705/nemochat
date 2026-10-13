# NeMo Lens 실습: 실제 CPU 학습 → 로컬 Jaeger

[English](README.md)

이 실습은 생성한 고정 데이터로 작은 MLP의 **실제 forward/backward/SGD 업데이트**를
실행하는 오프라인 CPU 대조군입니다. NemoChat 하이브리드 모델 학습, GPU 성능,
모델 품질을 검증하지 않습니다. Lens의 의미 단위 span은 CUDA 커널 프로파일이
아닙니다. Nsight 커널 시간, launch 상관관계, 커널→Python 소스 매핑도 제공하지
않습니다. 클라우드, W&B 계정, API 키, 모델/데이터 다운로드는 필요하지 않습니다.
패키지와 Docker 이미지 설치에는 네트워크가 필요하지만 학습/전송은 로컬입니다.

## 버전과 격리

- 공식 [NVIDIA-NeMo/Lens 소스](https://github.com/NVIDIA-NeMo/Lens/tree/4d045c03186b90afdc815db46b671f6a21bc7e98):
  `4d045c03186b90afdc815db46b671f6a21bc7e98` 고정 Git 설치,
  `nemo-lens==0.2.0+4d045c0`. 소스 전체를 복사하지 않았습니다. upstream 라이선스는 Apache-2.0입니다.
- OTel SDK/API/exporter `1.43.0`. `requirements.lock`은 실습 패키지의 해결된 버전을
  고정합니다. 프로젝트 torch의 전체 의존성을 고정하는 파일은 아닙니다.
- Jaeger `1.76.0` 이미지 digest는 `compose.yaml`에 고정했습니다.
- 실제 실행: Python `3.11.15`, torch `2.14.1+cpu`.
  자체 `.venv`에 Lens/OTel을 설치하고 `.pth`로 프로젝트 site-packages를 읽기 전용
  재사용합니다. 자체 설치 패키지가 우선합니다. 프로젝트 `.venv`나 루트 의존성은
  수정하지 않습니다.
- 실행 전 OTEL/Lens/W&B/scheduler/proxy 환경 설정을 제거합니다. metric/log는
  비활성화하고 인증 정보 없는 loopback HTTP exporter를 명시합니다. 원격 endpoint는
  거부하며 `.env`를 읽지 않습니다. 명시적인 requests Session의 `trust_env=False`로
  주변 netrc 인증 정보와 환경 proxy를 조회하지 않습니다. `allow_redirects=False`를
  강제하여 307/308의 trace 본문 재전송을 포함한 모든 3xx 응답을 export 실패로
  처리하며 두 번째 요청을 보내지 않습니다. 임의 비밀값 검출 도구라는 의미는 아닙니다.

## 실행

프로젝트 CPU `.venv`와 `uv`가 준비된 상태에서 저장소 루트부터:

```bash
cd tutorials/nemo_lens
python3 setup_env.py                  # 이미 있는 실습 .venv는 덮어쓰지 않음
ss -ltn                              # 사용할 포트가 비어 있는지 확인
docker compose up -d
.venv/bin/python verify_jaeger.py      # 실제 API 요청으로 제한 시간 내 준비 상태 확인
.venv/bin/python train.py --mode otlp --steps 20 --output runs/my-smoke
.venv/bin/python verify_jaeger.py --result runs/my-smoke/result.json
```

실제 시작 전 UI `127.0.0.1:16687`, OTLP gRPC `127.0.0.1:14317`,
HTTP `127.0.0.1:14318` 포트가 비어 있음을 확인했습니다. 여기서는 HTTP
`/v1/traces`를 사용합니다. 포트를 바꾸면 compose host 포트와
`--endpoint http://127.0.0.1:PORT/v1/traces`, `--api http://127.0.0.1:UI_PORT`도
같이 바꾸세요. 기존 결과 디렉터리를 덮어쓰지 않습니다.
Docker 프로젝트 이름은 `nemochat-lens-tutorial`이며 다른 컨테이너는 건드리지 않습니다.

## 코드 읽기와 UI 확인

1. `setup_env.py`: torch 재설치 없이 별도 환경 생성.
2. `train.py`: seed 고정, CPU thread 1개, 고정 tensor와 작은 MLP.
   소비자 코드가 `run`, `step`, `phase` 그룹을 `SpanRegistry`에 먼저 등록하고
   `setup_telemetry`를 호출합니다. `managed_span`으로
   `train.run → train.step → train.forward/backward/optimizer`를 중첩합니다.
   `nemo.run.id`, rank, CPU 속성으로 실행을 구분합니다. 종료 시 `handle.shutdown()`으로 배출합니다.
3. `verify_jaeger.py`: 해당 trace ID의 모든 span 수, operation, 실행/service 속성,
   부모-자식 연결을 검증하고 API 전체 응답과 검증 영수증을 저장합니다.
4. `benchmark.py`: subprocess 격리 비교와 실제 exporter 완료 검증.

브라우저에서 **http://127.0.0.1:16687**를 열고 Search 화면에서 service
`nemochat-lens-cpu`, operation `train.run`, Last Hour를 선택하여 Find Traces를 누릅니다.
trace를 열어 `train.run`, `train.step`을 펼치고 세 phase와 `train.step` tag,
process tag의 `nemo.run.id`, `nv.dl.rank`, `tutorial.device`를 확인합니다.
`result.json`의 trace ID로 `/trace/<trace_id>`에 직접 접근할 수도 있습니다.
막대는 계측 비용을 포함한 CPU 코드 블록 wall time이지 GPU 커널 시간이 아닙니다.

원격 머신이면 **사용자 노트북에서** 다음을 실행하세요:

```bash
ssh -N -L 16687:127.0.0.1:16687 ubuntu@YOUR_HOST
```

노트북의 `http://127.0.0.1:16687`로 접속합니다. UI를 공인 인터페이스에 노출하지
마세요. 부모가 API 수집과 실제 Chromium 렌더링을 검증했습니다. Screenshot은
private ignored host-local artifact이며 배포 파일로 포함하지 않습니다.
[부모 실행 검증](../../docs/ko/OBSERVABILITY_VERIFICATION.md)을 참고하세요.
Jaeger는 메모리 저장이므로 재시작하면 trace가 사라집니다. 이 실습만 종료:

```bash
docker compose down
```

## RL Rollout / Tool 병목 진단

[Rollout 실습](rollout/README.ko.md) ([English](rollout/README.md))에서 여러 trajectory의 generation 대기·모델 생성·실제 HTTP tool·검증과 마지막 완료 trajectory를 본다. Parent/child·동시 실행 span을 중복 합산하지 않는다. 의도적인 delay는 교육용 fixture이며 실제 NeMo RL/vLLM/Gym 성능이 아니다. [Native RL/Gym 연동 가이드](../../docs/ko/RL_ROLLOUT_OBSERVABILITY.md)에서 기존 span group·HTTP context·connection-pool wait·coverage 한계를 확인한다.

## 오버헤드 실험

```bash
.venv/bin/python benchmark.py --output runs/my-benchmark --steps 400 --warmup 40 --repetitions 8
# Docker 없이 baseline/disabled/file만 실행; OTLP 미실행을 명시적으로 기록
.venv/bin/python benchmark.py --output runs/my-offline --skip-otlp
.venv/bin/python -m pytest -q test_tutorial.py
uv pip check --python .venv/bin/python
```

각 arm은 **새 Python 프로세스**에서 실행합니다. baseline은 Lens를 import하지 않고
wrapper도 호출하지 않습니다. disabled는 동일 wrapper를 통과하지만 no-op provider를
사용합니다. file은 OTel `ConsoleSpanExporter` 출력을 JSONL 파일로 쓰며 OTLP는
로컬 Jaeger로 실제 전송합니다. 동일한 비계측 warmup 업데이트 이후 측정합니다.
import/모델 생성은 training 시간 밖이며 SDK setup, shutdown, 전체 프로세스 시간은
별도 기록합니다. 비동기 export가 학습과 겹칠 수 있으므로 shutdown만으로 전체
export 비용을 해석하면 안 됩니다.

기본 8회는 실행 순서를 회전시켜 네 위치를 각각 두 번 배정합니다. 실행 순서,
모든 raw duration, median, baseline median 대비 차이를 저장합니다. 모든 loss 값과
최종 parameter SHA256이 정확히 같아야 합니다. file span이 완전해야 하며 모든
OTLP 반복이 API 수집 검증을 통과해야 합니다. 누락 export는 빠른 성공으로 처리하지
않고 실패합니다. queue는 65,536이며 훨씬 큰 실험에는 별도 drop/queue 분석이 필요합니다.
이 결과는 단일 머신의 잡음이 있는 작은 CPU workload 측정입니다. 순수 wrapper와
직렬화/스케줄링 비용을 분리하지 않았으며 일반 SDK 오버헤드나 GPU/분산 성능으로
확장할 수 없습니다. **zero overhead라고 주장하지 않습니다.**

## 과거 실행 증거 (exporter 개인정보 보호 수정 전)

아래 측정값과 `executed_results.json`은 변경하지 않은 **수정 전** 결과입니다.
강화한 Session의 성능 측정이 아니며 이번 개인정보 보호 수정에서는 새
오버헤드 benchmark를 실행하지 않았습니다.

이 디렉터리에서 실행한 명령(저장소 전체 테스트가 아님):

```bash
.venv/bin/python train.py --mode otlp --steps 20 --output runs/smoke
.venv/bin/python verify_jaeger.py --result runs/smoke/result.json
.venv/bin/python benchmark.py --output runs/benchmark-8x400 --steps 400 --warmup 40 --repetitions 8
.venv/bin/python -m pytest -q test_tutorial.py
uv pip check --python .venv/bin/python
.venv/bin/python -m compileall -q train.py benchmark.py verify_jaeger.py setup_env.py test_tutorial.py
docker compose config --quiet
```

**9 tests passed** 및 실습 의존성 호환 확인. smoke **81 span**, 8개 OTLP 반복 각각
**1,601 span**의 실제 수집/계층/identity를 확인했습니다. 32개 subprocess의 loss
전체와 parameter hash가 정확히 일치합니다.

| 모드 | 400 업데이트 median 학습(ms) | median shutdown(ms) | baseline 대비 학습 비용 |
|---|---:|---:|---:|
| baseline | 176.995 | 0.000315 | 기준 |
| disabled wrapper | 187.138 | 0.012530 | +5.73% |
| file 활성화 | 426.042 | 6.384910 | +140.71% |
| 로컬 OTLP 활성화 | 323.105 | 3.955455 | +82.55% |

이 작은 workload에서는 단계당 4 span의 비용이 큽니다. 더 큰 연산이나 거친 그룹
설정은 새로 측정해야 합니다. JSON 파일 formatting은 여기서 OTLP보다 비쌌습니다.

- 버전 관리 가능한 [실측 요약과 raw duration](executed_results.json).
- 머신 내 전체 결과(ignored): `runs/smoke/{result,jaeger_trace,jaeger_receipt}.json`,
  `runs/benchmark-8x400/benchmark.json`.
- 각 반복 디렉터리에는 loss, stdout/stderr, file span 또는 전체 API 응답/영수증이
  있습니다. 재실행은 다른 출력 경로를 사용하세요.

## Exporter 개인정보 보호 수정 검증

새 로컬 검증이며 시간 비교 실험이 아닙니다:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider test_tutorial.py test_exporter_privacy.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python verify_jaeger.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python train.py --mode otlp --steps 20 --output runs/privacy-fix-smoke
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python verify_jaeger.py --result runs/privacy-fix-smoke/result.json
```

**15 tests passed** (기존 9개 + 개인정보 보호 회귀 6개). 회귀 테스트는 netrc 조회,
환경 proxy, HTTP transport를 mock하고 가짜 인증 정보와 본문만 사용합니다.
네트워크 요청이나 실제 인증 정보 읽기는 없습니다. 명시적으로 redirect를 요청해도
307/308은 export 실패이며 loopback POST는 정확히 한 번임을 확인합니다.
별도의 실제 로컬 smoke에서 **81 span**의 identity/계층을 검증했습니다.
trace: `a9e445e371045b8abdcfae78165c65c3`.
새 산출물: `runs/privacy-fix-smoke/{result,jaeger_trace,jaeger_receipt}.json`.
새 성능 주장은 하지 않습니다.

SDK 설치/import, 실제 optimizer, 정확성 동등성, 로컬 완전 export, readiness와
집중 테스트는 실행했습니다. GUI 렌더링, GPU/Nsight, CUDA source attribution,
네이티브 NemoChat 통합, 운영용 관측 시스템은 이 실습으로 검증되지 않았습니다.
