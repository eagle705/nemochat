# NeMo Lens ↔ Nsight Systems 브리지 MVP

[정식 영문 문서](README.md). 별도 웹 앱이 아니라 **기존 Nsight 안에서 읽을 수
있는 NVTX 역할/설명/식별자**와 로컬 소스 증거를 연결하는 명시적 opt-in CPU
교육용 구현입니다. NeMo Lens는 OTel 계측 라이브러리이지 GPU 프로파일러가
아닙니다. Nemotron의 네이티브 fused MoE나 Megatron 분산 EP 구현도 아닙니다.

## 검증 범위

- 실제 CPU sparse MoE forward/backward 및 token-major 참조와 출력/gradient
  비교를 실행했습니다. 토큰 8, expert 4, top-2입니다.
- 고정 Lens 커밋 `4d045c03186b90afdc815db46b671f6a21bc7e98`로 실제 로컬 OTel
  span 21개와 operation record 21개의 공통 bridge/source ID를 확인했습니다.
- 실제 Python 소스 snapshot, SHA256, 파일/행 범위/스니펫 검증과 경로 이탈,
  변조, 현재 checkout 변경, 로드한 코드와 디스크 코드 불일치를 테스트합니다.
- CUDA NVTX 코드는 구현했지만 이 호스트는 GPU와 `nsys`가 없어 **실제 NVTX
  캡처/설치된 plugin 검증/커널↔소스 대응/GUI 동작은 미검증**입니다.
- 네이티브 Nsight 설명/코드 패널은 **미구현**입니다. 공개 configuration 또는
  collection plugin API를 커스텀 GUI 위젯 API로 설명하지 않습니다.

## CPU 실행

**소스 트리 전용 설치/사용:** 이 checkout에서 실행하거나 checkout을 유지하는
editable 설치를 사용하세요. 현재 wheel에는 Nsight manifest가 포함되지 않으며
`cli.BASE/nsight`는 소스 트리 구조를 가정합니다. 독립 wheel 설치를 지원되는
Nsight plugin 배포로 주장하지 않습니다. packaging/resource 지원은 미완료입니다.

```bash
cd /home/ubuntu/nemochat/plugins/nemo_lens_nsight_bridge
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m lens_nsight_bridge doctor
.venv/bin/python -m lens_nsight_bridge run --lens \
  --output /home/ubuntu/nemochat/results/my-lens-moe-cpu
.venv/bin/python -m pytest -q
```

Torch/pytest는 실행 환경에 있어야 합니다. 이 호스트의 무시되는 로컬 `.pth`는
프로젝트 `.venv`의 site-packages를 **읽기 공유**하며 프로젝트 환경을 변경하지
않습니다. 새 호스트는 플러그인 전용 환경에 Torch를 설치하거나 공유 인터프리터를
사용하세요. 기존 output 디렉터리는 거부합니다. `--lens`를 빼면 Lens import도
없는 source/operation 검증입니다. CPU 기본값은 NVTX 호출이 없습니다.
`--nvtx` 요청 시 CUDA/NVTX가 없으면 명확히 실패하며 fallback 성공으로
표시하지 않습니다. CUDA는 `--device cuda`로 명시합니다.

Lens exporter는 **로컬 파일 전용**, metrics/logs는 off입니다. OTLP collector나
외부 telemetry가 없습니다. 설치 때만 네트워크를 사용합니다. Lens/OTel 설정은
process-global이므로 새 프로세스에서 기존 provider보다 먼저 Bridge 하나를
생성해야 합니다. 기존 provider, SDK 비활성화, 기록되지 않는 sampler span은
거부합니다. 이미 실행 중인 native Lens 앱에 자동으로 끼워 넣지 않습니다.

## 실제 Nsight와 연결하는 방식

`nsight/nemo_lens_bridge.yaml`은 공식 `TargetEnvironment` configuration
manifest입니다. 환경 플래그만 설정하며 Python 자동 주입, `sitecustomize`,
`LibraryPath`, GUI 패널은 없습니다. 사용자 Python은 **명시적으로 Bridge를
import하고 해당 함수 호출을 감싸야 합니다**. 예:
`examples/explicit_import.py`. 플러그인 활성화만으로 임의 학습 코드가
자동 계측되는 것은 아닙니다.

```bash
# 실제 호환 Nsight 및 CUDA 호스트에서만 가능; 현재 호스트에서는 차단됨
.venv/bin/python launch.py launch \
  --output /home/ubuntu/nemochat/results/my-lens-moe-cuda
```

런처는 `NSYS_PLUGIN_SEARCH_DIRS`를 추가하고 실제 `--version`, `profile --help`,
`plugins --help`, `plugins list`를 확인한 후
`nsys profile --enable=nemo_lens_nsight_bridge --trace=cuda,nvtx ...`로
명시적으로 import하는 Python control을 실행합니다. help/plugin/GPU 검사 실패와
기존 output prefix를 거부합니다. `doctor`에는 원문 probe 출력/종료 코드 및
backtrace/sampling help 행이 포함됩니다. Python/CUDA backtrace 옵션은 버전별
실제 help와 sampling/correlation 캡처 검증이 필요합니다. 현재 검증된 옵션으로
주장하지 않습니다. 플러그인 검색 디렉터리는 신뢰하는 경로만 사용하세요.

기존 Nsight GUI에서 `.nsys-rep`를 열면 `MoE/activation: SwiGLU | bridge=... src=...`
형태의 NVTX host 범위를 볼 수 있도록 구현했습니다. 원래 CUDA 커널 이름은
바꾸지 않습니다. 세부 역할 카드는 [별도 로컬 문서](docs/ROLES.ko.md)이며
Nsight에 설치되는 패널이 아닙니다. 정확한 bridge ID를 복사해:

```bash
.venv/bin/python -m lens_nsight_bridge snippet \
  --output /home/ubuntu/nemochat/results/my-lens-moe-cpu \
  --bridge-id '<operation/NVTX의 정확한 ID>' --current-root "$PWD"
```

검증된 **실제 snapshot 소스**가 나옵니다. `--current-root`를 사용하면 현재
checkout 해시도 일치해야 합니다. 이를 빼면 현재 파일이 변경되어도 캡처된
snapshot만 검증해 읽습니다. source ID 자체가 정확한 커널의 Python 행을 증명하지
않습니다. 실제 CUDA launch correlation/backtrace/compiler 메타데이터가 필요합니다.
GUI 패널/에디터 이동은 이후 SDK 및 고객 워크플로 검증 게이트입니다.

## 소스/결과와 한계

`operations.jsonl`, `lens-spans.jsonl`, `sources/<file-sha256>.py`, 성공 시
`result.json`을 로컬 output에 저장합니다. NVTX push/pop 실패에도 error record
작성을 시도하며 `nvtx_error`에는 단계와 예외 타입만 기록합니다.
`nvtx_emitted`는 push가 성공적으로 반환했다는 뜻이지 실제 캡처 증명이 아닙니다.
pop 실패는 전파하지만 wrapped 함수의 원래 예외를 덮어쓰지 않습니다. 이미 예외가
전파 중이면 record I/O는 best-effort입니다. source root, compile한 코드 객체와
로드한 함수 일치, inspect 행/원본 bytes, 파일/스니펫 SHA256, reference identity,
1-based inclusive 행 범위, 선택적 현재 checkout을 검증합니다. 절대경로, `..`,
역슬래시, symlink 경로 이탈을 거부합니다. 일반 undecorated Python 함수 대상이며
동적/C-extension/변환된 코드는 추후 별도 adapter가 필요합니다.
해시는 우발적 변경 검출용이지 snapshot과 manifest를 함께 바꾼 공격자의
인증 증명이 아닙니다. 고객 코드 저장 권한/보존 정책을 검토하고 기본적으로
외부 업로드하지 마세요. 단일 프로세스 동기 구현이며 동시/분산 exporter가
아닙니다. 계측 및 snapshot 컴파일 오버헤드가 있으므로 성능 수치가 아닙니다.

한 Python operation에서 여러 eager 커널이 나올 수 있고 fusion은 여러 소스 op를
하나로 합칩니다. backward 진입 함수는 gradient 커널별 forward source가 아닙니다.
CUDA Graph replay는 현재 host stack과 캡처 시 stack이 다를 수 있습니다.
시간 근접/NVTX nesting만으로 역할을 추측하지 않고 미확인을 보존합니다.
참조 출력/gradient 검사는 routing weight를 공유하며 계측되지 않은 검증 연산도
실행됩니다. 모든 캡처 커널에 annotation이 있다는 주장은 하지 않습니다.

로컬 research의 Megatron 부분 snapshot에서 `moe_layer.py:39` NVTX import 및
`:873–875` `delayed_expert_wgrad` annotation을 확인했습니다. 전체 설치 런타임
inventory가 아닙니다. native adapter 전에 실제 Megatron commit 및 router/
dispatch/grouped experts/combine/backward NVTX와 Lens group을 조사하고 중복을
피해야 합니다. 분산/CUDA/오버헤드 검증은 미완료입니다. 전역 monkeypatch는 없습니다.

공식 Nsight manifest/discovery, 고정 Lens file exporter/setup API 링크와 실제
명령/결과는 [영문 문서](README.md), [작업 로그](WORK_LOG.md)에 있습니다.

## Runtime layout / 실행 자료 경로

Use repository-root `datasets/`, `checkpoints/`, `results/`, and `logs/`; see [runtime layout](../../docs/en/RUNTIME_LAYOUT.md) and [한국어](../../docs/ko/RUNTIME_LAYOUT.md). Relative CLI output paths use the caller working directory. Historical recorded paths/measurements below remain unchanged; prior host artifacts physically migrated, with relative compatibility links and SHA-256 inventory in root `results/runtime-layout-migration.json`. Tutorial benchmark stdout/stderr now go to root `logs/nemo_lens/<output-name>/` (override `--log-output`); use a fresh name for both results and logs.
