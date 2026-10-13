# 에이전트 실험 계약 (스키마 1)

이 경로는 **오프라인 사용자 정의 PyTorch dense 대조군**입니다. 네이티브
NemoChat hybrid, Bridge/MCore, MoE/Mamba, 분산 학습, 모델 품질 또는 속도 향상을
검증하지 않습니다. 모델·토크나이저·데이터 다운로드나 온라인 추적이 필요하지
않습니다. 정확한 규약은 [영문 원문](../en/AGENT_EXPERIMENT_CONTRACT.md)과 같습니다.

## 명령과 출력

저장소 루트에서 실행합니다. validate/dry-run은 Python과 PyYAML만 필요하며
torch/CUDA를 가져오거나 하드웨어를 확인하지 않습니다. **출력 파일을 쓰지
않습니다**. run에는 torch와 새로운 출력 디렉터리가 필요합니다.

```bash
.venv/bin/python scripts/run_experiment.py validate --config configs/experiments/dense_cpu.yaml
.venv/bin/python scripts/run_experiment.py dry-run --config configs/experiments/dense_cuda.yaml
.venv/bin/python scripts/run_experiment.py run --config configs/experiments/dense_cpu.yaml --output runs/dense-new
.venv/bin/python -m pytest -q tests/test_experiments.py
```

run의 마지막 디렉터리는 기존 파일/디렉터리/심볼릭 링크가 없어야 합니다.
부모 디렉터리는 만들 수 있지만 최종 디렉터리는 배타적으로 예약합니다.
실패/차단된 디렉터리도 재사용하거나 덮어쓰지 않습니다. dry-run의 --output은
표시용이며 생략하면 자리표시자 경로를 보여 줍니다. 기본값은 없습니다.
CPU 예제는 batch 2, sequence 16, hidden 32, FFN 64, layer 1, vocabulary 128,
warmup 1, measured 3, repeat 2, profile false, 60초, GPU 0의 작은 대조군입니다.

## 필수 필드와 상한

각 매핑에 아래 키가 **모두, 그리고 오직 이 키만** 있어야 합니다.
정수 자리에 bool을 허용하지 않으며 bool은 실제 YAML bool이어야 합니다.

| 영역 | 필드 | 허용 값 |
|---|---|---|
| root | schema_version | 정수 1 |
| experiment | name | ASCII 영문/숫자/`_ . -`, 1–64자; 첫 자는 영문/숫자 |
| experiment | seed | 정수 0–4294967295 |
| workload | device | cpu 또는 cuda |
| workload | dtype | float32 또는 bfloat16; CPU는 float32만 허용 |
| workload | batch_size | 정수 1–64 |
| workload | sequence_length | 정수 1–4096 |
| workload | hidden_size | 정수 1–4096, 4의 배수 (명시적 bounded-control 계약; 아키텍처 필수 조건 아님) |
| workload | ffn_size | 정수 1–16384 |
| workload | layers | 정수 1–24 |
| workload | vocab_size | 정수 2–65536 |
| workload | learning_rate | 유한한 숫자, 0 초과 1 이하; 정수 1도 허용 |
| workload | activation_checkpointing | bool |
| measurement | warmup_steps | 정수 0–100 |
| measurement | measured_steps | 정수 1–1000 |
| measurement | repeats | 정수 1–20 |
| measurement | profile | bool |
| resources | max_seconds | 정수 1–3600 |
| resources | max_gpu_count | 0 또는 1; CPU는 0, CUDA는 1 필수 |

B=batch, S=sequence, H=hidden, F=FFN, L=layers, V=vocabulary로 두면:

- `(warmup_steps + measured_steps) * repeats <= 2000`
- `L*(4*H*H + 2*H*F) + 2*V*H <= 100000000`
- `B*S*V`, `B*S*max(H,F)*L`, `B*4*S*S*L` 각각 `<= 16777216`

모델은 **single-head** causal attention입니다. `B*4*S*S*L`은 실제 head 개수나
측정 할당량이 아니라 단일 head 기준의 4배 보수적 안전 상한입니다.

이 상한은 정확한 최대 메모리 보장이 아닙니다. optimizer/gradient/workspace/
profiler 메모리 때문에 유효한 설정도 OOM으로 차단될 수 있습니다. GPU 개수
예산은 VRAM 제한이나 스케줄러 예약이 아닙니다. CPU worker는 CUDA를 보지
못하며 CUDA worker는 기존 가시성 목록의 첫 GPU만 사용합니다(목록이 없으면
GPU 0). CUDA 가용성과 bfloat16 지원은 run에서만 확인합니다.

## 입력 및 실행 안전성

UTF-8 YAML 한 문서, 최대 32768바이트, 매핑/시퀀스 중첩 최대 8단계만 허용합니다.
SafeLoader에 중복 키 거부를 추가했으며 매핑 키는 고유 문자열이어야 합니다.
명시적 태그(안전한 태그도 포함), alias, anchor를 거부합니다. 명령/모듈 경로/
환경 변수/다운로드/데이터 주소/shell 문법을 지정하는 필드는 없습니다.
YAML 암묵적 타입 규칙은 유지됩니다. bool은 true/false를 쓰고 필요한 문자열은
인용하십시오. NaN/Infinity 학습률은 거부합니다.

직접 benchmark JSON도 동일한 필드 검증기를 사용하며 입력은 최대 32768바이트입니다.
비교 JSON은 산출물당 최대 8 MiB입니다. 중복 키, NaN/Infinity/지수 overflow를
거부하고 UTF-8/JSON 오류, parser 재귀 및 거대 정수 변환 오류는 출력 생성 없이
깨끗한 CLI 거부로 반환합니다.

원본을 한 번 읽고 정확한 바이트의 SHA256을 기록합니다. 메모리의 resolved
설정은 frozen dataclass와 재귀 불변 매핑이며 반환 dict는 복사본입니다.
resolved_config.json에는 합의한 스키마만 담고 한 번 쓴 뒤 0444 읽기 전용으로
설정합니다. 이는 owner/root에 대한 보안 경계가 아닙니다. SHA256은 result.json에
저장됩니다.

유일한 실행 대상은 현재 Python의 고정 benchmark_training.py입니다:

```text
<current-python> <repository>/scripts/benchmark_training.py --config <output>/resolved_config.json --output <output>/measurement
```

argv 리스트, shell=False, 저장소 cwd를 사용합니다. .env를 읽지 않고 환경 전체를
기록하지 않습니다. 호출자 환경을 상속하되 CUDA_VISIBLE_DEVICES 및
HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, WANDB_MODE=disabled를 적용합니다.
이 플래그는 임의로 변경된 코드에 대한 네트워크 sandbox가 아닙니다. 신뢰하는
로컬 코드만 실행하며 custom worker는 로컬 합성 텐서만 사용해야 합니다.

worker 제한 시간은 torch/하드웨어 사전 확인 뒤 남은 max_seconds입니다.
초과 시 새 POSIX 프로세스 그룹 전체에 SIGTERM, 이어 SIGKILL을 보내고 부모를
회수합니다. 정리에 최대 약 1초가 추가되며 로컬 import/사전 확인은 강제 중단되지
않습니다. 의도적으로 그룹을 탈출하는 프로세스까지 sandbox하지 않습니다.
Linux/POSIX 대상입니다. Ctrl-C 시 그룹을 정리하지만 완료된 실험 보고서는 아닙니다.

## 상태, 산출물, 증거 라벨

예약된 run은 resolved_config.json, result.json, report.md를 기록합니다. 실행된
worker에는 stdout.log, stderr.log와 worker가 만든 measurement/도 있습니다.
사전 차단에는 로그/측정이 없을 수 있습니다. 저장소 오류로 쓰기가 불가능하면
보고서를 만들었다고 주장하지 않습니다.

| 종료 코드 | 상태 | 의미 |
|---|---|---|
| 0 | success / valid / dry_run | 실제 worker 완료 또는 설정 전용 확인 완료 |
| 1 | failed | 3/4 이외의 비정상 worker 종료 또는 일반 실행 예외 |
| 2 | invalid | 계약/입력 오류, run 출력 누락, 기존 출력, I/O 오류 |
| 3 | blocked | torch/CUDA/bfloat16 불가, worker 코드 3, 감지된 OOM |
| 4 | timeout | 제한 시간 소진/worker 시간 초과 |

worker JSON stdout의 `out-of-memory`는 종료 코드 3 → runner blocked,
`time-budget-exceeded`는 코드 4 → runner timeout으로 매핑하며 정확한 reason을
보존합니다. worker 측정은 status failed와 failure_kind를 유지합니다.
보조 경로로 마지막 stderr 65536바이트에서 out of memory/OutOfMemoryError를 감지합니다.
설명 없는 OS kill을 OOM으로 추정하지 않습니다. timeout이 부분 OOM 로그보다
우선합니다. result에는 status, exit_code, benchmark_returncode(미실행은 null),
reason, source_sha256, config, argv, elapsed_seconds, evidence_label이 있습니다.
실패한 측정의 처리량/loss를 만들어 내지 않습니다.

모든 보고서 라벨은 `custom_offline_dense_control; native hybrid NOT validated`입니다.
preview는 **설정 전용 증거**, 단위 테스트는 **CPU 계약/runner 증거**, 실제 완료된
worker는 해당 장치의 **사용자 정의 dense 측정 증거**입니다. 네이티브 hybrid,
분산 정확성, WSD/WSM 재개, 품질 또는 전체 학습 개선의 증거가 아닙니다.
프로파일 실행을 비프로파일 처리량 A/B로 바꾸어 부르면 안 됩니다.
[네이티브 성능 후속 게이트](PRETRAIN_PERFORMANCE_LAB.md)를 별도로 수행하십시오.

테스트는 엄격 필드/YAML/예산, 불변 설정, CLI 무쓰기, 배타 예약, 모든 상태 및
실제 로컬 자식 그룹 종료를 확인합니다. mocked worker 상태는 성능 측정이
아닙니다. 실제 worker가 준비된 후 새 디렉터리에서 실행하고, CUDA 미실행은
미검증으로 유지하십시오.
