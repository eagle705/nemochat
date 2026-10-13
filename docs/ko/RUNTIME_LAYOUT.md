# 저장소 루트 런타임 디렉터리 규약

[English](../en/RUNTIME_LAYOUT.md) · [README](../../README.md)

## 네 가지 런타임 루트

CLI/config의 상대 경로는 config나 문서 위치가 아니라 **저장소 루트 working directory** 기준입니다. Tutorial이 절대 루트 경로를 명시하지 않는 한 예시는 저장소 루트에서 실행합니다.

| 루트 | 내용 | 예시 |
|---|---|---|
| `datasets/` | 다운로드한 입력 shard/corpus, 준비·토큰화한 데이터, 생성한 평가 입력 | `datasets/climbmix/<run>/`, `datasets/niah/<run>/fixtures.jsonl` |
| `checkpoints/` | 다운로드한 pretrained 모델/tokenizer, 학습이 생성한 raw 재개 snapshot, model-only export/WSM merge, 포장·복원 번들 | `checkpoints/pretrained/<model>/`, `checkpoints/cpu/<run>/`, `checkpoints/exports/<run>/` |
| `results/` | 실험 보고서, raw prediction, metric, receipt, profiler trace, screenshot, plot | `results/training_perf/<run>/`, `results/niah/<run>/`, `results/observability/<run>/` |
| `logs/` | 텍스트/stdout/stderr 로그와 local/offline W&B 로그 | `logs/<run>/train.log`, `logs/wandb/` |

이는 versioned source가 아닌 로컬 런타임 산출물입니다. 체크인한 입력 config는 `configs/`, 작성한 test/source fixture는 기존 위치에 둡니다. Fixture를 런타임 루트로 옮기지 않습니다. 실행마다 고유 이름을 쓰고 실패한 실행을 포함한 기존 산출물을 덮어쓰지 않습니다.

`--output`은 도구별 산출물 종류를 뜻하며 **항상 보고서는 아닙니다**. `prepare_data.py --output`은 `datasets/`, `train_cpu.py --output`·`--checkpoint-dir`·`--resume`은 `checkpoints/`, 평가/benchmark/report 출력은 `results/`에 둡니다. 다운로드한 checkpoint 입력과 학습이 생성한 checkpoint는 같은 루트의 별도 하위 폴더로 구분하고 provenance를 기록합니다. Optimizer/RNG를 포함한 raw checkpoint가 재개 원본이며 model-only export와 WSM merge는 대체할 수 없습니다. Long-context schedule cursor는 `results/`의 계획 산출물이지 학습 완료 증거가 아닙니다.

CPU checkpoint 안의 history/config/inventory, experiment 안의 resolved config처럼 도구가 하나의 번들로 만드는 파일은 함께 보존합니다. 확장자별로 나누기 위해 inventory payload를 분리하거나 수정하지 않습니다.

## 명시적 CLI 경로와 기존 schema

다음은 경로 사용 예시입니다. 새 이름을 선택하고 환경 설치·다운로드 승인은 별도로 처리합니다.

```bash
mkdir -p results/planning logs/my-run
.venv/bin/python scripts/estimate_corpus.py \
  --config configs/planning/corpus_local_smoke.json \
  --output results/planning/my-corpus-estimate.json
.venv/bin/python scripts/train_cpu.py \
  --texts tests/fixtures/texts.jsonl --steps 2 \
  --output checkpoints/cpu/my-step-2 > logs/my-run/train.log 2>&1
.venv/bin/python scripts/train_cpu.py \
  --texts tests/fixtures/texts.jsonl --steps 2 \
  --resume checkpoints/cpu/my-step-2 --output checkpoints/cpu/my-resumed
.venv/bin/python scripts/run_experiment.py dry-run \
  --config configs/experiments/dense_cpu.yaml \
  --output results/training_perf/my-dense-control
```

도구가 요구하는 **부모 디렉터리**만 미리 만듭니다. No-overwrite 도구가 직접 예약해야 하는 최종 폴더는 미리 만들지 않습니다. Shell redirection도 기존 로그를 truncate할 수 있으므로 새 log/run 경로를 사용합니다. Experiment YAML schema 1에는 output 경로 필드가 없습니다. `--output`을 전달하고 `output_dir`, dataset 경로나 새 환경 변수를 만들어 넣지 않습니다. W&B는 기존 `NEMOCHAT_WANDB_DIR=logs/wandb` 또는 지원되는 CLI/config directory 설정을 사용하며 명시적으로 활성화하지 않으면 disabled입니다. [Experiment 계약](AGENT_EXPERIMENT_CONTRACT.md), [checkpoint pipeline](HF_CHECKPOINT_PIPELINE.md), [W&B 단계](WANDB_STAGES.md)를 참고합니다.

자체 폴더에서 실행하는 tutorial은 루트 변수와 절대 산출물 경로를 사용합니다.

```bash
# tutorials/nemo_lens/ 안에서; 다른 checkout이면 루트를 맞춥니다.
REPO_ROOT=$(git rev-parse --show-toplevel)
# Tutorial 문서의 interpreter/명령과 기존 --output 옵션을 사용합니다.
# <tutorial command> --output "$REPO_ROOT/results/observability/<unique-run>"
# 그 외 지원 경로/redirection에는 "$REPO_ROOT/datasets/...",
# "$REPO_ROOT/checkpoints/...", "$REPO_ROOT/logs/..."를 사용합니다.
```

## Hugging Face cache는 별개

이 규약은 HF Hub/Transformers/Xet cache, `from_pretrained`의 model ID, 외부 upstream checkout/가상환경을 **자동 이동하지 않습니다**. 사용자가 지원되는 upstream 설정인 `HF_HOME` 등을 명시하지 않으면 기존 cache 기본값을 유지합니다. 해당 CLI가 지원하는 명시적 download 목적지나 local model 경로를 사용합니다. 존재하지 않는 cache 옵션을 만들거나 `checkpoints/`가 공유 HF cache를 대체한다고 가정하지 않습니다. 데이터 준비 과정의 임시 download/cache 처리도 각 도구 계약을 따릅니다.

## 과거 증거와 migration 호환성

과거 append-only log, timestamp, 측정 결과 및 receipt 경로는 변경하지 않습니다. 검증 기록의 `runs/`, `outputs/`, `exports/`, `figures/`, `/tmp/...`, host 절대 경로는 원래 실행 위치이지 새 실행의 권장 목적지가 아닙니다. 해당 검증 보고서와 저장소 runtime migration이 생성한 manifest의 old → new 로컬 경로 매핑을 참고합니다. 과거부터 새 규약을 사용했던 것처럼 증거를 고쳐 쓰지 않습니다.

로컬 migration은 payload bytes와 inventory 상대 경로를 보존해야 합니다. Remote Hub receipt의 `path`, 불변 `revision`, hash, stage/run namespace는 로컬 폴더 예시가 아닌 영구 handle이므로 기록된 값을 그대로 사용합니다. Migration은 remote namespace 변경, 검증 receipt 재발행, 사라진 임시 파일 복구 또는 checkpoint 재개 검증을 의미하지 않습니다. 새 경로 확보를 위해 기존 산출물을 덮어쓰거나 삭제하지 않습니다.

## 실제 이동 및 로그 검증

현재 호스트의 이동 내역은 `results/runtime-layout-migration.json`에 있다. 기존 720개 파일(72,960,623 bytes)의 SHA-256이 모두 일치하며 147개 이동에 대한 상대 호환 링크가 정상이다. 보존 여부와 로그 경로 변경은 독립 readonly 재검토 PASS. `results/runtime-layout-verification.json`에는 옮긴 WSD checkpoint 재개 후 정확한 상태 일치 등 실제 실행 증거를 기록했다.

실험 runner의 stdout/stderr는 저장소 루트 `logs/experiments/<unique-id>/`에 기록하고 `result.log_directory`로 참조한다. Lens benchmark는 루트 `logs/nemo_lens/<output-name>/` 또는 명시적 `--log-output`을 사용하며 결과·로그 디렉터리는 새 경로이고 서로 겹치면 안 된다. W&B는 기본 disabled이며 활성화 후 별도 override가 없으면 CWD-relative `logs/wandb`를 사용한다. 이번 경로 변경은 stage·commit·push하지 않았다.
