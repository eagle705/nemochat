# 사전학습 체크포인트의 공식 NeMo Gym 평가

Gym checkout 자체에서 실행할 때는 `NEMOCHAT_ROOT=/absolute/path/to/nemochat`처럼 NemoChat 루트를 명시합니다. 모델은 `"$NEMOCHAT_ROOT/checkpoints/exports/step_100/raw"`, 출력은 `"$NEMOCHAT_ROOT/results/gym/step_100/raw/ruler.jsonl"`을 사용하고 실제 materialized 입력은 `datasets/gym/<run>/`에 provenance와 함께 보존합니다. Gym-relative benchmark 입력과 upstream preparation 계약은 바꾸지 않습니다. 정확한 command template은 [영문 문서](../en/GYM_PRETRAIN_EVALUATION.md)에 있습니다.

새 CLI 실행은 저장소 루트 기준 경로와 새 산출물 이름을 사용합니다. [런타임 디렉터리 규약](RUNTIME_LAYOUT.md)에 따라 입력 데이터는 `datasets/`, checkpoint는 `checkpoints/`, 보고서·trace·plot은 `results/`, 텍스트/W&B 로그는 `logs/`에 둡니다. 필요한 부모 폴더만 만들며 기존 fixture/config와 과거 증거 경로는 보존합니다.

## 구현 범위와 실제 결과

`scripts/gym_eval.py`는 **공식 Gym 결과를 가져와 raw/WSM merged 체크포인트를 비교하는 도구**입니다. 새 scorer를 구현하거나 Gym 결과를 임의로 대체하지 않습니다. 모델·judge API 호출, 데이터 다운로드, upstream 준비 스크립트 실행도 하지 않습니다. 기존 `scripts/base_eval.py`의 CE/BPB 및 CORE 진단은 그대로 유지합니다. 생성 벤치마크와 다음 토큰 손실은 서로 다른 관측이므로 함께 보아야 합니다.

공식 저장소는 **NVIDIA-NeMo/Gym**, 고정 revision은 `804c12a7d56a151a98ae150a352350e3c856ba47`입니다. `NVIDIA/NeMo-Gym`이 아닙니다. 자세한 immutable source 링크, manifest 예제, 실행 계약은 [영문 문서](../en/GYM_PRETRAIN_EVALUATION.md)에 있습니다.

이번 환경에서 확인한 수준:

- **소스 검증 완료:** 고정 revision의 benchmark 설정, 공식 eval CLI, rollout 집계 코드, RULER pretraining completion profile 및 verifier.
- **CPU 테스트 완료:** 명시적으로 synthetic fixture라고 표시한 importer 단위 테스트 및 CLI 성공/실패 경로. 실제 모델 성능 측정이 아닙니다.
- **미검증:** 공식 Gym inference, NemoChat export의 vLLM 지원, 모델 endpoint, GPU 실행, 실제 benchmark score.
- 기본 Python 환경에서 `nemo_gym` 및 `gym` 실행 파일이 없습니다. upstream prepare를 실행하거나 judge 비용을 지출하지 않았습니다. 따라서 실제 사전학습 벤치마크 점수는 **미측정**입니다.

## 어떤 과제를 선택할 것인가

고정 `benchmarks/` 트리에는 RULER, MMLU 계열, GSM8K, MATH 계열, GPQA, AIME, IFEval, code·번역·멀티모달·agent 과제가 있습니다. 하지만 과제가 존재한다는 사실과 base checkpoint에 적합하다는 사실은 다릅니다.

| 과제 | 공식 평가 방식 | 사전학습 해석 |
|---|---|---|
| `ruler/config_pretrain` | chat template 없이 vLLM `/v1/completions`, 기존 RULER verifier | base/pretraining을 위한 명시적인 공식 경로. 모델이 실제로 지원하는 context 길이를 선택해야 합니다. |
| `mmlu` | 생성한 boxed 정답 문자, strict boxed MCQA scorer, 기본 1 repeat | 지식과 출력 형식 준수가 함께 측정됩니다. 선택지 likelihood 방식의 MMLU가 아닙니다. |
| `gsm8k` | math prompt, math-with-judge resource | 상속된 설정은 `should_use_judge: false`입니다. 이름만 보고 유료 judge가 필요하다고 단정하거나 임의로 활성화하면 안 됩니다. |
| `ifeval` | instruction-following 평가 | 정렬되지 않은 base model에는 지시 준수 confound가 큽니다. 기본 사전학습 지표로 권장하지 않습니다. |
| code/agent/multimodal | sandbox·tool·task별 backend | 각 설정과 모델 capability를 별도 검증해야 합니다. 전체 과제 지원을 주장하지 않습니다. |

이 pin에서 `benchmarks/hellaswag/config.yaml`, `benchmarks/arc_challenge/config.yaml` 직접 조회는 404였습니다. 이 과제들을 있다고 가정하거나 다른 과제로 몰래 교체하지 않습니다.

RULER `default` 모드는 질문에 answer prefix를 바로 붙이고, `base` 모드는 newline을 추가합니다. 별개의 프로토콜입니다. QA verifier는 reference 중 하나의 case-insensitive substring 일치를 사용하고, 다른 subset은 모든 reference의 일치 비율을 사용합니다. NIAH/VT/CWE/FWE/QA의 생성 한도는 각각 128/30/120/50/32 토큰이며 준비된 row에 들어갑니다. 이 공식 규칙을 유지합니다.

## 공식 실행 전제

별도 설치·검토한 Gym checkout을 지정 revision으로 고정합니다. 준비 스크립트는 virtualenv 생성, dependency 설치, 데이터 다운로드와 subprocess 실행을 할 수 있으므로 자동 실행하지 않습니다. 공식 local vLLM completion profile에는 `trust_remote_code: true`가 있으므로 checkpoint 코드를 검토하고 지원된다면 명시적으로 false로 설정합니다.

NemoChat training checkpoint를 해당 vLLM 버전이 실제 지원하는 model/tokenizer export로 먼저 변환해야 합니다. HF CPU inference 성공이 vLLM architecture 지원의 증거는 아닙니다. 일반 OpenAI chat client도 Gym의 Responses API model-server/agent/environment 경로를 자동으로 대체하지 못합니다.

영문 문서의 `gym eval prepare --benchmark ruler/config_pretrain` 및 `gym eval run ...`은 **소스로 확인한 실행 템플릿**이며 이번 환경에서 실행한 benchmark 명령이 아닙니다. 실제 모델 context에 맞는 길이를 지정합니다. raw/merged 두 모델에 동일한 tokenizer와 byte-identical input을 사용하고, randomized RULER dataset을 각각 재생성하지 않습니다. 출력 디렉터리는 분리하고 해당 training step의 두 weight export를 명시합니다.

공식 aggregate artifact는 `/aggregate_metrics` 호출 결과이며 agent별 `agent_metrics`, `key_metrics` 등을 포함합니다. `*_benchmark_metrics.json` 준비 파일은 데이터셋 통계일 수 있고 모델 점수가 아닙니다. 예를 들어 `ruler.jsonl`의 집계 파일 경로는 `ruler_aggregate_metrics.json`입니다. 실제 command output과 내용에서 확인합니다.

## Manifest 작성과 비교

`configs/eval/gym_pretrain.json`은 runs가 비어 있는 **미측정 템플릿**입니다. 그대로 실행하면 실패해야 정상입니다. 실제 결과가 생긴 후 각 `checkpoint_step × task × variant(raw/merged)`에 한 record를 작성합니다. 필드는 영문 문서의 JSON 예제를 따릅니다.

- `checkpoint`: 실제 weight 파일 또는 모든 shard hash를 포함한 canonical inventory 파일.
- `tokenizer`: tokenizer와 special-token/config 파일 전체를 식별하는 파일 또는 inventory.
- `input_jsonl`: 실제 collection에 사용한 input. 전처리가 바뀌었다면 이전 준비 파일과 혼동하지 않습니다.
- `rollouts`, `metrics`: 공식 Gym rollout JSONL 및 공식 aggregate JSON.
- `metric_pointer`: 실제 집계 파일의 숫자 지표에 대한 JSON pointer. 단일 agent artifact만 허용합니다. 예: `/0/agent_metrics/mean~1reward`.
- `protocol_files`: 실제 prompt/scorer/serving 설정 snapshot. raw/merged 모두 동일한 순서와 내용.
- `sample_count`, `num_repeats`: 양의 정수. cap을 사용했다면 full leaderboard 결과가 아니라 partial scope로 표시합니다.

`protocol.json`에는 `generation`, `context_length`, `prompt_mode`, `scorer`, `scope`, `runtime`, `wsm`을 기록합니다. 생성 seed/temperature/top-p/output·stop 한도, context/BOS/truncation, precision 및 runtime version을 빠짐없이 기록합니다. `wsm`은 두 arm이 공유하는 merge policy이며 window, 구성 체크포인트 hash, 정규화된 coefficient, 미래 checkpoint 미사용 cutoff를 명시합니다. 실제 merge를 적용하는 것은 merged arm뿐이지만 비교 provenance는 두 arm에 공유합니다.

```bash
PYTHONPATH=src .venv/bin/python scripts/gym_eval.py \
  --manifest results/gym/my-run/manifest.json \
  --gym-root /absolute/pinned/Gym \
  --output results/gym/my-run/gym_comparison.json
```

Importer는 source pin, checkout의 tracked 변경 여부, benchmark YAML, 각 step의 task coverage와 raw/merged pair, weight identity, input/tokenizer/protocol hash, rollout identity의 완전한 grid를 확인합니다. 누락·중복·masked·error·비유한 reward를 거부합니다. 실패한 실행을 0점으로 만들지 않고 살아남은 sample 교집합만 몰래 비교하지도 않습니다. 출력에는 원본 공식 점수, task별 `merged_minus_raw`, artifact 경로와 SHA-256이 들어갑니다. 공식 점수를 재계산하거나 서로 다른 scale의 task를 임의의 composite로 합치지 않습니다. 기존 출력 덮어쓰기도 거부합니다.

## 정직한 한계

저장된 metric과 실제 serving checkpoint 사이의 연결은 **사용자 선언에 의존**합니다. hash와 clean checkout은 어떤 모델이 답변을 생성했는지를 증명하지 않습니다. official launch log, resolved config, weight inventory, environment version, failure sidecar를 보존해야 합니다. importer는 조작된 upstream artifact, 누락된 inventory shard, 실제로 적용되지 않은 설정, 잘못 붙인 task 이름을 검출할 수 없습니다. snapshot을 upstream과 자동 대조하지 않으며 tracked-source 검사는 untracked Python 파일을 보장하지 않습니다.

`fixture: true`인 manifest는 `--allow-fixtures`가 있어야 읽을 수 있고 결과에도 `synthetic fixture only`라고 남습니다. 테스트용 숫자를 실제 모델 점수로 인용하지 않습니다. 영어 코드/설정을 canonical 구현으로 사용하고 이 문서는 해석과 운영상의 주의점을 설명합니다.
