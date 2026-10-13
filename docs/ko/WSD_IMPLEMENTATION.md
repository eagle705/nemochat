# WSD 구현: CPU HF 학습 제어 실험

## 구현 범위와 실험 의도

`scripts/train_cpu.py`에 warmup → stable → linear decay 학습률 정책과 주기적인 학습 체크포인트 저장을 추가했습니다. 이 코드는 여전히 **작은 Hugging Face 모델의 CPU 제어 실험**입니다. NemoChat 본체, NeMo Bridge 또는 Megatron-Core 사전학습을 구현하거나 검증한 것이 아닙니다.

사용자가 원하는 실험은 **하나의 WSD 사전학습 경로 + PR5114 WSM의 stable 구간 sliding-window 평가**입니다. 평가 시점마다 별도의 decay 분기를 학습하는 방식이 아닙니다. WSD는 학습률 정책이고 WSM은 오프라인 가중치 병합/평가이므로 역할을 분리했습니다. 이 스케줄러는 가중치 병합, 분기 실행, 평가 또는 병합 가중치로 학습 모델 교체를 하지 않습니다. 병합 도구는 별도 담당 코드입니다. [사전학습 실험 설계](../en/PRETRAIN_WSD_WSM_EXPERIMENT.md)를 함께 참고하세요.

## 경계 정의

`u`는 1부터 시작하는 **전역 optimizer update 번호**, `W/S/D`는 각 구간 길이, `T=W+S+D`는 전역 학습 horizon, `p`는 peak LR, `r`은 최저 LR 비율입니다. LR은 해당 optimizer update **직전**에 설정됩니다.

- `1 ≤ u ≤ W`: warmup, `p*u/W`.
- `W < u ≤ W+S`: stable, `p`.
- `W+S < u ≤ T`: linear decay, `p*(r+(1-r)*(1-(u-W-S)/D))`.

warmup과 decay는 길이 0이면 생략됩니다. stable은 최소 1 update가 필요합니다. 마지막 warmup update는 peak LR이더라도 phase는 warmup입니다. decay 길이가 1이면 그 update에서 즉시 floor에 도달합니다. 기본 `r=0`에서는 마지막 update의 LR이 0이어서 가중치는 이동하지 않지만 AdamW 상태는 진행합니다. 이산 update 경계를 명시적으로 정한 것이며 외부 구현과 수치적 동등성을 주장하지 않습니다.

`--steps`는 **이번 실행에서 추가할 update 수**이고, `--schedule-total-steps`는 실행 횟수와 독립적인 전체 horizon입니다. 후자는 `W+S+D`와 정확히 같아야 합니다. resume할 때 horizon을 새 실행의 steps로 다시 계산하지 않습니다. horizon 초과 실행, 음수/잘못된 phase 길이, NaN/Inf LR 및 floor 비율을 거부합니다. peak LR은 유한한 양수, floor 비율은 유한한 `[0,1]` 값이어야 합니다.

## 읽기 쉬운 코드와 정확한 resume

- `src/nemochat/training/wsd.py`: 작은 불변 설정 데이터 클래스와 직접 LR을 계산하는 함수. factory나 숨은 counter는 없습니다.
- `scripts/train_cpu.py`: update 직전 LR 설정, 실제 사용한 LR/phase 로그, update 경계 체크포인트 저장.
- `tests/test_wsd.py`: 구간 경계, 유한값 검증, CLI 거부, local tiny-HF fixture의 경계별 exact resume, 주기 저장 resume와 충돌 방지.

기본값은 기존과 동일한 constant LR입니다. 기존 contract의 data SHA256, context length, LR, model, revision 키를 변경하지 않았으므로 inventory가 없는 예전 체크포인트는 `--allow-unverified-legacy-resume`를 명시한 미검증 모드에서만 resume할 수 있습니다. WSD일 때만 `contract.lr_schedule`에 phase 길이, 전역 horizon, decay shape와 floor 비율을 기록합니다. 데이터나 설정을 바꾸면 resume을 거부합니다. WSD resume 명령에는 원래의 WSD 옵션을 모두 다시 지정해야 합니다. horizon 확장은 기존 exact-resume contract에서 허용하지 않습니다.

다음 LR은 `saved_step+1`의 순수 함수이므로 scheduler 상태 counter가 필요 없습니다. optimizer 텐서, Torch CPU RNG, deterministic chunk cursor와 처리한 target token 수를 기존 방식으로 복원합니다. stdout JSON과 `run.json`의 history에는 `learning_rate`, `lr_phase`가 추가됩니다. 예전 history에 이 필드가 없어도 위의 legacy opt-in 조건으로 resume할 수 있습니다.

## 주기 저장과 WSM 입력

`--checkpoint-every N --checkpoint-dir ROOT`를 함께 지정하면 **전역 update 번호가 N의 배수인 update를 완료한 뒤** `ROOT/step-NNNNNNNN`에 저장합니다. 마지막 `--output`도 항상 저장합니다. 모델/토크나이저, optimizer, 정확한 RNG와 cursor/token-count를 함께 저장하며 저장 작업이 RNG를 사용하더라도 원래 RNG를 복원합니다. 따라서 저장 빈도가 학습 경로를 바꾸지 않습니다.

기존 target 디렉터리가 있으면 실행 전 검사에서 거부하고, 실제 저장도 `exist_ok=False`로 덮어쓰기를 막습니다. 주기 저장 root와 마지막 output은 서로 포함 관계일 수 없습니다. 단일 writer용이며 원자적인 다중 파일 transaction은 아닙니다. 저장 실패 시 불완전한 디렉터리가 남을 수 있으므로 새 경로를 선택해야 합니다. 이 control이 사용하지 않는 Python/NumPy RNG, GPU RNG, distributed optimizer 또는 dataloader worker 상태는 저장/검증하지 않습니다.

실행한 8-update 예제는 `W=2, S=4, D=2`입니다. 주기 저장은 2/4/6/8 update에 발생했으며, **stable 구간에 속하는 것은 4와 6만**입니다. WSM stable sliding window에서는 warmup의 2와 decay의 8을 제외해야 합니다. 이 스케줄러 자체는 WSM을 실행하지 않았습니다. 병합 결과는 평가용 가중치 산출물이지 optimizer-bearing resume 체크포인트가 아닙니다.

## 체크포인트 identity 무결성

새 `run.json`에는 `checkpoint_format: cpu-hf-sha256-v1` 및 `artifacts_sha256`가 있습니다. SHA256 inventory는 **`run.json`을 제외한 모든 체크포인트 파일**을 포함합니다: 모든 모델 shard와 index, 모델/generation config, tokenizer 직렬화 파일/vocab/merges, tokenizer config 및 존재하는 special/added-token 파일, `optimizer_rng.pt`. 기존 constant contract 키는 그대로입니다.

resume은 **tokenizer/model/optimizer를 로드하기 전에** inventory를 검사합니다. 파일 변경/누락/추가, inventory entry 누락, 필수 모델/tokenizer/optimizer 누락, 잘못된 digest, 경로 traversal, symlink/디렉터리, 검증한 shard 밖을 가리키는 index를 거부합니다. 새 형식의 inventory가 손상/누락된 경우 legacy 플래그로 우회할 수 없습니다. 형식과 inventory가 모두 없는 예전 manifest는 `--allow-unverified-legacy-resume`가 필요하고 미검증 경고를 출력합니다. 이 실행의 새 산출물에는 inventory가 생기지만 과거 parent를 소급 검증하지는 않습니다.

SHA inventory는 **우발적인 변경/손상 및 token ID 교체를 검출**하는 용도이며, **악의적인 manifest 재작성 방지나 인증이 아닙니다**. `run.json`은 서명되지 않았습니다. 파일과 hash를 함께 바꾸거나 두 형식 필드를 제거해 legacy로 위장하는 공격은 보장 범위 밖입니다. 검증/로드 중에는 신뢰할 수 있는 단일 writer의 불변 디렉터리를 사용해야 합니다. 동시 변경 방지는 제공하지 않습니다.

이 파일들은 **HF CPU 체크포인트이며 Megatron DCP가 아닙니다**. stable snapshot도 Megatron DCP WSM merger 입력으로 사용할 수 없습니다. 실제 target-stack WSM에는 호환되는 Megatron DCP와 별도의 검증이 필요합니다.

## clean clone에서 재현하는 무결성 검증

untracked `runs/` 기록은 clean clone에 없습니다. 의존성을 `.venv`에 설치한 뒤 체크인된 text fixture와 새 root를 사용합니다:

```bash
ROOT=$(mktemp -d /tmp/nemochat-cpu-identity-XXXXXXXX)
WSD_ARGS="--texts tests/fixtures/texts.jsonl --context-length 32 --lr-schedule wsd --warmup-steps 2 --stable-steps 4 --decay-steps 2 --schedule-total-steps 8 --min-lr-ratio 0.1"
.venv/bin/python -m pytest -q tests/test_wsd.py
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 4 --output "$ROOT/split"
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 4 --resume "$ROOT/split" --output "$ROOT/resumed"
.venv/bin/python scripts/train_cpu.py $WSD_ARGS --steps 8 --checkpoint-every 2 --checkpoint-dir "$ROOT/snapshots" --output "$ROOT/continuous"
.venv/bin/python scripts/verify_cpu_resume.py "$ROOT/resumed" "$ROOT/continuous"
```

실제 새 root: `/tmp/nemochat-cpu-identity-hX2aU1zi`. `split`, `resumed`, `continuous`, `snapshots/step-00000002`, `step-00000004`, `step-00000006`, `step-00000008`이 생성됐습니다. 로그는 같은 root의 `split.log`, `resumed.log`, `continuous.log`입니다. targeted 테스트: **53 passed in 6.78s**. 전체 테스트: **222 passed, 1 skipped, 25 subtests passed in 16.40s**. 새 체크포인트 7개의 inventory 및 4+4와 8의 전체 history 동등성도 검증했습니다. pretrained tiny-HF 4+4 대 8 검증은 `PASS: exact model, optimizer/RNG, and run cursor/token-count equality (CPU control only)`를 반환했습니다. 새 inventory를 사용했으며 legacy 플래그는 사용하지 않았습니다. token ID swap, config/weight/optimizer/special-token 변경, inventory 누락/추가, 경로/symlink, 다중 shard 및 legacy opt-in 양성/음성 테스트를 포함합니다.

## 과거 실제 실행 근거

`/home/ubuntu/nemochat`에 이미 존재한 `.venv/bin/python`을 사용했습니다. 버전은 `torch 2.14.1+cpu`, `transformers 4.57.6`입니다. pretrained control은 `sshleifer/tiny-gpt2`, revision은 `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`입니다.

```bash
.venv/bin/python -m pytest -q tests/test_wsd.py
# 34 passed in 4.92s
.venv/bin/python -m pytest -q
# 첫 실행: 152 passed, 23 subtests passed in 16.32s
# 동시 작업 추가 후 최종 재실행: 173 passed, 23 subtests passed in 15.19s

.venv/bin/python scripts/train_cpu.py \
  --texts tests/fixtures/texts.jsonl --context-length 32 \
  --lr-schedule wsd --warmup-steps 2 --stable-steps 4 --decay-steps 2 \
  --schedule-total-steps 8 --min-lr-ratio 0.1 --steps 8 \
  --checkpoint-every 2 --checkpoint-dir /tmp/nemochat-wsd-20261007/snapshots \
  --output /tmp/nemochat-wsd-20261007/continuous
```

아래 수치/비교는 inventory 도입 전의 역사적 실행 결과입니다. 과거 `runs/cpu-train-*`는 로컬 호환성 fixture이며 clean clone에는 없습니다. 지금 예전 체크포인트를 resume하려면 `--allow-unverified-legacy-resume`를 추가해야 합니다. 예: `.venv/bin/python scripts/train_cpu.py --texts tests/fixtures/texts.jsonl --steps 2 --resume runs/cpu-train-2 --allow-unverified-legacy-resume --output /tmp/new-legacy-resumed` (새 output 경로 사용). 과거 산출물 root는 `/tmp/nemochat-wsd-20261007`이며 임시 파일이고 커밋하지 않았습니다. 재실행 시 새 root를 선택하세요. 전체 split/resume/검증 명령과 8개 update의 실제 loss/LR 표는 [영문 실행 기록](../en/WSD_IMPLEMENTATION.md)에 있습니다.

검증 결과:

- 총 8 update, 처리한 target token 총 197.
- LR은 `0.00005 → 0.0001` warmup, 3~6 update의 `0.0001` stable, `0.00005500000000000001 → 0.00001` decay.
- `split`에서 4 update 학습 후 `resumed`에서 4 update 추가: `continuous`와 모델/optimizer/RNG/cursor/token-count가 **정확히 동일**.
- `snapshots/step-00000004`에서 시작한 `periodic-resumed`도 동일.
- constant-LR `constant-continuous`는 수정 전의 `runs/cpu-train-uninterrupted`와 동일.
- 예전 `runs/cpu-train-2`를 resume한 `legacy-resumed`도 `constant-continuous`와 동일.

네 번의 실제 체크포인트 비교 모두 다음 메시지를 반환했습니다.

```text
PASS: exact model, optimizer/RNG, and run cursor/token-count equality (CPU control only)
```

Transformers의 기존 `loss_type=None` 경고와 기본 `ForCausalLMLoss` fallback은 나타났지만 실행과 비교는 성공했습니다. 단위 테스트의 random tiny-HF fixture와 위 pretrained CPU 실행은 별개의 근거입니다. **NemoChat/Bridge/MCore/B300 학습, 대규모 사전학습 수렴, WSM 평가 및 품질 향상은 이 스케줄러로 검증하지 않았으며 주장하지 않습니다.** 작은 수작업 corpus의 CE를 품질 benchmark로 해석하면 안 됩니다.

## 런타임 경로 migration 참고

위 과거 실행 기록의 경로·timestamp·결과는 원래 증거로 보존합니다. 새 실행은 [저장소 루트 런타임 규약](RUNTIME_LAYOUT.md)을 따르고, 이동된 로컬 산출물은 runtime migration manifest의 old → new 매핑으로 찾습니다. Receipt의 remote path/revision/hash와 inventory 상대 경로를 변경하거나 기존 산출물을 덮어쓰지 않습니다.
