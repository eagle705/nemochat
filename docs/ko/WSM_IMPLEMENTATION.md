# 공식 Megatron WSM 연결: 한국어 해설

이 문서는 [영문 구현 명세](../en/WSM_IMPLEMENTATION.md)의 해설이다. 구현·실행 명령과 revision의 기준은 영문 명세와 실제 CLI다.

## WSD와 WSM의 차이

- **WSD**는 optimizer update마다 적용할 learning rate를 정한다: warmup → stable → decay.
- **WSM**은 이미 저장된 여러 checkpoint의 모델 가중치를 offline으로 합쳐 평가용 checkpoint를 만든다.
- 이번 실험은 **stable 학습 경로를 계속 유지하고, 평가 시점까지의 최근 checkpoint window를 merge**한다. 시점마다 별도 decay branch를 학습하는 방식이 아니다.
- 동일 시점의 raw와 merged 모델을 동일 benchmark 설정으로 평가한다. 성능 향상은 보장하지 않으며 하락도 기록한다.

## 구현 범위

`scripts/merge_checkpoints.py` → `src/nemochat/checkpoints/wsm.py`는 [Megatron-LM PR #5114](https://github.com/NVIDIA/Megatron-LM/pull/5114)의 공식 도구를 실행하는 wrapper다. 자체 HF averaging으로 대체하지 않는다.

고정 revision: `ca4826e9555032b8710dfb4a4e5693c46c1613dd`.

입력은 같은 run/model/sharding layout의 신뢰할 수 있는 **Megatron `torch_dist` DCP checkpoint**다. `train_cpu.py`의 HF safetensors checkpoint는 이 도구의 입력이 아니다. 키·shape·dtype·chunk layout이 다르면 거부해야 한다.

공식 도구는 GPU 모델을 구성하지 않고 CPU fp32로 가중치를 누적한다. 따라서 merge를 CPU에서 할 수 있지만, 실제 NemoChat 로딩·inference·benchmark까지 검증됐다는 뜻은 아니다.

## 계수와 window

Wrapper는 `minus-sqrt`를 사용한다. 공식 구현은 decay 함수의 이산 차분을 checkpoint 계수로 사용하므로 `1-sqrt(x)` 값을 그대로 정규화하는 방식과 다르다. 계수는 선택된 checkpoint의 **순서/개수**에 의존하며, iteration 간격이나 실제 token 간격에 비례하는 가중치가 아니다.

입력 구간은 `[max(0, target-window), target]` 양 끝을 포함한다. 최소 iteration 간격 필터는 target에서 뒤로 걸으며 target을 보존한다. Wrapper 자체는 checkpoint가 stable 구간인지 판단하지 않으므로, 호출자는 phase metadata를 확인하고 warmup/decay/future checkpoint가 섞이지 않도록 구간을 정해야 한다.

Token 단위 window는 공식 CLI에서 직접 지원하지 않는다. 실제 누적 consumed/loss-bearing token counter를 사용해 iteration 구간으로 변환하고, 요청한 window와 실제 선택된 범위를 모두 기록한다. 가변 길이·packing·loss mask가 있는 경우 nominal batch×sequence length만으로 환산하지 않는다.

## 신뢰·state·출력

- Git revision과 공식 도구 내용, dependency 디렉터리의 수정 여부를 검사한다. 이는 sandbox가 아니다.
- DCP metadata/common state/extra state의 역직렬화는 코드 실행 위험이 있다. **Dry-run도 신뢰된 입력만** 사용하고 `--trusted-inputs`로 명시한다.
- 모델 tensor만 평균내며 Transformer Engine `_extra_state`는 target checkpoint에서 복사한다.
- 기본적으로 optimizer 등 non-model DCP entry는 거부한다. `--ignore-non-model-state`를 명시하면 지원되는 upstream 옵션으로 해당 entry를 제외한다. 이 옵션이 optimizer만 제외한다는 뜻은 아니다.
- Common state를 복사하더라도 그 optimizer가 merged weights와 일관된 학습 상태라는 증거는 없다. **원래 학습은 raw checkpoint에서 재개**한다.
- 출력은 새로운 경로를 사용하며 input/output root가 겹치면 안 된다. 이름이 `iter_<숫자>`인 output root는 upstream이 부모 tracker를 덮어쓸 수 있어 실행 전에 거부한다. Wrapper provenance와 upstream provenance를 보존한다.

## 실제 실행 환경

해당 revision은 `typing.override` 때문에 기존 Python 3.11에서 import가 실패했다. 별도 Python **3.12.13** 환경과 torch **2.14.1+cpu**를 사용했다. Package initializer가 Triton을 transitively import해 wheel도 필요했지만, merge가 CUDA kernel을 실행한 것은 아니다. 기존 tutorial `.venv`와 분리하고 [영문 문서의 환경 구성](../en/WSM_IMPLEMENTATION.md)을 따른다. Git과 [uv](https://docs.astral.sh/uv/getting-started/installation/)가 PATH에 있어야 한다. 명령은 자신의 checkout root에서 실행한다.

호스트에서 검사한 외부 경로:
- Upstream: `/home/ubuntu/runs/wsm-upstream/Megatron-LM`
- Interpreter: `/home/ubuntu/runs/wsm-upstream/venv312/bin/python`

다른 머신에서는 이 경로가 존재한다고 가정하지 않는다.

```bash
CACHE=${HOME}/runs/wsm-upstream
# Wrapper는 project interpreter, upstream은 분리된 interpreter로 실행한다.
.venv/bin/python scripts/merge_checkpoints.py --help
FIXTURE_PARENT=$(mktemp -d "${TMPDIR:-/tmp}/nemochat-wsm.XXXXXXXX")
"$CACHE/venv312/bin/python" -O scripts/verify_wsm_merge.py \
  --upstream "$CACHE/Megatron-LM" --work-dir "$FIXTURE_PARENT/fixture"
NEMOCHAT_WSM_UPSTREAM="$CACHE/Megatron-LM" \
  "$CACHE/venv312/bin/python" -m pytest tests/test_wsm.py -q
```

Review blocker 수정 후 분리된 환경에서 **25 passed**, project 환경에서 **24 passed, 1 skipped**를 확인했다. 공식 fixture의 8개 case는 `python -O`에서도 PASS했다. `--work-dir`은 존재하지 않는 새 디렉터리여야 한다. 기본 project suite는 외부 환경이 필요하므로 이 integration을 opt-in으로 skip한다. 부모 tracker sentinel 보존·subprocess 미호출과 `-O`에서 잘못된 state를 거부하는 회귀 테스트도 포함한다. 보고서의 extra-state iteration은 재로딩된 tensor에서 읽는다.

## 검증의 의미

작은 신뢰된 synthetic DCP 입력으로 실제 공식 merge, 독립 값 재검사, extra-state 복사, dry-run, overwrite 거부, 입력 부족/target 누락, target 보존과 layout mismatch 거부를 검사했다. `/home/ubuntu/runs/wsm-fixture-5/verification.json`은 이전 10-test 실행의 호스트 전용 기록이다. `/home/ubuntu/runs/wsm-review-opt-qSfc76JH/fixture/verification.json`은 이번 optimized 검증 기록이다. 둘 다 repository 파일이나 다른 머신의 재현 전제 조건이 아니다.

이것은 **CPU merge wiring 검증**이다. 실제 NemoChat의 GPU 학습 checkpoint, hybrid model reload/export, NeMo Gym benchmark 실행 및 raw 대비 성능 개선은 별도 검증이 필요하다. Upstream Nemotron 결과를 이 프로젝트의 결과로 인용하지 않는다.

## 런타임 경로 migration 참고

위 과거 실행 기록의 경로·timestamp·결과는 원래 증거로 보존합니다. 새 실행은 [저장소 루트 런타임 규약](RUNTIME_LAYOUT.md)을 따르고, 이동된 로컬 산출물은 runtime migration manifest의 old → new 매핑으로 찾습니다. Receipt의 remote path/revision/hash와 inventory 상대 경로를 변경하거나 기존 산출물을 덮어쓰지 않습니다.
