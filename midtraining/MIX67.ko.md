# 승인된 67/10/3/20 native LM+KD 복구 혼합

[English companion](MIX67.md)

`longrun24h.mix67.yaml`은 별도의 production 설정이다. 기존 `calibration.yaml`과
`longrun24h.yaml`의 60/10/10/20 기본값은 바꾸지 않는다. 소스 순서는 Code /
Algorithmic / Formal Logic / ClimbMix replay이며, 이 두 혼합만 허용한다.
임의 비율을 받아들이는 API가 아니다.

새 초기 예산은 optimizer call **3,200회**, WSD warmup **32**, final decay
**320**이다. global batch 128, sequence 4096, BF16, EP8/TP1/PP1/CP1 및 표시 이름
`nemochat baby midtraining super35 24h`를 유지한다. 학생은 original BASE의
model-only warmstart이며 optimizer/RNG를 초기화한다. 고정된 Base teacher는
학습하지 않는다. native Bridge의 학습 실행, loss, optimizer, scheduler,
save 및 teacher-free reload 경로는 그대로다. offline KD, TopK, vLLM 변경은 없다.

## 측정 근거와 parent 승인

기존 승인 receipt는
`results/midtraining/longrun/super35-newdata-calibration-01/controller/accepted.json`이다.
**60/10/10/20**으로 측정한 16 calls / 15 positive-LR updates 중 warm updates
9..16 평균은 24.29540627438837초다. seed manifest SHA256은
`52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308`이다.
새 비율의 실측 benchmark가 아니다. 3,200회 예산은 사용자가 승인한 혼합 변경
이후 기존 동일 shape profile에서 추정한 **잠정 projection**이다. 새 main run의
warm-step 시간은 해당 run의 실제 native update receipt로 확인해야 한다.
이 패치는 추가 GPU calibration을 만들지 않는다. 24시간 이전 iteration budget
종료를 24시간 완료로 승인하지 않는 기존 정책도 유지한다.

Parent는 새 corpus와 heldout exclusion을 실제 감사한 뒤
`--calibration-compatibility` JSON을 새로 작성해야 한다. 정확한 필드는 다음과
같으며 production digest만 실제 감사한 값으로 넣는다. 추가 필드는 허용하지 않는다.

```json
{
  "status": "parent_accepted_approved_mix_change_profile_projection",
  "calibration_manifest_sha256": "52d0d44fce30ea1737396fad9b2d95c3040835769eaa089df1d9bfa1f1d1e308",
  "production_manifest_sha256": "<actual audited recovery manifest SHA256>",
  "old_weights": [0.6, 0.1, 0.1, 0.2],
  "new_weights": [0.67, 0.1, 0.03, 0.2],
  "same_provider_tokenizer_sequence_batch_topology": true,
  "heldout_exclusions_preserved": true,
  "user_approved_mix_change": true,
  "projection_only": true,
  "fresh_new_mix_benchmark": false
}
```

기존 `parent_accepted_same_recipe_expansion` 분기는 유지하지만 새 혼합 승인에
사용할 수 없다. 승인된 calibration과 8개 rank 전체의 save/reload duration-control
검증도 계속 필요하다. 실행에 소비되는 projection에는 compatibility dictionary,
기존 timing weights, 새 production weights, 실제 main warm 측정 대기 상태를 기록한다.

## 공유 API

`longrun_native.py`:

- `WEIGHTS`는 기존 tuple, `MIX67_WEIGHTS`는 새 승인 tuple이다.
- `recognized_weights(value) -> tuple`: 정확한 list/tuple 또는 완전한 소스-key
  map만 검증하며 임의 비율과 잘못된 key를 거부한다.
- `manifest_weights(manifest) -> tuple`: schema-1 입력은 weights map이 필수다.
  schema_version과 status가 모두 없는 기존 메모리 adapter fixture에만 구형 기본값
  fallback을 둔다. 실제 production 파일 입력에는 이 fallback이 적용되지 않는다.
- `dataset_weights(c, manifest=None) -> tuple`: 설정 비율을 검증하고 manifest가
  있으면 두 비율이 정확히 일치하는지 확인한다.
- `capacity_gate(manifest, updates, batch, sequence, cursor_samples=0)`: 기존
  signature를 유지하고 manifest 비율을 사용한다. native 0.5% builder surplus,
  정수 반올림 reserve, shifted token, 소스별 heldout 용량 조건을 유지한다.
- `load_prepared_manifest(path) -> (manifest, sha256)`: 두 비율 모두 기존 schema와
  source-order 및 SHA로 묶인 companion success를 요구한다. 명시적인 실패 status를
  성공으로 덮어쓰지 않고 거부한다.

`launch_longrun.py`:

- `validate_calibration_compatibility(accepted, manifest_hash, weights,
  compatibility) -> dict | None`: 정확한 legacy 또는 승인된 projection dictionary를
  요구한다. 새 분기는 실제 측정된 기존 seed SHA와 다른 새 manifest SHA를 연결한다.
  `execute`에서 capacity/Engine/tracking 동작 이전에 호출한다.
- `SOURCES`는 기존 항목을 보존하고 새 YAML을 추가한다(15개 경로). 두 writer의
  작업이 끝난 뒤 parent가 새 source review를 고정해야 한다.

Capacity, builder, observer, rank-local 실행 모두 같은 실제 비율을 사용한다.
각 소스의 native epoch는 여전히 정확히 1회여야 하며 첫 global batch는 Formal
Logic 3%를 포함한 네 소스를 모두 포함해야 한다. 모든 rank의 finite loss,
positive gradient, frozen teacher, 최종 full-state save 및 teacher-free reload
조건을 낮추지 않는다. 학생 checkpoint byte audit의 명시적 면제와 별개로 새 실제
`.bin`/`.idx` SHA 및 reserved-ID 18..29가 모두 0인지는 계속 확인해야 한다.

## CPU 검증 한계

`tests/test_midtraining_mix67.py`에서 실패를 확인한 뒤 구현하는 red/green 단계를
실행했다. 두 비율의 용량, 설정/manifest 불일치, companion 실패, native config 및
observer 전달, one-epoch 거부, compatibility 필드 오염 거부, launcher 분기를
CPU로 확인했다. 합성 builder/index fixture는 실제 native/GPU 증거가 아니다.

```bash
.venv/bin/python -m pytest -q tests/test_midtraining_mix67.py tests/test_midtraining_longrun.py tests/test_midtraining_dataset_observer.py tests/test_midtraining_production_corpus.py tests/test_midtraining_production_pipeline.py
.venv/bin/python midtraining/launch_longrun.py --config midtraining/longrun24h.mix67.yaml --run-id super35-mix67-01 --preview
.venv/bin/python midtraining/longrun_native.py --config midtraining/longrun24h.mix67.yaml --preview
```

이 명령은 CPU test와 inert preview이며 학습이 아니다. 이 패치로 Engine container,
service, download, tracking run, checkpoint, compatibility receipt를 생성하지 않는다.
실제 production orchestration, 새 corpus 승인, source review는 parent의 별도
release gate로 남는다.
