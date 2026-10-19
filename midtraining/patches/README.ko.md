# 호환성 수정의 별도 관리

Megatron Bridge의 native `ConfigContainer`·YAML·optimizer·WSD scheduler·masked LM/KD loss·checkpoint 기능을 우선 사용합니다. 설치된 Bridge/Core/ModelOpt/Transformers 소스를 직접 고치지 않습니다.

현재 필요한 checkpoint-layout 예외는 `teacher_text_view.py`입니다. 이는 SDK monkeypatch가 아니라 **별도 출력 경로를 만드는 offline checkpoint-layout adapter**입니다. 현재 Bridge가 정확한 Super Omni outer architecture를 인식하지 못하므로, 원본 tensor payload를 그대로 복사해 지원되는 텍스트 `NemotronHForCausalLM` 형식을 만듭니다. 접두어를 제거하고 Vision/projector/MTP 제외를 명시하며, 원본 가중치는 보존합니다.

현재 별도 관리하는 항목은 다음과 같습니다.

- `teacher_text_view.py`: 원본 가중치를 보존하는 offline 텍스트 백본 추출.
- `distillation_provider_shape_guard.py`: 실제 native config 검사에서 student finalization이 teacher `num_layers`를 88→52로 덮어쓰는 문제를 재현했습니다. upstream source hash를 고정한 process-local setter 보정으로 architecture 필드만 독립적으로 유지합니다. 공통 runtime 설정은 native setter를 사용하며 설치된 SDK 파일은 수정하지 않습니다. 실패01/02와 수정 후 config03 성공 기록을 보존합니다.
- `tokenizer_alias_gate.py`: Base teacher와 기존 BASE의 reserved ID18–29 이름 차이를 기존 승인된 migration과 전체 ClimbMix zero-occurrence 감사에 대조합니다. reserved 의미가 같다고 주장하지 않고 명시적 이름 전환으로 기록하며, weight permutation은 하지 않습니다.

- `student_embedding_import.py`: student의 실제 HF serialization은 singular `backbone.embedding.weight`, native mapping은 plural을 기대합니다. 해당 student load hook의 native AutoMapping만 보정하고, source6289개 mapping과 실제 embedding 변환 호출을 확인합니다. teacher·원본 weight·SDK 파일은 변경하지 않으며 random fallback은 허용하지 않습니다. 실제 native import-plan(metadata-only) 검사는 통과했고 weight load/학습 검증은 별도입니다.
- `kd_finite_probe.py`: opt-in 진단 forward hook으로 student/teacher activation finite 여부를 관측합니다. 출력·gradient·Loss를 수정하거나 NaN을 보정하지 않습니다. CUDA 동기화로 timing이 달라지므로 성능 측정으로 사용하지 않습니다.

- `native_dataset_observer.py`: 장기 실행의 process-local native GPT registry 관측 adapter입니다. 실제 blend·문서 shuffle·sample index는 기존 native builder가 생성합니다. adapter는 source별 비중과 training의 단일 epoch를 검사하고 startup evaluation용 window 하나를 추가 예약하며 종료 시 원래 registry를 복원합니다. Bridge0.6.1 pinned runtime과 parent source hash 검사를 유지합니다. 회귀 테스트는 `test_midtraining_dataset_observer.py` 및 `test_midtraining_longrun.py`; 실제 GPU 실행은 별도 검증입니다. native callback에서 해당 관측·예약을 지원하면 제거합니다.

현재 적용된 upstream source `.patch` 파일은 없습니다. 필요해지면 이 폴더에 원본 revision·파일, 최소 diff, 적용/검사 명령, 회귀 테스트, 실행 증거 및 제거 조건을 함께 기록합니다. 실제 teacher load·수치 호환성·GPU update/save/reload는 fixture 테스트와 별도로 검증합니다.

자세한 범위와 제거 조건은 [English inventory](README.md)를 확인하십시오.
