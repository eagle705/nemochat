# Future work: Megatron Bridge 기반 teacher-assisted pretraining

**아래는 과거 제안이며 2026-10-17의 후속 사용자 결정으로 실행 단계가 시작되었습니다.** D까지 완료한 BASE에서 새 LM+KD midtraining을 시작하며, teacher는 정확한 Nemotron 3.5 Super VL **Base**입니다. 현재 계약·코드·native YAML·별도 패치·실행 증거는 [`midtraining/`](../../midtraining/README.ko.md)를 따릅니다. 아래의 비승인 문구는 과거 문서화 전용 요청의 기록이며 이후 명시적인 다운로드·pilot 승인을 취소하지 않습니다. 기존 BASE 자체는 LM-only입니다.

**당시 제안 상태: 향후 실험 계획이며 NemoChat 구현·학습 실행은 하지 않음.** 이 항목 추가는 새 학습, 추가 다운로드, 현재 SFT 변경에 대한 승인이 아니다. 완료된 10B-token BASE pretraining은 KD 없이 LM Loss로 학습했다.

## 제안 실험

- NemoChat 10BA1B hybrid student와 별도로 확정한 frozen teacher를 Bridge의 `convert_to_distillation_provider()` 및 native `distill()` 경로로 연결한다.
- 우선 logits Knowledge Distillation(KD)과 next-token LM Loss를 함께 사용한다. 문서상 기본값은 LM Loss를 생략하므로 `skip_lm_loss=false`를 명시한다. Loss weighting과 temperature는 설치된 Model Optimizer의 실제 계산식을 확인한 뒤 확정한다.
- 처음에는 intermediate-layer distillation을 끈다. Student hidden width는 1024, 제안된 Super teacher는 4096이므로 직접적인 표현 정렬의 호환성을 가정하지 않는다. Projection 도입은 별도 실험이다.
- 새 from-scratch pretraining인지 기존 BASE의 continued pretraining인지 먼저 구분한다. 기존 LM-only 학습을 distillation 결과로 소급 표기하지 않는다.

## 실행 전 검증 및 비교

1. Teacher/checkpoint/tokenizer revision을 고정하고 vocab 크기와 token-ID 의미를 확인한다. BASE와 post-training의 reserved-token alias 차이도 구분한다.
2. 실제 설치 환경에서 HybridModelProvider 연결, loss mask와 per-token normalization, teacher/student TP·EP·CP 호환성, teacher freeze 및 student-only save를 확인한다.
3. 제한된 실제 GPU update/save/reload pilot으로 finite Loss·Grad Norm, peak memory, throughput과 teacher forward 추가 비용을 측정한다. Provider가 인자를 받는 것만으로 실행 검증을 대체하지 않는다.
4. 동일 initialization·data·token budget·평가 프로토콜로 LM-only와 LM+KD를 비교한다. Wall-clock 및 compute 비용도 별도로 보고한다. Teacher 일치도뿐 아니라 heldout CE/PPL과 동일 조건의 downstream benchmark를 평가한다.
5. 현재 SFT·원본 체크포인트·기존 검증 면제 정책은 변경하지 않는다. 본 KD 학습은 별도 실험 조건과 예산 승인을 받은 뒤 진행한다.

## 확인한 기능 범위 — 학습 실행 증거가 아님

2026-10-17 read-only 조사에서 현재 전처리 컨테이너의 Megatron Bridge 0.6.1 및 NVIDIA Model Optimizer 0.46.0rc1을 확인했다. 설치된 `models/distillation_provider.py`는 GPTModelProvider와 HybridModelProvider를 허용하며, `training/distill.py`는 `pretrain(config, forward_step_modelopt)`를 호출한다. API는 experimental이고 NemoChat-specific KD 실행은 아직 검증하지 않았다.

공식 참고: https://docs.nvidia.com/nemo/megatron-bridge/latest/modelopt/distillation.html

[English](../en/PRETRAIN_DISTILLATION_FUTURE_WORK.md)
