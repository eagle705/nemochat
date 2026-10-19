# Native LM+KD control 실행 검증

## 2026-10-17 19:12 KST parent 재검증

**기존 ClimbMix를 사용한 동작 검증 control**입니다. 새 Specialized-v1.1 corpus 본학습이나 품질 개선 검증이 아닙니다.

- Student: D까지 완료한 원본 BASE update19,074의 HF 변환본. 가중치만 가져오고 optimizer/scheduler/RNG는 초기화했습니다.
- Teacher: 정확한 Nemotron3.5 Super VL **Base**, revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`의 frozen 텍스트 백본입니다. Instruct/3.0 대체나 다운로드한 모델 Python 실행은 없습니다.
- Bridge native LM+KD 경로에서 **8개 rank 모두 optimizer 호출4회 / 양의 LR learning update3회 / zero-LR 호출1회**를 완료했습니다. 모든 LM Loss·KD Loss·Total Loss·Grad Norm은 finite였고 teacher gradient는 없었습니다.
- 실제 `lr_used`: `[0, 1e-5, 1e-5, 1e-5]`. 설정된 minimum1e-6은 마지막 호출 후 scheduler가 도달한 값이며, 이 control에서는 실제 학습 update에 사용되지 않았습니다. 긴 decay 학습 구간을 검증한 결과가 아닙니다.
- Processed positions131072. BASE가 이미 학습한 train corpus의 명시적 stage 간 replay입니다.
- 마지막 LM Loss **2.63259649**, raw logits KD Loss **1.09624374**, Total Loss **3.94889426**, Grad Norm **1.36860561**.
- Grad Norm은 네 호출에서 약1.33~1.39였습니다. 큰 finite 값 때문에 실패시키지 않았고 native `clip_grad=1.0`은 유지했습니다. NaN 보정이나 검증 조건 완화는 없습니다.
- 각 rank의 forward activation 관측876건에서 nonfinite event0건을 확인했습니다. 동기화를 추가한 진단 실행이므로 **timing을 처리량 근거로 사용하지 않습니다.**

## 저장 및 teacher 없는 reload

- 저장 checkpoint: `checkpoints/midtraining/super35-kd-pilot-01/iter_0000004`.
- 8개 rank 모두 save4를 기록했습니다. Native checkpoint metadata6683건에 teacher 이름의 record는0건입니다. 추가 weight payload 무결성 감사는 하지 않았습니다.
- 별도 native reload는 **teacher 없이 / optimizer update 없이** 8개 rank 모두 finite heldout LM evaluation을 완료했습니다.
- 고정 heldout window LM Loss: 저장 전2.38600159 / reload 후2.38597488. Operational reload 증거이며 bitwise 동일성이나 matched 품질 개선의 증거가 아닙니다.
- Pilot은 optimizer/RNG를 저장하지 않은 model-only checkpoint입니다. 장기 본학습 optimizer resume용으로 사용하면 안 됩니다.
- 원래의 `not_checked_explicit_user_waiver` 무결성 정책은 유지했습니다.

## 정확한 실행 증거

Pilot Engine `906d2c5e1824579c76f9fe3f825994c4bd94c1f83a08fb85c15d8a5a13d7d662`, reload Engine `639d42ced1bc0e901bd7b1e9a105460456698f45c71235b0538a69b944a65c32` 모두 exit0/OOMKilledfalse입니다. Parent가 정확한 Engine 결과와 모든 rank 파일을 다시 읽어 acceptance를 재실행했고 supervisor success 및 종료 후 GPU compute PID가 없는 것도 확인했습니다.

`results/midtraining/super35-kd-01/pipeline-02/`의 `parent-accepted.json`, `checkpoint-metadata-student-only.json`, segment별 accepted/state/log 및 source allowlist를 따릅니다. Rank별 기록은 `pilot-02/`, `reload-02/`에 있습니다.

## 실패 보존과 수정 범위

Pilot01은 accepted update0/save 없음으로 실패했고 OOM은 아니었습니다. 당시 통합 오류에는 실제 실패 지표가 남지 않았으므로 원래 수치 원인을 단정하지 않습니다.

별도로 확인한 결함은 **student embedding import 이름 불일치**입니다. 실제 HF index는 singular `backbone.embedding.weight`, native mapping은 plural을 기대했습니다. 초기화를 끈 native import가 이를 건너뛰지 않도록 `patches/student_embedding_import.py`에서 해당 instance의 native mapping만 보정했습니다. 실제 SDK 검사에서 source6289개 mapping, 실제 재실행에서 모든 rank의 embedding native transform1회를 확인했습니다. 성공한 재실행은 수정의 동작 근거이나 원래 NaN/Norm 발생 메커니즘까지 입증한 것은 아닙니다.

Teacher layer count가 student 값으로 덮어써지던 별도 문제는 `patches/distillation_provider_shape_guard.py`로 격리했습니다. 설치 SDK·원본 BASE는 수정하지 않았고 실패01의 실행 source snapshot과 로그도 보존했습니다.

## 아직 남은 것

새 corpus 준비와 별도 pilot, 장기 token/LR 예산과 성능 측정, teacher HF/native 수치 parity, matched LM-only 비교, 새 SFT source 연결은 별도 검증 사항입니다. SFT 자동 시작은 계속 보류합니다.

[English](VERIFICATION.md)
