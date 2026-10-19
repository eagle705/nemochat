# BASE → LM + Knowledge Distillation → SFT

이 작업은 완료된 BASE의 학습 이력을 바꾸는 것이 아니라, **BASE에서 시작하는 별도 midtraining 실험**입니다. 기존 ClimbMix native LM+KD control의 update/save/teacher 없는 reload는 [실제 실행 검증](VERIFICATION.ko.md)을 통과했습니다. 새 Specialized corpus의 [실제 전처리](CORPUS.ko.md)는 native writer/readback 검사를 통과했습니다. Train 10,000,610 tokens / Validation 406,814 tokens, train·heldout token-document overlap 0, 8개 파일의 reserved ID18–29 출현 0입니다. 이는 calibration 용량이며 하루치 데이터가 아닙니다. [24시간 실행 경로](LONGRUN.ko.md)는 실제 pinned-image config 검사를 통과했고, duration-control/save/reload와 full-batch calibration은 별도 실행 검증입니다. 본학습·품질 향상 완료는 아직 주장하지 않습니다.

- Teacher: `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`
- Revision: `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`
- 가중치: 63 shards, 248,913,061,312 bytes. Instruct나 기존 3.0 Super로 대체하지 않습니다.
- Student: pretraining update19,074의 실제 BASE. 기존 SFT391에서 이어서 학습하지 않습니다.
- Objective: **LM Loss + logits KD**, `skip_lm_loss=false`, temperature1, intermediate-layer matching 비활성화.
- Pilot: 4 updates / sequence4096 / global batch8 / microbatch1 / BF16 / TP1·PP1·EP8·CP1. optimizer/RNG/scheduler는 새 stage로 초기화합니다.
- 실제 pretraining LR은 peak `1e-4`, 마지막 `1e-5`였습니다. Pilot은 보수적으로 peak `1e-5`를 출발점으로 사용합니다. 최적 LR을 확인했다는 의미는 아닙니다.
- 기존 ClimbMix train/validation indexed data를 사용합니다. BASE가 이미 본 train을 stage 간 재사용하는 것이며 새로운 unique corpus라고 보고하지 않습니다.

## 확정한 출발점 — 2026-10-17 사용자 결정

```text
기존 Pretraining W → S → D 완료(BASE, update19,074)
    → 가중치만 가져와 새 Midtraining WSD 시작(LM + Base teacher KD)
    → student-only save/reload 검증
    → 별도 source 계약을 고정한 SFT
```

**D까지 끝난 BASE에서 별도 continued-pretraining stage를 시작**하는 방식으로 확정했습니다. 기존 S 단계의 연장이나 optimizer/scheduler까지 복원하는 exact resume가 아닙니다. 원본 BASE와 tokenizer는 보존하고 optimizer moments·scheduler/sample cursor·RNG는 초기화합니다.

Pilot의 peak LR은 `1e-5`, minimum LR은 `1e-6`, warmup1 / 전체4 / 마지막 decay1 iteration입니다. Bridge 기본 warmup·step 이후 scheduler 갱신을 그대로 사용하므로, 과거 pretraining runner의 수동 step 직전 LR 조정과 경계가 다릅니다. 실제 `lr_used`를 기록하고 **LR=0인 optimizer 호출과 양의 LR에서 학습한 update를 구분**합니다. 이 네 번의 호출은 동작 검증이며 장기 학습 예산·최적 LR을 뜻하지 않습니다.

장기 midtraining의 데이터 혼합·학습량·LR horizon은 실제 pilot 측정 후 별도로 고정합니다. 기존 pretraining 재시작이나 S checkpoint로의 회귀는 선택하지 않았습니다.

## VL teacher의 텍스트 백본 사용

현재 Bridge의 Super Omni 직접 변환은 확인되지 않았고, Nano Omni mapping을 그대로 쓰면 latent projection을 놓칠 수 있습니다. 따라서 `patches/teacher_text_view.py`는 원본의 `language_model.backbone.*`, `language_model.lm_head.*` tensor만 **payload byte를 유지해 복사**합니다. 접두어를 제거하고 stock `NemotronHForCausalLM` config를 만듭니다. Vision/projector와 next-token logits에 불필요한 MTP head는 명시적으로 제외합니다. 시각 지식 distillation을 수행한다는 주장이 아닙니다.

ModelOpt 기본 loss balancer는 raw `LM + 0.5*KL`이 아닙니다. 양의 loss에서는 detached LM/KL 비율로 KD 크기를 맞춘 뒤 더하므로 total 값이 LM의 1.5배를 따라갈 수 있습니다. LM Loss, raw logits KD Loss, Total Loss를 구분해야 합니다.

## SFT와 실행 경계

2026-10-17 기존 `nemochat-nemotron-48h-pipeline-03.service`의 SFT 자동 시작을 의도적으로 보류했습니다. CPU SFT 데이터 준비는 유지했습니다. 기존 controller는 SFT391 source에 고정되어 있으므로 midtraining 완료 후 그대로 재시작하면 안 됩니다.

완료 조건은 다운로드 → 텍스트 변환·token-ID 호환성 → 실제 native update/save → student-only reload입니다. 각 단계의 검증 결과를 `results/midtraining/super35-kd-01/`에 구분하여 남깁니다. 다운로드 진행 중은 학습 진행 중이 아니고, fixture 테스트 통과는 GPU 실행 증거가 아닙니다. 장기 본학습·품질 개선·SFT 연결 완료는 아직 주장하지 않습니다.

기존 student checkpoint의 무결성 검사는 명시적 waiver를 유지합니다. 새로 다운로드하는 teacher 파일에 대한 공식 digest 검증과 구분합니다. 원본 checkpoint 덮어쓰기나 모델 업로드는 하지 않습니다.

## 본 Midtraining corpus 전환 결정

사용자는 Specialized-v1.1의 **Code-Concepts + Unconditional-Algorithmic + Formal-Logic**와 ClimbMix replay 제안을 수락했습니다. Economics/Multiple-Choice는 별도 검토 전 제외하고 Agentic/Instruction-Following/BBQ는 SFT용으로 유지합니다. 별도 데이터 recipe에서 revision·파일 allowlist·bounded 준비 예산·heldout/benchmark 제외를 고정합니다. 전체 corpus 무제한 다운로드나 장기 학습량이 승인·확정되었다는 뜻은 아닙니다.

현재 ClimbMix 네 번짜리 실행은 KD/import 결함을 분리하는 동작 검증 control입니다. 새 corpus pilot이라고 소급 표기하지 않으며, 데이터 준비를 통과한 후 **새 runner-consumed data 계약**으로 전환합니다.

## 실제 실행과 문서 위치

- 실험 설정: `midtraining/pilot.yaml`
- SDK가 생성한 전체 native YAML: `results/midtraining/super35-kd-01/config-native-03/native-config.yaml`. 실제 pilot/reload도 각자의 runner-consumed YAML을 생성하고, native 직렬화에서 빠지는 teacher/KD 설정은 별도로 보존합니다.
- 실행 controller: `midtraining/run_pipeline.py`. 검증된 전체 다운로드 → 텍스트 변환 → GPU lock/점유·recovery 검사 → native pilot → teacher 없는 student reload/evaluation 순서를 강제합니다. 소스 10개 checksum을 고정하며 실패한 native writer를 자동 재시작하거나 SFT를 자동 시작하지 않습니다.
- Supervisor: `nemochat-midtraining-base35-pipeline-01.service`. 현재 상태·로그·정확한 Engine 결과는 `results/midtraining/super35-kd-01/pipeline-01/`, rank별 학습 증거는 `pilot-01/`, reload 증거는 `reload-01/`에 남깁니다. `NATIVE_PILOT_STARTING`은 시작/로딩 단계이지 학습 update 완료가 아닙니다.
- Pilot은 student model-only DCP를 저장합니다(`save_optim=false`, `save_rng=false`). 다음 stage warmstart용이지 장기 학습의 optimizer resume checkpoint는 아닙니다. Reload 통과는 별도로 확인합니다.

## 첫 native 실패와 진단 generation02

첫 pilot01은 OOM 없이 exit1로 종료됐고, 8개 rank 모두 **accepted update 0 / save 없음**을 기록했습니다. Teacher freeze 검사를 통과하고 native forward/backward까지 도달했지만 원래 통합 오류 문구에는 어떤 수치가 실패했는지 남지 않았습니다. 큰 finite Loss/Grad Norm 때문에 중단됐다고 추정하지 않습니다. 크기 상한은 없습니다.

별도로 student HF index의 embedding 이름이 singular `backbone.embedding.weight`인데 native mapping은 plural `backbone.embeddings.weight`를 기대하는 결함을 확인했습니다. `patches/student_embedding_import.py`는 해당 student import instance만 보정하고 native AutoMapping/loader를 유지합니다. 원본 weight payload는 수정·재감사하지 않습니다. 이는 확인된 import 결함이며 원래 수치 실패의 원인이라고 아직 단정하지 않습니다.

진단 generation02는 실패한01 source/proof를 보존하고 activation finite 관측 및 실제 Loss/Grad Norm/skip 기록만 추가합니다. 출력·Loss·clipping·검증 기준은 바꾸지 않으며 진단 timing을 처리량으로 보고하지 않습니다. 실제 SDK import-plan 검사에서 student6289개 source key mapping을 확인했지만 weight load/update 검증과는 구분합니다. 현재 supervisor는 `nemochat-midtraining-base35-pipeline-02.service`, 상태와 증거는 `pipeline-02/`, `pilot-02/`, `reload-02/`를 따릅니다. 실행 source allowlist는12개입니다.

실제 source 경로와 세부 acceptance는 [English contract](README.md)를 확인하십시오.
