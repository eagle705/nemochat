# Offline cached KD — 최초 64 calls 결과

- 기록 대상: **2026-10-18, `offline2b-prefixtrain-01`의 최초 student 실행**.
- 범위: native optimizer calls **1–64만**. 이후 continuation·재시작·filtered·dynamic 실행은 합산하거나 그래프에 이어 붙이지 않았다.
- **현재 학습은 사용자 요청으로 중단된 상태다. 이 문서 작성으로 학습을 재개하지 않았다.**
- [English](../en/MIDTRAIN_OFFLINE_KD_FIRST64_RESULTS.md) · [계산 결과 JSON](../assets/offline-kd-first64/summary.json) · [64-call 원시 지표 CSV](../assets/offline-kd-first64/metrics.csv)

## 1. 요약

8개 rank 모두 **64 optimizer calls**를 완료했다. 첫 call은 실제 사용 LR이 0이므로 **Positive-LR update는 63회**, zero-LR call은 1회다. Skipped 및 Nonfinite는 모두 0이고, Grad Norm은 모든 call에서 finite·positive였다. 처리량은 **33,554,432 positions**, global samples는 **8,192개**다. 처리량에 첫 zero-LR call의 입력도 포함하며, 8개 rank에 중복 곱하지 않는다.

첫 8 calls와 마지막 8 calls의 평균을 비교하면 LM Loss·KD Loss·Primary Total Loss가 모두 내려갔다. 이는 **실제로 관측한 training Loss의 하락**이지, validation·benchmark 개선이나 긴 학습의 안정성을 입증한 결과는 아니다.

## 2. Learning Rate / Loss 그래프

![최초 64 native optimizer calls의 실제 Learning Rate와 Training Loss](../assets/offline-kd-first64/lr-loss-first64.png)

- 위: native optimizer가 **해당 call에서 실제 사용한 Learning Rate**. post-step scheduler 값으로 대체하지 않았다.
- 아래: LM Loss, Ghost-tail KD Loss, `Primary Total = 0.5 × LM + 0.5 × KD`.
- 옅은 선은 raw 지표, 진한 선은 **trailing 8-call mean**이다. 각 곡선은 같은 관측 자료를 사용한다.
- Call 1은 LR=0이다. 37-iteration warmup 설정에 따라 **call 38에서 처음 LR=1e-5를 사용**했고, call 64에서도 LR=1e-5였다.
- Router auxiliary regularizer는 이 Primary Total 표시값과 별도다. 미래 Loss나 아직 실행하지 않은 schedule 구간을 합성하지 않았다.
- 원본 PNG를 수정 없이 문서 asset으로 복사했다. 두 PNG의 SHA256은 `69ee1db30128699f551922042a88da556dcb58e32d81a11b5d796a3157af2ccb`로 동일하다. 이는 **그림 파일의 동일성**이며 checkpoint payload 검증이 아니다.

## 3. 실제 수치

### 첫 8 calls 평균 → 마지막 8 calls 평균

| Metric | Calls 1–8 | Calls 57–64 | 상대 감소율 |
|---|---:|---:|---:|
| LM Loss | 2.312244 | 1.718245 | 25.69% |
| KD Loss | 1.744951 | 1.137173 | 34.83% |
| Primary Total Loss | 2.028598 | 1.427709 | 29.62% |

상대 감소율은 `(첫 window 평균 − 마지막 window 평균) / 첫 window 평균`이다. 서로 다른 training batches를 비교한 기술 통계이며, 동일 고정 문항의 전후 평가나 KD 효과의 인과 비교가 아니다.

### Raw endpoints

| Metric | Call 1 | Call 64 |
|---|---:|---:|
| LM Loss | 2.328561 | 1.703239 |
| KD Loss | 1.747200 | 1.117284 |
| Primary Total Loss | 2.037881 | 1.410262 |

- Grad Norm 범위: **0.231944–1.385862**. Native gradient clipping 설정은 1.0이며, 보고된 Grad Norm을 clipping 후 값으로 해석하지 않는다.
- 마지막 8 calls의 **각 call별 최대 rank callback 시간 평균**: **9.730136초/call**. CUDA 동기화된 학습 callback 구간의 시간이며 모델 초기화·전체 wall time·checkpoint 저장 비용을 모두 포함하는 지표가 아니다. Online KD 또는 기존 pretraining과의 통제된 A/B 속도 비교로 해석하지 않는다.
- Raw Loss, LR, Grad Norm은 64 calls 전체에서 rank 간 동일함을 확인했다. 이미 reduction된 값이므로 8개 rank의 Loss를 다시 더하지 않았다.

## 4. 실행 조건과 lineage

| 항목 | 기록된 조건 |
|---|---|
| Student 시작점 | 완료된 BASE19074의 `checkpoints/exports/nemochat_baby_base_hf_bf16_02`; model-only warmstart, optimizer/RNG/cursor 초기화 |
| Teacher | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16` |
| Teacher revision | `4ba52e8ffa79e9693936c1a819ac89f551ceeef6` |
| Teacher 사용 방식 | 기존 frozen teacher의 offline Top-32 cache; student 실행에 teacher forward 없음 |
| Framework 경로 | Native Megatron Bridge pretrain + 격리된 Core-first cached-KD adapter; 손으로 작성한 optimizer loop 아님 |
| Objective | LM + native Ghost-tail KD, fixed alpha=0.5; 온라인 ModelOpt의 detached magnitude-balancer objective와 동일하다고 주장하지 않음 |
| Precision / 병렬 구성 | BF16 student, FP16 teacher cache; 8 GPUs, TP1 / PP1 / CP1 / EP8, student DP8 |
| Sequence / batch | Sequence Length 4096; Global Batch Size 128; Micro Batch Size 1; rank당 16 microbatches/call |
| 데이터 혼합비 | Code / Algorithmic / Formal / Replay = 67 / 10 / 3 / 20 |
| Seed | 43 |
| 전체 source sampler 예산 | `[468736, 120, 8]` — train / validation / test |
| 전체 schedule | WSD horizon 3662, warmup 37, decay 366, LR 1e-5 / min LR 1e-6 |
| 당시 준비된 cache prefix | 공통 source iterations 0–1142; 1143은 준비된 범위이지 이 문서의 학습 완료량이 아님 |
| 이 보고서의 학습 범위 | Source iterations 0–63, optimizer calls 1–64; 이후 static/dynamic exclusion 이전 실행 |

전체 sampler와 전체 schedule은 유지한 채 완료된 cache prefix를 소비한 실행이다. 이 결과를 새로 만든 64-step sampler나 전체 약 1.920B 학습 완료로 표시하지 않는다. Cache ID는 `8e2ba57bb9e10d4b6b5ca2fe82a7e1969c285acb392afca9121cbda29e36bd4d`다.

## 5. 입력 정렬 및 checkpoint 증거

- `student-rank{0..7}-updates.jsonl`을 다시 읽어 각 rank에 정확히 calls 1–64가 있는지, LR·Loss·Grad Norm·Skipped 상태를 검증했다.
- 최초 **1,024 microbatches/rank**, 총 **8,192개 microbatch digest rows**를 원래 teacher의 같은 source iteration/rank와 다시 대조했다. Tokens·labels·loss mask·position IDs를 묶은 실제 기록이 모두 일치했다. 이 8,192개는 위의 global samples 8,192개와 **서로 다른 집계 단위**다.
- Native synchronous checkpoint save의 8-rank transaction이 같은 `next_iteration=64`, `consumed_samples=8192` 및 cache/sampler binding을 확인한다.
- 저장 위치: [`results/midtraining/offline2b-prefixtrain-01/checkpoint/iter_0000064/`](../../results/midtraining/offline2b-prefixtrain-01/checkpoint/iter_0000064).
- Model·optimizer·RNG 저장이 기록되어 있다. 이후 별도 continuation에서 8-rank step64 복원 및 후속 update가 실제 관측되었지만, **그 수치는 이 보고서 그래프에 포함하지 않았다**.
- 사용자 면제에 따라 **model/optimizer checkpoint payload SHA·CRC 감사는 하지 않았다**: `not_checked_explicit_user_waiver`. 문서용 로그·그림 SHA 검증과 혼동하지 않는다. 이 문서 작성 중 checkpoint를 다시 로드하거나 모델을 실행하지 않았다.

## 6. 해석 제한과 현재 상태

1. 이 보고서는 초기 training Loss와 operational checkpoint 증거를 보존한다. Matched validation 전후 비교, benchmark 성적, 일반화 성능 또는 online/offline objective parity 증명은 포함하지 않는다.
2. 이후 continuation과 재시작에서 별도의 cache read/SHA 문제가 발생했다. 최초 64 calls의 기록을 전체 prefix 무결성·장기 학습 성공으로 확대 해석하지 않는다. 이후 실행은 독립된 generation이다.
3. 사용자의 중단 요청에 따라 학습은 중지되어 있다. [의도적 중단 receipt](../../results/midtraining/offline1920-parent-01/user-training-stop-20261018-1342.json)를 보존하며, 이 문서는 재개 승인이 아니다.

## 7. 원본 증거와 재현 범위

- [원본 64-call PNG](../../results/midtraining/offline2b-prefixtrain-01/lr-loss-first64.png) / [plot data JSON](../../results/midtraining/offline2b-prefixtrain-01/lr-loss-first64.json)
- [최초 64-call 검증 proof](../../results/midtraining/offline2b-prefixtrain-01/first64-proof.json)
- [실행 config](../../results/midtraining/offline2b-prefixtrain-01/run-config.json) / [segment plan](../../results/midtraining/offline2b-prefixtrain-01/segment-plan.json)
- [Checkpoint64 transaction](../../results/midtraining/offline2b-prefixtrain-01/checkpoint-step64-transaction.json)
- [별도 operational resume proof](../../results/midtraining/offline2b-prefixtrain-01/operational-resume-proof.json) — 당시 snapshot이며 현재 live 상태가 아님
- Rank별 원본: `results/midtraining/offline2b-prefixtrain-01/student-rank{0..7}-{updates,samples}.jsonl`
- Teacher 입력 기록: `results/midtraining/offline2b-production1920-01/teacher-rank{0..7}-samples.jsonl`
- 문서용 [summary JSON](../assets/offline-kd-first64/summary.json)에 원본 텍스트 증거와 그림의 SHA256 및 재계산한 지표를 기록했다. [CSV](../assets/offline-kd-first64/metrics.csv)는 rank별 일치가 검증된 global-reduced 지표를 call당 1행씩 내보낸 것이다.

기존 renderer를 다시 실행하면 원본 run 디렉터리의 그래프가 갱신되므로, 이 보고서에서는 원본을 다시 그리지 않고 보존된 그림을 그대로 사용했다. 문서 검증은 로그·JSON·CSV·그림·상대 링크만 대상으로 했으며 GPU 학습은 수행하지 않았다.
