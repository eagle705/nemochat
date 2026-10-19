# 제한된 native offline KD (DP8 / EP8)

기존 online/ModelOpt 경로와 설치 SDK는 변경하지 않습니다. Production 시작은 부모 에이전트가 소스와 실제 GPU gate를 검토한 뒤 수행합니다. Controller 시작만으로 학습 성공을 의미하지 않습니다.

## 현재 승인된 약 1.920B 고유 토큰 profile

사용자가 기존 혼합비·중복 없는 권장안을 승인했습니다. **3,662 calls / 1,919,942,656 positions**, 원본 BASE에서 새로 시작하고67/10/3/20 비율을 유지합니다. 실행 profile은 `--mode production1920`이며 기존 `production`/`gate` 예산은 보존합니다. WSD warmup37/final decay366, LR1e-5/min1e-6이며 native 모델·forward·Loss·adapter·topology 코드는 바꾸지 않습니다.

새 corpus 경로는 `datasets/midtraining/super35-kd-01/offline1920-v1/data/manifest.json`입니다. 특수 alias ID가 검출된 검토 완료 Algorithmic 문서7개는 통째로 제외하고 ID를 다시 매핑하지 않습니다. 실제 train/validation의 IDs18..29 전수 검사가 모두0이어야 합니다. 새로운 raw 다운로드나 학습 반복은 허용하지 않습니다.

부모 release는 실제 과거 GPU gate source map에 결합된 명시적 `budget_compatibility`와 실제 manifest SHA·정확한 owned CPU producer의 clean Engine exit에 결합된 `corpus_approval`을 요구합니다. 수정은 부모가 검토한 launch/supervisor 제어 코드로 제한하며 기존 native 실행 소스 hash는 유지합니다. 과거 gate를 재사용하는 것이지 새 성능 측정으로 바꾸지 않습니다. 장시간 작업의 owned scratch는 실행 중 idle 만료되지 않도록 유지합니다. 승인 계약은 `results/midtraining/offline1920-parent-01/request.json`입니다.

실제 corpus 자료화·producer 검증 뒤 teacher cache를 생성하며 모든 native shard가 준비된 뒤 student 학습을 시작합니다. Runtime 준비는 본학습 update 시작을 뜻하지 않습니다.

## 이전 2B 계획 — 용량 미달, 호환성을 위해 보존

- 학생은 원본 `checkpoints/exports/nemochat_baby_base_hf_bf16_02`입니다. Smoke에서 학습한 체크포인트를 사용하지 않습니다.
- 교사는 Nemotron3.5 Super VL **Base** BF16의 revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6` text view입니다. Plain AutoBridge, frozen BF16, dropout0, DDP/optimizer 없음 조건을 유지합니다.
- GPU0–7, TP1/PP1/CP1, DP8/EP8, GBS128/MBS1, rank당 native microbatch16, sequence4096, seed43입니다.
- Core native tar writer/reader와 DP mapping, top32 ghost-tail 고정 alpha objective를 사용합니다. `primaryTotal=.5 LM+.5 KD`, temperature1입니다. Router auxiliary loss는 native 모델 측 항으로 별도이며 primaryTotal에 합쳐서 보고하지 않습니다. Raw metric 이름은 `lm loss`, `logits distillation loss`, `total loss`입니다.
- Native Bridge `pretrain`, Adam, clip1, full optimizer/RNG save를 사용합니다. Production은3815 optimizer call, 2,000,158,720 processed positions입니다. WSD LR1e-5/min1e-6, warmup38/final decay382입니다. 24시간 성능 약속이 아니라 token/update 예산입니다.
- Mix67/10/3/20, training cycling 금지, 실제 native 단일 epoch index 검사와 source별0.5% 여유 용량을 확인합니다. Production은 새 `offline2b-v1/data/manifest.json`과 인증된 producer success만 허용합니다. Pilot 대체는 금지합니다.
- Cache600GiB/data64GiB, 전체 watchdog72시간입니다. Watchdog 종료는 실패이며2B 완료로 처리하지 않습니다.
- 사용자가 면제한 checkpoint payload SHA/복사는 수행하지 않습니다. 상태는 `not_checked_explicit_user_waiver`입니다. 실제 native load, finite positive gradient, skip 검사와 optimizer update 검증은 유지합니다.

## 실제 분산 gate

저장소 루트에서 실행합니다. 실패한 output은 재사용하지 말고 새 generation을 사용합니다.

```bash
.venv/bin/python midtraining/offline2b/launch.py --mode gate \
  --output results/midtraining/offline2b-gate-01 --start
```

Dump/첫 학습/resume 모두 총512개 training sample을 갖는 동일 GPTDataset과 동일 validation 예산을 사용합니다. Seed만 일치한다고 가정하지 않고, 모든 rank의 실제 tokens/labels/positions/masks 데이터 SHA를 교사 batch와 비교합니다. 첫 학습은2와4에서 저장하고, payload 복사 없이 metadata/symlink로 checkpoint2를 선택하여3–4 update를 native full-state resume으로 다시 수행합니다. 추가 gate update2회는 production2B와 별도로 집계합니다.

성공은8개 rank의 finite positive nonskipped update, all-rank masked-token 수, 동기 full-state 저장 뒤 cache cursor 원자적 commit, 분리된 heldout native LM 평가, resumed positive update 및 정확한 Engine clean exit를 모두 요구합니다. 최종 `report.json`은 전체 chain 성공 후에만 생성됩니다. CPU test/provider load는 이 GPU 검증을 대신하지 않습니다.

User-systemd는 UID/GID를 상속하며 `User=`/`Group=`을 설정하지 않습니다. MainPID/실제 UID/전체 argv/source hash를 검사합니다. 모든 GPU 단계는 cooperative all8 lock과 실제 idle 검사를 수행합니다. 단계별 컨테이너는 network를 차단하며 새 output과 task scratch만 쓰기 가능합니다. Cleanup은 기록된 동일 owner container만 중지합니다.

## 부모의 Production release

부모가 소스, 실제 gate 및 새 corpus를 검토하고 `source_sha256` 전체 mapping과 accepted `gate_output` 절대 경로를 가진 review JSON을 작성해야 합니다. 과거 freeze를 수정하거나 review를 자동 생성해서 통과시키지 않습니다.

```bash
.venv/bin/python midtraining/offline2b/launch.py --mode production \
  --output results/midtraining/offline2b-production-01 \
  --review results/midtraining/offline2b-runtime-01/parent-review.json
# 부모 검토 후에만 위 명령에 --start를 추가합니다.
```

소비한 `run-config.json` 및 SDK native YAML은 새 output에 저장됩니다. Checkpoint transaction은 cache identity, 전체 sampler budget, seed, corpus, topology, 다음 iteration을 결합합니다. Sidecar commit이 없는 save는 accepted resume cursor가 아닙니다. Corpus 용량/producer 실패는 교사 load 전에 차단해야 합니다. Mix 변경,10M cycling, online KD fallback을 하지 않습니다.

## 실행된 gate 증거

`results/midtraining/offline2b-gate-02/report.json`은 `gate_accepted`입니다.8개 rank 모두 positive-LR native update4회, full-state checkpoint2/4 저장, checkpoint2에서 optimizer/RNG/cursor 복원, 추가 positive resumed update2회를 완료했습니다. 모든 실제 sample digest가 교사와 일치했습니다. Native cache32개 shard는172,247,040 bytes이며 update당 all-rank masked token은524,288개입니다. 실제 heldout LM과 raw router auxiliary metric은 Engine log에 보존했습니다. Bitwise trajectory 동등성이나 모델 품질 검증을 의미하지 않습니다.

독립 readback, 정확한 Engine ID/종료 상태, source freeze, 부모 시작 argv, CPU 결과는 `results/midtraining/offline2b-runtime-01/`에 있습니다. 최종 CPU focused test는99 passed/SDK 전용6 skipped이며 실제 GPU 검증은 별도로 실행했습니다. Gate01은 compiler worker 과다 생성과 CPU JSON-sidecar 회귀를 확인하여 학생 update 전에 의도적으로 중지했고 증거를 보존했습니다. 수정본은 Inductor compile threads를1로 제한합니다. EP8에서 실제 optimizer group은 EP1용 TE-Adam wrapper가 필요하지 않았습니다.

최종 readback에서 모든 owned container가 종료되었고 systemd unit은 success/inactive, GPU8개는 idle, cooperative lock 재획득이 가능했습니다. Production은 시작하지 않았으며 새 corpus manifest도 아직 없습니다. 부모 검토와 실제 전체 corpus 용량 승인이 남아 있습니다.

초기 adapter/patch snapshot은 `results/midtraining/offline2b-runtime-01/initial*`에 보존됩니다. 검증 명령과 상세 계약은 [English README](README.md)를 참고하십시오. Commit/public push/download/GPU reset은 수행하지 않습니다.
