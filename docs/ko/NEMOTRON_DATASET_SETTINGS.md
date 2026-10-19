# 버전 관리하는 Nemotron 데이터셋별 SFT 설정

[English (canonical)](../en/NEMOTRON_DATASET_SETTINGS.md) · [SFT/토크나이저 이력과 실행 증거](NEMOTRON_SFT_AND_TOKENIZER_TRANSITION.md)

## 범위와 버전 경계

정책 기준은 **2026-10-17**, 16K pilot 수락 및 pipeline-03 시작 영수증 이후다. 현재 데이터셋 정책이며 과거 4K 실행 기록을 덮어쓰지 않는다. 실제 작성자 카드 다섯 개는 `results/pretraining/nemochat_baby/nemotron_sft/discovery/<release-name>.json`의 `id`, `sha`, `card`, `cardData`, 파일 inventory에서 읽었다. `results/` 실행 증거는 local/ignored이므로 새 clone에 없을 수 있다.

버전 관리 인터페이스:

- [현재 명시적 16K/48h trainer template](../../configs/posttraining/nemochat_baby_nemotron_sft_16k_48h.yaml): pipeline-03의 고정 실행 입력이다. Calibration receipt/iteration estimate는 실제 측정 전까지 의도적으로 null이다.
- [추후 128K 계획](../../configs/posttraining/nemochat_baby_long_context_128k.plan.yaml): **계획 전용**이며 바로 실행할 trainer config가 아니다.
- [데이터셋 정책 registry](../../configs/posttraining/nemotron_dataset_settings.yaml)(`nemochat-nemotron-dataset-settings-v1`), [검증 모듈](../../src/nemochat/data/sft_dataset_settings.py), [조회/준비 wrapper](../../scripts/posttraining/nemotron_dataset_settings.py). 문서 작업 중 제공됐으며 실제 `--help`, 기본 검증, 모듈의 plan compile을 GPU/가중치 읽기 없이 통과했다.
- [실제 normalization/preparation 구현](../../src/nemochat/data/nemotron_sft.py), [native adapter](../../scripts/posttraining/native_nemotron_sft.py), [48h controller](../../scripts/posttraining/nemotron_48h_pipeline.py).

새 registry는 명시적 config와 정책을 대조하고 연결된 후속 호출에서 소비할 수 있다. 이미 시작된 pipeline-03의 입력 원천이 소급해서 바뀌거나 고정 입력이 자동 재작성되지는 않는다. 재시작 승인도 아니며, 과거 process가 registry를 읽었다는 증거도 아니다. Registry 검증, wrapper/plan 소비, 실제 native 실행은 구분한다.

## 현재 SFT의 공통 실행 설정

활성 릴리스는 **Agentic v4, Instruction-Following/Chat v4, BBQ v1**, labeled 파일 여섯 개다. Long Transduction은 전체 보류, RL Seed는 별도 예약이다. 이미 받은 검증 원본을 사용한다. 활성 원본은 **99,631,965,701 bytes**, Long을 포함한 labeled SFT 전체 일곱 파일은 **100,853,847,138 bytes**다. 이는 원본 크기이지 채택 토큰 수나 추가 전송량이 아니다. Full-03의 추가 다운로드는 0 bytes다.

- Sequence length **16,384**, CP **1**, EP **8**, 물리 GPU 8개, microbatch **1**, production/calibration global batch **128**, BF16. 수락한 소형 pilot의 global batch는 **8**이므로 구분한다.
- `results/pretraining/nemochat_baby/sft/runs/sft-main-01/checkpoint/iter_0000391`에서 model-only 초기화하고 optimizer/scheduler/RNG/cursor는 재설정한다. Long이 포함된 예전 4K pilot이나 저장되지 않은 RLVR 상태를 이어 쓰지 않는다.
- `official-super35-selected-assistant-v1`, 승인된 migrated tokenizer manifest `results/pretraining/nemochat_baby/tokenizer_migration/manifest.json`을 사용한다. 공식 nonthinking 렌더링(`enable_thinking=False`)으로 `reasoning_content`를 빼고 최종 응답/tool call 및 공식 empty think marker를 보존한다. Reasoning trace를 학습했다는 뜻이 아니다.
- Assistant header **뒤**의 선택된 assistant content/function call/응답 종료자만 loss 대상이다. System/user/tool 메시지, 선택 안 된 assistant turn, header, padding은 loss에서 제외한다. 계측 렌더링과 공식 렌더링의 bytes/전체 token IDs가 정확히 같아야 한다. 조각별 독립 토큰화가 아니라 전체 렌더링으로 shifted mask를 만든다.
- 전체 formatted 예제만 사용한다. 길이 초과는 이유/수량을 기록해 제외하고 정답을 자르지 않는다. 데이터셋 단계별 제외가 길이 적합 여부보다 우선한다.
- Unique-first, 승인 최대 1 epoch, `allow_repetition: false`. 마지막 batch의 zero-loss padding은 실제 대화 재노출이 아니다. Main은 **48시간**과 고유 데이터 가용량/실측 처리량으로 제한하며, 이틀을 채우기 위한 반복 학습을 약속하지 않는다.
- Trainer template: LR `0.00001`, minimum LR `0.000001`, warmup ratio `0.03`, weight decay `0.01`, seed `42`, save/validation interval `250`, final validation 필수, optimizer/RNG 저장, 최근 checkpoint `3`개 유지. 품질 기반 early stopping/best-validation 보존은 미구현이다. Artifact upload는 금지다.

## 비파괴적 조회 인터페이스

저장소 루트에서 실행한다. 아래는 검증/미리보기만 한다.

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/posttraining/nemotron_dataset_settings.py --help
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/posttraining/nemotron_dataset_settings.py --check --json
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/posttraining/nemotron_dataset_settings.py --preview --stage deferred_long_context --json
```

기본/`--check`/`--preview`는 실행을 시작하지 않는다. JSON에는 검증과 stage별 data plan이 있고 `native_training_argv`는 null이다. 명시적인 `--raw`와 새 `--output`을 함께 주면 producer 명령을 만들 수 있다. `--write-plan`은 새 JSON plan만 쓴다. `--execute-prepare`만 실제 CPU 준비를 수행하며, 이미 승인된 호환 cached CPU runtime에서만 사용한다. 학습/container launcher가 아니다. 여기서 검증했다고 준비를 실행했다는 뜻은 아니다. 로컬 discovery/migration/reservation/audit 증거가 필요하며, 없는 fresh clone에서는 metadata를 만들지 말고 실패해야 한다.

Long의 registry `train: true`/`heldout: true`는 **보류된 해당 stage 범위**이며 현재 단계 전체 제외를 뒤집지 않는다. 선택은 full eligible이며 고정 mix weight/quota가 없다. Wrapper의 큰 physical-byte 상한은 legacy producer의 quota 조기 종료를 막기 위한 값일 뿐 요청 예제 수나 corpus 수량이 아니다.

## 데이터셋별 설정

### 1. Agentic v4 — 현재 labeled SFT

- **ID/revision:** `nvidia/Nemotron-SFT-Agentic-v4` @ `a4593a26ea83a700035bbe0ec1edc9ef3810669d`.
- **작성자 목적/subset:** `tool_calling`(작성자 수량 549,707)은 tool 선택·인자 생성·결과 활용, `knowledge_agent`(84,957)는 tool이 반환한 문서/정책에 근거한 응답을 학습한다. 별도 문서 backend는 제공하지 않는다.
- **Mask/reference:** gold assistant 응답과 구조화된 call을 보존한다. 별도 작성자 mask가 없으면 모든 assistant turn을 선택한다. Tool 결과는 context이지 target이 아니다. 현재 nonthinking branch에서는 `reasoning_content`만 제거한다.
- **구조/출처:** tool 정의, parallel call 순서, argument mapping, call ID와 대응 `tool_call_id`를 구조화된 record/원본 provenance에 보존한다. 유효한 JSON object 문자열만 공식 template용 mapping으로 변환한다. Array/scalar/null/잘못된 JSON을 `{}`로 수리하지 않는다. 공식 렌더링 텍스트는 call ID를 생략하므로 구조적 ID 보존이 ID의 렌더링·지도학습을 뜻하지는 않는다.
- **필터/split:** 아래 공통 dedup/prompt-group 예약/benchmark/완전한 길이 gate를 적용하고, typed 잘못된 call·orphan response는 대화 전체를 quarantine한다. 현재는 16K/CP1이며 subset별 CP 변경이 아니다.
- **실행/한계:** 두 subset 모두 수락된 16K pilot에 기여한다. Knowledge-agent는 전체 길이 4,679–15,786인 16개(13 train/3 heldout)를 채택했다. 과거 제한된 4K prefix의 채택 0개는 전체 subset 사용 불가를 뜻하지 않는다. Full 16K 최종 채택 수는 full manifest 수락 전까지 미확정이다. 준비 중 tool을 실행하지 않는다.
- **License:** OpenMDW-1.1과 카드의 Nemotron-Personas-USA CC-BY-4.0 출처 고지를 유지한다.

### 2. Instruction-Following/Chat v4 — 현재 labeled SFT

- **ID/revision:** `nvidia/Nemotron-SFT-Instruction-Following-Chat-v4` @ `1cd362945d7a00b136b95b96892bbfea8798686e`.
- **작성자 목적/subset:** `chat`(914,716), `multi_turn_instruction_following`(2,669,676), `instruction_following_for_response_length_control`(266,190). 각각 이전 대화를 이용한 follow-up, 여러 turn에 걸친 지시 준수, 명시적 응답 길이 제어다. 원본 수량이지 적격/학습 수량이 아니다.
- **Mask/reference:** `metadata.train_turns_mask`는 필수이며 assistant만 나열한 배열이 아니라 **모든 메시지**와 정렬된다. True인 assistant 위치만 지도하고 false assistant는 context로 남긴다. 메시지 수와 같은 길이의 boolean 배열을 요구한다. 누락/오정렬은 fatal이며 모든 assistant 지도 방식으로 우회하지 않는다. `reasoning_content`를 빼도 최종 응답과 작성자 mask를 유지한다.
- **필터/split/출처:** subset, UUID/metadata, turn 선택을 보존하고 공통 gate를 적용한다. 길이 제어 정답을 임의 축약하지 않는다. 현재 16K/CP1이다.
- **실행/한계:** 세 subset 모두 실제 16K pilot target에 기여한다. Full-03 렌더링/수량은 아직 수락 전이다. 영어 자료만으로 한국어 품질 개선을 주장할 수 없다.
- **License:** OpenMDW-1.1, WildChat ODC-By v1.0 및 LMArena prompt CC-BY-4.0 고지를 유지한다.

### 3. BBQ v1 — 현재 공정성 지향 labeled SFT

- **ID/revision:** `nvidia/Nemotron-SFT-BBQ-v1` @ `4efdb045cda11918f29065a2e0fd5000b73b8c22`; 원본 split `train`.
- **목적/reference:** 나이·외모·장애의 세 범주를 다루는 영어 합성 free-text 대화다. 애매한 경우 근거 부족을 인정하고 명확한 경우 인구학적 추정 대신 행동 근거를 따른다. 공식 객관식 BBQ 평가 benchmark가 아니며 공정성 점수의 증거도 아니다.
- **Mask:** assistant 최종 응답을 지도하고 `reasoning_content`는 제거한다. System/user는 context로 mask한다. 16K/CP1에서 공통 전체 렌더링/assistant 선택 규칙을 적용한다.
- **필터/split/출처:** 공통 품질/split gate를 적용하고 원문과 source offset을 유지한다. 카드의 **6,000**은 필수 release filtering 이전 수량이다. 로컬 전체 schema audit도 실제 6,000행을 방문했지만 전부 적격/train/학습 완료라는 뜻은 아니다.
- **실행/한계/license:** 수락한 16K pilot에 기여한다. Full 적격 수량은 manifest를 기다린다. 영어/세 범주만 다루며 모든 인구학적 공정성을 포괄하지 않는다. OpenMDW-1.1.

### 4. Long-Context Transduction v1 — 현재 제외, 추후 128K 계획

- **ID/revision:** `nvidia/Nemotron-SFT-Long-Context-Transduction-v1` @ `90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0`; 원본 split `train`.
- **목적/reference:** 산술, 16진 token 정렬, 변수 확장, CSV 변환 각 2,500개, 총 10,000개의 프로그램 생성 지시/완전한 정답 쌍이다. User+assistant 메시지이며 tool이나 별도 reasoning trace가 없다. 추후 loss 대상은 user 입력이 아니라 완전한 assistant reference다.
- **현재 실행:** 데이터셋 전체 제외로 **train 0/heldout 0**이다. 2K/16K에 들어가는 짧은 행도 제외한다. Pin된 원본과 기존 heldout blocklist를 보존하며 대체 heldout을 만들어내지 않는다. 과거 4K pilot 참여는 현재 정책이 아니다.
- **추후 context/필터:** 계획 sequence는 **131,072**다. 실제 migrated tokenizer로 **공식 전체 prompt+전체 reference**를 측정한다. `metadata.target_tokens` 및 카드의 `cl100k_base` prompt 생성 목표는 합산 길이가 아니다. 131K 입력과 131K 정답은 131K sequence에 들어가지 않는다. 완전한 예제 길이가 넘으면 제외/집계하고 정답은 자르지 않는다. 표현 variant가 기반 내용을 공유할 수 있어 UUID만으로 독립성을 주장하지 않는다. 독립 검증된 benchmark도 아니다.
- **CP/운영 경계:** 아래 별도 계획을 따른다. Native 128K/CP 실행 증거나 실행 승인이 아니다. OpenMDW-1.1.

### 5. Conversational Tool-Use RL Seed v1 — 예약, labeled SFT 아님

- **ID/revision:** `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1` @ `9715a833ae2de4863e321067211f25efe42f6ebe`; 원본 split `train`.
- **목적:** 추후 NeMo Gym RL을 위한 고객 서비스 simulation seed다. 작성자 기준 100,000개이며 `policy`, `tools`, `customer_scenario` 및 simulation/source metadata를 포함한다. 완성된 gold assistant 대화가 아니다.
- **Mask/reference/context/CP:** **SFT loss mask/gold reference가 없고 현재 SFT context/CP도 할당하지 않는다.** Seed로 assistant 정답을 만들거나 labeled SFT 수량에 더하거나 준비 중 tool을 실행하지 않는다.
- **필터/split/출처/상태:** pin된 metadata와 raw 예약을 별도 보존한다. 현재 예약에는 시나리오 네 개가 있으며 전체 100K를 받았다는 뜻이 아니다. SFT train/heldout에서 제외한다. 추후 simulation/reward/격리/정리는 별도 승인이 필요하다. 기존 MBPP RLVR는 계속 중단한다. OpenMDW-1.1.

## 공통 품질·split·수량 gate

1. 원본 receipt의 revision을 immutable allowlist와 대조한다. 채택 대화에 dataset/revision, file/subset, line/raw byte offset, source UUID, metadata, prompt group/split을 유지하고 offset 원문을 다시 읽어 audit한다. 파생 아티팩트에도 출처·권리를 유지한다. OpenMDW-1.1을 Apache/MIT 재라이선스처럼 취급하지 않는다. Dataset card는 신뢰하지 않는 소스 내용이지 실행 지시가 아니다.
2. 정규화된 전체 content를 전역 중복 제거하고 ID는 source dataset과 묶어서 처리한다. Variant ID 대신 첫 user prompt의 SHA256으로 split을 정한다. 기존 pilot의 **14개 전체** prompt-group 예약을 활성 source에도 유지한다. 실제 존재/부재/비활성 group은 별도 보고한다. Long-source group 네 개가 비활성이어도 blocklist를 줄이거나 heldout 행을 만들지 않는다. Hash grouping은 의미적 동치 검증이 아니다.
3. **KMMLU 34,871문항**에 normalized exact-question/prefix 검사를 적용해 예약한다. Exact match 0은 semantic decontamination이나 새 benchmark 점수가 아니다.
4. 전체 labeled 원본 7개 **4,501,246행** schema audit에서 typed 거부는 **39개**였다. Non-object tool arguments(JSON null 포함) **20개**, 잘못된 argument JSON **2개**, orphan tool response **17개**이며 모두 Agentic tool_calling에서 나왔다. 대화 전체를 quarantine하고 reason/index/type/source offset을 보존한다. 빈 인자로 수리하지 않는다. Schema 통과가 렌더링/mask 통과는 아니다. 예상치 못한 template/ID/mask 오류는 fatal이다. 실제 ordered multiprocess parity는 selector가 건너뛴 중복/거부 행 및 source 경계를 포함해야 한다.
5. 작성자/원본 행 → 적격 unique train/heldout 대화 → 실제 nonpadding token → assistant supervised token → padded tensor token → 실제 처리 노출/반복을 구분한다. 활성 6파일/3 SFT 릴리스는 다섯 릴리스 모두 학습했다는 뜻이 아니다. 작성자 수량에서 mix weight나 최종 채택 수를 추정하지 않는다.

## 실행 증거와 남은 gate

별도 표기가 없으면 아래 경로 앞에 `results/pretraining/nemochat_baby/nemotron_sft/`가 붙는다.

- `seq16k_protocol/parent-pilot-accepted.json`: 실제 16K CP1/EP8 pilot, **8개 rank 각각 11 updates**, final validation, optimizer/RNG save11, 정확한 Engine exit0/OOMfalse. Prepared manifest는 `datasets/nemotron_sft/nemochat_baby/prepared-16k-pilot-03/data/manifest.json`, **83 train/13 heldout**, train supervised token **47,177**, 그중 **21,687**은 position **4,096** 뒤에 있다. 보고된 pilot 최대 reserved memory는 **114.96 GiB**다. Full-run peak 보장이 아니다. Pilot은 CP1에 들어갔고 CP2는 필요하지 않아 운영 미검증 상태다. EP8과 수학적으로 불가능하다는 뜻이 아니다.
- `cpu-full-03/FULL_HANDOFF.json`: 영수증 시점에는 전체 CPU 준비가 실행 중이며 미완료였다. 목표는 `datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json`이다. Writer 정상 종료, counted/re-render audit, 파일/lineage 안정성, atomic `cpu-full-03/completion-ready.json` 이후에만 준비 완료로 판정한다. 중간 manifest 존재만으로는 부족하다.
- `longrun_protocol/pipeline-03/parent-start-readback.json`: controller PID **827560**, `WAITING_DATA`, 당시 owned native run 없음. **기록된 시점에 48h 본 학습은 시작 전**이었다. 이 문서 작업은 live job을 poll/제어하지 않는다.
- 다음 순서: full data 수락 → **그 manifest 전용** 16K GB128 native calibration → 실측 unique-first horizon → parent-owned main supervisor. 실제 `seq16k_protocol/calibration.yaml`은 **15 updates**, cold **5** 제외/warm **10** 측정, 고정 final validation 최대 **128**대화, 최대 **0.75h**다. Production template의 validation cap 256과 다르다. 4K 처리량이나 GB8 pilot 시간을 GB128 calibration으로 재사용하지 않는다.

Checkpoint byte integrity는 계속 `not_checked_explicit_user_waiver`다. 이 문서 때문에 가중치 SHA/CRC, 직접 byte 읽기, forensic 복사를 하지 않는다. 데이터/토크나이저/소스 검사는 별도다. RLVR는 계속 중단한다. 이 설정 문서는 품질 향상, 전체 실행 완료, 업로드, root 성능곡선의 증거가 아니다.

## 추후 128K CP/메모리 계약 — 현재 production 아님

[128K 계획](../../configs/posttraining/nemochat_baby_long_context_128k.plan.yaml)은 `auto_launch: false`, `native_launch_ready: false`, `source_checkpoint: null`(수락된 최종 16K SFT 대기), `global_batch_size: null`(추후 native calibration 대기)이다. 현재 16K SFT를 중단/변경하지 않는다. 현재 native adapter는 CP1의 4K/16K만 허용하므로 이 계획을 그대로 실행 config로 전달할 수 없다.

후보는 물리 GPU 8개, TP1/PP1, EP8 candidate, BF16, microbatch1이다. 보수적으로 **CP8**에서 시작하고 **[8, 4, 2]**를 검토하되 실제 학습 메모리/처리량 증거 뒤에만 CP를 낮춘다. 실제 training step 이후 **15% 여유 메모리**를 요구한다. Pin된 dense/expert grid 및 native Mamba CP partitioning을 각각 검증한다. Dense DP 계산만으로 EP 가능 여부를 단정하지 않는다. CP1 pilot 성공은 후속 CP 후보 검증이 아니다.

후보 확정 전에 131072 길이의 실제 input/label/position/shifted mask, 한계에 가까운 실제 nonpadding forward/backward/optimizer update, CP sample/token/loss 소유권 및 중복 집계 방지, all-rank 논리 parameter shape 불변, 유한한 양수 gradient, full-state save, 정확한 Engine 정상 종료/OOMfalse, 실측 peak memory/headroom/throughput을 검증한다. 지원되는 activation recomputation/microbatch 제어는 이 probe에서 선택하며 CPU 계획에 framework 실행 필드를 설정한 척하지 않는다. 전부 실패하면 중단/보고한다. Driver reset을 하지 않으며 OOM 불가능을 보장하지 않는다. 추후 128K 단계의 **예산은 별도 승인** 대상이다. 현재 SFT의 48h 승인을 자동 이전하지 않는다.

## Pin된 1차 출처

아래는 로컬에서 읽은 카드의 정확한 revision이며 floating `main`이 아니다.

- [Agentic v4 카드](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4/raw/a4593a26ea83a700035bbe0ec1edc9ef3810669d/README.md)
- [Instruction-Following/Chat v4 카드](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4/raw/1cd362945d7a00b136b95b96892bbfea8798686e/README.md)
- [BBQ v1 카드](https://huggingface.co/datasets/nvidia/Nemotron-SFT-BBQ-v1/raw/4efdb045cda11918f29065a2e0fd5000b73b8c22/README.md)
- [Long Transduction v1 카드](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Long-Context-Transduction-v1/raw/90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0/README.md)
- [RL Seed v1 카드](https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1/raw/9715a833ae2de4863e321067211f25efe42f6ebe/README.md)
