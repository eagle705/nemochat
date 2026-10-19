# NemoChat — 실제 최종 SFT1248 (저장소 이름은 SFT-1200)

[English model card](README.md)

**이 private 저장소의 이름은 사용자 요청 그대로 `eagle705/nemochat-10BA1B-BF16-SFT-1200`이지만, 실제 가중치는 native `iter_0001248`, 누적 SFT 1,248 update의 최종본입니다. 1,200 step 모델이 아닙니다.** 비상업 교육/연구용 실험 모델이며, 모델 전용 HF BF16 safetensors를 배포합니다. Native optimizer/RNG 재시작 묶음이 아닙니다.

**지시 이행이 불안정합니다.** 이번 최종1248에서 새로 실행한 영/한 6개 생성 모두 EOS 없이 128-token 상한에 도달했습니다. 반복, 잘못된 한국어 답변, 형식 무시, 잘못된 JSON을 그대로 공개합니다. 유한한 수치 연산은 품질·안전·실용성의 증거가 아닙니다. `en`/`ko` 태그는 의도한/시험한 입력 언어이지 검증된 언어 능력 표시가 아닙니다.

## 실제 학습 계보와 배포 범위

1. BASE native 사전학습 최종 step19,074.
2. C: BASE에서 실제 mid-training1,000 calls, 처리 positions524,288,000. 초기 zero-LR1회와 positive-LR999회. Teacher는 정확히 **`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Base-BF16`**, revision `4ba52e8ffa79e9693936c1a819ac89f551ceeef6`의 88-layer **Base** text backbone이며 Instruct가 아닙니다. Offline cached objective는 `0.5 LM + 0.5 Ghost-tail KL`, T=1 전역 정규화/Top32 FP16 log-probability와 잔여 tail입니다. Online teacher가 아닙니다. [이전 C 카드의 계보·수식·설정](training/inherited_midtraining_modelcard.md)은 당시 릴리스 기록이며 그 카드의 당시 미완료 평가/구형 runtime 제한을 현재 상태로 읽지 마세요.
3. D: C의 **모델만** 로드하고 SFT optimizer/scheduler/RNG/cursor를 새로 시작한 1–905 updates.
4. E, **현재 릴리스**:905에서 **optimizer/scheduler/RNG/cursor 전체 상태를 복구**,343회(906–1248) 추가 진행. **905에서 optimizer를 초기화하지 않았습니다.** 복구한 cursor는115,840 conversations입니다. 최종 model+optimizer+RNG 저장 후 training Engine은 **2026년10월19일14:39:16 KST, exit0**으로 정상 종료했습니다. Hub에는 모델만 올립니다.

실패한 calibration03/04 및 폐기된 runtime05 calibration은 이 누적1,248 trajectory/weights에 포함하지 않습니다. SFT905 가중치 대체, averaging, 이번 배포를 위한 추가 학습, RL/RLVR, WSM merge는 없습니다. **SFT objective는 selected-assistant LM + native router auxiliary 동작이며, SFT loop에는 KD teacher가 없습니다.**

## 구조와 학습 설정

| 항목 | 실제 값 |
|---|---|
| 공식 HF class | `NemotronHForCausalLM`, Transformers5.12.1, `trust_remote_code=False` |
| Parameter | **9,790,967,416**; inherited dense+top-k weight-count 관례상 약964,054,648 active (**FLOPs 아님**) |
| Hybrid 구조 | Mamba/attention/latent-MoE52 serialized blocks; hidden1024/latent512 |
| Expert | routed128/top6; routed width3072/shared width1024 |
| Attention |16heads/2KV/head_dim128 |
| Vocab/embedding |131072/untied; row resize·permutation 없음 |
| Precision |BF16 + FP32 router correction buffer23개 |
| 학습 context |16384; 실제 usable context 품질 검증은 아님 |
| Native |Megatron Bridge0.6.1, DDP/distributed AdamW, B3008장 |
| 병렬/배치 |EP8/TP1/PP1/CP1, global128/micro1 |
| LR/minLR/WD/seed |1e-5/1e-6/0.01/42 |
| 원래 sampler/scheduler horizon |34135 updates, 조기 시간 종료에 맞춰 재설정하지 않음 |
| 최종 optimizer-boundary LR |9.99900171673013e-6; post-scheduler9.998992703267726e-6 |

1–905의 callback LR와 SDK post-step 출력은 서로 다른 관측 stream입니다. 이 구간에 대해 조정·추정한 실제 optimizer-boundary LR를 주장하지 않습니다.906–1248에는 optimizer-step parameter-group LR와 post-scheduler LR를 명시적으로 기록했습니다. 다음 scheduler 값이 현재 update에 적용된 값이라는 뜻은 아닙니다. [설정](training/hyperparameters.json), [전체 관측 CSV](training/observed-metrics.csv).

## 데이터·마스크·실제 소비량

준비된 전체 train pool은 **4,369,280 conversations /3,219,012,753 assistant targets /11,383,390,631 nonpadding tokens**입니다. 별도 heldout은 **45,090 conversations /33,153,239 assistant targets**입니다. **이 전체 pool을 학습 완료한 것이 아닙니다.**

| 선택한 NVIDIA SFT source / 고정 revision | 준비된 train 대화 수 |
|---|---:|
| Instruction-Following-Chat-v4, `1cd362945d7a00b136b95b96892bbfea8798686e` |3,800,207 |
| ↳ chat/response-length/multi-turn |895,553/263,314/2,641,340 |
| Agentic-v4, `a4593a26ea83a700035bbe0ec1edc9ef3810669d` |563,131 |
| ↳ tool_calling/knowledge_agent |511,106/52,025 |
| BBQ-v1, `4efdb045cda11918f29065a2e0fd5000b73b8c22` |5,942 |

정확한 repo/subset/revision은 [provenance](training/data_provenance.json)에 있습니다. 위 수치는 준비된 pool 기준이며 실제 소비한 subset 비율을 측정한 값이 아닙니다. Long-Context-Transduction은 제외했고 RL seed는 SFT에 사용하지 않았습니다. 초과 길이 대화는 전체를 버렸고 잘라 붙이지 않았습니다. 실제 대화의 반복 학습은 없습니다. 선택된 assistant message만 supervision하며 user/system/tool/header는 mask 처리했습니다. Nonthinking에서 reasoning_content를 빼되 공식 empty-think marker는 유지했습니다. Source schema 오류는 대화 전체를 격리했고 정답처럼 고치지 않았습니다. 기존 reservation/exact-match 검사는 완전한 contamination 배제 증거가 아닙니다.

**실제 누적 SFT1–1248 소비량:**
- 고유 대화 **159,744**개, 반복/cycling 없음.
- 로그 합산 **117,409,886 assistant-target exposures**.
- **2,617,245,696 processed positions, padding 포함**. 이를 supervised tokens/고유 텍스트 토큰/전체 corpus 완료라고 부르면 안 됩니다.

Tokenizer는 migrated Super3.5 revision **`fc067b34f6b514014d2f8bddba855d2e4bfdfba6`**, official `chat_template.jinja`를 포함해 accepted export의 바이트를 유지합니다. Reserved alias migration에서 embedding row 및 나머지 ID를 바꾸지 않았습니다. [Upstream README](TOKENIZER_UPSTREAM_README.md)는 attribution이며 이 실험 모델의 카드/성능 보증이 아닙니다. Official `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`를 사용합니다. 직접 확인한 EOS/pad는 **`<|im_end|>` ID11**이고 기존 generation_config도11입니다. 아래 예제는 이 값을 명시적으로 다시 지정합니다. Tokenizer의 큰 model_max_length는 검증된 context 길이가 아닙니다.

## 결합 학습 그래프

![실제 SFT1–1248 trajectory](graphs/sft1248-training-summary.png)

Training LM Loss raw와 trailing16-update 평균, native clipping 이전 Grad Norm, 관측 시점이 다른 LR stream을 표시했습니다. 점선905는 full-state resume이며 새 optimizer 시작이 아닙니다. 전체8개 rank에서 연속1–1248 및 metric 일치를 검사했습니다. Calibration은 제외합니다. 첫16/마지막16 LM 평균은 **2.964211/2.045009**이며 cross-stage Total Loss 비교나 추론 품질 증명이 아닙니다.

Native validation은 heldout45,090개 전체가 아니라 동일한 seeded/source-stratified **고정128대화 window**입니다. 관측 LM Loss:250 **2.299372**,500 **2.132026**,750 **2.028384**,1000 **1.954848**. **최종1248 validation point는 없으며 만들지 않았습니다.** 아래 pretraining-retention NLL과 다른 평가입니다. [원본](training/validation-observations.json).

## 고정 5-arm 평가 — 작은 subset의 기술적 관측

![5-arm benchmark](graphs/benchmark-five-arms-SFT1248.png)
![별도 pretraining-retention NLL/PPL](graphs/heldout-five-arms-SFT1248.png)

| Arm |MMLU acc/200|ARC acc_norm/100|HellaSwag acc_norm/100|Retention NLL/65536 targets|PPL|
|---|---:|---:|---:|---:|---:|
|A BASE19074|24.0%|33.0%|52.0%|2.192594311851152|8.958423939014796|
|C LM+KD1000|22.5%|35.0%|52.0%|1.278155480928284|3.590011769576160|
|B LM-only1000|20.0%|32.0%|52.0%|1.288966996663131|3.629035812717152|
|D SFT905|22.0%|33.0%|48.0%|1.373007705764395|3.947204890471098|
|**E FINAL SFT1248**|**22.5%**|**35.0%**|**48.0%**|**1.399376922740647**|**4.052674050965295**|

E만 새로 평가했고 A/C/B/D 보고서는 변경 없이 재사용했습니다. D 대비 E는 +0.5/+2.0/0.0%p이나 retention NLL/PPL은 **악화**했습니다. 일반적인 품질 개선, 인과관계, 통제된 KD ablation이라고 주장하지 않습니다.

MMLU10subjects×20 test/5-shot, ARC-Challenge100test 및 HellaSwag100validation/0-shot: 총400questions/1600choices입니다. **Chat template 없는 raw prompt**, full-vocab likelihood, MMLU acc 및 문자 길이 정규화 ARC/HellaSwag acc_norm입니다. Official task source에서 유도한 scorer이며 **공식 harness backend 실행이 아닙니다**. 별도 retention은 고정 pretraining heldout64windows/65536targets로 SFT validation과 다릅니다. Reserved alias의 frozen1600encodings 일치 및 heldout에 해당 ID 부재를 검사했습니다. 초기 model-load 전 comparison-tokenizer/receipt permission 실패는 보존했고 성공 E scoring은1회입니다. Full benchmark/leaderboard 점수, 통계 유의성, contamination 해소, chat/tool 안전성을 입증하지 않습니다. [한국어 보고서](evaluation/FINAL_REPORT.ko.md), [영어 보고서](evaluation/FINAL_REPORT.md), [CSV](evaluation/scores.csv), [JSON](evaluation/summary.json), [E 상세](evaluation/E-report.json).

## 이번 최종1248에서 새로 만든 실제 영/한 생성

이전 진단과 **동일한6개 prompt**를 model load 전에 고정하고 각1회만 생성했습니다. Retry, best-of 선택, prompt 수정, 정답 대체, 예전905 생성문 재사용은 없습니다. Greedy/seed42/num_beams1/max_new_tokens128, official nonthinking template, offline BF16/eager B3001장, Transformers5.12.1입니다. 모든 loaded tensor와 generation score는 finite, loading error list는 비어 있고 DynamicCache가 반환되었습니다. **6개 모두128tokens/EOS0개**입니다. 수동 실패 진단이지 품질 benchmark가 아닙니다.

아래는 실패를 숨기지 않은4개 **원문 prefix 발췌**입니다. 화면 표시만 줄였으며 **전체6개의 무편집 결과/입출력 token IDs/시간/cache/EOS/settings**는 [examples.json](examples/examples.json), [사전 고정 prompts](examples/prompts-predeclared.json), [load 전 generation plan](examples/generation-plan-before-load.json)에 있습니다.

@@EXAMPLES@@

## 실행 예제와 검증 범위

Private repo 접근 권한으로 먼저 Hugging Face에 로그인하세요. 코드/로그에 credential을 넣지 마세요. 시험한 runtime은 Transformers**5.12.1**, Torch**2.13.0a0+8145d630e8.nv26.06**, Python3.12, B300 BF16/eager입니다. 가중치 약20GB 외 cache/runtime memory가 필요합니다. 다른 backend 호환성을 보장하지 않습니다.

@@SNIPPET@@

공식 text AutoTokenizer/AutoModel 경로를 검증했으며 remote code는 필요 없습니다. 상속된 multimodal AutoProcessor 지원을 뜻하지 않습니다. Optional Mamba fast kernel은 꺼져 있고 `time_step_min=0` native scan lower bound를 유지했습니다. Export는 검토한 pinned Bridge registry/global-DCP adapter를 사용했으며 stock Bridge export CLI를 실행한 것이 아닙니다. Native619→serialized6289→runtime447 records, grouped experts 및 FP32 router buffer를 보존하고 missing-key random fallback은 사용하지 않았습니다. 전체 공식 GPU reload/finite forward/cached generation과 제한된 likelihood 평가를 실제 실행했습니다.

- **Native/HF 수치 logit parity는 미실행**입니다. Finite/shape/benchmark 검사가 parity를 대신하지 않습니다.
- **원본 checkpoint payload SHA/CRC forensic read/copy 감사는 사용자 명시 면제로 미실행**입니다. 필요한 Hub upload protocol hash는 허용되지만 원본 감사 완료로 바꾸지 않습니다. Remote 검증은 immutable commit/inventory/작은 asset byte 일치이며 weight를 다시 다운로드하지 않습니다.
- Native optimizer/RNG/DCP, 학습 원문, teacher weights/cache, credential/cache는 배포하지 않습니다. [Publication manifest](publication_manifest.json)에 실제1248 source와 native save/model-only export 연결을 남겼습니다.
- Long-context 품질, 안전/red-team, production, vLLM/TensorRT-LLM, native kernel parity, 언어 숙련도/신뢰할 수 있는 tool 실행을 보장하지 않습니다. 영어와 특히 한국어 실패를 그대로 남겼습니다.
- **비상업 교육/연구 목적, 새로운 learned-weight 권리 부여 없음**. NVIDIA source attribution, inherited ClimbMix **CC-BY-NC4.0**, selected Specialized-v1.1 **CC-BY4.0**, pinned teacher/SFT data의 적용 가능한 **OpenMDW1.1** 조건을 보존합니다. 선택한 SFT source는 manifest상 모두 OpenMDW1.1입니다. Repack의 MIT/Apache code/data 표기가 upstream 제약이나 learned-weight 권리를 덮어쓰지 않습니다. [USE_CONSTRAINTS.md](USE_CONSTRAINTS.md)는 새 상업/재배포 권한이나 법률 보증이 아닙니다.
