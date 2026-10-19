# NVIDIA 데이터셋 추가 SFT, Super3.5 토크나이저 전환, 비공개 BASE 게시

[English](../en/NEMOTRON_SFT_AND_TOKENIZER_TRANSITION.md) · [저장소](../../README.md) · [버전 관리 데이터셋별 설정](NEMOTRON_DATASET_SETTINGS.md)

현재 데이터셋 정책, mask·필터·출처, 16K 실행 설정과 별도 보류한 128K CP 계획은 [데이터셋별 설정 문서](NEMOTRON_DATASET_SETTINGS.md)에 모았다. 아래 과거 snapshot은 그대로 보존한다.

## 결정과 증거 범위

기준 시점: **2026-10-17 10:16 KST**. 사용자는 MBPP RLVR를 일시 중단하고, 지정한 NVIDIA 릴리스 다섯 개를 검토해 추가 SFT를 준비하며, 예약 토큰 이름만 전환하고, **사전학습 BASE**를 비공개 게시하도록 요청했다. 서로 보완하는 작업이며 RLVR 자동 재개나 SFT 체크포인트를 BASE로 게시하라는 뜻이 아니다.

**실행 완료**, **준비됨**, **미완료**는 구분한다. `results/`의 실행 증거는 로컬/ignored 아티팩트이므로 새 clone에는 없을 수 있다. 아래 경로는 저장소 루트 기준의 증거 위치이며 데이터셋·가중치를 Git으로 버전 관리한다는 뜻이 아니다. 기존 README 구현 표는 이전 CPU-first 기준선 설명이다. 이 실행의 상태는 아래 단계별 증거를 보되, 나머지 로드맵까지 실행됐다고 해석하지 않는다.

| 작업 | 기준 시점에 관측된 상태 | 남은 승인 조건 |
|---|---|---|
| 기존 Smol 한국어/영어 SFT | **실행 완료:** 8개 rank 각각 native 391회 업데이트와 최종 저장 | 새 NVIDIA 데이터 SFT와 별개 |
| SFT 후 평가 | **실행 완료:** 요청한 KMMLU 100문항 부분집합에서 0-shot 21/100; 실제 한국어 생성은 반복 발생 | 새 후보는 조건을 맞춰 재평가 |
| MBPP RLVR | **중단 요청됨; 관측기에서 controller inactive.** 모델 rollout 59개 모두 reward 0; all-rank 공통 optimizer update 0회 | 정확한 Engine 종료 및 소유 sandbox 전체 inventory/정리 영수증; 자동 재시작 금지 |
| NVIDIA 소스 릴리스 5개 | **탐색 준비 완료:** pin된 카드 5개, 메타데이터, 파일 inventory 확인; 원격 카드와 로컬 사본 일치 | 제한된 다운로드, 실제 선택·mask·토큰 집계와 native 연결 |
| Super3.5 토크나이저 | **소스 확인:** 예약 이름 12개 교체; 구현 준비 중 | 전체 기존 corpus/SFT ID 검사 manifest 및 호환 runtime 토큰화 검증 |
| 새 NVIDIA 데이터 native SFT | **미완료:** 새 pilot/학습 완료 증거 없음 | 제한된 본 실행 전에 실제 짧은 native pilot |
| 비공개 HF BASE 게시 | **요청됨/미완료:** `eagle705/nemochat-10BA1B-BF16-base` | 실제 export 형식, 업로드, immutable revision, 원격 private/card/inventory 재조회 |

### 기존 실행 증거와 부정적 결과

- 기존 SFT: `results/pretraining/nemochat_baby/sft/runs/sft-main-01/verification.json`은 rank별 391회 업데이트와 250/391 저장을 기록한다. `results/pretraining/nemochat_baby/SFT_COMPLETION_REPORT.md`에 **2026-10-17 02:45:06 KST** 종료, KO/EN 쌍 25,000개 주제에서 얻은 대화 50,000개, heldout 대화 1,000개, assistant loss-bearing target 34,535,226개가 기록돼 있다. 최종 heldout CE **2.5068976879119873**, PPL **12.26681547947325**는 assistant-target SFT 지표다. KMMLU도, 사전학습 loss와 직접 비교 가능한 값도 아니다.
- 완료 보고서의 당시 문구인 “post-SFT KMMLU has not been run”은 이후 증거로 갱신됐다. `results/pretraining/nemochat_baby/post_sft_eval/engines/native-0shot-100-01/verification.json`은 **0-shot 21/100**, 45개 과목/400개 logical choice score, 8개 rank 복원·동결, native exit 0을 확인한다. 이전 **28/100**은 **PRETRAIN 4-shot**이었다. 모델과 prompting 조건이 모두 달라 SFT 향상이나 인과적 퇴보를 입증하지 않는다.
- 같은 평가 영수증에 비 오는 날 준비물 세 가지를 묻는 실제 greedy 한국어 생성이 남아 있다. `이 경우`를 반복하다 128토큰 제한에 도달해 잘렸고 정상 종료하지 않았다. 좋은 답변이나 전체 성능 추정이 아니라 instruction-following 실패 사례다.
- `results/pretraining/nemochat_baby/post_sft_handoff/rlvr-observer-latest.json`의 **2026-10-17 10:07:34 KST** 관측은 controller `inactive`, 판정 rollout 59개, all-eight 공통 accepted update 0회를 기록한다. `rlvr/runs/rlvr-main-02/reward_responses/*.json` **59개**를 직접 집계한 결과 **reward 0이 59개, 양수 reward 0개**였다. 이전 09:42 progress의 40개는 오래된 값이다. 이 영수증은 실제 모델의 MBPP 시도이지 성공한 학습 업데이트가 아니며, RLVR 체크포인트 저장도 입증되지 않았다.
- RLVR Engine 식별자: `ab1d48467964e1bb70f2d0cdc19d2f35cd7aa1c44c36416898aae8e96152e3f6`. Controller inactive나 서비스 포트 종료만으로 모든 소유 guest 제거가 입증되지는 않는다. 종료 요청·exit 및 정확한 guest/container inventory를 별도로 보존한다. 이 문서는 전체 정리 완료를 인증하지 않는다.

체크포인트 byte integrity는 계속 **`not_checked_explicit_user_waiver`**다. 이 문서를 완성한다는 이유로 가중치 SHA/CRC 검사, 직접 읽기, forensic 복사를 다시 시작하지 않는다. 토크나이저·데이터·소스 검사는 별도이며 허용된다. Native 복원·저장 영수증은 byte integrity나 HF 로딩 호환성 인증이 아니다.

## 사용자가 선택한 NVIDIA 릴리스 다섯 개

Pin된 카드 다섯 개 모두 영어 텍스트와 OpenMDW-1.1 조건을 표기한다. 실제 NVIDIA 데이터 릴리스이며 임의로 만든 대체 이름이 아니다. 아래 수량은 **작성자 카드의 소스 수량**이지 다운로드·선별·중복 제거·학습된 수량이 아니다.[1][2][3]
RL seed와 BBQ 수량의 추가 조건은 따로 명시한다.[4][5]

| 릴리스 | 고정 revision | 작성자 표기 수량 | 로컬 사용 방침 |
|---|---|---|---|
| `nvidia/Nemotron-SFT-Agentic-v4` | `a4593a26ea83a700035bbe0ec1edc9ef3810669d` | tool_calling 549,707; knowledge_agent 84,957 | 정답 대화가 있는 SFT; tools와 call/result 연결 보존.[1] |
| `nvidia/Nemotron-SFT-Instruction-Following-Chat-v4` | `1cd362945d7a00b136b95b96892bbfea8798686e` | 길이 제어 266,190; chat 914,716; multi-turn IF 2,669,676 | 정답 SFT; 작성자의 turn mask 필수.[2] |
| `nvidia/Nemotron-SFT-Long-Context-Transduction-v1` | `90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0` | 완전한 지시/정답 쌍 10,000개 | 현재 context에 **전체 입력과 정답**이 모두 들어가는 사례만 사용.[3] |
| `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1` | `9715a833ae2de4863e321067211f25efe42f6ebe` | 시뮬레이션 seed 100,000개 | 추후 agentic Gym RL용으로 보류; **gold assistant-answer SFT가 아님**.[4] |
| `nvidia/Nemotron-SFT-BBQ-v1` | `4efdb045cda11918f29065a2e0fd5000b73b8c22` | **필수 release filtering 이전 원본** 6,000개 | 근거 기반 공정성 SFT; 실제 공개·선별 행 수 별도 집계.[5] |

작성자의 고정 카드 링크는 아래 **Sources**에 있으며 floating `main` 링크가 아니다. 탐색 파일은 `results/pretraining/nemochat_baby/nemotron_sft/discovery/<release-name>.json`이다. 각 파일에 `id`, `sha`, `card`, `cardData`, 소스 파일 inventory가 있다. 문서 작성 중 pin된 원격 README 다섯 개를 다시 읽었으며 모두 저장된 `card` 문자열과 정확히 일치했다.

### 텍스트만 옮기지 말고 작성자의 의미를 보존

1. **Agentic v4:** `tool_calling`, `knowledge_agent`는 두 SFT subset이다. 메시지 순서, tool 정의, assistant tool call, call ID와 대응 tool-result context를 보존한다. Knowledge-agent는 대화 궤적이며 별도의 문서 corpus/backend를 제공하지 않는다. 카드는 non-reasoning 학습 시 `reasoning_content` 제거를 허용하지만 응답과 tool call은 유지하도록 안내한다.[1]
2. **Instruction-Following/Chat v4:** `metadata.train_turns_mask`로 loss를 줄 assistant turn을 선택하고 나머지 메시지는 context로 보존한다. `reasoning_content`를 제거해도 작성자 mask를 버리거나 모든 assistant turn을 몰래 지도하면 안 된다. Final-response-only/non-thinking 방침은 이 필드를 무시해도 된다는 뜻이 아니다.[2]
3. **Long-context transduction:** 규칙으로 생성된 완전한 정답은 산술, 16진 토큰 정렬, 변수 확장, CSV 변환을 포함한다. 2,048–131,072 prompt budget은 `cl100k_base` 기준 생성 목표이지 Super3.5 실측 길이나 입력+출력 총길이가 아니다. 실제 토크나이저로 전체 렌더링 길이를 재고 초과 사례는 이유·수량을 기록해 제외한다. 정답을 조용히 자르지 않는다. 이 데이터로 현재 **4,096토큰** native 모델에 131K 능력이나 독립 benchmark 결과가 생기지는 않는다.[3]
4. **RL seed:** `policy`, `tools`, `customer_scenario`와 시뮬레이션 메타데이터를 제공하는 NeMo Gym 입력이다. 완성된 gold assistant 대화가 아니다. 정책·시나리오로 가짜 assistant 정답을 만들거나, seed 100,000개를 labeled-SFT 수량에 더하거나, 데이터 준비 중 tool을 실행하지 않는다. 후속 RL에는 별도 승인된 시뮬레이션·reward·격리·정리가 필요하다.[4]
5. **BBQ SFT:** 애매한 context에서는 근거 부족을 인정하고, 명확한 context에서는 인구학적 추정 대신 행동 근거를 사용한다. BBQ 공정성 설계를 참고한 합성 free-text 학습 자료이지 **공식 객관식 BBQ benchmark 재사용**이 아니다. 카드의 원본 6,000개와 실제 공개·선별 행 수를 구분한다.[5]

각 소스와 export에 출처·권리 메타데이터를 유지한다. 카드의 OpenMDW-1.1 표기가 파생 구성요소까지 Apache/MIT로 바꾸지는 않는다. Agentic은 CC-BY-4.0 persona grounding, IF/Chat은 WildChat ODC-By와 LMArena CC-BY-4.0 prompt를 명시한다. 해당 고지를 보존하고 재배포 조건은 별도로 검토한다.[1][2] 다섯 소스는 모두 영어다. 이름만으로 한국어 instruction-following 개선을 주장할 수 없으며 실제 평가할 가설이다.[1][2][3]

## 제한된 추가 SFT 계획 — 실행 결과가 아님

위 **labeled SFT 릴리스 네 개**를 사용하고 RL-seed 릴리스는 보류한다. 이미 캐시돼 있다는 이유로 Smol KoreanTalk이나 ClimbMix로 조용히 되돌아가지 않는다. 첫 증분은 실제 선별 가능 수량·소스별 quota·실측 **24시간** 예산을 조건으로 **적격 대화 약 50K–100K**를 검토한다. 기준 시점에 적격 수량, 정확한 mix, 예상 개선, 처리량은 확정되지 않았다.

- JSONL/prefix 확보는 명시된 다운로드 예산(**초기 최대 8 GB**) 안에서 제한한다. 전체 대형 소스 모음을 몰래 받지 않는다. Prefix/선택 표본은 릴리스 전체를 대표하는 조사가 아니다.
- Pin된 행의 출처, subset, 작성자 mask, tool 연결, split grouping, 제외 사유, 채택 수량, 렌더링 토큰 길이, assistant-target 수를 남긴다. 학습/heldout 생성 전에 중복 제거와 분리를 수행한다. Exact-match 검사만으로 의미적 benchmark decontamination을 주장하지 않는다.
- 호환 runtime에서 승인된 실제 Super3.5 토크나이저/template를 쓴다. 실제 선별 사례로 assistant-only loss, 선택-turn mask, 응답 종료자, tool 문법, non-thinking 동작을 검증한다. 합성 fixture만 통과한 것은 충분하지 않다.
- **SFT391**에서 model-only로 이어가되 SFT optimizer/scheduler/RNG/data cursor는 새로 시작하고 native normalization/router 동작은 유지한다. 원래 PRETRAIN과 SFT391을 보존한다. Pretraining optimizer를 재개하거나 저장되지 않은 RLVR 후보로 학습하지 않는다.
- 실제 짧은 native pilot(예정된 16/4-update 검사), all-rank load/update/save 영수증, 유한한 양수 gradient, 선택-target 집계와 heldout/생성 검사를 본 실행의 gate로 둔다. 준비 테스트나 CLI help 성공은 이 gate가 아니다.
- 현재 native context는 **4K**다. 완전한 long-transduction 사례 중 들어가는 것만 유지하고 초과 제외 수량을 명시한다. 더 긴 context 적응은 별도 미완료 작업이다. 특정 요청 소스의 채택 수가 0이면 이를 보고하며 다섯 개 모두 학습했다고 주장하지 않는다.

## 토크나이저 전환: 예약 ID 이름만 변경, 가중치 크기·행 순서 불변

기존 토크나이저는 `checkpoints/tokenizers/2dc98e2afe4face0e4ce40972a915c45368bd34a/`에 보존한다. Vocab 크기는 **131,072**다. 대상은 `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`의 **토크나이저 파일만**이며, 10:16 조회에서 revision `1f8953f7f0935358e3605150d22bc15df062061c`를 확인했다. 해당 대형 모델 가중치 다운로드나 Baby 아키텍처 교체 요청이 아니다.[6]
이후 제공된 migration config는 탐색 당시 floating HEAD가 아닌 **`fc067b34f6b514014d2f8bddba855d2e4bfdfba6`**를 pin한다. 이 revision의 토크나이저도 직접 읽어 아래 이름 12개가 같음을 확인했다. 재현 시 latest HEAD로 바꾸지 말고 **config/manifest pin**을 따른다.[7]

| 기존 ID | 기존 이름 | Super3.5 이름 |
|---|---|---|
| 18 | `<SPECIAL_18>` | `<image>` |
| 19 | `<SPECIAL_19>` | `<img>` |
| 20 | `<SPECIAL_20>` | `</img>` |
| 21 | `<SPECIAL_21>` | `<quad>` |
| 22 | `<SPECIAL_22>` | `</quad>` |
| 23 | `<SPECIAL_23>` | `<ref>` |
| 24 | `<SPECIAL_24>` | `</ref>` |
| 25 | `<SPECIAL_25>` | `<box>` |
| 26 | `<SPECIAL_26>` | `</box>` |
| 27 | `<SPECIAL_27>` | `<so_embedding>` |
| 28 | `<SPECIAL_28>` | `<so_start>` |
| 29 | `<SPECIAL_29>` | `<so_end>` |

새 이름 12개는 pin된 대상 토크나이저에서 직접 확인했다.[6] 전환 승인 조건은 다음과 같다.

1. 전체 backend를 비교한다. 정확히 ID 18–29의 승인된 이름만 달라질 수 있다. 다른 모든 text/control ID, vocab 크기, BPE merge, normalizer, pre-tokenizer, decoder는 유지한다. 추가 차이가 있으면 자동 보정하지 말고 거부한다.
2. **적용 전에** 실제 packed 사전학습 corpus와 기존 prepared SFT token ID에서 18–29 각각을 전수 검사한다. 전체 검사 경로·토큰 수·ID별 출현 수·누락을 기록한다. Fixture/부분 표본은 요청한 전체 10B-corpus 검사가 아니며, manifest가 생길 때까지 결과는 미완료다.
3. 기존 토크나이저/체크포인트 lineage를 변경하지 않고 별도 migrated tokenizer를 만든다. 이름 변경은 embedding/output vocab 확대, 행 재정렬, 재초기화, 가중치 재작성을 승인하지 않는다.
4. 대표 기존/신규 텍스트와 실제 학습 사례에서 chat/template 및 text-ID parity를 확인한다. 예약 이름 변경의 영향을 받는 literal 문자열과 응답 종료자를 명시한다.

이미지를 연상시키는 토큰 이름이 **시각 입력, vision encoder, 멀티모달 능력을 추가하지는 않는다**. 입력/정답 ID 출현이 0이어도 해당 output 행이 “한 번도 학습되지 않았다”는 뜻은 아니다. 양성 target 출현이 없어도 softmax gradient가 output 행에 영향을 줄 수 있다. Corpus 검사와 행 정렬 불변은 더 좁은 호환성 질문을 확인한다.

호스트 Transformers **4.57.6**의 `AutoTokenizer`가 새 `TokenizersBackend`를 읽는다고 가정하지 않는다. 실제 토큰화는 이미 사용 가능한 Transformers **5.12.1** native NeMo runtime에서 수행한다. 공유 호스트를 업그레이드하거나 임의 remote tokenizer code를 켜서 우회하지 않는다. 실제 runtime 버전을 migration/preparation 증거에 남긴다.

예정된 migration 증거 위치(기준 시점에는 아직 미생성): `results/pretraining/nemochat_baby/tokenizer_migration/manifest.json`.

## 비공개 HF 게시: PRETRAIN BASE는 별도 lineage

대상은 **`eagle705/nemochat-10BA1B-BF16-base`**, visibility는 명시적으로 **private**다. 소스는 `results/pretraining/nemochat_baby/training/runs/baby-main-01/checkpoint/iter_0019074`이며, **SFT391도 RLVR도 새 토크나이저 전환/SFT branch도 아니다**.

BASE 패키지는 실제 사전학습 lineage를 설명하도록 **기존 토크나이저/예약 이름**을 유지한다. Posttraining alias 전환은 별도 variant다. 로컬 raw native resumable checkpoint는 보존하며 BF16 model-only 게시물과 optimizer/RNG를 포함하는 native 학습 상태를 구분한다.

모델 카드에는 아키텍처/config와 실제 직렬화 형식, 사전학습 소스·step과 기존 tokenizer pin, instruction-tuned가 아닌 BASE의 한계, 해당 출처·권리, 평가 조건·주의점, 명시적 byte-audit 면제를 기록한다. Native DCP archive를 “HF Transformers safetensors”라고 부르거나 실제 exporter/loader 검증 없이 `AutoModel.from_pretrained` 지원을 약속하지 않는다. 저장소 이름 끝의 BF16은 형식·호환성 증거가 아니다.

정확한 원격 대상 재조회로 **private visibility**, immutable commit revision, 카드 내용, 기대 업로드 inventory를 확인한 뒤에만 게시 완료로 처리한다. 로컬 export, 저장소 생성, upload 요청 성공, 다른 worker의 의도는 완료 영수증이 아니다. 문서 worker는 가중치를 업로드하지 않았으며 이 snapshot에서 게시 상태는 미완료다. [기존 checkpoint 패키징 경계](HF_CHECKPOINT_PIPELINE.md)도 참고한다.

## 재현 인터페이스와 남은 영수증

저장소 루트에서 실행한다. 아래는 code worker와 합의한 entrypoint이며 **기준 시점에 구현·실행됐다고 단정하지 않는다**.

| 예정 entrypoint | 목적 | 관련 계약 |
|---|---|---|
| `scripts/posttraining/prepare_tokenizer.py` | 소스 pin, backend 호환성, 기존-ID 검사, migration 아티팩트 | `configs/tokenizers/nemotron_super35.yaml`; `src/nemochat/tokenization/posttraining.py` |
| `scripts/posttraining/nemotron_sft.py` | 제한된 소스 준비와 선택-turn/token 집계 | `configs/posttraining/nemochat_baby_nemotron_sft.yaml`; `src/nemochat/data/nemotron_sft.py` |
| `scripts/posttraining/publish_hf_base.py` | 별도 비공개 PRETRAIN BASE 패키징/게시 | `src/nemochat/hub/base_checkpoint.py` |

10:16 KST에 스크립트 세 개와 해당 config는 아직 없어 `--help` 성공이나 subcommand/flag 이름을 주장하지 않는다. 토크나이저 모듈은 작성 중이었으며 파일 존재만으로 전체 CLI 준비가 입증되지는 않는다. 작업 명령을 만들기 전에 아래 비파괴적 존재/help 검사를 사용한다. 이 검사는 **데이터 준비, GPU 학습, 업로드를 하지 않는다**.

```bash
for entry in \
  scripts/posttraining/prepare_tokenizer.py \
  scripts/posttraining/nemotron_sft.py \
  scripts/posttraining/publish_hf_base.py
do
  if test -f "$entry"; then
    .venv/bin/python "$entry" --help
  else
    printf 'PREPARING: %s\n' "$entry"
  fi
done
```

**준비 상태 갱신, 2026-10-17 10:21 KST:** `scripts/posttraining/prepare_tokenizer.py`와 `configs/tokenizers/nemotron_super35.yaml`이 생성됐다. 문서 worker가 `.venv/bin/python scripts/posttraining/prepare_tokenizer.py --help`를 실제 실행해 exit 0을 확인했다. Positional stage `download`, `audit`, `verify-runtime`, `publish`와 `--config`를 제공한다. 여기의 `publish`는 HF 가중치 업로드가 아니라 **로컬 tokenizer migration manifest 작성**이다. 소스가 안내하는 순서는 위와 같다. `verify-runtime`은 호환 native SDK CPU runtime에서 실행해야 하며 비호환 호스트에서 수행하지 않는다. 문서 worker는 이 네 단계를 실행하지 않았고, 준비 상태 확인 시 migration manifest는 여전히 없었다. SFT와 BASE 게시 CLI는 아직 없어 flag를 명시하지 않는다.

파일명만 보고 준비/학습/업로드 option을 추측하지 않는다. Code worker가 CLI를 제공하면 실제 `--help`를 저장하고 지원 flag만 사용하며 정확히 승인·실행된 명령과 영수증을 여기에 추가한다. Native launch에는 호환 SDK와 parent의 GPU 승인/gate가 추가로 필요하다. 이 문서에서 실행 명령을 만들어내지 않는다.

미완료 항목을 실행 완료로 바꾸기 전에 다음을 보존한다: tokenizer migration/전체-ID 검사 manifest, 실제 소스별 채택·제외·target 수가 있는 pin된 다운로드/prepared-data manifest, source/mask/template gate, 실제 native pilot 후 본 실행 영수증, 동일 조건 품질 진단, 명시적 RLVR stop/소유 자원 정리 증거, 비공개 BASE 원격 게시 영수증. 문서와 CPU 검사는 이 결과를 대체하지 않는다.

## 실행 결과 갱신 — 2026-10-17 10:46 KST 이후

이 절은 위 준비 시점의 상태를 갱신한다. 과거의 `pending` 기록은 당시 상태이며 현재 실패를 뜻하지 않는다.

- **비공개 BASE 게시 완료:** `eagle705/nemochat-10BA1B-BF16-base`, immutable commit `94422e79dd31ede40e52a9940367115b690d5048`. Parent가 `publish_hf_base.py verify --revision ...`를 재실행해 private 설정, payload 17개/19,620,927,468 bytes 및 모델카드 일치를 확인했다. Hub 자동 생성 `.gitattributes`는 payload 수에서 별도 취급한다.
- **형식 제한:** model-only native Megatron/PyTorch DCP이며 optimizer/scheduler/RNG/cursor를 포함하지 않는다. Transformers `from_pretrained` 변환이 아니다. 전체 export의 native reload/forward는 아직 검증하지 않았고 모델카드에도 명시했다. 원래 Super3 tokenizer를 보존했으며 새 post-training tokenizer와 섞지 않았다.
- **Tokenizer 전환 완료:** `tokenizer_migration/manifest.json` passed. 전체 131072 ID 중 18–29만 이름이 다르고 나머지 backend 구조는 동일하다. ClimbMix train/validation/calibration과 기존 SFT train/heldout의 총 10,227,216,313 stored IDs를 스캔해 18–29 모두 0회를 확인했다. 기존 가중치와 tokenizer는 그대로 보존했다. 실제 pinned CPU runtime의 production loader와 no-thinking 렌더링도 통과했다.
- **실제 pilot 데이터 준비 완료:** 82 train + 14 heldout 대화, supervised tokens 46,900/11,706. 네 SFT 데이터셋이 train에 기여하며 RL Seed는 별도 보존했다. 전체 96개 원본 재렌더링으로 공식 template/token IDs/shifted masks를 확인했다. Long Transduction 16개는 완전한 prompt+reference가 2,831–3,506 tokens에 들어간다. Knowledge-agent는 표본 344개 모두 4K를 초과해 이번 pilot에 포함하지 않았다.
- **Native pilot 실행 시작:** `nemochat-baby-nemotron-pilot-01.service`, Engine `eab7fdb86e7c48945ae5a01d9a1175411527870d4f0f3ef0785442742e3f256f`. SFT391 model-only warmstart, EP8/GBS8, 11 updates 계획이다. 실행 시작은 update/save 완료의 증거가 아니다. 최종 `runs/nemotron-pilot-01/verification.json`과 all8 receipts로 수락한다.
- **본 학습은 아직 미완료:** 50K–100K main corpus의 bounded acquisition/preparation을 별도 CPU 작업으로 진행하며 pilot heldout 14개 전체를 예약한다. tiny pilot을 반복해 main으로 부르지 않는다. RLVR는 계속 중단 상태다.
- Parent focused tests: tokenizer/data/publication 합계 **24 passed**. 전체 suite의 기존 scratch-filesystem `renameat2 EINVAL` 실패는 해결/통과했다고 주장하지 않는다.

재검증/실행 CLI (repo root):

```bash
.venv/bin/python scripts/posttraining/publish_hf_base.py verify \
  --revision 94422e79dd31ede40e52a9940367115b690d5048
.venv/bin/python scripts/posttraining/launch_nemotron_sft.py \
  --name nemotron-pilot-01 \
  --manifest datasets/nemotron_sft/nemochat_baby/prepared-pilot-01/data/manifest.json \
  --pilot --online --run-id nemochat-baby-nemotron-sft-pilot-01 --execute
```

두 번째 명령은 이미 실행된 정확한 명령이다. 기존 이름/출력 경로로 재실행하지 않는다. 증거: `results/pretraining/nemochat_baby/{hf_base_publication/publication_receipt.json,tokenizer_migration/manifest.json,nemotron_sft/README.md,nemotron_sft/pilot-01-controller.log}`. Checkpoint byte waiver는 그대로 유지했다.

### Pilot 수락 — 2026-10-17 10:53 KST 종료

Parent가 native verifier를 다시 실행하고 정확한 Engine을 재조회했다: **all8 11/11 updates, optimizer/RNG final save11, entrypoint0, Engine exit0/OOMfalse**. 실제 checkpoint는 `results/pretraining/nemochat_baby/nemotron_sft/runs/nemotron-pilot-01/checkpoint/iter_0000011`이다. W&B도 exact run `nemochat-baby-nemotron-sft-pilot-01`을 재조회해 `finished`/step11을 확인했다. Final Training Loss0.7940755486488342, fixed-window Validation CE1.7139155864715576. 이는 82개 대화 pilot의 실행 증거이며 전체 추가 SFT 완료나 일반화 품질 개선의 증거가 아니다. 본 학습용 CPU 데이터 준비는 진행 중이며 아직 main GPU run은 실행하지 않았다. 증거는 `parent-pilot-accepted.json`과 `parent-pilot-wandb-readback.json`에 남겼다. Main은 tiny pilot을 반복하는 대신 SFT391에서 독립적으로 시작한다. Checkpoint byte audit/reload를 새로 수행했다고 주장하지 않는다.

### 48시간/전체 데이터 승인 및 GB128 calibration 갱신

사용자가 약 48시간 SFT와 **네 labeled SFT repo의 전체 7개 파일, 100,853,847,138 bytes** 확보를 명시적으로 승인했다. 위 8GB/24h 표본 제한은 이전 준비 정책으로 대체되었다. RL Seed는 계속 별도 예약이며 정답을 만들지 않는다. Parent가 `nemochat-nemotron-full-acquire-01.service`와 `nemochat-nemotron-full-prepare-01.service`의 실제 실행을 확인했다. 전체 파일의 source digest/길이 검사와 24-worker CPU 준비·mask audit를 거친 후에만 full manifest를 수락한다.

GB128 calibration01은 20 updates/save/heldout을 수행했지만 adapter 강제 평가와 SDK 자동 최종 평가가 중복되어 iterator 소진/exit1로 실패했다. 실패를 보존하고 RED→GREEN 회귀검증 후 수정했다. 새 calibration02는 Parent 재검증으로 **all8 20 updates, save20, 최종 heldout868, Engine exit0/OOMfalse**를 확인했다. Warm15 updates 평균12.352091671002563 sec, heldout CE2.424147129058838이다. 이는 전체 모델 품질 향상의 증거가 아니라 운영·처리량 검증이다. 해당 receipt는 82,557-row manifest에 묶여 있으며 full manifest로 임의 재사용하지 않는다.

**48시간 본 학습은 아직 시작 전이다.** Full CPU 준비와 그 manifest에 대한 calibration이 통과하면 실제 대화 수·노출 횟수·시간 예산을 고정하고 독립 supervisor에서 시작한다. 품질 기반 early stopping/best-checkpoint retention은 구현 완료라고 주장하지 않는다. 증거: `longrun_protocol/{calibration-01-failure-and-fix.json,parent-calibration-02-accepted.json}`와 `cpu-full-01/`.

### 최신 사용자 정책: 16K, Long-Context Transduction 제외

현재 단계는 seq_len16384이며 **Nemotron-SFT-Long-Context-Transduction-v1 전체를 train/heldout 모두에서 제외**한다. 짧아서16K에 들어가는 예제도 제외하고 원본은 향후 별도 단계용으로 보존한다. Agentic/Instruction-Following-Chat/BBQ 세 SFT dataset과 별도 RL Seed 예약만 유지한다. 모든 기존14 prompt-group blocklist는 유지하되 제외된 source의 group을 새 heldout에 억지로 만들지 않는다. 이전4K generation02는 사용자 변경으로 중단했고 재시작하지 않는다.

Parent는 실제16K CP1/EP8 pilot을 재검증했다: **all8 11 updates, final validation, optimizer/RNG save11, Engine exit0/OOMfalse**. 83train/13heldout, supervised targets47177 중21687개가 position4096 뒤에 있어 metadata만16K로 바꾼 실행이 아니다. CP1이 실제로 맞았으므로 CP2는 활성화하지 않았다. SDK는 별도 expert grid/Mamba CP training support를 갖지만 CP2는 운영 검증하지 않았다. GB8 pilot timing을 GB128 처리량으로 부르지 않는다.

16K 전체 CPU 준비는 `cpu-full-03`/`prepared-full-03` 새 경로에서 실행 중이다. Parent가 source-only12개 입력을 검토·고정하고 `nemochat-nemotron-48h-pipeline-03.service`를 시작해 실제PID/argv/WAITING_DATA를 확인했다. 전체 준비→그 manifest의16K GB128 calibration→실측48h 예산·독립 main을 연결한다. **48시간 본 학습은 아직 시작 전**이다. 증거: `seq16k_protocol/parent-pilot-accepted.json`, `longrun_protocol/pipeline-03/parent-start-readback.json`, `cpu-full-03/FULL_HANDOFF.json`. 앞선4K 토큰예산 추정은 새 조건에 재사용하지 않는다.

## Sources

[1] https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4/raw/a4593a26ea83a700035bbe0ec1edc9ef3810669d/README.md — Nemotron-SFT-Agentic-v4 (NVIDIA pinned dataset card)
[2] https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v4/raw/1cd362945d7a00b136b95b96892bbfea8798686e/README.md — Nemotron-SFT-Instruction-Following-Chat-v4 (NVIDIA pinned dataset card)
[3] https://huggingface.co/datasets/nvidia/Nemotron-SFT-Long-Context-Transduction-v1/raw/90fa9816dd6e9a6bdb123a73fc1a2ed6408783e0/README.md — Nemotron-SFT-Long-Context-Transduction-v1 (NVIDIA pinned dataset card)
[4] https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1/raw/9715a833ae2de4863e321067211f25efe42f6ebe/README.md — Nemotron-RL-Agentic-Conversational-Tool-Use-Seed-v1 (NVIDIA pinned dataset card)
[5] https://huggingface.co/datasets/nvidia/Nemotron-SFT-BBQ-v1/raw/4efdb045cda11918f29065a2e0fd5000b73b8c22/README.md — Nemotron-SFT-BBQ-v1 (NVIDIA pinned dataset card)
[6] https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/raw/1f8953f7f0935358e3605150d22bc15df062061c/tokenizer.json — NVIDIA Super3.5 tokenizer JSON (pinned; tokenizer only)
[7] https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16/resolve/fc067b34f6b514014d2f8bddba855d2e4bfdfba6/tokenizer.json — NVIDIA Super3.5 tokenizer JSON (implementation-config pin)
