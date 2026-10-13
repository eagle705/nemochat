# NemoChat 3부 진행·블로그 시리즈 해설

Canonical 계획: [English execution/blog plan](../en/EXECUTION_AND_BLOG_SERIES_PLAN.md)

상태: 진행·문서화·블로그 계획만 작성했다. 학습 결과, 실제 Figure, 블로그 게시 또는 공개 release가 생성된 상태는 아니다.

## 1. 사용자 지정 묶음으로 진행한다

| 묶음 | 포함 단계 | 종료 산출물 | 블로그 주제 |
|---|---|---|---|
| Part I | [1] 데이터 구축, [2] pretraining, [3] mid-training, [4] long-context | LC-2까지 검증한 **base checkpoint** | 데이터부터 장문 hybrid Latent MoE까지 |
| Part II | [5] SFT, [6] RLVR | Post-SFT와 Post-RLVR checkpoint | Assistant 정렬과 검증 가능한 RL |
| Part III | [7] MOPD | Super-distilled checkpoint와 최종 serving bundle | 한 노드에서 Super 증류와 전체 reference 완성 |

전체는 세 편의 큰 진행·결과 공개 단위다. 각 단계의 상세 기술 문서는 계속 따로 유지하지만, 블로그를 일곱 편으로 분리하지 않는다. 한 묶음은 여러 번의 일주일 임대에 걸칠 수 있으며, '이번 주 종료'와 '묶음 완료'는 다른 개념이다.

영문 글/문서/Figure/code comments를 기준으로 하고 한국어 해설을 별도로 둔다. 최초 시리즈는 Tutorial budget으로 완주하고, Extended 실험은 checkpoint/config가 구분되는 후속 addendum으로 기록한다.

## 2. Part I: [1][2][3][4]

**가제:** Building NemoChat, Part I: From Curated Data to a Long-Context Hybrid Latent MoE

진행:
- 임대 전 데이터 source/license·quota streaming·fixture 준비.
- A/B만 비교하고 실제 parameter count, Nsight, checkpoint export/parity를 확인.
- Curator 정제/dedup/오염 검사, DataDesigner 검증 합성.
- WSD pretraining → domain-rich mid-training. Stable 평가 시점마다 PR #5114 minus-sqrt WSM과 raw를 동일 pinned 공식 Gym task로 비교하고 CE/PPL/BPB·CORE는 별도 진단으로 유지한다. CPU mechanics와 실제 target 학습·benchmark 증거는 구분한다.
- **64K long-only → 64K/4K 교대 CPT**.
- 동일 평가 pack의 Pre-LC/Post-LC-1/Post-LC-2 결과를 보존.

필수 Figure/근거:
- Architecture와 A/B의 loss·throughput·memory·expert load.
- Curation funnel, unique/consumed tokens, 실제 저장 footprint.
- Pre/mid source별 LM CE/validation BPB, consumed tokens·실제 training wall time 곡선, nanochat-compatible CORE와 task raw/centered scores.
- ClimbMix subset/provenance/원본 비상업 조건 및 [metric 상세 설명](PRETRAIN_METRICS_AND_CLIMBMIX.md).
- LC long/short token exposure.
- NIAH heatmap 및 수학·코드 retention 비교.
- 실제 GPU 사용 시간·evaluation/checkpoint overhead와 비용 조건.

**종료 gate:** 안정적인 train/resume/export, 실제 검증된 context 길이, 라이선스 review, immutable LC-2 checkpoint/tokenizer/data/eval manifests. 이 모델은 아직 instruction-tuned assistant가 아니라 base 모델이라는 점을 명확히 쓴다.

64K 목표를 못 달성한 경우에는 부분 결과/실패 분석 글을 쓸 수 있지만, 목표가 통과한 것처럼 다음 gate를 표시하지 않는다.

## 3. Part II: [5][6]

**가제:** Building NemoChat, Part II: Instruction Tuning and Verifiable Reinforcement Learning

진행:
- Part I checkpoint와 tokenizer/eval pack을 고정 입력으로 사용.
- SFT의 chat template, assistant-only loss mask, truncation을 검증.
- **Post-SFT checkpoint를 저장·평가한 뒤** RLVR 시작.
- Math/code/JSON verifier, reward-hacking fixture, sandbox 검증.
- GRPO/RLVR pilot → 승인 budget.
- Standard vLLM control 후 호환될 때 managed Dynamo rollout/refit/cache invalidation 비교.

필수 Figure/근거:
- SFT mixture/길이/유효 loss tokens.
- Base → SFT → RLVR heldout 비교, prompt/template 차이 공개.
- Reward와 실제 정답률을 나란히 표시; reward만 올라간 것을 성공으로 해석하지 않기.
- Entropy/KL/길이/zero-variance group와 rollout/update 비용.
- Train-generation logprob parity와 weight sync.
- SFT/RLVR 이후 NIAH·장문 task retention.

**종료 gate:** reward/보안/parity 테스트, post-SFT와 post-RLVR 결과 보존, 최종 학생 snapshot과 manifests. 성능 개선이 없더라도 사실대로 공개하는 것은 유효한 튜토리얼 결과다.

## 4. Part III: [7]

**가제:** Building NemoChat, Part III: On-Policy Distillation from Nemotron 3 Super on One Node

진행:
- 기본 입력은 Part II의 frozen RLVR checkpoint. SFT checkpoint로 분기하면 lineage에 명시.
- Self-distillation, causal shift/token/mask/logprob gold-case 검증.
- 동일 노드 **vLLM teacher scoring adapter + controller**: 순차 실행 correctness부터, 이후 4/4 placement 검토.
- 학생 rollout → teacher prefix scoring → 학생 update.
- Pre/post MOPD heldout·NIAH 평가.
- 최종 Dynamo export/serving/cache 검증과 전체 공개 bundle.

필수 Figure/근거:
- Rollout/scoring/update sequence diagram.
- Native reference와 custom same-node extension의 지원 경계.
- GPU timeline과 load/offload/scoring/update overhead.
- Self-distill parity, teacher-student gap, 실제 heldout 변화.
- Pre/post MOPD NIAH retention.
- 실제 serving TTFT/latency/throughput/memory.

**종료 gate:** 실제 on-policy scoring/update와 snapshot correctness. Teacher 답으로 offline SFT한 것은 MOPD 완료로 표시하지 않는다. 자원/구현이 막히면 blocker engineering report로 공개하되 허구의 결과를 만들지 않는다.

Final Dynamo serving은 Part III의 마무리에 포함하고, Part II의 managed Dynamo RLVR 실험과 역할을 구분한다. 필수 네 번째 글은 추가하지 않는다.

## 5. 블로그를 위해 실험 중부터 남길 것

각 묶음마다 다음을 보존한다:
- Hypothesis → input checkpoint/config/data IDs → 실제 결과 → 변경 이유/실패 사례.
- SDK/container/code revision, tokenizer hash, source/license/split manifests.
- Raw evaluation JSONL/CSV와 Figure 생성 입력·설정.
- 유효 tokens/sec, training/eval/profile/save/load 시간; RL 생성·teacher scoring·update tokens는 별도.
- 마지막 resume checkpoint와 milestone weight exports를 임대 종료 전 외부 저장소로 보존.
- 실제 공개 업로드 후 검증한 링크만 블로그에 넣는다. 계획된 URL을 존재하는 링크처럼 표시하지 않는다.

가칭 artifact labels: `part1-base-lc`, `part2-rlvr`, `part3-mopd`. 현재 생성된 tag/model ID가 아니다.

## 6. 예정 블로그 파일

```text
blog/en/part-01-data-to-long-context.md
blog/en/part-02-sft-and-rlvr.md
blog/en/part-03-super-mopd.md
blog/ko/part-01-data-to-long-context.md
blog/ko/part-02-sft-and-rlvr.md
blog/ko/part-03-super-mopd.md
blog/assets/part-01/
blog/assets/part-02/
blog/assets/part-03/
blog/manifests/part-01-artifacts.json
blog/manifests/part-02-artifacts.json
blog/manifests/part-03-artifacts.json
```

위는 예정 경로이며 현재 글이 작성된 상태는 아니다. `docs/`는 실행 recipe/reference, `blog/`는 선택 이유와 실측 결과의 서사다. 블로그가 복사한 명령을 따로 유지하기보다 pinned recipe/config를 링크해서 drift를 줄인다.

각 묶음 종료 후 artifact freeze → 실제 결과로 Figure 생성 → 영문 draft → 한국어 해설 → 기술·license·재현성 검토 → 사용자가 지정한 destination으로 게시 순서. 지금은 계획을 문서화했으며 외부 업로드/게시/commit을 수행하지 않았다.
