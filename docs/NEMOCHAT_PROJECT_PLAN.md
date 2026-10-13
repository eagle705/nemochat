# NemoChat Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task. 해당 skill이 제공되지 않는 실행 환경에서는 독립 작업 위임과 명시적인 코드/명세 검토로 대체한다.

**Goal:** NVIDIA 데이터·학습·프로파일링·서빙 SDK를 실제로 연결하여, 약 13.8B total / 2B active의 Nemotron-inspired 학생 모델을 데이터 구축 → pretraining → mid-training → long-context → SFT → RLVR → MOPD까지 만드는 재현 가능한 레퍼런스 튜토리얼.

**Architecture:** Nemotron 3 Nano의 52-block Mamba2/Attention/MoE 배치에 Super-inspired latent MoE를 적용한다. 데이터는 NeMo Curator와 DataDesigner, 언어모델 학습은 Megatron Bridge/Core/Transformer Engine, RL은 NeMo RL/Gym, 시스템 분석은 Nsight Systems/Compute, 최종 배포는 Dynamo로 연결한다. SDK의 공식 지원과 NemoChat이 추가 구현해야 할 통합 계층을 구분한다.

**Tech Stack:** NeMo Curator, NeMo DataDesigner, Megatron Bridge, Megatron Core, Transformer Engine, NeMo RL, NeMo Gym, Nsight Systems, Nsight Compute, NVIDIA Dynamo; 선택적으로 Model Optimizer, vLLM 또는 TensorRT-LLM, lm-evaluation-harness.

---

## Current positioning update — training systems reference lab

현재 사용자 선택은 **GPU 병목을 깊이 이해하는 교육 자료 + 에이전트가 안전하게 실험하는 YAML 계약**이다. 기존 hybrid/WSD/WSM 목표는 유지하되, 단일 dense Transformer control의 실측·검증 루프를 먼저 완결한다. SDK 연결 개수나 agent framework 추가를 성과로 삼지 않는다.

새 실행 경로와 증거 경계는 [TRAINING_SYSTEMS_LAB.md](en/TRAINING_SYSTEMS_LAB.md), [YAML 계약](en/AGENT_EXPERIMENT_CONTRACT.md), [성능 도구](en/TRAINING_PERFORMANCE_TOOLS.md) 및 한국어 companions를 참조한다. CPU control 결과를 B300/native hybrid/distributed 성능으로 해석하지 않는다. 아래 기존 아키텍처·자원 계획은 이 변경으로 자동 구현되지 않는다.

## 0. 문서 상태와 전제

- 조사 기준: 2026-10-06 UTC. **계획 문서이며 아직 학습·GPU 벤치마크·SDK 설치를 수행하지 않았다.**
- 프로젝트 루트: `/home/ubuntu/nemochat`. 작성 전 저장소에는 `.gitignore`만 있었고 tracked 변경 사항은 없었다.
- 사용자 확인: **표준 메모리 구성의 B300 8-GPU, 단일 NVLink 노드**. 일주일 단위로 여러 번 임대하여 학습한다. 실제 인수 시 GPU별 memory/topology를 기록하고 CPU/RAM/NVMe와 임대 간 persistent storage는 별도 확인한다.
- 현재 문서를 작성하는 서버에서는 `nvidia-smi`가 드라이버에 연결되지 않았다. 여기서의 파라미터 수는 계산값이며 throughput, GEMM 효율, 최대 context 길이는 실측값이 아니다.
- 사용자 확인: 최종 모델·공개 결과물·코드 주석은 영어. Public canonical docs는 영어로 만들고 한국어 해설을 `docs/ko/`에 별도로 제공한다. 이 초기 한국어 설계 문서는 작업 기록이며 향후 영문 canonical plan으로 정리한다. 한국어 설명 문서를 제공하는 것이 한국어 training corpus를 추가한다는 뜻은 아니다.
- 무작위 초기화로 학습한다. Nano/Super의 pretrained weight를 축소해서 가져오는 프로젝트가 아니다. Super는 후반 distillation teacher로 사용한다.
- 목표는 SOTA, 상용 서비스 품질, compute-optimal pretraining이 아니라 **모든 단계의 데이터/체크포인트/평가/프로파일을 남기는 교육용 reference**다.
- SDK가 32~128GPU recipe를 제공한다고 해서 8GPU custom 학생도 자동 지원된다고 판단하지 않는다.
- 전체 계획은 현재 SDK 조사 결과에 기반하지만, 실행 환경의 container digest와 실제 dependency 조합을 먼저 고정해야 한다.

## 1. 결정 요약

1. **기본 아키텍처 후보 A:** `H=2048, latent=512, routed FFN=4096, experts=128, top-k=6` → 약 **13.773B total / 2.003B active**.
2. **성능·품질 안전 후보 B:** `latent=1024, routed FFN=2048`, 나머지 동일 → 약 **13.821B / 2.052B**.
3. **보류된 비-latent 대조군 C(이번 pilot에서는 실행하지 않음):** `latent=None, routed FFN=1024` → 약 **13.724B / 1.955B**. 앞선 대화의 13.7B/2B는 이 대조군에 해당했다.
4. **A는 Super와 같은 latent/hidden=1/4 비율**, B는 1/2 비율. 사용자 결정대로 **A/B만** 프로파일링·짧은 학습에서 비교한다. C는 수식상 참고로 보관하며 별도 승인 전 실행하지 않는다.
5. Super tokenizer/vocabulary를 학생과 공유한다. 단순 vocab 크기 일치가 아니라 **token ID, special token, chat template, tokenizer 파일 hash**까지 검증한다.
6. 첫 마일스톤은 30B-token pretraining이 아니다. **smoke → 약 3.435B-token 전체 튜토리얼 → 약 34.17B-token 확장** 순서다.
7. RLVR와 MOPD는 tokens만으로 실험을 정의하지 않는다. prompt 수, rollout 수, update 수, 실제 생성/학습/teacher-scoring tokens를 각각 기록한다.
8. 가장 큰 운영 리스크는 작은 latent 차원이 아니라 **custom hybrid checkpoint의 학습–생성 round-trip과 Super teacher를 포함한 MOPD의 자원 배치**다.
9. Dynamo는 우선 최종 학생의 serving tutorial에 사용한다. **NeMo RL native MOPD teacher를 Dynamo OpenAI API로 그대로 대체할 수 있다고 가정하지 않는다.**

## 2. 모델 설계

### 2.1 원본에서 가져오는 것과 의도적으로 바꾸는 것

공식 Nano는 Mamba/Attention/MoE hybrid, ReLU² expert, shared expert, 128 routed experts 및 top-6를 사용한다. 공식 Super는 512 experts/top-22 및 latent=1024, hidden=4096과 MTP를 사용한다. NemoChat은 원본 Super의 축소 복제라기보다 **Nano 배치 + Super-style latent expert 공간의 교육용 hybrid**다.

| 항목 | NemoChat 기본 후보 A |
|---|---:|
| Hybrid block 수 | 52 |
| Mamba / MoE / Attention | 23 / 23 / 6 |
| Residual hidden H | 2048 |
| Mamba inner I | 3072 |
| Mamba head 수 / head dim | 48 / 64 |
| Mamba groups / state dimension / conv kernel | 8 / 128 / 4 |
| Attention Q heads / KV heads / head dim | 32 / 2 / 128 |
| Routed experts E / selected k | 128 / 6 |
| Latent dimension D | 512 |
| Routed FFN width F | 4096 |
| Shared FFN width S | 2048 |
| Expert activation | squared ReLU, 非GLU |
| Vocab | 131072、Super tokenizer 파일 기준으로 검증 |
| Input/output embedding tying | 없음 |
| MTP | 첫 버전은 끔 |
| 학습 정밀도 | BF16 correctness baseline, 이후 FP8 ablation |

원본 Nano block pattern:

```text
MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
M = Mamba2 block, E = MoE block, * = Attention block
```

주의: 여기서 52는 hybrid sub-block 개수다. 일반적인 attention+FFN의 52 Transformer layers와 같은 연산 규모로 비교하지 않는다. Attention projection 폭 `32 × 128=4096`은 residual H=2048과 다르며, 둘이 같아야 하는 구조를 가정하지 않는다. Mamba inner dimension도 H의 단순 2배가 아니라 `mamba_num_heads × mamba_head_dim`으로 확인한다.

### 2.2 Latent MoE 연산 구조

```text
                 ┌─ Router(H → E), top-6 ─────────────────────┐
x [T,H] ─────────┤                                           │
                 ├─ down-latent(H → D) → dispatch → experts   │
                 │   each expert: D → F → ReLU² → D           │
                 │   weighted combine → up-latent(D → H) ────┼─ sum → y
                 └─ Shared expert: H → S → ReLU² → H ────────┘
```

Router와 shared expert는 **원래 H 차원**에 남는다. Latent projections는 layer당 공통 가중치이며 expert마다 H↔D projection을 별도로 만드는 것이 아니다. Routing은 latent projection 이전 hidden에 대해 수행된다.

실행 버전에서 확인할 설정:

```python
# 실행 코드가 아니라 공식 Nano HybridModelProvider 기반 변경 명세
hidden_size = 2048
mamba_num_heads = 48
mamba_head_dim = 64
mamba_num_groups = 8
mamba_state_dim = 128
num_attention_heads = 32
num_query_groups = 2
kv_channels = 128
num_moe_experts = 128
moe_router_topk = 6
moe_latent_size = 512
moe_ffn_hidden_size = 4096
ffn_hidden_size = 4096
moe_shared_expert_intermediate_size = 2048
moe_shared_expert_overlap = False
share_embeddings_and_output_weights = False
mtp_num_layers = 0
```

`activation_func=squared_relu`, `gated_linear_unit=False`를 확인한다. Router score function, expert-bias/load-balancing, scaling factor는 Nano recipe를 기준으로 시작하되, expert FFN/latent 변경 후 로그를 보며 조정한다. Super의 top-22 또는 scaling factor를 이유 없이 복사하지 않는다.

**중요한 지원 제약:** 조사한 Megatron Core의 latent MoE projection은 Transformer Engine을 요구한다. 해당 소스는 학습 시 `moe_shared_expert_overlap=True`를 latent MoE와 함께 사용하는 것을 거부한다. 초기에는 overlap을 꺼야 한다. 공식 Nano recipe는 overlap을 켜므로 그 recipe를 수정 없이 사용할 수 없다.

### 2.3 파라미터 계산과 집계 정의

ReLU² expert의 주요 가중치:

```text
per-routed-expert = 2 × D × F
per-layer all routed = E × 2 × D × F
per-layer selected routed = k × 2 × D × F
per-layer shared = 2 × H × S
per-layer latent projection pair = 2 × H × D
```

본 계산은 모든 input embedding과 output head를 active에 포함한다. Input embedding은 lookup이므로 active count는 FLOPs-equivalent dense 크기가 아니다. Mamba 및 Attention도 FLOPs가 dense Transformer와 다르다. MTP, vocab padding 변화, 추가 norm 및 구현별 bias가 있으면 실제 `numel()`과 차이가 난다.

| 후보 | D | F | 전체 B | 활성 B | 해석 |
|---|---:|---:|---:|---:|---|
| A | 512 | 4096 | 13.7727 | 2.0035 | Super 압축비와 같은 1/4, 기본 후보 |
| B | 1024 | 2048 | 13.8209 | 2.0517 | 덜 공격적인 bottleneck, GEMM 대조 |
| C | 없음(H=2048) | 1024 | 13.7244 | 1.9552 | 비-latent 대조군 |

공통 부분은 C에서 약 1.3764B이며, A의 latent projection은 약 0.0482B, B는 약 0.0965B를 더한다. 세 후보는 `D×F`를 동일하게 맞춰 routed expert 파라미터와 주요 GEMM FLOPs를 비슷하게 유지한다. **A가 latent를 쓴다는 이유만으로 C보다 expert 연산량이 4배 줄어드는 것은 아니다.**

계산 재현용 독립 Python 예시(실제 계산 시 사용한 식; CUDA 불필요):

```python
H, I, E, k, V = 2048, 3072, 128, 6, 131072
M, A, N, G, state, heads = 23, 6, 52, 8, 128, 48
mamba = H*(2*I + 2*G*state + heads) + I*H
mamba += (I + 2*G*state)*5 + 3*heads + I
attention = 2*H*4096 + 2*H*256
base = 2*V*H + M*mamba + A*attention
base += 23*2*H*2048 + 23*H*E + N*H + H
for D, F in [(512, 4096), (1024, 2048), (2048, 1024)]:
    projection = 0 if D == H else 23*2*H*D
    routed = 23*2*D*F
    print(D, F, (base+projection+routed*E)/1e9,
          (base+projection+routed*k)/1e9)
```

## 3. GEMM efficiency 분석: 작은 모델에서 latent가 적절한가?

### 3.1 판단

**가능성이 충분히 있고 tutorial 가치도 높다. 단, throughput 개선을 실측 전 주장하지 않는다.** D=512는 128/256과 같은 더 작은 bottleneck보다 보수적인 선택이며 F=4096은 지나치게 얇지 않다. A는 입력/출력 latent 차원이 작아 dispatch payload를 줄이지만, 추가 projection과 긴 직사각형 GEMM의 효율이 이득을 상쇄할 수 있다. B는 더 균형적인 GEMM과 더 큰 latent 표현력을 가진 대조군이다.

| 후보 | Expert up GEMM | Expert down GEMM |
|---|---|---|
| A | `[m,512] @ [512,4096]` | `[m,4096] @ [4096,512]` |
| B | `[m,1024] @ [1024,2048]` | `[m,2048] @ [2048,1024]` |
| C | `[m,2048] @ [2048,1024]` | `[m,1024] @ [1024,2048]` |

치수 alignment만으로 높은 Tensor Core 효율이 보장되지는 않는다. 실제 TE/cuBLASLt kernel, forward/dgrad/wgrad, FP8 recipe, routing imbalance 및 microbatch shape에 따라 달라진다.

### 3.2 Expert별 token 수가 핵심

TP=1, CP=1, EP=8, 각 rank가 다른 input microbatch를 처리하고 8-rank EP domain이 함께 dispatch하는 시작점에서:

```text
T_domain = 8 × micro_batch_size × sequence_length
평균 tokens/expert m = T_domain × top_k / num_experts
```

| Seq | Rank별 microbatch | EP domain tokens | 평균 m/expert |
|---:|---:|---:|---:|
| 2048 | 1 | 16384 | 768 |
| 2048 | 2 | 32768 | 1536 |
| 4096 | 1 | 32768 | 1536 |
| 4096 | 2 | 65536 | 3072 |
| 8192 | 1 | 65536 | 3072 |

이는 balanced routing 가정이다. 적게 선택된 expert, padding, masked tokens, 다른 CP/TP 배치에서는 실제 m이 달라진다. 각 GPU는 layer당 16 expert를 소유하므로 local grouped GEMM workload까지 기록한다.

**Gradient accumulation은 optimizer batch를 키우지만 각 forward의 expert m을 키우지 않는다.** 작은 GEMM 문제는 microbatch/sequence/packing/dispatcher 수준에서 해결해야 한다. Decode rollout에서는 pretraining보다 m이 훨씬 작을 수 있어 학습 GEMM 효율과 rollout 속도를 별개로 측정한다.

### 3.3 Latent의 통신 이득과 품질 리스크

- H=2048 → D=512이면 **routed activation element payload**는 이론상 1/4이다. Router metadata, probabilities, padding, local traffic 및 latency까지 포함한 전체 all-to-all 시간이 1/4이 된다는 뜻은 아니다.
- D=1024는 payload가 1/2이며 projection pair 파라미터/연산은 A보다 크다.
- Shared expert는 H 차원으로 유지되어 정보의 우회 경로를 제공한다. 하지만 layer당 공통 down/up projection은 routed branch에 실질적인 bottleneck을 만들므로 표현력 손실은 실험 대상이다.
- ReLU² expert이므로 SwiGLU recipe, gate/up tensor mapping 및 FP8 fusion을 혼동하지 않는다.
- BF16 품질이 확인되기 전에 FP8, CUDA graphs, flex dispatcher와 MTP를 동시에 켜지 않는다.

### 3.4 채택 gate와 프로파일링 계획

1. A/B 각각 동일 데이터·seed·LR·token budget으로 30~100M tokens pilot. 모델별 loss 곡선과 domain holdout을 비교한다. 이 pilot만으로 최종 품질을 확정하지 않는다.
2. 동일 S/MBS의 BF16 warm-up 후 수백 step에서 p50/p95 step time, useful tokens/sec, peak memory, expert counts CV/max-to-mean, padding fraction 기록.
3. Nsight Systems: CPU stalls, kernel launch gaps, dispatch/combine collective, compute/communication overlap 및 dataloader stalls 분석. 짧은 step 범위에 NVTX markers 사용.
4. Nsight Compute: 선택된 grouped-GEMM kernel의 forward/dgrad/wgrad를 제한해서 profile. Tensor Core utilization, occupancy, DRAM/L2 traffic, achieved FLOPs와 kernel time 확인. 전체 training job을 ncu로 감싸지 않는다.
5. EP=8 vs EP=4, MBS=1/2/4, S=2K/4K/8K, A/B를 비교하되 고정 world size에서 **같은 유효 global token batch**로 비교한다. EP 변경은 expert data-parallel 그룹도 바꾸므로 optimizer/sharding 설정까지 확인한다.
6. FP8은 BF16 correctness를 통과한 조합에만 적용하고 동일 token budget의 수렴·router 안정성과 속도를 기록한다.
7. 내부 결정 기준(설계값, 공식 보증 아님): A가 B 대비 throughput 손해 10% 이내이고 heldout loss 악화 2% 이내이면 A를 우선 채택. 그 이상이면 B를 검토한다. 두 후보 모두 불안정하면 먼저 원인을 조사하고, C 추가 실행은 사용자와 별도 결정한다.
8. 최종 평가는 downstream 점수와 학습 안정성도 포함한다. 작은 loss 차이만으로 bottleneck 영향을 결론내리지 않는다.

## 4. SDK 역할과 지원 경계

| SDK | 실제 사용할 위치 | 튜토리얼 산출물 | 지원 경계/추가 작업 |
|---|---|---|---|
| NeMo Curator | 텍스트 정제·품질·dedup·오염 제거 | 입력/출력 manifest, filter 통계, dedup report | 원문 license/PII 자동 면책 도구가 아님; 단계별 감사 필요 |
| NeMo DataDesigner | seed 기반 task/QA/structured instruction 합성 | generation spec, provenance, validator report | teacher endpoint와 실행 비용 필요; 자연 코퍼스 전체 대체용 아님 |
| Megatron Bridge | pre/mid/long/SFT, checkpoint conversion | stage config, checkpoints, resume tests | custom hybrid의 HF ↔ MCore tensor mapping/round-trip 검증 필요 |
| Megatron Core + TE | Mamba/MoE/latent/FP8 kernels | model census, profile, dtype ablation | version 조합 고정; latent shared overlap 제약 |
| NeMo RL + Gym | RLVR, multi-step 환경, native MOPD | rewards, rollout logs, update metrics | 공식 large-model recipe를 1-node로 단순 복사 불가 |
| Nsight Systems / Compute | 모든 주요 학습 단계의 제한된 profile | `.nsys-rep`, `.ncu-rep`, 해석 문서 | 측정 overhead와 profile 기간 분리 |
| Dynamo | 최종 학생 serving, 검증 후 RLVR managed rollout | endpoint, weight refit/cache invalidation, latency/throughput | NeMo RL main의 managed vLLM 경로만 지원; custom hybrid 검증 및 native teacher scoring은 별개 |
| 선택: Model Optimizer | teacher/학생 quantization, export | BF16 vs quantized quality/throughput | post-training 핵심 경로 통과 후 확장 |

패키지를 무조건 단일 venv에 설치하지 않는다. `data`, `train`, `rl`, `serve` container를 분리하고 tokenizer/checkpoint/data manifests를 계약으로 연결한다. 각 container의 image digest, driver/CUDA/NCCL/TE/MCore/Bridge/vLLM 버전·commit을 기록한다. 데이터 구축/학습/서빙을 동시에 돌려 GPU 또는 NVMe를 경쟁시키지 않는다.

설치/실행 전 확인한 구체적인 주의점:

- 현재 Curator README는 Ray 기반 stage/executor pipeline을 설명한다. 예전 Dask API 예시를 최신 recipe와 혼합하지 않는다.
- 조사한 Curator GPU text 경로의 `text_cuda12` extra는 RAPIDS/vLLM Numba dependency 충돌 때문에 일반 `pip install`이 지원되지 않는다고 명시한다. 공식 `uv` override 또는 검증된 container를 사용한다. 이 서버에는 uv가 없을 수 있으므로 설치 유무를 확인하고, 무조건 pip로 대체하지 않는다.
- DataDesigner Python 패키지 이름은 `data-designer`, import namespace는 `data_designer`다. 공식 model/provider configuration을 통해 endpoint를 연결한다.
- DataDesigner의 library telemetry는 `NEMO_TELEMETRY_ENABLED=false`로 끌 수 있다. 이것이 외부 inference provider의 로그 수집/보존 정책을 끄는 것은 아니다.
- NVIDIA Build endpoint는 평가/테스트 용도 및 provider terms를 검토하고, 비밀·개인정보를 보내지 않는다. 대규모 합성은 허용된 dedicated/self-hosted inference 또는 계약된 endpoint를 사용한다.

## 5. 데이터 구축과 공통 계약 [1]

### 5.1 데이터 계약

정제된 record에는 `record_id, source, source_revision, source_uri, license, language, domain, text/messages, quality_scores, dedup_cluster_id, split, tokenizer_hash, token_count, generation_provenance`를 저장한다. 합성 데이터는 seed 원문 ID, 생성 모델/revision, generation settings, prompt template hash, validator 결과도 남긴다.

- 전체 data split과 evaluation freeze를 **합성 및 packing 이전**에 수행한다.
- 동일 원문/문제 family/코드 repository/near-duplicate cluster가 train/eval에 함께 가지 않도록 group split.
- exact dedup → near dedup → benchmark contamination scan → tokenization → packing 순서.
- Raw count, unique accepted tokens, consumed tokens(재사용 포함), loss-bearing tokens를 구분한다.
- 압축 JSONL/Parquet 원문과 tokenized mmap binary 사이 lineage를 유지한다.
- Heldout은 source-balanced이며 budget에 포함하지 않는다. 작은 fixture 100~1000 records로 계약 테스트.
- 데이터셋 전체 다운로드 전에 필요한 shard/subset만 streaming하고, 사용 tokenizer로 실제 token count를 측정한다.

### 5.1.1 Quota-based acquisition: 전체 다운로드 금지

**18B는 전체 원천을 다운로드한 뒤 잘라내는 계획이 아니다.** 원격 source에서 필요한 부분만 읽고, 정제·선택·토큰화를 통과한 제한된 subset만 training-ready artifact로 저장한다. 아래 제어 계층은 NemoChat이 구현할 orchestration이며, 모든 Curator stage가 임의 iterator를 바로 지원한다는 의미가 아니다.

```text
Pinned remote dataset + 선택된 shard manifest
 → bounded batch/shard read (HTTP streaming 또는 선택 shard만 임시 다운로드)
 → Curator cleaning/quality + selected-pool dedup + contamination scan
 → Super tokenizer로 실제 token count
 → accepted-token quota ledger
 → 확정된 subset만 Megatron mmap .bin/.idx로 저장
 → 임시 원문/cache 삭제, 다음 batch/shard 처리
```

- Tutorial v1의 general-text quota는 1.8B, Extended는 18B. 주 source는 ClimbMix, FineWeb-Edu는 fallback이다. 먼저 v1 크기로 구축하고 Extended는 필요할 때 추가한다.
- FineWeb-Edu 공식 sample config는 `sample-10BT`, `sample-100BT`, `sample-350BT`다. 이 숫자는 **GPT-2 tokenizer 기준 대략적인 tokens**이며 Super tokenizer의 실제 count와 다르다.
- **주 source 변경:** 비상업적 교육 track은 ClimbMix의 선택 subset을 Super tokenizer로 재토큰화하여 v1 1.8B/Extended 18B quota를 확보한다. 원본 CC BY-NC·출처 조건과 커뮤니티 card 표기의 차이를 기록하며 migration은 아직 실행하지 않았다. `karpathy/climbmix-400b-shuffle` 마지막 val shard를 train에서 제외한다. FineWeb-Edu는 fallback/독립 heldout으로 유지하고, fallback에서는 v1 `sample-10BT` 일부/Extended `sample-100BT` 일부만 읽는다. 어느 source도 전체 다운로드하지 않는다.
- Revision 및 shard 목록을 고정하고 shard 순서를 seeded shuffle/stratified sampling한다. Iterator 앞부분만 취하면 crawl/source 편향이 생길 수 있으므로 dump/domain 분포를 기록한다. Buffered shuffle은 전역 균일 sampling이 아니라 근사 shuffle이다.
- Quota는 원문 크기나 행 수가 아니라 **필터 후, 실제 학생 tokenizer로 센 accepted tokens**로 관리한다. 필터 drop과 dedup 제거 때문에 네트워크로 읽는 원천량은 최종 quota보다 크다.
- Dedup stage는 global state/shuffle 또는 추가 pass가 필요할 수 있다. **선택한 candidate pool만** batch/partition 처리하고, 필요한 dedup fingerprints/index를 저장한다. 원천 전체에 대한 global dedup을 수행했다고 주장하지 않는다. 제거로 quota가 부족해지면 다음 shard를 보충한다.
- Streaming에서도 Parquet row group, prefetch, HTTP range/fsspec cache 등의 단위 때문에 사용하지 않은 bytes가 일부 읽힐 수 있다. Range access가 안 되거나 SDK stage가 파일을 요구하면 **선택 shard만** 내려받고 처리 후 지운다. 무저장/무초과 네트워크 전송을 보장하지 않는다.
- 여러 worker는 작은 quota reservations와 중앙 ledger로 overshoot를 제한한다. 재실행 시 shard/row progress와 accepted IDs를 복원해 중복 append를 막는다.
- 최종 tokenized pool과 packing manifest를 freeze하고 HF remote streaming을 training hot path에서 직접 수행하지 않는다. 원격 네트워크 장애가 GPU 학습을 정지시키지 않게 한다.
- 기존 blend의 18B는 **training consumed token 목표**다. 데이터 구축 단계에서는 unique accepted count를 따로 기록한다. 1-pass 기준으로 시작하되 반복 소비가 있으면 epoch/reuse factor를 공개한다.
- 고정 sequence packing/update 단위 때문에 source별 consumed count는 목표 근처에서 약간 달라질 수 있다. 엄밀히 맞춰야 하면 마지막 chunk의 초과 label을 loss-mask하거나 source별 최종 segment를 조정하고, 유효 loss-bearing token 수를 집계한다. 18B가 자동으로 정수 optimizer step 수에 맞는다고 가정하지 않는다.
- 보존: 선정 record IDs/shard-row 위치/revision/tokenizer hash/filter config/dedup manifest + 소량 audit samples + 최종 `.bin/.idx`. 재구축에 필요한 원문 보존은 라이선스와 upstream 가용성에 따라 선택한다. 임시 다운로드·정제 원문·tokenized artifact를 모두 영구 중복 저장하지 않는다.

Storage 하한 예시: vocab=131072이므로 일반 token IDs는 uint16에 들어가지 않는다. **uint32 token IDs만** 계산하면 web 18B는 72GB(decimal), 전체 pretrain 30B는 120GB다. `.idx`, manifests, shuffle/sample indices, dedup state, temporary raw/cache 및 checkpoints는 별도다. 저장소 절감 대안은 3B v1부터 시작하거나 bounded tokenized window를 순차 처리하는 방식이지만, 후자는 global shuffle·resume·반복 학습 비용이 더 복잡하므로 최초 reference는 선택 subset의 local mmap을 권장한다.

참고: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu (공식 card의 streaming 사용 예시 및 sample tokenizer 정의 확인).

### 5.2 후보 소스: 검증된 ID와 제약

| 역할 | Dataset ID | 조사 시 card license | 사용 원칙 |
|---|---|---|---|
| 일반·교육 텍스트 (주 source) | `nvidia/Nemotron-ClimbMix` / `karpathy/climbmix-400b-shuffle` | 원본 CC BY-NC 4.0 / 재포장 card MIT | 비상업 교육용; 원본 조건 유지, provenance/재배포 gate; 선택 subset retokenization |
| 교육 웹 fallback/독립 holdout | `HuggingFaceFW/fineweb-edu` | ODC-BY | URL·attribution·원문 권리 확인 |
| 코드/수학 보조 | `HuggingFaceTB/smollm-corpus` | ODC-BY | Python-Edu/Cosmopedia 등 subset을 확인; 다른 blend와 중복 제거 |
| 긴 문서 | `HuggingFaceFW/finepdfs` | ODC-BY | 문서 길이/추출 품질 검증; PDF의 원문 라이선스 별도 감사 |
| 수학 SFT/mid | `nvidia/OpenMathInstruct-2` | CC-BY-4.0 | train split/subset만; 평가 문제 family와 overlap 제거 |
| 일반 instruction | `allenai/tulu-3-sft-mixture` | ODC-BY | 각 constituent의 원래 조건 검토; 이미 평가에 가까운 source 제외 |
| 수학 reasoning | `open-r1/OpenR1-Math-220k` | Apache-2.0 | 합성 trace correctness·중복·문제 출처 검토 |
| 코드 RL 후보 | `open-r1/verifiable-coding-problems-python` | API card license 미표기 | **기본 blend에 넣지 않음.** source/unit-test rights 확인 후 opt-in |

집합 license는 constituent 원문/코드의 권리를 일괄 보증하지 않는다. 특히 코드 샘플의 원 repo license와 attribution을 유지한다. 카드만 보고 데이터 재배포 허용을 선언하지 않는다.

코드 RL의 첫 버전은 직접 작성한 작은 Python task generator + unit tests, 수학 RL은 자체 procedural arithmetic/algebra generator로 시작한다. DataDesigner가 만든 문제도 verifier를 통과해야 하지만 verifier 통과가 문제의 유일성/교육적 품질을 자동 보증하지 않는다.

### 5.3 Curator pipeline

- 텍스트 정규화, URL/HTML boilerplate 정리, 언어/길이/반복/품질 필터.
- 코드/수학에는 일반 prose 품질 필터를 과도하게 적용하지 않고 domain 별로 분기.
- Exact + near dedup; cross-dataset 중복 제거; train/validation/test contamination scan.
- PII/secret 후보 탐지와 샘플 감사. 허용/차단 기준 및 false positive 예시 공개.
- 필터별 pass/drop 수와 tokens, 비용, runtime 저장. GPU 경로를 사용했다면 CPU baseline과 실제 이득도 문서화.

### 5.4 DataDesigner pilot

- 첫 합성 run은 1000 examples, 다음은 1~5M tokens 이하. 대량 생성보다 validation 설계가 우선.
- Task axes: domain, difficulty, instruction style, answer format, number of turns, reasoning budget.
- Math: deterministic answer validator; code: sandbox test; grounded QA: cited source span/answer consistency; JSON: schema validator.
- Teacher endpoint의 실패/timeout/retry/cache/provenance를 저장. 비검증 LLM judge는 보조지표일 뿐 RLVR 정답 보증으로 사용하지 않는다.
- 평가 문제를 seed로 넣지 않으며 synthetic teacher outputs에 대한 모델 license·distillation 조건을 검토한다.

## 6. 단계별 token budget: 세 가지 실행 규모

**모든 숫자는 추천 시작값이지 최적값이나 실측 성과가 아니다.**

| 단계 | Smoke(연결 검증) | Tutorial v1(먼저 권장) | Extended reference(후속) |
|---|---:|---:|---:|
| [2] Pretraining | 30M | 3B | 30B |
| [3] Mid-training | 5M | 300M | 3B |
| [4] Long context | 2M | 100M | 1B |
| [5] SFT | 1M processed | 20M processed | 100M processed |
| [6] RLVR | 32~128 prompts × 4 rollouts, 2~10 updates | 5M generated-token cap | 20M generated-token cap |
| [7] MOPD | self-distill correctness test | 10M generated-token cap | 50M generated-token cap |

Tutorial v1의 이종 token cap을 단순 합산하면 약 3.435B, Extended는 약 34.17B다. **이 합계는 학습 FLOPs나 wall-clock 추정을 위한 동등 token 수가 아니다.** RL의 생성/재학습과 teacher scoring은 다른 비용이므로 분리 집계한다. Smoke도 13.8B full model을 짧게 돌리는 GPU smoke와 작은 CPU fixture 테스트를 분리한다.

3B pretraining은 이 모델을 잘 학습시키기에 매우 짧다. 30B도 완성도/범용성을 보장하지 않으며 128 experts 중 각 expert가 충분한 다양성을 학습했는지 확인해야 한다. 작은 run에서 RL이 작동하지 않는 이유가 SDK가 아니라 base model capability일 수 있다. 이때 tutorial에서는 작은 verifier-friendly tasks를 쓰고, pretrained Nano를 이용한 별도 SDK control run을 허용하되 **NemoChat 학습 성과와 혼동하지 않는다.**

## 7. Pretraining [2]

### 데이터 및 curriculum

Extended 30B 기준 mixture(소비 tokens 기준):

| Blend | 비율 | Tokens |
|---|---:|---:|
| ClimbMix 중심 일반·교육 텍스트 (비상업 교육 track) | 60% | 18B |
| 허용된 Python/code subset | 20% | 6B |
| 수학/과학 설명·검증 합성 | 10% | 3B |
| 긴 문서에서 얻은 자연 텍스트 | 10% | 3B |

Tutorial v1은 같은 비율로 3B. 목표 code/math 물량이 license/중복 필터 후 부족하면 동일 작은 set을 무작정 반복하지 말고 교육 웹으로 재배분하거나 총 예산을 낮춘다. 2K부터 시작해 4K로 올린다. S=2K/4K 비율과 warm-up은 profile/안정성 확인 후 고정한다.

### Training 시작값

- BF16, distributed AdamW optimizer, TP=1/PP=1/EP=8/ETP=1/CP=1부터.
- MBS=1→2를 sweep; global batch는 **유효 tokens/update**로 정의. 초깃값 후보 128K~256K tokens/update, 안정화 후 512K 검토. attention DP와 expert DP가 다르므로 단순 EP 숫자로 batch를 계산하지 않는다.
- Peak LR 2e-4~4e-4의 짧은 sweep, Adam beta=(0.9,0.95), weight decay=0.1, grad clip=1.0를 starting hypothesis로 둔다.
- 토큰 기반 warm-up 약 1%를 후보로 하고 smoke는 최소 여러 optimizer step의 warm-up을 확보. 사용자 선택은 WSD이며 전체 horizon과 stable/최종 decay 구간을 recipe에 명시. Stable 평가 시점에는 Megatron PR #5114의 minus-sqrt WSM과 raw checkpoint를 동일한 pinned NeMo Gym benchmark로 비교한다. 별도 decay branch는 선택한 방식이 아니다.
- FP8과 CUDA graphs는 correctness 이후 한 변수씩 켠다. 속도보다 router collapse와 수치 안정성 우선.
- S=2K와 4K를 섞을 때 optimizer batch token 수를 맞추고, GPU peak memory와 useful packing ratio를 측정한다.

### Gate

- 단일 step forward/backward, BF16 finite loss, expert counts, save/resume 동일 config 확인.
- 동일 checkpoint에서 interrupted/uninterrupted 학습이 허용 수치 오차 내에서 재현.
- **선택한 checkpoint 비교: 공식 NeMo Gym benchmark의 raw vs. stable-phase WSM 평가.** Validation BPB/CE/PPL과 기존 nanochat-compatible CORE는 별도 진단 지표로 유지한다. Source별 heldout LM CE/PPL은 auxiliary MoE loss와 분리하고, math/code executable diagnostics와 tokenizer/domain coverage를 유지한다. CORE quick/full을 구분하고 source/domain별 BPB·task별 raw/centered score를 보고한다.
- Metric 수식·해석·Super tokenizer raw-byte/분산 집계·teacher-forced CORE 주의점 및 ClimbMix adoption: [영문 명세](en/PRETRAIN_METRICS_AND_CLIMBMIX.md), [한국어 상세 설명](ko/PRETRAIN_METRICS_AND_CLIMBMIX.md). CPU HF evaluator/inference·bounded data acquisition은 실행 검증했으며 [검증 기록](IMPLEMENTATION_VERIFICATION.md)을 참고한다. 실제 NemoChat GPU/분산 평가 및 공식 Gym inference는 미실행.
- EMA loss, grad norm, routing CV/max expert load, router bias 변화, tokens/sec 저장.
- Holdout이 악화하거나 NaN/empty expert가 반복되면 mid 단계로 넘어가지 않는다.

### Pretraining performance lab — 사용자 추가 확정

Nsight Systems로 실제 학습 critical path의 CPU launch/dispatch overhead와 작은 kernel 연속 실행을 조사하고, 적합한 후보에 optional Triton fusion을 시도한다. PyTorch eager/가능한 compile baseline 대비 forward/backward correctness, profiler를 끈 전체 update throughput·rank별 memory, checkpoint/resume까지 검증한다. 작은 SwiGLU activation은 학습 후보일 뿐 실제 병목으로 확정하지 않는다.

실행 gate: GPU baseline → trace/source mapping → standalone fusion → native training A/B → 채택/거부. 기존 단일 WSD 및 stable-window 공식 WSM을 유지하고 기본 경로 변경은 별도 검토한다. 현재 구현/실험은 GPU 환경 대기이며 성능 개선 수치는 없다. [영문 명세](en/PRETRAIN_PERFORMANCE_LAB.md), [한국어](ko/PRETRAIN_PERFORMANCE_LAB.md).

## 8. Mid-training [3]

정의: chat alignment가 아니라 base LM에 task-rich domain/reasoning distributions를 추가하는 **continued next-token training**. SFT와 loss mask/format을 구분한다.

Extended 3B, Tutorial 300M:

| Blend | 비율 | Extended |
|---|---:|---:|
| 수학 problem/solution + 검증 reasoning | 40% | 1.2B |
| Python/code explanations + tested code | 30% | 0.9B |
| Source-grounded technical QA/문서 | 10% | 0.3B |
| Pretrain replay | 20% | 0.6B |

- OpenMathInstruct-2/OpenR1 train subsets의 조건을 검토하고, problem family 기준 평가 오염을 제거한다.
- Raw LM 형태로 모든 적절한 tokens에 loss. Assistant-only SFT mask는 다음 단계에서 사용한다.
- S=4K~8K, peak LR은 pretrain peak의 약 10~30% 후보부터. 분포 급변 시 짧은 re-warm-up.
- Replay는 general-domain forgetting을 감시하며 조정. Mid corpus의 unique tokens와 반복 epoch를 기록한다.
- Gate: math/code heldout 개선, general PPL와 instruction diagnostics의 회귀 공개. 작은 모델에 매우 긴 teacher CoT를 무조건 주입하지 않는다.

## 9. Long-context training [4]

사용자 결정: **64K를 기본 학습·평가 목표**로 올린다. 32K는 중간 correctness/quality gate다. **128K → 256K는 후속 agentic-coding stretch track**으로 두고, 실제 memory·throughput·장문 품질 검증 후 진행한다. Context window 설정이 256K라고 실제 repo-scale coding 능력이 확보된 것은 아니다.

**Super 논문 §2.6(PDF p.16) 확인 후 수정:** 기존 8K→64K 단계별 예산 배분은 Super 재현 schedule이 아니라 자체 제안이었다. 원문은 constant LR 4.5e-6, GBS16, CP64/TP2/EP64, document QA 20% + downscaled Phase 2 80%로 1M CPT 34B tokens를 진행하고, 이후 수학 성능 저하를 완화하려고 1M/4K 교대 학습 17B tokens를 수행했다. 해당 절에는 교대 빈도·길이별 batch·token 비율이 명시되지 않는다.

단일 노드에서는 이 패턴을 **64K long-only → 64K/4K alternating CPT**로 적용한다. 8K/16K/32K는 correctness/resource smoke로 먼저 확인한다.

| LC 단계 | Tutorial tokens | Extended tokens |
|---|---:|---:|
| LC-1: 64K long-only | 67M | 670M |
| LC-2: 64K/4K alternating | 33M | 330M |

대략 2:1 단계 예산은 논문의 34B:17B를 참고한 시작값이다. LC-1은 장문 grounded QA 20% + selected pre/mid replay 80%, LC-2의 4K stream은 math/code/general replay를 강화한다. 이는 공개 가능한 대체 blend이지 원본 QA dataset의 정확한 재현이 아니다.

LC-2는 **update당 유효 token batch를 맞추고 long/short update를 교대**하는 50:50 token baseline으로 시작한다. 이 비율은 NemoChat 설계값이며 논문이 명시한 값이 아니다. 같은 sequence GBS로 64K/4K step을 1회씩 하면 장문 token 비중이 94.12%가 되므로 step/token/time 비율을 따로 기록한다.

두 local data streams와 rank-synchronized update controller, 길이별 accumulation/token-normalized loss, optimizer/scheduler/RNG 및 두 data cursor의 resume가 필요하다. 고정 topology로 시작하며 CUDA graphs는 초기 correctness run에서 끈다. 기존 fixed-seq Bridge recipe가 교대를 자동 지원한다고 주장하지 않는다.

**NIAH + Figure 필수 산출물:** 길이×needle-depth accuracy heatmap(PNG/SVG), pre-LC/post-LC-1/post-LC-2 비교, 길이별 정확도/uncertainty, 별도 short-math/code retention panel, raw predictions JSONL 및 cell CSV. 64K main grid는 5 lengths × 11 depths × 10 seeds = 550 requests/checkpoint. 256K extended grid는 770 requests/checkpoint. 실제 tokenizer 길이/needle 보존/answer exact-match를 검증하고 미지원/OOM/timeout을 accuracy 0과 섞지 않는다. Figure는 실제 model inference 후에만 생성한다.

Canonical 세부 명세: [English LC/NIAH plan](en/LONG_CONTEXT_AND_NIAH_PLAN.md). 별도 해설: [한국어 LC/NIAH 해설](ko/LONG_CONTEXT_AND_NIAH_PLAN.md). Vanilla NIAH 외 RULER·멀티문서 QA·실행 가능한 cross-file coding으로 능력을 확인한다.

- FinePDFs의 자연 긴 문서와 license-approved technical docs를 사용. 특정 corpus 최대 길이/유효 토큰 물량은 사전 측정.
- 임의 문서를 EOS 없이 붙여 long context 이해를 얻었다고 주장하지 않는다. attention segment 경계와 **Mamba state reset**의 의미를 검증한다.
- Packing은 Mamba state leakage, attention masking, position IDs, labels, loss mask가 모두 맞아야 한다. 독립 sequence의 batched 실행과 packed 실행을 비교한다.
- 먼저 microbatch/recompute로 길이를 늘리고, 필요 시 CP=2/4를 검토. **hybrid Mamba의 CP 지원을 버전별 확인**하고 TP×CP×EP 등 그룹을 추측해 구성하지 않는다.
- 원본 hybrid의 실제 attention positional encoding을 inspect한다. config에 rope 필드가 있다는 이유만으로 RoPE scaling을 무조건 적용하지 않는다. Mamba에는 동일한 RoPE 처방을 적용할 수 없다.
- 평가: RULER/needle retrieval, multi-hop document QA, distractor robustness, 길이별 heldout NLL, 2K/4K 회귀, Mamba state correctness.
- 32K에서 retrieval/QA가 개선되지 않으면 64K 확장보다 데이터·mask·state reset 문제를 먼저 해결한다.
- 64K 통과 후 128K/256K는 각각 inference/cache/memory smoke → retrieval → repo-grounded QA → 실행 검증 가능한 cross-file coding 순서. 승인 시 추가 long-context budget 후보 100~300M tokens를 독립 실험으로 잡고, 기존 1B 예산에 완료된 것처럼 포함하지 않는다.
- 짧은 전체 repo를 무작정 반복/연결해서 256K를 만들지 않는다. license-approved 실제 repository, symbols/dependencies, issue/task, gold patch와 tests를 연결하고 repo/family 단위 heldout을 유지한다. Retrieval 또는 summarization 기반 agent memory도 full-context baseline과 비교한다.
- Mamba가 포함되어 있어도 6개 full-attention blocks의 attention-core 연산은 sequence에 대해 quadratic이다. **32K→64K는 같은 sequence batch에서 attention-core 연산 약 4배, 64K→256K는 약 16배**다. 전체 모델 비용이 그만큼 늘어난다는 뜻은 아니며, 고정 token budget에서는 sequence 수가 달라진다. FlashAttention은 attention score의 메모리 materialization을 줄여도 이 연산량 증가를 없애지 않는다.

## 10. SFT [5]

Extended 100M processed tokens, Tutorial 20M processed tokens. Loss-bearing assistant tokens는 별도 집계한다. 시작은 약 20K~100K unique examples의 길이 분포를 측정한 뒤 1~2 epochs 내에서 budget을 맞춘다. example 수를 tokens와 고정 환산하지 않는다.

| Mix | 비율 |
|---|---:|
| 일반 assistant/instruction 및 구조화 응답 | 30% |
| 수학·reasoning | 30% |
| Python 코드·debugging | 25% |
| Grounded QA 및 긴 문서 task | 10% |
| Safety·refusal·정직성 | 5% |

- Tulu-3의 허용된 source subsets, OpenMathInstruct-2, verified synthetic code, DataDesigner JSON/schema/grounded tasks를 사용.
- Student/teacher tokenizer, special tokens, system/user/assistant template 동일성 확인. Thinking/non-thinking 출력 정책을 명시한다.
- Assistant-only loss; multi-turn mask와 truncation이 답을 제거하지 않게 검증한다. 원문이나 system/tool tokens에 우발적인 loss가 들어가지 않게 테스트.
- 4K/8K를 주로 쓰되 긴 context 유지용 일부 16K/32K/64K. 한 batch에 길이가 크게 다른 sample을 섞는 비용 확인.
- Full-parameter SFT를 기본으로 한다. LoRA는 별도 비용 절감 비교 실험이지 main 단계의 대체라고 암묵 처리하지 않는다.
- Gate: instruction following, math answer accuracy, code unit-test pass, valid JSON, grounded answer consistency, safety fixture; base model과 비교.

## 11. RLVR [6]

NeMo RL + NeMo Gym 환경, GRPO 계열 시작. RLVR는 rule/execution 기반 reward를 쓰며 LLM judge-only reward를 정답 보증으로 부르지 않는다.

### 초기 task blend

- 60% 수학: procedural arithmetic/algebra → 검증 가능한 공개 train 문제. Final answer extractor와 symbolic/numeric equivalence verifier.
- 30% Python 코드: 직접 작성한 task/unit tests부터. container sandbox, timeout, memory limit, network off, filesystem 제한.
- 10% structured output/constraint satisfaction: JSON schema, 계산/추론 결과, format compliance. format만 맞춰 reward를 얻는 loophole 방지.

### Budget와 실행

- Smoke: 32~128 prompts, 4 rollouts/prompt, 2~10 updates. 정확한 reward와 zero-variance 그룹 처리 확인.
- Tutorial: 생성 5M tokens 상한, 약 200~500 updates 후보, 4 rollouts/group부터.
- Extended: 생성 20M tokens 상한, 약 500~1500 updates 후보, 4~8 rollouts/group. **token cap 또는 update cap 먼저 도달 시 정지**하고 실제 수치를 보고.
- Prompt batch/response max length는 pilot의 pass rate/rollout 비용으로 정한다. 첫 response cap 1K~2K; 바로 8K CoT를 강요하지 않는다.
- LR 후보 1e-6~1e-5, clipping/KL/reference 및 entropy를 작은 sweep으로 결정. 레시피 버전별 파라미터 semantics 확인.
- Base student가 거의 전부 실패하는 task는 난이도를 낮추거나 SFT 데이터로 보강한다. Reward가 전부 0이면 학습 진행을 성공으로 표시하지 않는다.
- 단일 노드에서 policy train/generation 배치를 사전 산정한다. Colocated 또는 sequential weight synchronization 지원을 검증하고, 무조건 8GPU train+8GPU generation을 동시에 요구하지 않는다.

### 필수 검증

- Verifier gold cases: correct/incorrect/malformed/timeout/unsafe 코드/answer spoofing.
- Fixed heldout pass@1 및 pass@k, reward vs true correctness 차이, response length, entropy, clipping, KL, zero-variance group rate.
- Train-generation token/logprob parity, exported weight synchronization, policy version lag.
- 문제와 unit tests가 heldout benchmark를 유출하지 않았는지 검사.

## 12. MOPD [7]: Super teacher부터 시작

### 12.1 용어와 지원 사실

조사한 NeMo RL에서 **MOPD = Multi-Teacher On-Policy Distillation**이다. 하나 이상의 teacher를 지원하므로 **처음에는 Super 단일 teacher**로 시작해도 된다. 다중 teacher ensemble averaging을 기본으로 하는 방식이 아니다. Gym agent/task 이름별로 teacher를 선택한다.

Native 경로는 async GRPO의 `adv_estimator: opd`이며 학생이 생성한 tokens에 대해:

```text
advantage_t = stop_gradient(log p_teacher(token_t|prefix)
                           - log p_student(token_t|prefix))
```

Teacher가 만든 답을 저장해 SFT 하는 것은 offline distillation이고, 이 on-policy MOPD가 아니다. RLVR 다음 단계로 연결하되, MOPD를 reward-based GRPO와 같은 objective로 설명하지 않는다.

### 12.2 가장 중요한 단일 노드 제약

공식 MOPD 문서는 **non-colocated, dedicated teacher worker groups**를 사용하고, teacher는 Megatron inference-only, unquantized 경로다. 예시 배치는 policy node + rollout node + teacher node의 3-node 구성이다. 따라서 **B300 1노드에서 공식 native MOPD가 아무 수정 없이 작동한다고 보장할 수 없다.**

사용자 확인: 추가 teacher node 없이 **동일 B300 노드에서 vLLM teacher를 띄우는 경로**를 main tutorial로 선택한다. 이는 native Megatron teacher scheduler가 아닌 scoring/placement adapter가 필요한 extension이다. 가능한 경로:

1. **SDK-native reference(현재 main 범위 밖, 추가 자원 확보 시):** 별도 teacher GPU/node 확보 + student train/rollout resourcing. Super BF16 teacher 주요 가중치만 대략 240GB 규모이며 activation/logprob workspace가 추가되므로 dedicated teacher topology를 실측 결정한다.
2. **선택된 단일 노드 vLLM educational extension:** frozen student snapshot으로 rollout → Super teacher scoring → student update를 serial/time-multiplex. 정확한 policy snapshot/old logprobs를 유지하고 async drift를 없앤다. 이것은 알고리즘상 on-policy를 지향하지만 **native async MOPD runtime 그대로가 아닌 custom controller 작업**이다. Checkpoint loading/offload overhead까지 보고한다.
3. **Teacher scoring adapter(선택한 local vLLM endpoint에도 필수):** dedicated endpoint가 generated text뿐 아니라 **학생 prefix의 정확한 token-level teacher logprobs**를 제공해야 한다. 표준 chat-completion API만으로는 부족하다. Adapter와 scoring alignment 검증이 추가 구현 범위다.
4. 자원이 없으면 teacher-generated offline SFT를 fallback demo로 제공하되 **MOPD 완료로 체크하지 않는다**. Stage 7의 native 실행은 resource-blocked 상태로 정직하게 기록한다.

Teacher를 quantize하면 memory를 줄일 수 있지만 현재 native 문서는 teacher quantization을 적용하지 않는다고 명시한다. Dynamo/ModelOpt quantized teacher를 사용한 별도 실험을 native MOPD와 동일 지원으로 표현하지 않는다.

### 12.2.1 선택한 same-node vLLM 운영 설계

- 첫 correctness 경로는 **time-multiplex**: 학생 snapshot으로 rollout → 학생 actor를 내려 teacher scoring → teacher를 내려 학생 update. 통신/weight load 비용이 크지만 모든 모델이 동시에 GPU를 차지할 필요가 없어 adapter 검증이 단순하다.
- 이후 throughput 후보는 **GPU 0–3 Super teacher vLLM(TP4), GPU 4–7 학생 train/rollout pool**이다. 학생 pool에서는 train과 generation을 교대로 사용하거나 검증된 refit/sleep/resume를 사용한다. 단순히 student training/rollout/teacher 세 fleet를 모두 독립 상주시킨다고 가정하지 않는다. GPU 4/4 split 및 BF16 teacher 수용 여부는 실측 gate.
- Teacher는 최초 BF16 기준. Quantized teacher는 별도 ablation으로 label하고 sampled logprob 품질 변화와 KL/학습 안정성을 비교한다.
- 필요한 것은 teacher의 새 completion이 아니라 **학생 prompt+response token IDs 전체를 teacher 입력으로 넣어 얻는 causal prompt-token logprobs**다.
- 조사한 vLLM source는 engine `SamplingParams.prompt_logprobs`, OpenAI-compatible completion protocol의 `prompt_logprobs`와 `prompt_token_ids` 응답을 제공한다. Pinned serving 버전에서 token-ID 입력/응답 지원을 확인하고, 필요하면 engine-side scoring service를 구현한다. 일반 chat completion의 generated-token logprobs로 대체하지 않는다.
- 요청 전체 vocab logprobs(-1)는 피한다. Prompt의 실제 student token logprob을 얻는 최소 payload를 사용하고, top-k 제한 아래에서도 해당 token 값이 정확히 제공되는지 fixture로 확인한다.
- Prompt 최초 token의 logprob 부재, BOS/EOS 추가, causal shift, prefill chunk 경계, temperature/logit processors, prefix caching을 검증한다. Correctness baseline에서는 scoring cache 최적화를 끄거나 지원 조건을 확인한다.
- Snapshot hash, generation policy version, raw student logprobs, teacher revision/scoring config를 rollout batch에 고정한다. Teacher scoring과 update 사이에 정책을 바꾸거나 다른 snapshot rollout을 섞지 않는다.
- 학습은 NeMo RL objective/update와 연결하되 **native async MOPD teacher worker를 vLLM로 교체하는 adapter/controller 작업**을 명시한다. 최종 tutorial은 native reference와 custom same-node extension의 차이를 공개한다.
- 초기 rollout/scoring 길이는 2K~8K로 제한한다. 모델의 64K context 지원이 모든 MOPD rollout을 64K로 만들어야 한다는 뜻은 아니다. 256K teacher/student scoring은 독립 stretch gate.

vLLM 조사 소스(실행 시 호환 version/commit을 고정):
- https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
- https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/protocol.py

### 12.3 Tokenizer와 logprob parity

- Super tokenizer 파일을 처음부터 학생에 사용한다. HF token IDs, special tokens, chat template, EOS/PAD, thinking token 일치 테스트.
- Prefix tokenization, next-token shift, trained assistant mask, tool/environment mask, temperature, padding을 비교한다.
- Native sampled-token teacher gap을 1차 목표로 한다. Full-vocab KL로 처음부터 확장하지 않는다.
- 8192 tokens × 131072 vocab × BF16만으로 full logits tensor가 약 **2GiB**이고 FP32/grad/양쪽 분포까지는 더 커진다.
- 조사 문서의 full-vocab hidden-state payload 경로는 student와 teacher hidden-size 조건을 요구한다. H=2048 학생과 H=4096 Super 조합을 그 경로에 무검증 투입하지 않는다. logits fallback도 memory/transport/chunked KL 구현을 검증해야 한다.
- Self-distillation에서 gap/divergence가 거의 0인지 확인하는 것이 tensor alignment/shift 검증의 출발점. 단순 decomposition residual이 0이라고 payload가 올바른 것은 아니다.

### 12.4 데이터와 budget

- Tutorial: 생성 10M tokens cap. Extended: 생성 50M cap. Teacher-scored tokens와 train replay tokens는 각각 별도 계수.
- Prompt mix 초깃값: 수학 40%, Python/code 35%, grounded/structured instruction 25%.
- 학생이 적절히 시도할 수 있는 문제부터 시작. Super의 매우 긴 thinking output 길이를 학생에 강제하지 않는다.
- Teacher checkpoint/revision 고정. Frozen rollout policy ID와 logprobs를 저장하고 on-policy drift gate를 적용.
- 첫 단계는 Super 하나. 이후 정말 필요한 경우 math/code/grounded specialist teacher alias를 분리한다. 여러 alias가 동일 Super checkpoint면 물리 worker 중복을 제거한다.
- TROPD/proximal teacher와 advantage clipping은 native docs를 따라 안정화 선택지로 검토하되 다른 objective와 혼합해 성과를 해석하지 않는다.
- Gate: self-distill parity 통과 → Super logprob alignment → 소규모 학습 → heldout 향상/회귀 확인. gap 감소만으로 문제풀이 향상을 주장하지 않는다.

## 13. Dynamo: RLVR managed rollout + 최종 Serving

### 13.1 NeMo RL managed Dynamo integration(추가 공식 확인)

조사한 NeMo RL 고정 main commit에는 `policy.generation.backend: dynamo`가 실제로 구현되어 있다. Slurm-backed Ray allocation 안에서 NeMo RL이 Dynamo frontend/control plane와 고정 vLLM worker fleet를 관리하며, GRPO/Gym rollout, NCCL weight refit, cache invalidation, telemetry를 지원한다. 따라서 Dynamo를 마지막 배포에만 쓰는 것보다 **RLVR의 standard vLLM control run 후 managed Dynamo 비교 tutorial**을 추가하는 것이 좋다.

- 공식 1-node 2-GPU smoke는 train 1GPU + TP1 generation 1GPU다. 이것은 작은 1B 모델의 integration smoke이며 13.8B custom 학생 placement 보증이 아니다.
- Nemotron 3 Nano용 3-node 검증 recipe가 있으나 NemoChat의 latent 크기와 tensor mapping/refit은 별도 검증해야 한다.
- 현재 경로는 **managed vLLM-only**이며 외부 Dynamo deployment, Kubernetes/DGD, SGLang/TRT-LLM, quantized/speculative generation, cross-node model-parallel engine groups를 지원하지 않는다.
- Generation EP는 1 또는 TP와 같아야 한다. Training EP=8 설정을 generation에 그대로 복사하지 않는다.
- 별도 Dynamo venv의 vLLM/NCCL pin과 layerwise reload backport가 결합되어 있어 버전을 따로 올리지 않는다. 이미지의 opt-in build가 필요하다.
- main 기능을 release container에 자동 존재한다고 가정하지 않는다. MOPD 기본 지원과 최신 full-vocab/managed Dynamo 지원은 entrypoint와 commit별로 구분한다.
- 비교 gate: identical checkpoint/tokenizer/seed/tasks → 첫 rollout 유효성 → update/refit → cache invalidation 후 출력·logprob parity → telemetry → cleanup.
- Native MOPD teacher는 여전히 별도 Megatron scoring worker다. **Dynamo rollout 지원 발견이 Super teacher scoring 연결 문제를 해결하는 것은 아니다.**

공식 소스: https://github.com/NVIDIA-NeMo/RL/blob/7409c28e0071194af12fb8e36ca2dd7a6eb725c0/docs/guides/dynamo-generation.md

### 13.2 최종 학생 serving

- SFT/RLVR/MOPD 각각 checkpoint export 후 frozen inference correctness suite.
- Dynamo 자체는 serving orchestration이며 실제 hybrid inference 지원은 선택 backend(vLLM/TensorRT-LLM 등)의 config/kernel 지원에 달려 있다.
- Custom `H=2048, mamba_inner=3072, latent=512` config를 export할 수 있어도 backend가 자동 지원한다고 가정하지 않는다. 먼저 단일 요청·batch·cached continuation을 실행한다.
- Training logits와 generation backend logits/logprobs를 짧은 gold sequences에서 비교. Mamba cache/reset, latent expert, output head, token shift 검증.
- Serve benchmark: TTFT, inter-token latency, output tokens/sec, p50/p95, concurrency별 memory; short/long prompt와 reasoning length별 결과.
- KV-cache에 해당하지 않는 Mamba recurrent state도 cache correctness/요청 격리 테스트. Stateless and cached 결과 비교.
- Single-node monolithic serving 먼저, disaggregated prefill/decode는 선택 tutorial. Hybrid model에서 disaggregation/state transfer 지원을 확인한 경우에만 진행.
- **Dynamo endpoint가 NeMo RL generation backend 또는 native teacher scorer에 바로 연결된다는 주장은 하지 않는다.** Serving demo와 RL worker integration을 별개 gate로 둔다.

### 13.3 독립 Inference Performance 트랙 — 사용자 확정

학습 단계의 부속 demo가 아니라 [독립 inference 트랙](en/INFERENCE_PERFORMANCE_TRACK.md) ([한국어](ko/INFERENCE_PERFORMANCE_TRACK.md))을 병행한다. 지원되는 pretrained model/engine control에서 workload/SLO와 P99/goodput, operator roofline, profiler 기반 실제 코드 수정, quality/regression 및 upstream-ready evidence를 검증한다. Custom NemoChat serving/export/state support는 별도 gate다.

첫 코드는 saved-request 분석과 single-operator roofline CPU 도구이며 load generation/GPU serving/최적화 결과는 대기다. Pretraining WSD/WSM 정책과 resume는 유지한다.

## 14. 자원·시간·저장소 계획

### 14.1 Memory

A의 파라미터당 16B를 가정한 전형적인 mixed-precision Adam 전체 모델 상태는 약 **220.4GB**다. 이는 노드 전체 aggregate 계산일 뿐 GPU별 요구량이 아니다. Non-expert replication, expert DP/EP, optimizer sharding, activations, MoE buffers, checkpoint gather와 generation/reference copies 때문에 달라진다. Profile 후 rank별 peak memory 및 optimizer state dtype를 공개한다.

단일 학생 학습의 메모리보다 Super teacher + 학생 optimizer + student rollout actor/reference를 동시에 올리는 운영 구성이 더 어렵다. 단순 전체 HBM 합으로 실행 가능을 판정하지 않는다.

### 14.2 Time: 실측 throughput으로 정하기

현재 tokens/sec 측정은 없다. 다음은 **pretraining 30B만의 가상 민감도**, eval/checkpoint/idle overhead를 제외한 계산이다.

| 노드 useful training tokens/sec (가정) | 30B 소요 일수 |
|---:|---:|
| 2000 | 173.6일 |
| 5000 | 69.4일 |
| 10000 | 34.7일 |
| 20000 | 17.4일 |

위 수치는 B300 성능 예측이 아니다. 실측 pre/mid/long throughput을 각각 적용하고, RL은 generation + scoring + policy update 시간으로 따로 추정한다. 3B tutorial pretrain은 같은 throughput에서 위 시간의 1/10이다. **최초 profile 결과 후 사용자의 GPU 가용 시간/비용에 맞춰 budget을 줄이거나 늘린다.**

### 14.2.1 일주일 단위 임대와 재개 전략

고정 token budget을 무조건 한 주에 넣지 않는다. `tokens_per_week = useful_tokens_per_second × 실제 training-active seconds`로 측정 후 배분한다. 7일 중 75%를 학습에 쓴다는 **예시 가정**에서는:

| Useful train tokens/sec(가정) | 주당 train tokens |
|---:|---:|
| 5000 | 2.268B |
| 10000 | 4.536B |
| 20000 | 9.072B |

이는 GPU 성능 예측도 RL/MOPD 비용 추정도 아니다. setup/profile/eval/checkpoint/export와 long-context/rollout 비용을 별도 기록한다.

- GPU 없는 시간: 데이터 subset 구축, license audit, fixture/CI, docs, next-stage preparation. 가능한 CPU 작업을 임대 시작 전에 끝낸다.
- 첫 임대: 환경 compatibility + checkpoint/export/inference round-trip + A/B short pilot을 최우선; 선택 모델의 pretrain을 남은 시간에 진행. 모든 7 stages를 첫 주 완료한다고 약속하지 않는다.
- 이후 임대: pretrain 이어서 수행 또는 완료 checkpoint에서 다음 stage 진행. 날짜가 아니라 각 stage quality gate와 누적 consumed token 수로 전환한다.
- 체크포인트에는 model/optimizer/scheduler/RNG, consumed tokens, source quota counters, packing/dataloader cursor, tokenizer/env/data manifests를 보존한다.
- Distributed checkpoint는 동일 8-GPU topology로 resume하는 것을 기본으로 한다. GPU split을 바꾸는 MOPD 전환은 reshard/conversion smoke가 별도로 필요하다.
- 임대 종료 직전이 아니라 종료 수시간 전에 final save와 외부 persistent storage 업로드를 완료한다. Checksum/read-back을 확인한 후 로컬 임시 artifact를 정리한다. Bandwidth/파일 크기를 측정해 실제 필요한 lead time을 정한다.
- Latest resume checkpoint + 직전 정상 checkpoint + milestone weight export를 보존한다. 모든 step 체크포인트를 무한히 보관하지 않는다.
- 다음 임대 시작 시 checkpoint download/read-back → 짧은 resume smoke → loss/LR/token cursor 연속성 확인 후 full run. 모델 weights만 저장해서 optimizer/scheduler가 초기화되는 것을 재개로 부르지 않는다.

### 14.3 Storage

- 모델 BF16 weights만 약 27.5GB이지만 optimizer/distributed checkpoint는 훨씬 크다.
- 데이터 원문/tokenized/cache와 checkpoints를 분리. checkpoint 크기는 첫 저장 후 실제 측정하며 full checkpoints 3~5개 + milestone exports를 보존하는 정책부터.
- Raw corpus 전체 mirroring은 피하고 streaming/subset 처리. 데이터 lineage manifest와 source revisions는 삭제하지 않는다.
- Nsight 결과는 작은 window만 저장. Secrets, raw PII, teacher API credentials 및 sandbox outputs는 Git에 넣지 않는다.
- 전체 필요한 NVMe/RAM은 pilot 이후 산정한다. 다운로드 규모를 dataset 이름만으로 확정하지 않는다.

## 15. 구현 roadmap과 파일 명세 (CPU 기반 구현 + GPU 통합 예정)

아래 체크박스는 GPU end-to-end 목표이며 CPU fixture 통과가 전체 milestone 완료를 의미하지 않는다. **현재 CPU-first 코드가 추가되어** bounded ClimbMix acquisition, HF base evaluation, NIAH pipeline, LC scheduler 및 tiny optimizer/resume smoke를 실제 실행했다. 구현/검증 범위와 아직 미완료인 GPU 통합은 [CPU walkthrough](en/CPU_IMPLEMENTATION.md), [한국어 해설](ko/CPU_IMPLEMENTATION.md), [실행 검증 report](IMPLEMENTATION_VERIFICATION.md)에 구분한다. Bridge provider는 설정 코드만 있으며 B300 학습·분산 동작·SFT/RLVR/MOPD는 아직 검증되지 않았다.

### 15.1 사용자 지정 3부 진행·블로그 공개 단위

세 묶음은 내부 engineering milestone과 별개의 **execution/publication package**다. 각 묶음이 검증 gate를 통과한 후 해당 블로그를 작성·검토·공개하며, 임대 한 주와 묶음 하나가 반드시 일치하지는 않는다.

| Part | 포함 단계 | 종료/handoff artifact | 블로그 중심 내용 |
|---|---|---|---|
| I | **[1][2][3][4]** | LC-2 후 long-context base checkpoint, data/tokenizer/eval manifests | 데이터 구축·A/B latent MoE·pre/mid·long/short CPT·NIAH |
| II | **[5][6]** | Post-SFT 및 Post-RLVR exports/results, frozen RLVR student | SFT·검증 rewards·GRPO/RLVR·Dynamo managed rollout·retention |
| III | **[7]** | Super MOPD checkpoint, final serving/public reference bundle | Same-node vLLM scoring/controller·distillation·최종 Dynamo serving |

- Part I 모델은 아직 assistant가 아니라 **base**라고 명시한다. Part II 입력은 Part I의 immutable LC checkpoint. Part III 기본 입력은 Part II RLVR checkpoint이며 SFT 분기는 별도 lineage로 기록한다.
- Pre-LC/post-LC-1/post-LC-2, post-SFT/post-RLVR, pre/post-MOPD 비교 결과를 보존하여 글에서 각 단계의 효과를 분리한다. NIAH heatmaps와 실제 short-task retention Figure를 동일 pack으로 비교한다.
- Canonical 영어 블로그 3편 + 별도 한국어 해설. 세부 기술 문서는 stage별로 계속 유지하지만 블로그를 일곱 편이나 네 번째 serving 편으로 자동 분할하지 않는다.
- 최초 시리즈는 Tutorial-budget reference, Extended 결과는 checkpoint/config가 구분되는 versioned addendum으로 공개한다.
- 실제 실험 중 hypothesis/config/input/output/실패/비용 diary와 raw evaluation·Figure 입력을 남긴다. 묶음 종료 시 artifact freeze와 license/재현성 review. 결과 없는 초안에 추정 성능 Figure나 미검증 공개 링크를 넣지 않는다.
- 사용자가 지정한 게시 destination과 공개 artifact 조건을 확인한 뒤 업로드/게시하며, 현재 단계에서는 글 게시·Git tag·release upload를 하지 않았다.

세부 목차·필수 Figure·종료 gate·예정 블로그 파일: [English execution/blog plan](en/EXECUTION_AND_BLOG_SERIES_PLAN.md), [한국어 해설](ko/EXECUTION_AND_BLOG_SERIES_PLAN.md).

### Milestone 0 — Reproducibility와 resource gate

- [ ] `README.md`: 프로젝트 목적, smoke/tutorial/extended 구분, native MOPD 자원 제약.
- [ ] `environments/{data,train,rl,serve}/`: 분리된 container/dependency 명세.
- [ ] `configs/environment.lock.yaml`: image digest, SDK commit, tokenizer/model/data revisions.
- [ ] `docs/hardware.md`: 실제 GPU topology, driver, NVLink, CPU/RAM/NVMe 및 SDK compatibility.
- [ ] `tests/test_manifest_schema.py`: 필수 lineage/license 필드 검증.
- 검증: clean environment에서 fixture 실행; 필요한 resources가 없으면 설치나 학습을 시작하지 않는다.

### Milestone 1 — Architecture와 GEMM gate

- [ ] `configs/model/{latent512,latent1024}.yaml`. Nonlatent config는 보류.
- [ ] `src/nemochat/models/hybrid_provider.py`: 공식 HybridModelProvider 기반 explicit architecture.
- [ ] `tools/count_parameters.py`: block별·공통/routed/shared·active 계산과 실제 instantiated weights 검증.
- [ ] `tests/test_model_config.py`: pattern count, non-GLU, shared overlap off, tokenizer size 검증.
- [ ] `tests/test_checkpoint_roundtrip.py`: HF/MCore tensor mapping 및 logits round-trip.
- [ ] `benchmarks/moe_gemm.py`, `benchmarks/train_step.py`: A/B fixed-shape 및 end-to-end 비교.
- [ ] `docs/profiling/latent_moe.md`: 실제 nsys/ncu 결과와 채택 이유.
- 검증: BF16 fwd/bwd/save/resume, A/B pilot, profile gate 통과. GPU 없는 CPU CI는 pure config/계산 테스트만.

### Milestone 2 — Data [1]

- [ ] `src/nemochat/data/{schemas,curate,split,tokenize,pack}.py`.
- [ ] `configs/data/{sources,filters,blends}.yaml`.
- [ ] `configs/datadesigner/{math,code,grounded,json}.yaml`.
- [ ] `tests/test_data_lineage.py`, `tests/test_dedup_split.py`, `tests/test_contamination.py`.
- [ ] `docs/data/{licenses,curation_report,synthetic_report}.md`.
- 검증: fixture → sampled shards → token count → split/dedup reports. 不허용 license source가 기본 blend로 들어오면 실패.

### Milestone 3 — LM stages [2–5]

- [ ] `configs/train/{pretrain,mid,long_context,sft}.yaml`.
- [ ] `src/nemochat/train/{launch,stage_registry}.py`: checkpoint/tokenizer/data manifests 계약.
- [ ] `tests/test_loss_masks.py`, `tests/test_packed_mamba_reset.py`, `tests/test_resume.py`.
- [ ] `src/nemochat/train/lc_controller.py`, `tests/test_lc_schedule.py`, `tests/test_lc_token_accounting.py`, `tests/test_lc_resume.py`: long/4K update alternation와 checkpoint resume.
- [ ] `src/nemochat/eval/niah_{generate,run,score,plot}.py`, `configs/eval/niah_{smoke,64k,256k}.yaml`.
- [ ] `tests/test_niah_fixture.py`, `tests/test_niah_scoring.py`, `tests/test_niah_plot_inputs.py`: 실제 tokenizer 길이/needle 보존·score·raw-results 기반 Figure 검증.
- [ ] `results/niah/<checkpoint>/{predictions.jsonl,cells.csv}`, `figures/niah/` 실제 inference Figure. CI fixture Figure를 모델 측정 결과로 공개하지 않는다.
- [ ] `evals/{base,math,code,long_context,instruction}/` 및 stage별 result schema.
- [ ] `src/nemochat/eval/{token_bytes,bpb,core_adapter,base_eval}.py`, `configs/eval/{base_quick,base_full}.yaml`: pinned nanochat-compatible evaluation pack, Super raw-byte/TP-CP-EP parity, CORE baseline/answer-span/teacher-forced tests 및 실제 inference Figure.
- [ ] ClimbMix revision/shards/quota/provenance/license manifest와 last validation shard exclusion; full-corpus downloader/repackaging script 실행 금지.
- [ ] `docs/tutorials/02_pretrain.md`부터 `05_sft.md`까지 실제 launch/결과/실패 사례.
- 검증: 모든 stage smoke → short pilot → v1, 완료 checkpoint와 evaluation JSON이 다음 stage 입력 조건.

### Milestone 4 — RLVR [6]

- [ ] `src/nemochat/verifiers/{math,python_sandbox,json_schema}.py`.
- [ ] `configs/rl/rlvr.yaml`, `configs/gym/tasks.yaml`.
- [ ] `configs/rl/rlvr_dynamo.yaml`: 공식 managed Dynamo/vLLM 경로를 standard vLLM control과 비교; train-generation GPU 분할과 custom hybrid weight refit 검증.
- [ ] `tests/test_dynamo_refit.py`: policy version 일치, refit 후 cache invalidation, rollout token/logprob metadata 검증.
- [ ] `tests/test_verifiers.py`, `tests/test_reward_hacking.py`, `tests/test_generation_logprob_parity.py`.
- [ ] `docs/tutorials/06_rlvr.md`: reward correctness, pass rate, rollout/policy costs.
- 검증: verifier fixtures → policy/generation parity → tiny GRPO → v1 token cap. Unsafe code sandbox 실패 시 차단.

### Milestone 5 — MOPD [7]

- [ ] `configs/rl/{mopd_self,mopd_super}.yaml`.
- [ ] Native scheduler 가능 여부와 teacher hardware gate를 먼저 판정.
- [ ] 필요하면 `src/nemochat/distill/{teacher_scoring_adapter,sequential_controller}.py`를 별도 experimental module로 구현.
- [ ] `tests/test_teacher_token_alignment.py`, `tests/test_self_distill.py`, `tests/test_policy_snapshot.py`.
- [ ] `docs/tutorials/07_mopd.md`: native/extension/offline fallback를 명확히 구분.
- 검증: self teacher gap≈0 → Super teacher scoring → tiny updates → v1. 별도 자원 없는 native 경로는 not-targeted/resource-blocked로 기록하고, 선택한 same-node vLLM extension의 실행 여부는 따로 보고.

### Milestone 6 — Serving와 reference publication

- [ ] `configs/serve/dynamo.yaml`, `tools/export_checkpoint.py`.
- [ ] `tests/test_inference_cache.py`, `tests/test_export_logits.py`.
- [ ] `benchmarks/serving.py`, `docs/tutorials/08_dynamo.md`.
- [ ] `docs/results/{model_card,data_card,stage_matrix,limitations}.md`.
- 검증: clean-start end-to-end walkthrough, SDK별 최소 하나의 실제 기능/산출물, checkpoint → serving logits parity, 버전 재현성, license review.

각 code task는 failing test → 구현 → targeted test → GPU integration(필요 시) → review 순으로 작게 쪼갠다. 문서에 아직 실행하지 않은 명령의 출력이나 성능 수치를 기입하지 않는다. Git commit/push는 별도 승인/요청 없이 하지 않는다.

## 16. 공통 평가와 성공 기준

- Frozen evaluation pack: source-heldout PPL, math exact-answer, Python tests, instruction/JSON compliance, grounded QA, long-context retrieval, safety fixtures.
- 공개 benchmark는 evaluation-only이며 exact/near-duplicate contamination scan 후 점수 보고. Tiny custom tests를 공개 benchmark 점수로 포장하지 않는다.
- 모든 stage: inputs hash → data manifest → SDK/env config → checkpoint ID → evaluation → profile → limitations.
- RLVR/MOPD: reward/gap와 downstream heldout 점수를 분리. 반복 evaluation에 따른 selection bias를 기록.
- Tutorial 성공은 각 SDK를 import했다는 뜻이 아니다. **Curator는 정제 report, DataDesigner는 validated generated data, Bridge는 train/resume/export, RL/Gym은 실제 verifier-based updates, Nsight는 해석된 profile, Dynamo는 실제 요청과 latency 측정**이 있어야 한다.
- GPU/teacher 자원이 없는 단계는 resource-blocked로 표시. 완료하지 않은 native MOPD를 완료로 주장하지 않는다.
- Tutorial v1과 extended checkpoint를 명확히 구분하고, 매우 짧은 from-scratch 모델의 낮은 능력을 솔직히 공개한다.

## 17. 사용자 확정 사항과 남은 운영 항목

### 확정 사항

1. **Hardware:** 표준 메모리 구성 B300 8장, 단일 NVLink 노드. 일주일 단위 임대를 여러 번 사용.
2. **Teacher:** 추가 node 없이 동일 node에서 vLLM Super teacher. Native MOPD placement 대신 local scoring adapter와 single-node controller를 구현할 범위로 선택.
3. **Language:** 최종 결과물과 canonical docs/code/comments는 영어. 이해를 돕는 한국어 문서는 `docs/ko/`에 별도 제공. 영어 training mixture 유지.
4. **Publication:** 비상업적 **교육용** tutorial이며 code/config/data/weights 모두 공개하는 방향. ClimbMix 원본 NC·attribution 조건은 유지한다. 다만 공개 의도와 법적 재배포 가능 여부는 다르므로 license gate 통과 artifact만 배포.
5. **Context:** 64K 기본 목표. 128K/256K agentic coding은 staged stretch track이며 실제 측정과 추가 데이터/예산이 필요.
6. **Pilot:** A/B만 비교. C는 계산 참고이며 실행 보류.

### 공개 artifact 정책

- 영문 `README.md`, `docs/en/` canonical tutorials, `docs/ko/` 해설 및 model/data cards. 예제 코드의 identifiers/comments/docstrings는 영어.
- Code/config license를 선택한 뒤 model/tokenizer/teacher-output 및 각 dataset source의 조건을 별도로 기록한다. Code license를 모든 데이터/weights에 일괄 적용하지 않는다.
- 재배포 가능 정제/합성 데이터는 provenance/attribution/필터 report와 공개. 원문 재배포가 불가한 source는 acquisition recipe + source IDs/revisions + 허용된 derived artifact만 공개한다. 이 예외를 숨기지 않는다.
- Weights는 tokenizer 권리/teacher distillation 조건/학습 데이터 이용조건 review 후 license와 limitations를 함께 공개한다.
- Secrets/API keys/PII/개인 경로를 artifact에서 제거하고, teacher 사전학습 오염은 완전히 통제할 수 없다는 한계도 명시한다.

### 아직 필요한 운영 확인

- 임대 종료 후에도 보존 가능한 object storage/NVMe, bandwidth와 egress 비용, checkpoint 업로드 권한.
- 실제 CPU RAM/local NVMe, driver/CUDA/Slurm/container 환경. GPU가 있는 임대 node와 현재 문서 서버는 별개.
- 같은 node의 Super vLLM BF16 TP4 + 학생 EP4 후보 placement가 correctness와 peak memory를 통과하는지. GPU split은 확정 topology가 아니라 benchmark 후보.
- 첫 주 실측 throughput 이후 v1/extended token budget과 추가 임대 회수를 조정한다.


## 18. 조사 출처와 snapshot

### 공식 모델·학습 구조

- Nano config: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json
- Super config: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json
- Super implementation(latent/shared 구조): https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/modeling_nemotron_h.py
- Bridge Nano recipe: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c860f8a5bc5fddd78690f32baa0b8696774308b4/src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_nano.py
- Bridge Super recipe: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c860f8a5bc5fddd78690f32baa0b8696774308b4/src/megatron/bridge/recipes/nemotronh/h100/nemotron_3_super.py
- Core latent implementation: https://github.com/NVIDIA/Megatron-LM/blob/fbbe6eb5490de4c81c51bd8f715f333c91837fe4/megatron/core/transformer/moe/moe_layer.py
- Bridge parallelisms: https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html

### RL/MOPD와 serving

- Native MOPD: https://github.com/NVIDIA-NeMo/RL/blob/7409c28e0071194af12fb8e36ca2dd7a6eb725c0/docs/about/algorithms/mopd.md
- 별도 OPD 경로(지원 문구가 native MOPD와 다름): https://github.com/NVIDIA-NeMo/RL/blob/7409c28e0071194af12fb8e36ca2dd7a6eb725c0/docs/about/algorithms/on-policy-distillation.md
- Nano post-training guide: https://github.com/NVIDIA-NeMo/RL/blob/7409c28e0071194af12fb8e36ca2dd7a6eb725c0/docs/guides/models/nemotron/nemotron-3-nano.md
- Dynamo repo: https://github.com/ai-dynamo/dynamo
- Nsight Systems: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Compute: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

### 데이터 SDK 및 후보 데이터

- Curator: https://github.com/NVIDIA-NeMo/Curator
- DataDesigner: https://github.com/NVIDIA-NeMo/DataDesigner
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- https://huggingface.co/datasets/HuggingFaceFW/finepdfs
- https://huggingface.co/datasets/nvidia/OpenMathInstruct-2
- https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
- https://huggingface.co/datasets/open-r1/OpenR1-Math-220k
- Opt-in, license 미확정: https://huggingface.co/datasets/open-r1/verifiable-coding-problems-python

조사 시 upstream main commit(실행용 compatibility lock을 의미하지 않음):

```text
Megatron Bridge c860f8a5bc5fddd78690f32baa0b8696774308b4
Megatron Core/LM fbbe6eb5490de4c81c51bd8f715f333c91837fe4
NeMo RL         7409c28e0071194af12fb8e36ca2dd7a6eb725c0
Curator         73e83f4165d849e123770d4d9e621b62202d6a35
DataDesigner    f9aff6ea5f81f913111f76ded3684330bf25beac
Dynamo          1f7a233393bbff0eedee9e12286fd4424c88b6fa
```

Model source revisions(API 검증):

```text
Super BF16 2dc98e2afe4face0e4ce40972a915c45368bd34a
Nano BF16  bf77c3174f68ad409e1c2aa60daeb46e32d1c606
```

두 model card의 API license 값은 `other`이므로 dataset의 Apache/CC license처럼 취급하지 않는다. 실제 model license/추가 조건을 검토한 뒤 teacher outputs, tokenizer files 및 학생 weights의 배포 정책을 확정한다.

Dataset revisions(API 검증, 실행 시 재고정):

```text
FineWeb-Edu       87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
FinePDFs          220bac3acbf07789502c621d2d33952f51ac7f86
SmolLM-Corpus     3ba9d605774198c5868892d7a8deda78031a781f
OpenMathInstruct2 469216e3f46f4dacf476b382e192485ea51a143e
Tulu3-SFT         b14afda60f1bbebe55d5d2fa1e4df5042f97f8be
OpenR1-Math       e4e141ec9dea9f8326f4d347be56105859b2bd68
```

公式 docs와 repo main 사이에 지원 문구가 상충하면 실행 코드를 확인하고 검증된 dependency 조합의 integration test로 판정한다. 특히 별도 OPD 문서의 Megatron 미지원 설명과 repo의 Megatron distillation examples가 공존하므로 entrypoint별 지원을 구분한다.
