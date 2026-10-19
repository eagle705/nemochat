# 고정 A/B/C 평가 인계

영문 상세 규약과 실행 명령은 [README.md](README.md), 실행 증거는 저장소 루트 기준 `results/midtraining/offline2b-ablation-eval-01/`에 있다.

## 비교의 의미

- A는 원본 BASE, B는 **1.0 LM-only** 1000 accepted updates, C는 배포된 **0.5 LM + 0.5 KD** 1000 accepted updates이다.
- B/C는 accepted batch IDs·순서·초기화·optimizer/LR·precision 등 설정 일치 여부를 부모가 확인한다.
- **일반 LM 학습 대비 실제 KD recipe의 실용적 효과**를 비교한다. 공통 LM coefficient를 고정한 KD 항의 인과적 분리 실험이 아니다. 별도 0.5 LM control은 요청되거나 실행되지 않았다.

## 동결된 입력

`suite.json`의 canonical-content SHA256:
`07609a0cf53e4a63d555a3e50938adfed1cb05471d02b37fb93461e7879af19e`.

- MMLU: 지정된 10개 subject마다 첫 test 20개, **총 200개**. 각 subject의 명시적 dev 5개를 고정된 순서로 쓰는 5-shot.
- ARC-Challenge: 첫 test 100개, zero-shot.
- HellaSwag: 첫 validation 100개, zero-shot.
- 총 **400개 질문**의 원본 row·ID·gold·prompt·continuation·token IDs와 revision을 결과 전에 고정했다. chat template/BOS 추가·padding·truncation은 없다.
- 비표준 first-N 진단 subset이다. full MMLU나 leaderboard 점수가 아니며, benchmark train contamination이 없다는 증거도 아니다.

공식 EleutherAI v0.4.8 commit은 `6d2abda4fd171e68a8789330c4149e37c1ca0bda`이다. 데이터 revision 및 primary-source 원문은 `research/`, SHA256은 `contract.json`에 저장했다. 실제 image의 NVIDIA fork는 nvidia_lm_eval 25.11.1이며 보고 버전은 lm_eval 0.4.8이다.

## Heldout LM Loss

기존 reserved validation 4개 source에서 각각 16개의 1025-ID window를 고정했다. window당 next-token target은 1024개로, **64 windows / 65,536 Loss-bearing positions**이다. 경계의 context ID만 하나 공유하고 target은 반복하지 않는다. 기존 EOS를 유지한 document concatenation이며 padding은 없다.

8개 전체 validation bin/idx 파일을 O_DIRECT로 읽어 manifest SHA256과 일치함을 확인했다. 인증한 immutable bytes에서 token을 복사했으며 임시 mmap reader view를 사용하지 않았다. 모든 validation ID는 vocabulary 범위 안이고 18…29는 없었다. 기존 manifest의 train/heldout exclusion 증거를 계승했지만 새로운 전체 train raw-text 중복 검사는 아니다. host read fault를 복구했다고 주장하지 않는다. Model payload byte integrity waiver와 데이터 인증은 별개이다.

## 실제 실행 증거

- 선택된 실제 400개 row 전부의 prompt/request/gold/metric이 설치된 task logic과 일치했다.
- 내려받은 공식 v0.4.8 함수들을 실제 row/choice에 적용하는 검증을 포함해 **4 tests passed**.
- 공식 HFLM backend는 optional PEFT→Transformer Engine import에서 libcuda.so.1 부재로 실패했다. GPU를 추가하지 않고 명시적으로 표시한 source-derived custom evaluator를 사용했다. 설치 package는 수정하지 않았다.
- 실제 BASE CPU BF16 smoke는 질문 2개, 유한 choice score 8개, heldout 1 window/1024 targets를 처리하고 Engine exit 0으로 끝났다.
- Heldout full-vocabulary NLL **2.3266710285244585**, PPL **10.243783446046432**. 두 질문은 모두 오답이므로 이 smoke는 품질 개선의 증거가 아니라 실행 가능성의 증거이다.
- 네트워크는 준비 단계만 허용했고 scoring은 network none, CUDA_VISIBLE_DEVICES는 빈 값, DeviceRequests 없음, CPU 4 threads/4 CPUs, 최대 memory 128 GiB였다. HF/HOME/compiler cache는 모두 소유한 `/raid` 결과 디렉터리 안에 있다.

## 남은 작업

부모가 B/C native checkpoint와 HF export, accepted batch/settings 일치를 승인한 후 README의 full score 명령을 실행한다. `--smoke`를 빼면 400개 질문과 64개 window 전체를 평가한다. B export 경로는 실제 완료 후 지정하며 추측하지 않았다. C 예정 경로는 `checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01`이다.

`compare.py`는 완전한 입력 coverage와 suite hash를 요구하고 paired Accuracy/acc_norm·gold logp·heldout NLL delta 및 고정 seed의 5000 bootstrap 결과를 남긴다. MMLU는 subject 안에서, LM Loss는 source 안에서 resampling한다. 이는 training seed uncertainty를 추정하는 절차가 아니다.

Full scoring은 아직 실행하지 않았다. GPU·학습·다른 child artifact의 대기/수정·publication·root ledger 수정은 하지 않았다. KD의 이득은 아직 판정할 수 없다.
