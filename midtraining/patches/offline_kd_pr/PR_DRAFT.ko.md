# DRAFT — Megatron-LM native cached-logits KD의 Bridge port

[English](PR_DRAFT.md) · [구현/명령](README.ko.md) · [상세 gate](FEASIBILITY.ko.md)

**로컬 review 전용이다. commit/remote branch/PR, installed SDK 수정, active training 변경은 없다.** 실제 applyable diff를 만들고 disposable SDK copy에만 적용했다.

## 대상과 목적

installed **Megatron-Bridge 0.6.1, `2a8fcdd05181ffcef261192f71ff5910d0005bf2`**가 base다. image/source hash는 `evidence/integration-base.json`에 있다. 앞서 읽은 upstream `2311cf8c32ee895ab96db8391996c8f3732e1e7d`와 혼용하지 않으며 upstream main rebase는 별도다.

기존 Megatron-LM offline KD 설계를 최대한 유지한다. `LogitsSaverHooks`, native tar/zstd storage/reader, output-head capture, ghost-tail KL을 연결한다. 새 JSON production schema나 vLLM-first serving pipeline을 만들지 않고 default/online ModelOpt 경로를 보존한다.

## 실제 diff

1. `training/cached_kd/{logits_saver,utils_logits,cached_logits_loss}.py`를 installed Core 원문에서 port하고 copyright를 유지한다. tensor/storage/math 로직은 보존하고 CLI args/print/microbatch 의존성만 명시적으로 바꾼다. reader에 iteration 누락 거부를 추가했다.
2. scoped `CoreRuntime`, 검증된 `CachedKDConfig`를 추가한다. 존재하지 않는 ConfigContainer 필드나 global CLI init을 만들지 않는다. caller가 native sampler/blend/sample budget과 실제 Bridge step/microbatch getter를 고정한다.
3. `CachedKDForwardStep`은 teacher-only no_grad `cache_forward`(optimizer/backward 없음), complete-iteration 동기 flush, fresh-directory/partial-write guard, student hook/logits lifetime, sum/count/report와 fixed alpha, LM-only eval, failure/replay/cursor check, hook/worker cleanup 및 explicit checkpoint binding을 구현한다.
4. `gpt_step.make_cached_kd_forward_step(...)`를 덧붙여 실제 `_forward_step_common`을 연결한다. 그 앞 기존 source/AST와 online ModelOpt는 동일하다. cached adapter와 ModelOpt `DistillationModel` 동시 사용은 거부한다.

지원 objective는 **`megatron-lm-ghost-tail-fixed-alpha`**, T=1, `(1-alpha)*LM + alpha*KD`, 명시적 기본 alpha=0.5다. K<V이면 coarse-tail KL이며 full KL/conditional topK가 아니다. 현재 detached ModelOpt balancer와 별도 정책으로 선언한다. cached ModelOpt mode는 추가하지 않았다.

## 검증한 수용 범위

- [x] `evidence/bridge-tdd-*`에 RED→GREEN 구현 기록과 환경 실패를 보존했다.
- [x] writer/storage/reader는 실제 runtime 코드이며 import stub이나 AST 추출만으로 만든 production 구현이 아니다.
- [x] pinned CPU image에서 실제 unmodified SDK writer→unmodified SDK reader/port reader가 native tar/zstd tensor cache로 상호운용했다. 좁은 getter 주입은 테스트 메모리 내에서만 수행하고 복구한다.
- [x] port writer→reader→tiny student Loss/backward, mask/count/report, LM-only eval, cursor resume, mismatch/gap/replay/partial-write 거부를 검증했다.
- [x] 정확한 installed SDK copy에 `git apply --check`와 apply, 실제 staged Bridge factory/dependency import, default/ModelOpt source 보존을 확인했다.
- [x] 기존 JSON fixture/source-control suite도 보존·재실행했다. JSON/vLLM은 reference/optional future이며 native 통합 수용 필수 조건이 아니다.

최종 count, CPU-only Engine ID/scope/exit는 `evidence/integration-acceptance.json`에 있다. SDK warning을 숨기지 않았다. native Nemotron 모델이나 Bridge scheduler/optimizer/checkpoint 전체를 실행한 증거는 아니다.

## Production 완료로 주장하지 않는 미구현 사항

- [ ] 정확한 Base teacher checkpoint/view/tokenizer/model load, tensor 누락 검사와 GPU BF16 score/loss/gradient/update parity.
- [ ] TP/CP/DP/EP/PP/VPP와 native CP/MBS/GBS/sample ownership, MTP/recompute/overlap/graph. Core helper는 보존했지만 adapter는 미지원 기능을 거부한다.
- [ ] 실제 sampler/packing/position/mask/Mamba reset 계보와 immutable producer manifest. native hash/caller cache_id가 teacher/tokenizer byte를 인증하지 않는다.
- [ ] 성공 native optimizer checkpoint와 cache sidecar의 atomic transaction/reload. `checkpoint_binding()`/reader resume를 테스트했지만 TrainState가 자동 저장하지 않는다.
- [ ] rerun rewind, skip recovery, spawn/restart, remote MSC atomic readiness, native cache 보안 강화와 concurrent writer publication.
- [ ] 같은 조건의 native 전체 update, dump+I/O+student first-use GPU-hours, 장기 안정성과 heldout 품질. CPU 기반 speedup 주장은 없다.

향후 비교는 **같은 Core objective의 caching 효과**와 ModelOpt fullKL/detached balance→ghost-tail/fixed-alpha 정책 변경을 분리한다. tiny full-K/top32와 online on-the-fly/cache도 각각 조건을 고정한다. full FP16 values만 약439.8 TB이므로 전체 full-K dump는 하지 않는다. 부모 review 및 별도 GPU gate 승인 전 기존 online 계획을 유지한다.

## 인계

이 폴더와 `tests/test_offline_kd_pr_bridge.py`만 수정했으며 기존 `tests/test_offline_kd_pr.py`는 그대로다. ledger/controller/corpus와 채택 결정은 부모 소유다. snapshot은 primary evidence이고 isolated patch는 review artifact일 뿐, installed SDK 수정이나 학습 시작 권한이 아니다.
