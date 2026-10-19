# 실현 가능성: Megatron-LM cached KD를 Bridge에 충실하게 연결

[English](FEASIBILITY.md) · [구현/명령](README.ko.md)

## 고정한 범위

대상은 **installed Bridge 0.6.1, `2a8fcdd05181ffcef261192f71ff5910d0005bf2`**다. 기존 upstream `2311cf8c32ee895ab96db8391996c8f3732e1e7d`와 installed `gpt_step.py`가 상당히 다르므로 혼용하지 않는다. 정지된 pinned-image container에서 실제 config/state/train/pretrain/loss/forward seam을 읽었으며 `evidence/bridge-installed/`, `integration-base.json`에 근거를 남겼다. Core 원문 hash는 `installed-core-provenance.json`, upstream 참고 commit은 `upstream_refs.json`에 있다.

실행 가능한 isolated port와 applyable diff이지 **production distributed 지원 완료가 아니다**. frozen recovery/native15 source, online ModelOpt 설정, corpus/controller/ledger와 installed package를 변경하지 않았다.

## 원문 → 구현

| 원래 경로 | 유지한 로직과 Bridge 적응 |
|---|---|
| Core `logits_saver.py` | main output hook, global log-sum-exp/topK, FP16/BF16/FP32, topP/sentinel, 17-bit ID, buffer/take_pending/native tar writer를 유지한다. Bridge producer는 complete iteration마다 동기 flush하여 pending buffer를 제한한다. async checkpoint 통합을 구현했다고 주장하지 않는다. |
| Core `utils_logits.py` | tar member/metadata hash/zstd/torch/local·MSC/storage/prefetch 유지. CLI getter와 blend 의존성만 명시적으로 재결합한다. |
| Core `cached_logits_loss.py` | TeacherTarDataset, DP remapping, CachedLogitsKDLoss, full-vocab student normalizer와 ghost-tail 수식 유지. iteration 누락으로 stream이 조용히 밀리지 않도록 연속성 검사를 추가했다. |
| Native LossFuncCallable | masked LM/KD sum, token count/report key, LM-only eval, fixed alpha 의미를 따른다. deferred loss에서 global hook을 읽지 않고 해당 forward의 instance-owned capture를 사용한다. |
| Bridge `_forward_step_common` | 실제 batch/model-forward seam을 명시적 factory로 연결한다. 기존 `forward_step`/`forward_step_modelopt`는 byte 그대로 보존한다. 표준 loss tuple을 scheduler에 반환하며 optimizer/trainer를 새로 만들지 않는다. |
| Bridge TrainState/pretrain | `train_state.step`을 zero-based cache iteration으로 사용하고 최초 attach 때 consumed samples를 확인한다. sidecar binding은 cache ID/objective/alpha/dataset/fixed GBS·MBS/next iteration/consumed samples를 검증한다. 자동 checkpoint transaction은 미구현이다. |

`CoreRuntime`은 `megatron.training` import를 대체하는 scoped dependency다. `set_global_variables`, 가짜 Bridge CLI 인자, installed getter 수정은 하지 않는다. unmodified SDK 호환성 테스트만 CLI getter를 메모리 내에서 좁게 주입하고 pytest가 복구한다. fork 상속은 명시했지만 spawn/restart worker transport는 해결하지 않았다.

## Production 후보 wire format은 native 그대로

`cp{CP}_dp{DP}__{last_iteration}.tar`의 첫 member는 `_meta.json`, 이후는 `{iteration}.pt.zst`다. zstd를 풀면 torch dict의 `values`, uint16 `indices_low`, bool `bit_17` list가 있으며 microbatch마다 `[S,B,K]` tensor가 있다. reader가 global integer ID를 복구한다. local native writer는 `.tmp` 뒤 `os.replace`로 공개한다. 새로운 per-token JSON cache나 converter가 필수인 구조가 아니다. 신뢰한 tensor cache만 읽어야 한다.

native hash는 seed/sequence length/sample budget/blend identifier를 포함하지만 teacher revision/view, tokenizer byte, sample token/mask, reset/packing/backend kernel까지 인증하지 않는다. `cache_id`는 checkpoint와 대조하는 외부 immutable lineage handle이지 stock tar metadata에 내장된 cryptographic digest가 아니다. production에는 검토된 immutable producer manifest와 성공 completion receipt가 여전히 필요하다. single-writer/fresh-directory 가정이며 overwrite race/remote atomic readiness/전체 corpus 용량을 검증하지 않았다.

## Objective와 호환성

선택한 objective는 temperature 1의 **native ghost-tail forward KL + fixed alpha**다. teacher와 student 모두 full vocabulary normalizer를 유지하고, 버린 확률질량을 하나의 bucket에 합치며 native 1e-8 clamp를 쓴다. tail 내부 KL은 복구하지 못한다. `top_k=V`는 bounded tiny parity에만 쓰며 native vocabulary 상한은 131072다.

adapter alpha 기본 **0.5**는 native callable 기본값이며 native CLI의 **1.0**과 구분한다. 현재 online ModelOpt fullKL의 detached `LM/KD` magnitude normalization과 **동등하지 않다**. 기존 CE/KL entropy 반례 source-control 테스트를 보존했다. ModelOpt-compatible cached mode는 추가하지 않았고 online→Core objective 전환을 순수 caching 비교라고 부르면 안 된다.

## 실행한 증거와 남은 gate

정확한 pinned no-GPU CPU image에서 unmodified SDK writer→SDK reader/port reader, port writer→reader→CPU student gradient, loss/count/report, LM-only eval, 다음 iteration reader resume, mismatch/replay/gap/partial-write 거부와 scratch SDK `git apply --check`/apply 및 실제 Bridge factory/dependency import를 실행했다. default/ModelOpt 보존 검사는 source/AST 동등성이지 **전체 기본 training 실행 회귀가 아니다**. CPU model-forward/batch는 주입한 fixture이며 native hybrid model, scheduler/optimizer loop를 실행하지 않았다. warning과 실패한 import 시도도 보존했다.

남은 조건:

1. GPU native model forward/backward/optimizer, 실제 checkpoint save/reload, 성공 optimizer update와 sidecar의 transactional binding. Loss closure 완료만으로 optimizer 성공이라고 할 수 없다.
2. TP/CP/DP/EP/PP/VPP, 동일 MBS/CP layout, fixed GBS/global sample ownership과 DP remap divisibility. 원래 helper는 남았지만 adapter는 non-singleton TP/CP/DP/PP를 거부한다.
3. MTP/recompute/overlap/CUDA graph는 거부한다. packed attention/position/Mamba reset, rerun/skip/restart, spawn은 별도 gate다. cursor 오류로 중단하는 것은 rewind/recovery 구현이 아니다.
4. 정확한 Base teacher revision/view/tokenizer와 tensor 누락 검사, 같은 Core objective의 BF16 score/loss/gradient/update parity. 실제 teacher weight를 load하지 않았다.
5. remote MSC/atomic completion, inherited reader 이상의 corrupt/truncated native payload 강화, 실제 cache 용량/I/O/H2D/전체 update 시간.
6. dump+student first-use GPU-hours, reuse 손익분기와 heldout 품질. 기존 synthetic CPU timing을 production 예측에 쓰지 않는다.

`evidence/storage-estimate.json`의 full-day workload 계산은 **full FP16 values만 약 439.8 TB**이며 ID/overhead는 더해진다. 이전 저장공간 보고로는 전체 full-K dump가 불가능하다. top32 용량도 실측 압축률이 아닌 예시 계산이다. 별도 승인된 matched-objective 비교 없이 default/online을 바꾸지 않는다.

## vLLM은 optional future

이전 adapter/source evidence와 조사 내용은 `evidence/legacy-vllm-feasibility.md`, `.ko.md`에 보존했다. 이 통합을 위해 추가 vLLM 연구/engine 실행을 하지 않았다. 외부 producer는 native format을 출력하고 teacher-output/native-sampler parity를 별도로 증명해야 하지만, Core-first port의 필수 수용 조건은 아니다.
