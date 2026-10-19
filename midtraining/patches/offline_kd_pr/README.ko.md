# Megatron-LM 로직 중심의 Bridge cached KD 통합

**격리된 opt-in patch이며 현재 학습에는 적용하지 않았다.** 구조는 Megatron-LM writer → native tar/zstd storage → native reader → Bridge forward/loss closure다. vLLM-first나 새로운 JSON production cache가 아니다.

[English](README.md) · [설계와 gate](FEASIBILITY.ko.md) · [PR 초안](PR_DRAFT.ko.md)

## 실제 구현 파일

- `megatron-bridge-0.6.1-cached-kd.patch`: installed Bridge **0.6.1**, base **`2a8fcdd05181ffcef261192f71ff5910d0005bf2`** 대상 diff. `training/cached_kd/`와 `gpt_step.py`의 명시적 factory만 추가하며 기존 default/online ModelOpt 코드는 바꾸지 않는다.
- `bridge_cached_kd/{logits_saver,utils_logits,cached_logits_loss}.py`: 실제 installed Core 코드의 substantive port다. global-normalized top-K, uint16+17th-bit index, sentinel, `_meta.json`, tar/zstd/torch payload, storage abstraction, reader/prefetch, ghost-tail KL을 유지한다. AST stub이 아니다.
- `runtime.py`: dataset/iteration/microbatch 의존성을 명시적으로 주입한다. `megatron.training` global args를 만들거나 변경하지 않는다. Linux fork worker는 dataset identity를 상속하며 spawn은 gate다.
- `integration.py`: `CachedKDConfig`, backward/optimizer 없는 teacher-only `cache_forward`, student main-head hook, 해당 forward의 differentiable logits를 묶는 deferred loss, sum/token-count 반환, fixed alpha, LM-only eval, hook cleanup, cursor와 checkpoint binding을 구현한다.
- `build_patch.py`, `evidence/integration-base.json`, `evidence/bridge-installed/`: 재생성 가능한 diff, base/image/source hash, 실제 installed seam 원문이다.
- `tests/test_offline_kd_pr_bridge.py`: 신규 통합/SDK 테스트. 기존 `tests/test_offline_kd_pr.py`는 그대로 보존했다.

## Objective는 의도적으로 Megatron-LM-compatible

`objective='megatron-lm-ghost-tail-fixed-alpha'`, **temperature=1**, 명시적 alpha(기본 **0.5**; native callable 기본값이며 CLI의 1.0과 구분):

`total = (1-alpha)*masked_LM_sum + alpha*masked_KL_with_ghost_tail_sum`

`(total, num_tokens, reports)`를 반환하고 backward/normalization/optimizer/LR scheduler는 Bridge가 담당한다. native tail clamp는 1e-8이다. K<V이면 **full KL가 아니며**, conditional top-K나 **현재 ModelOpt detached magnitude balancer와도 다르다**. ModelOpt-compatible cached mode는 추가하지 않았다. 기존 online 경로는 분리·유지하며 cached adapter에 ModelOpt `DistillationModel`을 넣으면 거부한다.

## 저장소 root의 실제 실행 명령

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_offline_kd_pr.py tests/test_offline_kd_pr_bridge.py
python3 midtraining/patches/offline_kd_pr/build_patch.py
python3 midtraining/patches/offline_kd_pr/run_cpu_sdk.py --label review-fresh-01 --apply-patch --test tests/test_offline_kd_pr.py tests/test_offline_kd_pr_bridge.py
```

매번 새 label을 사용한다. 기존 pinned image, UID1001/GID985, `runc`, network none, GPU DeviceRequests=[], repository read-only와 새 owned `$TMPDIR` scratch만 RW로 사용한다. **정지된** 같은 image container `2f1750f9d121488f45a55d1ae66b8390b6ed758e2125e66747c010fe97bc2e92`의 SDK archive를 읽는다. image `/opt` ancestor 권한 때문에 정확한 원문을 scratch에 복사한다. TE의 eager dynamic link에는 **실제 host `libcuda.so.1` 복사본**을 사용했지만 GPU device는 노출하지 않았다. fake library/import stub/download/installed SDK 수정은 없다. patch는 scratch SDK에만 적용하며 Engine ID/scope/exit와 warning/실패를 `evidence/<label>.{json,txt}`에 저장한다.

실제 unmodified SDK writer → unmodified SDK reader 및 port reader, port writer → student Loss/backward, LM-only eval, 명시적 cursor resume, iteration gap와 GBS/MBS mismatch 거부, partial-write 거부, staged Bridge factory import를 검증했다. SDK 테스트는 CLI getter에만 **메모리 내 명시적 dependency injection**을 사용하며 module 대체나 Core global args 변경은 없다. 작은 CPU teacher fixture의 native 직렬화 검증이지 **실제 Nemotron teacher/model 학습 증거가 아니다**. 최종 count/receipt는 `evidence/integration-acceptance.json`에 있다.

## API와 미구현 gate

검토한 SDK copy에 적용한 뒤 `megatron.bridge.training.cached_kd.integration.CachedKDConfig`와 `gpt_step.make_cached_kd_forward_step(config, runtime_factory, resume_binding=...)`를 사용한다. 반환 객체는 `with`로 관리하며 Bridge의 `forward_step_func`로 전달한다. factory는 실제 `_forward_step_common`을 연결하지만 CPU end-to-end 테스트의 model-forward/batch seam은 작은 fixture로 주입했다. `ConfigContainer`/CLI에 없는 필드를 있다고 설명하지 않는다. 별도 dataclass와 frozen runtime identity를 명시적으로 저장해야 한다.

native saver hook은 `model.training=True`를 요구한다. `cache_forward`는 eval mode를 거부하며 no_grad로 실행한다. 실제 producer는 dropout/stochastic-layer 정책을 명시적으로 고정하고 의도한 frozen teacher와의 동등성을 확인해야 한다. no_grad가 eval 동작을 뜻하지는 않는다.

`runtime_factory(state)`는 실제 native sampler의 seed, sequence length, 총 sample budget, blend/blend_per_split, 고정 GBS/MBS, `iteration=lambda: state.train_state.step`, native microbatch count를 담은 `CoreRuntime`을 반환해야 한다. nonzero 시작은 **완료된 cache iteration**의 `checkpoint_binding()`과 일치해야 한다. 단순 Loss 계산 이후가 아니라 성공한 native optimizer checkpoint와 함께 저장해야 한다. **Bridge checkpoint sidecar의 자동 transactional save/reload는 미구현**이다. `cache_id`는 caller가 제공하는 immutable lineage handle이지 teacher/cache byte의 암호학적 인증이 아니다. native dataset hash도 teacher/tokenizer/packing 전체를 인증하지 않으므로 production producer manifest 검토가 남는다.

현재 adapter는 **분산 TP/CP/DP/PP**, VPP/MTP, recomputation, schedule-plan overlap, CUDA graph를 명시적으로 거부한다. 원래 Core의 TP/DP remapping 코드는 보존했지만 adapter 검증을 했다는 뜻은 아니다. GPU native model forward/backward/optimizer, 실제 checkpoint reload, packed/CP/Mamba reset, rerun/skip recovery, remote MSC publication, 정확한 Base teacher load/parity, BF16 tolerance, 전체 성능·품질은 미검증이다. 부모의 multi-GPU main에 바로 활성화할 수 없다.

기존 `cache_contract.py`, `source_controls.py`, `proof.py` 및 JSON/CPU evidence는 **fixture/reference**로만 남긴다. vLLM adapter/evidence와 과거 설명은 `evidence/legacy-vllm-feasibility*.md`에 optional future 자료로 보존했으며 acceptance 필수 조건이 아니다. 기존 용량 계산의 full FP16 values만 약 **439.8 TB**이므로 전체 full-K dump는 불가하고 별도 승인된 tiny bounded 비교만 허용한다.

active recovery/main source, corpus/controller/ledger, 실행 중 container, GPU job, commit/remote PR은 변경하지 않았다.
