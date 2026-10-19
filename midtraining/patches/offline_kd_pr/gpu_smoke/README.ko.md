# 실제 cached-KD GPU smoke

이 경로는 production 설정이 아니라 작은 실제 데이터에 대한 operational overfit 실험이다. `control.py`의 기존 Base teacher와 BASE-19074 student를 그대로 로드한다. checkpoint payload 해시 검사를 하지 않는 사용자 waiver를 유지하며, 다운로드·설치 SDK 수정·main 자동 실행·random model 대체는 하지 않는다.

## 고정 조건

- cooperative `gpu-all8.lock`을 잡고 GPU 점유를 다시 확인한 뒤 물리 GPU0만 사용한다.
- 기존 pinned NeMo image, network 없음, UID1001/GID985. 저장소와 원본 모델은 read-only이며 task output과 소유 scratch만 쓸 수 있다.
- EP/TP/PP/CP/DP1, BF16, sequence512, MB1/GB1. 네 실제 데이터 source에서 train window를 각각2개 선택하여 동일한8개 순서를4회 반복한다. 별도 producer-heldout window4개는 학습하지 않는다.
- native Bridge optimizer32회, LR1e-5, warmup0, native clip1.0. Loss를 고르기 위한 추가 epoch나 수제 학습 loop는 없다.
- teacher는 plain AutoBridge provider로 로드하고 DDP/optimizer 없이 freeze/dropout0/no_grad 조건에서 native cache를 쓴다. teacher process가 종료된 뒤 student를 로드한다.
- student는 scratch에만 적용한 Bridge cached forward factory와 실제 native writer/codec/reader, `pretrain`, optimizer, scheduler, checkpoint를 사용한다.

보고하는 primary Total Loss는 `.5*LM + .5*top32 ghost-tail KL`, temperature1이다. ModelOpt의 detached magnitude balancing이나 full-vocabulary KL과 다르다. 기존 BASE의 native router auxiliary 정책(`seq_aux_loss`, coefficient1e-4, expert bias)은 유지하며, 이 auxiliary term은 보고되는 primary Total Loss scalar에 포함되지 않는다.

고정 train/heldout의 전후 평가는 동일 입력·mask에서 `eval/no_grad`로 계산한 **LM-only** 평가다. heldout cached-KD 검증이라고 부르면 안 된다. epoch별 LM/KD/Total Loss 평균은 같은 window 순서끼리 비교한다.

## 실제 runtime 보정

`optimizer_compat.py`는 native TE Adam kernel의 tensor list를 작은 launch로 나눌 뿐 Adam이나 학습 loop를 대체하지 않는다. pinned TE의 고정 tensor-handle pool이 EP1 student의 큰 parameter group을 한 번에 처리하지 못하는 문제에 대한 task-local 보정이다. 모델 로드 전에 실제 CUDA에서 기존 native Adam과 parameter/moment parity를 비교하고6000개 scalar parameter update를 확인한다. 이 작은 fixture optimizer 호출은 student32회와 별도다.

원본241GB BF16 teacher는 처음 allocator fragmentation OOM을 만났지만 `PYTORCH_ALLOC_CONF=expandable_segments:True`로 모델·정밀도·topology를 바꾸지 않고 실제 로드되었다. 실패 로그와 receipt는 `failures/`에 보존한다.

## 결과 읽기

`results/midtraining/offline-kd-smoke-01/`의 `teacher-complete.json`, `updates.jsonl`, `initial-fixed-lm.json`, `final-fixed-lm.json`, `checkpoint-save.json`, `checkpoint-cache-binding.json`, `reload-restored.json`, `reload-fixed-lm.json`, `reload-complete.json`을 확인한다. 정확한 Engine ID·mount·GPU scope·종료 코드는 `*-engine-*.json`에 있다. `supervisor-complete.json`과 각 실제 Engine의 clean exit까지 확인하기 전에는 완료라고 하지 않는다.

Native saver는 teacher CE 계산을 생략한다. 초기 raw log의 `teacher_lm_loss=0`은 실제 teacher Loss가 아니라 writer placeholder이며 `teacher-output-correction.json`에서 명시한다. Student LM Loss 및 고정 LM 평가는 이 placeholder가 아니다.

작업 소유 user-systemd supervisor/deadline guard는 subagent가 끝나도 유지된다. deadline은 정확한 ID·소유 label·GPU0 scope가 맞는 task container만 중지한다. 고정 output 경로를 덮어쓰며 다시 실행하면 안 된다. `--reuse-teacher`는 이미 성공한 동일 cache를 보존하고 실패한 student만 원본 BASE에서 재시작하는 제한된 복구 기능이다. 새 실험에는 별도 검토된 identity/output이 필요하다.

로컬 control: `.venv/bin/python -m pytest -q tests/test_offline_kd_gpu_smoke.py`. 이 테스트의 성공만으로 GPU 학습을 증명하지 않으며 위 실제 native artifact를 함께 확인해야 한다.
