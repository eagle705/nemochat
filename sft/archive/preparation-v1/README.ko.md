# C LM+KD1000 → 정오 종료 SFT CPU 인계 v1

[English](README.md)

## 현재 결과

**이 준비 작업의 GPU optimizer update는 0이며 calibration/SFT를 실행하지 않았다.**
종료 시각은 **2026년10월19일 12:00:00 KST (+09:00)** 로 고정한다.
10월20일이 아니다. 실제 native calibration 전에는 정오까지 저장/수출 완료를 보장하지 않는다.

부모가 3모델 GPU 평가 종료를 알려준 뒤 실제 final marker와 모든 보고서를 한 번
독립 검증했다. `benchmark-acceptance.json`은 marker/report/source SHA와 실제 Engine
Exit0, inactive 서비스 상태를 기록한다. `benchmark_gate.py`는 생산자의 GPU 전용
검증기로 모델별400문항/1600선택지,64개 heldout window/65536 targets, 동일 ID/backend,
각 문항 원본 receipt와 PNG2개를 확인한다. marker는 flock 해제 전에 기록되므로
서비스 종료가 필수이며, **미래 launcher는 gpu-all8.lock 재획득 후 live idle도 확인해야 한다**.

## 실제 데이터 준비 상태: 파일은 있지만 launch는 BLOCKED

기존 승인 full-03 데이터 위치:
`datasets/nemotron_sft/nemochat_baby/prepared-full-03/data`

| Split | 실제 NPY row | 새 전체 mask 합계 |4096 이후 targets|
|---|---:|---:|---:|
|Train|4,369,280|3,219,012,753|898,456,733|
|Heldout|45,090|33,153,239|9,258,249|

모든 저장 mask를 새로 스캔했고 uint8/binary/manifest 합계 일치를 확인했다.
uint32 shifted token 배열 폭16385, mask 폭16384도 확인했다.
GB128에서 반복 없는 유한1epoch 용량은 **34,135 updates** 다.
`cpu-readiness.json`에 실제 파일 크기, mask census, heldout-mask SHA 일치를 기록했다.
이는 전체400GB 이상 파일의 현재 SHA/token 내용 검증 완료를 뜻하지 않는다.

과거 CPU 생산자 자체는 Exit0 및 counted/source-mask audit 통과였지만,
최종 publisher에서 `stable prepared file byte/hash gate failed`가 발생했다.
따라서 **completion-ready.json은 여전히 없다**. 오류는 파일명을 기록하지 않았으므로
특정 파일 손상이라고 추측하지 않는다. 기대 SHA를 바꾸거나 manifest의 passed만으로
release하지 말고, immutable 기대값에 대해 안정된 전체 데이터 검증을 완료해야 한다.
이번 mask scan이 호스트 read-path/스토리지 문제를 고쳤다는 주장도 하지 않는다.

기존 승인 recipe는16K CP1/EP8 BF16/MB1/GB128이며 Agentic v4, IF/Chat v4, BBQ를 쓴다.
Long Transduction 전체는 train/heldout 모두 제외하고 RL seed는 gold label 없이 보존한다.
기존14 prompt-group 예약, KMMLU34871문항 배제, structured tools와 완전한 reference,
공식 전체-render selected-assistant mask를 유지한다. IFC author mask는 전체 message에
정렬된다. 새로운 다운로드, 가짜 assistant 정답, 무단 epoch 반복은 하지 않는다.

## Source / tokenizer

유일한 source는 `checkpoints/exports/nemochat_mid_super35_distill_hf_bf16_01`이며,
C native `offline2b-prefixtrain-1000-01/checkpoint/iter_0001000`의 model-only lineage다.
B LM-only, 원래 BASE, 이전 SFT391, calibration 학습 weight는 main source가 아니다.
기존 `student_embedding_import.py`의 pinned native mapping/전체 key coverage 및
실제 learned-embedding transform 검증을 사용한다. embedding row 재배열은 없다.
SFT optimizer/scheduler/RNG/cursor는0부터 새로 시작하며 teacher/KD objective는 SFT에 없다.

Vocab131072/EOD11을 유지한다. migration manifest의18–29 alias 변경 승인은 존재하지만
과거 corpus scan은 새로운 C corpus를 포괄하지 않는다. **C corpus 및 현재 SFT 입력에
대한 예약 ID 감사가 남아 있다.** 예전 ClimbMix receipt로 새 의미 정렬을 인증하지 않는다.
공식 pinned tokenizer/template, trust_remote_code=False를 유지한다.
Checkpoint SHA/CRC/direct forensic 감사는 사용자 면제로 실행하지 않았다.
단, 실제 native restore/유한 양의 gradient/update/resumable save 검증은 여전히 필요하다.

## 생성한 코드와 실제 검증

- `native_sft_v1.py`: 검증된 native Bridge SFT의 task-local snapshot. C HF import/reset,
  source1000 receipt, 소비 row 범위 coverage, loss logging clone, absolute-stop/minimum LR
  제어를 추가했다. shared source/SDK는 바꾸지 않았다.
- `recipe-v1.yaml`:16K 승인 recipe. update/time budget은 새 C16K GB128 calibration 후 고정한다.
- `config_probe.py`: 실제 pinned image에서 SDK ConfigContainer 생성 성공.
  `config-probe-03/native-config.json`, Engine03 Exit0, DeviceRequests=[], weight load0,
  optimizer0이다. libcuda 미장착/abstract-provider 실패는 이전 attempt로 보존했다.
  성공 attempt는 driver 공유라이브러리만 읽기 전용 mount했고 GPU device는 요청하지 않았다.
- `controls.py`: 고정 deadline/유한 용량 budget/3모델 score gate/실제 kernel pidfd wait.
- `controller.py`와 `start.sh`: **아직 실행 예약이 아닌 inert preview**.
  --start는 stable-data proof가 없으면 거부한다. 실제 service/owned launcher 연결과
  부모 source-freeze 검토를 완료해야 한다. preview의 Restart=no이며 기존 service는 손대지 않았다.

Budget은 새 native warm timing과 남은 시간, 유한1epoch 용량의 최솟값이다.
최소1800초는 validation/save/선택적 export에 남긴다. early-duration stop에도 full sampler와
heldout budget을 보존하고 마지막 optimizer 호출의 실제 LR/save/clean exit를 확인해야 한다.
새 absolute-stop/final-save 경로는 CPU control 검증만 했고 **GPU 실행은 미검증** 이다.
최종 optimizer/RNG 포함 native resumable checkpoint가 필수이며 HF export는 선택사항이다.
이 준비 작업은 새 HF push를 수행하거나 승인 범위를 확대하지 않았다.

## 부모에게 남은 순서

1. full-03 stable-file 실패를 해결/재검증하여 실제 데이터 completion을 승인한다.
2. 새 C/SFT 예약 ID 감사와 tokenizer lineage를 고정한다.
3. preview→실제 launcher/service, source allowlist, 소유 container 한정 cleanup을 검토한다.
4. GPUlock/idle 확인 후 C restore/15update 및10개 이상 warm timing/save/heldout/Exit0를 검증한다.
5. 실제 budget을 고정하고 main은 C에서 다시 model-only 시작한다. 정오 이전 save/eval buffer를 둔다.

새 benchmark accuracy threshold를 만들지 않았다. root ledger, 기존 데이터/실험, Hub, git commit은 변경하지 않았다.
