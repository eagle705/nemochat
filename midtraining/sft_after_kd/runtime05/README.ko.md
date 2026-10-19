# runtime05 — 짧은 private TMPDIR checkpoint 수정

이번 작업은 **CPU 준비만** 수행합니다. GPU service/learner를 시작하지 않으며, 새 학습 시도는 사용자 재승인과 부모의 명시적 release 이후에만 가능합니다. 기본 start.sh는 inert이고 template은 NOT_RELEASED, automatic_main_after_calibration=false입니다.

runtime04의 16회 positive-LR update는 실제 실행됐지만 첫 save가 SyncManager/AF_UNIX 경로 길이 오류로 실패해 저장된 checkpoint가 없습니다. 이전 update를 runtime05 곡선·카운터에 합치지 않습니다. runtime04와 producer/READY/holder는 수정하지 않습니다.

모든 새 container의 Python 실행 전 환경을 TMPDIR=TMP=TEMP=/nct로 지정합니다. /nct는 container별 새 tmpfs(rw,exec,nosuid,nodev,mode=1777,size=8g)이며 compiled shared object 실행을 막는 noexec를 쓰지 않습니다. host 전역 /tmp를 사용하지 않습니다. HOME/Inductor/Triton/CUDA cache는 새 generation/stage 경로이며 compiled cache를 복사하지 않습니다.

실제 pinned CPU torch multiprocessing Manager.Queue의 put/get/proxy close/shutdown, spawn 자식 환경, 실제 UNIX socket 경로 byte 길이와 긴 경로 negative control을 확인합니다. /nct에서 C executable과 shared object를 빌드·실행합니다. Gloo + MCore ShardedTensor + 실제 TorchDistSaveShardedStrategy/backend torch_dist로 작은 CPU fixture tensor를 native save/load합니다. 이는 storage integration 증거이며 **전체 모델/optimizer save/reload 증거가 아닙니다**. 기존 모델 weight payload의 SHA/CRC/감사는 수행하지 않습니다.

actual post-install_runtime sourceguard 및 pinned SDK 4개 source check, 5개 negative control, 8-rank fixture와 기존 READY의 real public FD client 통합도 각각 검증합니다. holder unit과 peer prefix는 그대로 유지합니다. 모든 결과/정확한 test count/경로는 HANDOFF.json을 참조하십시오.

실제 storage attempt03은 Queue/spawn/socket 길이/compiled executable/shared object 검증에 성공했으나, pinned MCore filesystem_async.py:248 preload_tensors의 무조건 torch.cuda.synchronize() 호출로 RuntimeError: No CUDA GPUs are available에 막혔습니다. **tiny save 미완료, load 미실행**이며 CUDA mock이나 SDK 수정은 하지 않습니다. 실패 fixture·Engine·log를 보존합니다. attempt01의 spawned AutoProxy close 처리와 attempt02의 Docker tmpfs 기본 noexec 문제는 각각 실제 IPC 연결 close와 명시적 exec 옵션으로 수정했습니다. 부모는 별도 사용자 승인 후 release에서 cpu_native_storage_blocker_acknowledged=true를 명시해야 하며, 실제 GPU save16/reload gate는 그대로 필수입니다.

원본 C model-only fresh optimizer/scheduler/RNG/cursor, BF16 seq16384 CP1 EP8 TP1 PP1 GB128 MB1, 고유 1 epoch/34135 native horizon, FD-only row/dtype/label/mask/position/all-rank MIN, embedding/destination coverage guard를 보존합니다. 2026-10-19 11:30 KST 학습 중지, 12:00 KST save/eval/정리 마감도 같습니다.

**재승인 이후에만** 부모가 정확한 새 source/READY/producer canonical files digest/live holder/CPU proof들을 parent-release.json에 바인딩하고 다음 명령을 한 번 실행할 수 있습니다:

```bash
bash midtraining/sft_after_kd/runtime05/start.sh --calibration --release /raid/yoon/nemochat/results/midtraining/sft-after-kd-runtime05/parent-release.json
```

실제 GPU calibration16의 finite update, committed full-state save16와 operational native reload/eval이 성공한 뒤에만 원본 C를 새로 import한 main을 허용합니다. template 정책 false에서는 main도 별도 명시적 시작이 필요합니다. 자세한 절차와 acceptance는 README.md에 있습니다.
