# Continuation03 recovery contract

## English

The failure is a reproducibly corrupt compressed payload, not an established reader race. The pinned-image CPU decoder fails three times on `cp0_dp7__70.tar:70.pt.zst` with identical payload SHA256 `0c47f808a64d4d70b4cf54340a81516a6dec99d48facaea353909ed3fa46c523`; neighboring 68,69,71–73 decode successfully. This does not identify whether damage occurred at initial writing or subsequently in storage. No decoder, installed SDK, original cache or frozen earlier source is modified.

The parent owns the complete 9,144-shard codec audit and supplies the complete corrupt iteration set. `repair_native.py` loads the original pinned frozen teacher with DP8/EP8. For only those global iterations, the original native sampler seeks to `iteration*128` with unchanged full dataset budget `[468736,120,8]`. Every rank checks every actual microbatch digest against original teacher receipts before its no-grad forward. Original native writer hooks preserve k32/fp16/metadata/cache ID. Regenerated shards are independently decoded using the native tensor decoder before completion. All-rank sparse repair does not run an optimizer or restart the old teacher pipeline.

The latest user authorization permits fresh student step0, so `native.py` starts from the original completed BASE19074 HF model with optimizer/RNG/cursor reset. It does not resume checkpoint64 or pretend failed updates65–70 were persisted. The original checkpoint64 and old failed logs are preserved. Student uses unchanged full WSD3662/warm37/decay366 and unchanged full sampler, with a bounded execution stop1143 (599261184 positions). Cache ID remains that of the original full run. Native optimizer, DDP, clip1, alpha0.5 and ghost-tail codec/objective remain unchanged.

`prepare_overlay.py` requires the exact complete parent audit and clean all-rank sparse repair, then builds only a fresh symlink view: original good intervals and regenerated all-rank bad intervals. The parent must explicitly publish `cache-acceptance.json` with status `all_closed_shards_codec_validated`, shards9144, coverage `[0,1142]` before the student supervisor can launch.

Launches are separate fresh user-systemd units with exact Engine identity/readback, cooperative all8 lock, pinned image, read-only original files, and newly writable HOME/Inductor/Triton directories. Compiler artifacts are never copied. `start.py repair --execute` requires parent approval; `start.py student --execute` additionally requires the accepted complete overlay. A launch or restored model is not evidence of positive-LR student learning. Healthy handoff requires real all-rank update>=128 and the synchronous native checkpoint128 transaction.

## 한국어

원인은 rank7 iteration70의 실제 압축 payload 손상으로 재현됩니다. reader/decoder를 우회하거나 수정하지 않으며, 원래 teacher cache와 checkpoint64 및 실패 로그를 보존합니다. 부모의 전체 9,144-shard 검사가 끝나야 손상 iteration 목록을 확정하고, 해당 iteration에만 동일한 frozen teacher/원래 native sampler를 사용해 8개 rank의 cache를 새 경로에 다시 생성합니다. 모든 microbatch의 실제 입력 digest를 기존 teacher 기록과 대조합니다.

사용자의 최신 허가에 따라 student는 원래 BASE19074에서 step0으로 새로 시작하며 optimizer/RNG/cursor를 reset합니다. 전체 sampler와 WSD3662/warm37/decay366은 변경하지 않고, 완료된 teacher prefix1143 전체를 학습하는 실행 상한만 둡니다. 부모의 실제 codec 승인 전에는 student를 시작하지 않습니다. 정상 인계는 실제 모든 rank의 update128 이상 및 native checkpoint128 저장을 확인한 뒤에만 주장합니다.
