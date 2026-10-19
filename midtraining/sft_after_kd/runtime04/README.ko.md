# runtime04 — sealed RAM 전용 native SFT

GPU 학습은 아직 실행하지 않았습니다. CPU fixture/SDK 구성 증거와 실제 학습·save/reload 증거를 구분합니다. v1/runtime02 및 원본 mmap 실패 기록은 보존합니다.

부모는 `README.md`의 handoff 순서대로 새 source-freeze와 CPU 결과를 검토하고, 완전한 producer READY가 나온 뒤 `integrate_snapshot.py --ready /절대경로/READY.sealed.json`을 실행합니다. 이 경로는 pinned image의 GPU 없는 8-process CPU 실제 client 통합 검사이며, 학습을 시작하지 않습니다. 그 실제 receipt까지 새 parent-release에 바인딩한 뒤 이미 승인된 calibration을 `start.sh --calibration --release /절대경로/parent-release.json`로 시작합니다. 사용자에게 다시 승인을 요구하는 단계가 아닙니다.

`input_mode=sealed_scm_rights`만 허용합니다. 모든 train/heldout token, mask, row hash 및 heldout metadata는 seal15/O_RDONLY FD에서 읽습니다. 원본 RAID 데이터 fallback은 없습니다. 실제 shifted inputs/labels/mask/position 및 source row SHA256를 clone한 tensor에서 검증하고, 모든 rank의 MIN 성공 collective 뒤 같은 tensor를 native forward에 전달합니다. 오류 시 모든 rank가 fatal 종료하며 optimizer/LR 진행, 재표본추출, quarantine은 없습니다. 앞선 정상 microbatch forward는 이미 일어났을 수 있지만 해당 accumulated update는 수행하지 않습니다.

원본 C model-only로 optimizer/scheduler/RNG/cursor를 초기화합니다. BF16 seq16384 CP1 EP8 TP1 PP1 GB128 MB1, 고유 1 epoch/34135 native horizon을 유지합니다. calibration16 → native full-state reload/eval → 원본 C 재로딩 main 순서입니다. automatic_main_after_calibration=true는 부모가 명시할 때만 적용됩니다. warm update9..16 전체 rank 실측, finite gradient/update, save/reload 증거가 필수입니다.

2026-10-19 11:30 KST 학습 중지, 12:00 KST save/eval/소유 Engine 정리 정책을 유지합니다. 정오 완료는 보장하지 않습니다. checkpoint SHA/CRC는 명시적 사용자 waiver로 검사하지 않습니다. 자세한 release schema, CPU 증거와 남은 실제 GPU gate는 README.md 및 결과 HANDOFF.json을 참조하십시오.
