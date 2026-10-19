# Native SFT runtime02 — 부모 검토 전 실행 금지

이 폴더와 `results/midtraining/sft-after-kd-runtime02/`만 새로 작성했다.
기존 v1 코드·recipe·CPU 증거와 완료된 GPU benchmark는 변경하거나 재실행하지 않는다.
준비 단계에서 실제 GPU 학습은 실행하지 않았다. 상세 release JSON 계약은 영문 README를 따른다.

- 기본 `start.sh` / `--preview calibration|main`은 **INERT**다.
- DATA worker의 실제 `READY.json`과 7개 파일의 기존 expected/actual SHA, audit/provenance,
  fresh C 및 새 SFT reserved-ID 검사, buffered/mmap 읽기 허용 여부를 부모가 검토해야 한다.
  `direct PASS / buffered FAIL`은 이 mmap reader의 실행 허가가 아니다.
- 부모가 source-freeze SHA와 실제 READY SHA를 binding한 release를 작성한 뒤에만
  `start.sh --calibration --release /ABSOLUTE/RELEASE.json`을 실행한다.
  `automatic_main_after_calibration=true`이면 같은 durable user-systemd supervisor가
  calibration16 → native full-state reload/eval → **원래 C부터 새 main**을 순차 실행한다.
  false이면 calibration/reload 후 종료하고 `start.sh --run main --release ...`로 따로 실행한다.
- C LM+KD1000만 model-only로 읽고 optimizer/scheduler/RNG/cursor를 초기화한다.
  calibration에서 학습된 checkpoint는 operational reload 검사에만 사용하며 main에 연결하지 않는다.
- BF16, EP8/TP1/PP1/CP1, seq16384, GB128/MB1, LR1e-5/min1e-6, WD.01, seed42다.
  native train_iters/sampler/scheduler/eval allocation은 one epoch **34135 calls**로 유지한다.
  native exit_interval만 calibration16 또는 실제 측정 기반 main cap으로 제한한다. cycling은 없다.
- native Bridge pretrain/DDP/AdamW/masked Loss/full-state checkpoint를 재사용한다.
  기존 version-gated embedding adapter는 그대로이며, 추가 observer가 모든 native destination의
  실제 mapping/transform을 요구한다. 누락된 weight의 random fallback이나 SDK 파일 수정은 없다.
- calibration의 all-rank-max update9..16 timing으로 main budget을 계산한다.
  checkpoint/eval overhead15%와 fresh import300s는 **추정 가정**이며 완료 보장이 아니다.
- 절대 deadline은 **2026-10-19 12:00 KST**, training stop policy는 **11:30 KST**다.
  final evaluation/save에1800s를 남기고 별도 supervisor/Engine watchdog을 둔다.
  장시간 step/save 등으로 deadline을 지키지 못하면 실패/부분 저장 상태를 그대로 보고한다.
- save16, 이후 periodic250, retention3, native final full-state save를 요구한다.
  checkpoint payload SHA/CRC waiver는 유지하지만 데이터 검증까지 면제하지 않는다.
- 공유 all8 lock 및 매 job 직전 live idle 검사, exact MainPID/UID/argv/Engine ID/name/label,
  ExecStopPost의 해당 generation cleanup만 허용한다. 다운로드, publish, online W&B는 없다.

CPU fixture와 실제 pidfd/CLI 테스트, no-GPU/no-weight SDK ConfigContainer 구성 성공은
GPU 복원·실제 update·save/reload 또는 성능의 증거가 아니다. 최종 HANDOFF에서 구분한다.
