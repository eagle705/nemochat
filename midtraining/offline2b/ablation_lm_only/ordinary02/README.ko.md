# 일반 LM-only B02 — B01 대체

최신 사용자 수정에 따라 **B=1.0LM**, 기존 배포 레시피 **C=0.5LM+0.5KD**를 비교한다. 공통 LM 계수를 고정한 KD 항의 독립 인과효과가 아니라 실제 레시피 비교다. Primary Total Loss=LM Loss이며 native routeraux/expert bias는 별도다.

원본 BASE HF에서 fresh0로 시작하며 optimizer/RNG/cursor/scheduler를 초기화한다. B01/C 학습 가중치를 이어받지 않는다. 원본 tokenizer/alias importer, Adam beta2=.999 등 optimizer 설정, seed43, 전체 sampler468736, seq4096/GB128/MB1, BF16 TP1/PP1/CP1/EP8/DP8 및 WSD3662/warm37/decay366을 유지한다.

C에서 실제 accepted된1000 physical batch 순서/128000개 입력 digest를 그대로 고정한다. Physical*128+micro*8+rank로 sample을 선택하며 모든 rank가 forward 전에 검증한다. 새 오류는 증거를 보존하고 전체 실패시킨다. 새 제외/retry/rebase/fallback/teacher cache decode는 없다.

Native Bridge pretrain/forward/masked_next_token_loss/DDP/AdamW/scheduler/save를 사용한다. Checkpoint16 이후64주기/최종1000, 보관3개, optimizer/RNG 포함. Payload SHA/CRC 면제는 유지하지만 입력 검증은 면제되지 않는다. Parent가 나중에 최종 B와 BASE/C의 동일 heldout/benchmark 평가를 수행한다.

기존 B01은 중지/대체됐으며 변경 없이 보존한다. B01의 실제0-update 실패 원인은 NumPy scalar metadata에 clone을 호출한 adapter 오류였다. B02에서는 tensor만 clone하고 metadata는 보존한다. 해당 오류를 재현한 실패 test 및 loss/replay/error/save/native RED→GREEN 로그는 `results/midtraining/offline2b-lmonly-1000-02-preflight/`에 보존한다.

독립 user-systemd service `nemochat-offline2b-lmonly-1000-02.service`가 all8 lock/container/12h watchdog을 소유한다. 결과는 `results/midtraining/offline2b-lmonly-1000-02/`이며 자동 retry/benchmark/publication은 없다.
