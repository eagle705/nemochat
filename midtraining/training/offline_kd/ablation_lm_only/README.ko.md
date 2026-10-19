# 계수 일치 LM-only 대조군 B

B는 **0.5 LM**, 기존 C는 **0.5 LM + 0.5 ghost-tail KD**다. 일반적인 1.0 LM 레시피가 아니다. Router auxiliary loss / expert bias / clipping / regularization은 그대로 유지하며 primary Total Loss와 구분한다.

완료된 C의 8개 rank에서 1000개 accepted physical batch 순서와 실제 입력 digest 128000개를 고정한다. Teacher-era digest와도 일치해야 한다. Seed43, native 전체 sampler468736, seq4096, GB128/MB1, TP1/PP1/CP1/EP8/DP8 및 WSD3662/warm37/decay366은 유지한다. Sample index는 physical*128 + micro*8 + rank다. 새로운 읽기/digest 오류는 전체 rank에서 forward 전에 실패시킨다. 임의 제외, retry, fallback, digest 변경은 없다.

원본 BASE HF에서 model-only fresh0로 시작한다. Optimizer/RNG/cursor/scheduler를 초기화하며 학습된 C 가중치를 사용하지 않는다. Native Bridge pretrain/forward/masked_next_token_loss와 AdamW/DDP/save를 사용하고 backward LM 계수만0.5다. 표시 LM Loss는 계수 없는 값이며 Total Loss는0.5LM다. Checkpoint는16,64주기,최종1000, 보관3개이며 optimizer/RNG를 포함한다. Payload SHA/CRC는 사용자 면제 상태이며 입력 검증은 면제되지 않는다.

독립 user-systemd service `nemochat-offline2b-lmonly-1000-01.service`가 all8 lock과 소유 container를 관리한다. Pinned image, network none, repository read-only, 새 compiler cache와12h watchdog을 사용하며 자동 재시작은 없다. 결과는 `results/midtraining/offline2b-lmonly-1000-01/`이다. 최종 B checkpoint 평가/공개는 parent 소유이며 이 실행에서는 benchmark/teacher/cache 생성 또는 decoding을 하지 않는다.
