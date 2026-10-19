# 완료된 cached prefix의 실제 native student 학습

**현재 상태: 2026-10-18 사용자 요청으로 학습 중단.** 아래 실행·handoff 설명은 과거 설계/실행 기록이며, 현재 실행 중인 서비스나 재개 승인을 뜻하지 않습니다.

최초 64 calls의 결과와 LR/Loss 그래프는 [한국어 보고서](../../../docs/ko/MIDTRAIN_OFFLINE_KD_FIRST64_RESULTS.md)와 [English report](../../../docs/en/MIDTRAIN_OFFLINE_KD_FIRST64_RESULTS.md)에 정리했습니다. 검증된 64행 CSV, 계산 결과 JSON, checkpoint 증거와 해석 제한을 포함하며 이후 재시작·filtered·dynamic 실행을 합산하지 않습니다.

의도적으로 중지한 `offline2b-production1920-01` teacher의 닫힌 shard만 사용합니다. 8개 rank 공통 iteration 0..1142가 대상이며 전체 cache 완료나 teacher 정상 종료로 표시하지 않습니다. Teacher를 재시작하지 않습니다.

- 원래 config/cache ID, seed43, 데이터 혼합비와 전체 GPTDataset 예산 `[468736, 120, 8]`은 그대로 유지합니다. DatasetProvider가 짧아진 segment의 예산 대신 원래 예산을 native builder에 전달합니다. 모든 실제 학습 microbatch의 digest를 teacher 기록과 비교합니다.
- Native WSD 전체 길이3662, warmup37, decay366, LR1e-5/min1e-6을 유지하고 준비된1143까지만 실행합니다. Native Adam/DDP/EP8/gradient clipping/router 동작과 optimizer/RNG 저장을 유지합니다.
- 첫64회 checkpoint의 8-rank 동기 저장 transaction이 확인되면 실제 LR/Loss PNG를 생성합니다. 별도 durable handoff가 자신 소유의 첫 unit/container만 의도적으로 멈추고 checkpoint64의 전체 상태를 새 `continuation/` 출력에서 복원합니다. 64 이후 저장 전 실행이 있었다면 해당 상태는 폐기합니다. 실행 중인 소스는 수정하지 않습니다.
- Continuation의 Validation 주기는250입니다. 기존 전체 Validation 예산120을 바꾸지 않으면서64주기 평가로 인한 iterator 고갈을 피합니다. 최초 checkpoint64는 continuation의 checkpoint 정리와 분리하여 보존합니다.
- 사용자 면제에 따라 model/optimizer payload SHA 감사는 하지 않습니다(`not_checked_explicit_user_waiver`). 실제 native 복원/후속 update가 확인되어야 operational resume 성공으로 볼 수 있습니다.

결과: `results/midtraining/offline2b-prefixtrain-01/`. 실제 rank별 update 로그와 systemd/container 소유권·시작·상태 receipt를 보존합니다. 그래프는 실제 optimizer에 사용한 LR, raw/trailing-8 mean LM/KD/primary Total을 구분합니다. 첫 LR=0 호출은 positive-LR update와 별도로 집계합니다. Router auxiliary는 별도이며 미래 Loss를 생성하지 않습니다.

첫 continuation은 model/optimizer 복원 전에 실패했습니다. 복사된 Inductor artifact의 이전 scratch 절대경로 때문에 Triton autotune이 읽기 전용 위치에 쓰기를 시도한 오류입니다. 실패한 출력/소스/container receipt는 보존했습니다. 새 `continuation02/` 소스와 `continuation-02/` 출력은 빈 로컬 Inductor cache와 새 HOME/Triton cache를 사용합니다. 생성된 compiler 코드나 SDK는 수정하지 않았고 최초 checkpoint64도 그대로입니다. 현재 recovery unit은 `nemochat-offline2b-prefixtrain-01-continuation02.service`입니다. 자동 재시도 정책이 아닌 명시적인 소스 수정 후 새 generation입니다.

실행 unit과 그래프 재생성 명령은 영어 README를 참고하십시오. 기존 teacher 출력, 기존14개 동결 소스와 설치된 SDK는 수정하지 않습니다.
