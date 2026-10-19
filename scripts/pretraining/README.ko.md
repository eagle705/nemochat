# 순서가 있는 Megatron Bridge pilot 실행

현재 checkout의 `.venv`, 고정 NeMo 이미지, Engine 접근 권한과 준비된 pilot 데이터가 필요합니다. 실행 wrapper는 `results/`의 로컬 검증 helper에 의존하므로, 새 clone만으로 실행되는 독립적인 production trainer는 아닙니다.

```bash
bash scripts/pretraining/1.wandb_preflight.sh
bash scripts/pretraining/2.megatron_bridge_pretrain.sh bridge-my-fresh-run 1800
bash scripts/pretraining/3.verify_run.sh bridge-my-fresh-run
```

1. `.env`의 허용된 항목만 shell 실행 없이 읽고, W&B 온라인 인증과 별도 preflight run을 원격 확인합니다.
2. 실제 Megatron Bridge `pretrain` entrypoint로 GPU 8개에서 indexed 실제 데이터를 학습합니다. WSD W1/S2/D1 총 4 update와 native W&B 온라인 로깅을 사용합니다. **새 run 이름**을 지정해야 하며 기존 checkpoint/result를 덮어쓰지 않습니다. 먼저 GPU 작업 소유자를 확인해 다른 작업과 겹치지 않게 실행하세요.
3. 정확한 학습 컨테이너의 정상 종료와 8개 rank 기록을 확인합니다. 새 manifest는 배타적으로 생성하며 기존 manifest는 변경하지 않고 비교합니다. 전체 payload 해시와 W&B 학습 기록을 검증합니다. 불일치는 실패이며 자동으로 manifest를 고치지 않습니다. 이 단계만으로 실제 native checkpoint 재개까지 검증되는 것은 아닙니다.

검증한 실행은 `bridge-fresh-wsd4-01`, W&B run은 `nvidia/nemochat/bridge-train-9271cf68694e`입니다. 4 update·loss-bearing token 4,096개를 실행했습니다. 새 checkpoint의 SHA와 모든 metadata-indexed DCP record CRC 검사를 통과했지만, 이 checkpoint를 실제 재개하는 검증은 아직 하지 않았습니다. 다른 작업과 겹친 timing을 처리량 benchmark로 해석하면 안 됩니다.

상세 보고서와 parent 재검증 결과: `results/pretrain_followup/bridge-fresh-wsd4-01/`. 기존 손상된 `resumed-02`는 사용하지 않습니다.
