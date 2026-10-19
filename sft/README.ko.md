# sft 단계

SFT는 midtraining과 분리된 최상위 단계입니다. data/configs/training/resume/evaluation/publication/archive로 나눴습니다. 최신 수락1248의 full-state 재개는 resume/runtime, 실패/대체된 runtime02–05와 복구 실험은 archive에 보존합니다. 기존 payload와 실행 증거는 그대로 두며 새 GPU 검증을 주장하지 않습니다.

새 입력/상태/보고서는 datasets/sft, checkpoints/sft, results/sft를 명시적으로 사용하세요. 과거 경로는 증거를 보존하기 위해 유지합니다. [영문 상세](README.md), [경로 이전 명세](../docs/ko/STAGE_LAYOUT.md).
