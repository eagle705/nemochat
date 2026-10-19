# rl 단계

현재 버전 관리 소스에는 운영 가능한 RL 학습기가 없습니다. evaluation/gym_eval.py는 공식 저장 결과 가져오기/비교 도구이며 학습이나 모델 평가를 실행하지 않습니다. configs는 미측정 명세이며 RL seed를 SFT 라벨로 만들면 안 됩니다.

새 입력/상태/보고서는 datasets/rl, checkpoints/rl, results/rl를 명시적으로 사용하세요. 과거 경로는 증거를 보존하기 위해 유지합니다. [영문 상세](README.md), [경로 이전 명세](../docs/ko/STAGE_LAYOUT.md).
