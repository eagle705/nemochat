# pretraining 단계

LM-only BASE 사전학습 단계입니다. entrypoints는 기존 native 실행 래퍼, configs는 WSM/평가 설정, publication은 BASE 출판 도구입니다. 기존 래퍼는 results 아래 과거 실행 producer에 의존하며 새 범용 학습기가 아닙니다.

새 입력/상태/보고서는 datasets/pretraining, checkpoints/pretraining, results/pretraining를 명시적으로 사용하세요. 과거 경로는 증거를 보존하기 위해 유지합니다. [영문 상세](README.md), [경로 이전 명세](../docs/ko/STAGE_LAYOUT.md).
