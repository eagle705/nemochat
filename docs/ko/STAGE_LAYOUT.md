# 단계별 소스 구조 (v1)

버전 관리되는 단계별 소스 루트는 `pretraining/`, `midtraining/`, `sft/`, `rl/` 네 곳입니다. 런타임 페이로드 루트 `datasets/`와 `checkpoints/`는 Git에서 무시되며 인덱스에 소스 파일이 하나도 없으므로 깨끗한 checkout이나 `git archive`에는 없을 수 있습니다. 현재 로컬 호스트에는 여섯 루트가 모두 있지만 새 checkout에서도 존재한다는 보장은 아닙니다. 별도로 승인된 작업에 필요할 때만 페이로드 디렉터리를 만들고 페이로드나 자리표시자 파일을 강제로 추가하지 마세요.

최상위 `pretraining/`, `midtraining/`, `sft/`, `rl/`가 각 단계 소스를 소유합니다. `datasets/`는 데이터, `checkpoints/`는 가중치/상태 전용입니다. SFT는 더 이상 midtraining 아래에 없습니다. 공통 라이브러리 `src/nemochat`, 공통 scripts/configs/tests/docs, 대시보드/튜토리얼은 공유 구조를 유지합니다.

pretraining은 BASE 실행/설정/출판, midtraining은 data/training/evaluation/publication/configs/docs, SFT는 data/configs/training/resume/evaluation/publication/archive로 분리했습니다. 실패한 runtime02–05와 복구/스냅샷 실험은 `sft/archive`입니다. RL 폴더의 실제 기능은 Gym 결과 가져오기/비교이며, 재사용 가능한 RL 학습기를 구현했다고 주장하지 않습니다.

[영문 상세 매핑](../en/STAGE_LAYOUT.md), [원본 Git blob 기반 이전 명세](../stage-layout-migration-v1.json)를 확인하세요. 원본 소스 tar와 Git 이력, 불변 실행 증거, 기존 데이터/체크포인트는 보존합니다. 소스 경로가 바뀌었으므로 과거 GPU 소스 동결과 동일하거나 재검증되었다고 볼 수 없습니다.

최신 native1248은 `results/midtraining/sft-continue1500-02/main/checkpoint/iter_0001248`에 그대로 있습니다.905 및 기존 HF export도 이동/복사/해시 재검사하지 않았습니다. 새 출력은 명시적으로 `datasets/<stage>`, `checkpoints/<stage>`, `results/<stage>`와 새 이름을 사용해야 합니다. 과거 실행에 결박된 코드는 증거 경로 기본값을 유지하며 새 학습 실행 승인이 아닙니다. 수치 알고리즘/마스크/토큰 ID/예상 데이터 해시는 바꾸지 않습니다.

CPU Transformers4 환경과 native/export5.12.1 환경은 별개입니다. CPU 통과/도움말/소스 아카이브 검사는 GPU나 학습 품질 증명이 아닙니다. 기존 historical opt-in88함수/149케이스와 파일시스템 baseline 실패는 별도로 보고합니다.

Offline-KD preparation/materialization is owned by `midtraining/data/offline_kd`, recipes by `midtraining/configs/offline_kd`, guides by `midtraining/docs/offline_kd`, optional native-cache I/O by `midtraining/cache/directio`. Native/LM-only/recovery implementations remain in the training family.
