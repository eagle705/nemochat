# Training systems lab 검증 기록

Latest status: reviewed CPU/offline control scope. Historical pending notes below refer to earlier snapshots; [final whole-source review](../en/WHOLE_REPOSITORY_REVIEW.md) records repaired descriptors, fresh execution and independent approval. Native GPU/distributed/target-quality gates remain unverified.

Canonical: [English verification](../en/TRAINING_SYSTEMS_VERIFICATION.md).

## 범위

새 YAML 실험 계약, 독립 dense Transformer의 실제 full update, profiler/A-B 도구, linear forward/backward 연산 추정이다. 기존 native hybrid·WSD/WSM·NeMo Gym·분산/재개 경로의 구현 완료를 의미하지 않는다.

## 실제 실행

- 변경 전 전체 suite: 520 passed, 1 skipped, 25 subtests passed.
- 부모 artifact: `/tmp/nemochat-systems-parent/`.
- YAML baseline과 checkpointing candidate가 실제 CPU optimizer update를 완료했다.
- CUDA 설정은 exit 3/blocked로 끝났고 CPU로 자동 대체하지 않았다.
- 별도 CPU profiler trace에 data/forward/backward/optimizer label을 확인했다. GPU critical path나 CPU peak memory는 주장하지 않는다.
- 작은 CPU A/B에서 checkpointing 후보가 느려졌으며 experimental-only를 유지했다. GPU 성능이나 통계적 유의성의 증거는 아니다.
- Linear planner는 계산 결과만 출력하고 기존 출력 경로 재사용을 exit 2로 거부했다.

## 통합 결함과 검증 상태

YAML/direct JSON의 중복 검증이 서로 다른 한계를 적용했다. 부모 회귀 테스트 10개가 이를 재현했으며 shared validator와 실패 상태 전달을 수정 중이다. 최종 suite와 독립 리뷰 전까지 이 문서는 승인 기록이 아니다.

AGENTS.md 수정은 보호 파일 승인 timeout으로 차단되어 우회/재시도하지 않았다. README와 NemoChatAgent.md에 새 진입점과 목표를 연결했다. Commit/push/업로드/GPU 임대는 수행하지 않았다.

## 런타임 경로 migration 참고

위 과거 실행 기록의 경로·timestamp·결과는 원래 증거로 보존합니다. 새 실행은 [저장소 루트 런타임 규약](RUNTIME_LAYOUT.md)을 따르고, 이동된 로컬 산출물은 runtime migration manifest의 old → new 매핑으로 찾습니다. Receipt의 remote path/revision/hash와 inventory 상대 경로를 변경하거나 기존 산출물을 덮어쓰지 않습니다.
