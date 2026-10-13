# Inference 트랙 구현 검증

Canonical: [INFERENCE_IMPLEMENTATION_VERIFICATION.md](../en/INFERENCE_IMPLEMENTATION_VERIFICATION.md).

## 완료 범위

독립 트랙의 saved-request analyzer와 single-operator roofline 계산을 구현하고 CPU/synthetic 계약을 검증했다. 실제 engine/GPU 성능 검증은 아니다.

부모 실행 artifact: `/tmp/nemochat-inference-parent-qR1ewG/`.

- 요청 fixture 7개: success 4/error 1/timeout 1/cancelled 1. Success SLO는 met 1/violated 1/unavailable 2로 처리. Synthetic 라벨 유지.
- Roofline fixture는 model estimate/bandwidth 분류이며 exact input SHA256 포함. 실제 B300 수치가 아니다.
- Synthetic opt-in 누락과 output 재사용은 exit 2. 역전 timestamp/chunk timing/nonfinite 입력도 거부.
- Worker 완료 후 부모 전체 suite: **520 passed, 1 skipped, 25 subtests passed**, 19.70s.
- Pip check/compileall/git diff --check 및 문서 링크/fence/공백 검사 통과.
- 독립 read-only 리뷰: blocking security/logic finding 없음.

Nonblocking 후속 제안: 매우 깊은 malformed JSON의 RecursionError를 일반 CLI 입력 오류로 처리, request/config exact-byte provenance hash 추가. 현재 request CLI는 경로만 기록하며 roofline은 hash를 기록한다. 이 제안을 완료 기능으로 보고하지 않는다.

## 미검증

실제 load generator/serving, GPU memory/Nsight/Triton, model quality, 분산 inference, 성능 개선/regression 채택/upstream PR은 미실행이다. Imported timestamp·누락 요청·환경은 독립 인증하지 않는다. 기존 skip을 GPU 검증으로 해석하지 않는다. Commit/push/공개 작업 없음.

## 런타임 경로 migration 참고

위 과거 실행 기록의 경로·timestamp·결과는 원래 증거로 보존합니다. 새 실행은 [저장소 루트 런타임 규약](RUNTIME_LAYOUT.md)을 따르고, 이동된 로컬 산출물은 runtime migration manifest의 old → new 매핑으로 찾습니다. Receipt의 remote path/revision/hash와 inventory 상대 경로를 변경하거나 기존 산출물을 덮어쓰지 않습니다.
