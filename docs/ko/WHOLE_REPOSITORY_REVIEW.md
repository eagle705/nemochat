# 전체 저장소 리뷰·커밋 검증

Canonical: [영문 범위·실행 증거·gate](../en/WHOLE_REPOSITORY_REVIEW.md).

**현재 상태: 검토된 CPU·소스 트리 범위의 독립 재리뷰 PASS. 로컬 commit 승인됨; staged 전체 내용·최종 packaging gate 확인 후 실행한다.** 사용자가 전체 소스 검증·필요한 수정·로컬 commit을 승인했다. Push·공개·외부 telemetry·upload·유료 GPU 실행은 포함되지 않는다.

## 검토 범위

처음 HEAD는 `b5594a5debc6c090b16a33d7a9ebb4fb647a2d4e`이며 `.gitignore`만 tracked였다. 기존 untracked 소스도 사용자 작업이므로 삭제/reset/stash하지 않고 전부 검토한다.

초기 commit-visible 후보는 207개(103 Python, 70 Markdown)였고 진행 중 rollout 추가 파일에 따라 최종 count가 달라질 수 있다. Data/eval/model/WSD/WSM/Gym/long-context/operations/performance/실험 runner와 Lens/Bridge/browser/rollout을 포함한다.

## 부모 실제 실행

- Main suite **783 passed, 22 skipped, 25 subtests**. 20 browser opt-in은 이전 별도 실제 실행과 구분하며 기존 gate 2개가 남는다.
- Main pip check 통과.
- Ruff fatal/undefined 검사 통과. Full F 검사에서 unused import 3개를 정리하고 optional Megatron import를 source validation 뒤에 하는 이유를 주석으로 보완. 이후 E9/F 통과.
- 수정 후 planning/browser CLI **54 passed**.
- Root script help 21개 모두 exit 0.
- Wheel 실제 build/read-back: Python module 34개, 현재 README metadata와 notices 포함. 문서 최종 freeze 후 다시 build한다.
- 별도 공식 WSM runtime **25 passed**, 실제 CPU synthetic DCP merge 포함. Python `-O` fixture 검증 **PASS**(8 check).
- 실제 cached tiny-HF WSD 4 update: 2+2 resume와 연속 4 update의 model·optimizer/RNG·cursor/token count가 정확히 일치했다. GPU/분산/target-model 품질 검증은 아니다.
- Dense YAML baseline/checkpointing 후보 모두 실제 실행하고 A/B importer 통과. 비교는 `experimental-only`이며 짧은 CPU timing을 GPU speedup으로 해석하지 않는다.
- 초기 AST/credential pattern/파일 크기/symlink 검사에서 보안 flag 없음. 실제 credential은 읽거나 표시하지 않는다. 최종 staged contents에서 다시 검사한다.
- 초기 문서 링크 오류는 아직 작성 중인 rollout README 링크다. 최종 staging 전 해결해야 한다.

## 독립 리뷰·완료 기준

Core와 operations/performance를 나눠 읽기 전용 독립 리뷰를 진행한다. Rollout 새 코드에는 별도 frozen-source 리뷰가 필요하다. 이전 Lens 승인을 새 코드의 승인으로 재사용하지 않는다.

실제 blocker를 수정하고 실패 경로 테스트를 추가한다. 무리한 framework/refactor 대신 필요한 주석·설명과 작은 변경을 우선한다. 모든 writer 종료 후 main/isolated 테스트, CLI/문서/설정검사, 실제 control, packaging을 재실행한다.

Staged 전체 내용과 secret·raw trace·source snapshot·checkpoint/model/data·cache/venv/build 제외를 확인한 뒤 적절한 메시지로 commit하고 Git 상태를 확인한다. Push하지 않는다.

Native hybrid GPU 학습/분산 resume·export/target WSM·Gym 품질/native inference·EP/Nsight loading·mapping·panel/GPU overhead는 별도 미검증 gate로 유지한다. 리뷰 verdict·수정 내용·최종 commit 증거는 이 문서와 NemoChatAgent.md에 이어 기록한다.

## 첫 독립 리뷰 결과 — 수정 필요

Core/operations 첫 리뷰 모두 FAIL: NIAH 결과 덮어쓰기·뒤쪽 fixture 오류 전 요청 손실·Bridge cleanup의 원래 예외 가림을 재현했다. 별도 수정 context가 회귀 테스트와 함께 보완 중이다. Rollout 실패 경로는 별도 리뷰 중이며 아직 commit 승인 아님.

Unused import 3개, CPU 초기화 reseed·CUDA timing 안 scalar 동기화 금지·HF read-back 실패가 remote rollback은 아니라는 설명 주석을 보완했다. 더 강한 hosted revision/key schema/storage 검증은 비차단 follow-up으로 남긴다.

Rollout 세 scenario는 부모가 각각 실제 25-span API 계층을 확인했고 straggler를 실제 Chromium에서 7 operation·25 spans로 렌더링했다. 8개 focused 테스트 통과. 위 CPU/교육용 관측이 export·cleanup 실패 경로 또는 native RL의 승인이라는 뜻은 아니다.


## 최종 검증 snapshot — 2026-10-13T02:23:34Z

Cycle-2 독립 재리뷰 3건 모두 PASS이며 현재 reviewed source 17개 hash 일치를 확인했다. 부모 최종 main **828 passed, 22 skipped, 25 subtests**, Lens/rollout **38 passed, 57 subtests**, plugin **52 passed**, 실제 Chromium/CLI **39 passed**, 별도 공식 WSM **25 passed**와 optimized verifier PASS.

최신 세 rollout은 각각 실제 25-span ID·resource·trajectory 계층·원격 timestamp 검증 통과. 실제 Chromium에서 최신 straggler 7 operation·25 spans와 JS error 없음 확인. `outputs/precommit-whole/final-cycle2-ui/trace.png`는 private ignored artifact다.

NIAH partial 공개 결과는 삭제하지 않고 reservation tombstone을 남겨 retry마다 새 경로를 사용한다. Private staging 소유권 계약·각 cleanup 시도·원래 예외 보존을 검증했다. Bridge는 자기 stage/type 진단만 전달한다. Float64의 JSON integer 표현 차이와 SDK/Jaeger 1–2µs 정밀도 차이만 허용하며 bool/int·손상 timestamp는 거부한다.

최종 문서 freeze 후 wheel/read-back·staged 전체 scan을 통과한 경우 승인된 로컬 feature commit을 실행한다. 실제 commit ID는 실행 후 Git에서 확인하며 문서에 자기참조 hash를 넣지 않는다. Push·외부 upload·online W&B·유료 GPU와 native GPU/Nsight/RL/vLLM/Gym 승인은 포함하지 않는다. 비차단 follow-up과 전체 commands/evidence는 영문 canonical을 참고한다.
