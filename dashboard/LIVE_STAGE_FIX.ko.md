# 학습 단계 반영·타임라인 조작 수정

2026-10-14 KST에 실제 API·브라우저 검증을 완료했습니다. [상세 영문 기록](LIVE_STAGE_FIX.md).

- 누락 원인: 기존config에는 과거 pretrain8runs만 있었습니다. 승인된 Bridge/RL 실행 루트에서 한 단계 깊이만 검사해 명시적 prefix→stage와 지원 marker가 있는 새 run을 자동 등록합니다. 경로/ID/종류 검증, symlink 제외, 파일/실행 상한과 누락 경고를 유지합니다. 매 갱신마다 checkpoint/data를 재귀 탐색하지 않습니다.
- 현재13runs: MID2개/SFT2개/RLVR1개를 추가했습니다. MID warmstart02는4updates/4096tokens, SFT diagnostic03은4updates/16384processed/3025loss-bearing tokens입니다. 이전 실패도 보존합니다.
- RLVR는 rollout4개·sampling512calls·optimizer0updates·scheduler0samples·reward mean0, **Blocked / rollout_only**입니다. exit0를 정책 학습 완료로 표시하지 않습니다. raw rollout/prompt/token내용은 브라우저에 전달하지 않습니다.
- 첫 로딩은 최신 기록을 기본 선택하고, 실행을 바꾸면 profile도 해당stage를 따릅니다. 과거참조 선택은 별도 모드로 유지합니다. 새stage에 GPU캡처가 없으면 과거pretrain 그래프를 대신 표시하지 않습니다.
- 타임라인은 즉시 hover tooltip, 포인터 중심 wheelzoom, dragpan, kernel이름 검색, stream필터, 화면 내 이벤트 수, rawname과 대응API highlight를 제공합니다. 동일캡처의 SSE갱신에는DOM/줌/선택을 유지하며 update3→4에서는실제데이터/geometry를 변경합니다.
- 실제mouse조작에서AllGather필터25events, 휠/드래그에 의한시간창변화,6.5초이후동일canvas/검색/zoom유지, rawAllGather/correlation15483843 hover·click을 확인했습니다. MID/SFT선택의미수집GPUcapture와영어metric표기도확인했습니다.
- 관련254tests+32subtests 통과. 실제API/SSE13runs·데스크톱/모바일 overflow없음·browsererrors없음을 확인했습니다. 원본학습/소스/checkpoint·trainer는변경하지 않았고 GPU/profiler는실행하지 않았습니다.

**남은 경계:** live학습metrics와historicalCUDAcapture는별개입니다. MID/SFT/RLVR의실시간CUDA타임라인을얻으려면해당단계의실제수집자료가필요합니다. SFT로그에는host구간98개가있지만updateID미확인이라GPU커널/iteration타임라인으로가공하지않습니다. 전체GPUproject검증/zero-overhead검증을한것은아닙니다.

검증파일: `results/dashboard_live_fix/5eeb26c5dff3/`.
화면: `results/dashboard-live-sft-fixed.png`, `dashboard-live-rl-fixed.png`, `dashboard-timeline-interactive-fixed.png`.
