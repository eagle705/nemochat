# sealed_snapshot03 — CPU 전용 full03 불변 RAM 전송

원본 manifest가 지정한23개 파일(SFT7개, C bin8개, C idx8개)의 바이트를 변경하지 않고 RAM으로 캡처합니다. 이어서 행별 SHA256(raw32) 카탈로그2개를 파생합니다. 원본 수정, 다운로드, GPU 할당, 캐시 제거, 모델 가중치 읽기 및 기존 mmap 입력 승인은 하지 않습니다.

## 실행과 수명

`/raid/yoon/nemochat`에서 기존 `.venv/bin/python`을 사용합니다.

- `-m unittest discover -s midtraining/sft_after_kd/sealed_snapshot03 -p 'test_*.py' -v`
- `-m midtraining.sft_after_kd.sealed_snapshot03.holder fixture`: 작은 실제 pinned Docker SCM_RIGHTS 시험만 수행
- `-m midtraining.sft_after_kd.sealed_snapshot03.driver start`: 승인된 단일 세대 user-systemd 서비스 시작
- `-m midtraining.sft_after_kd.sealed_snapshot03.driver status`: 실제 증거 확인

서비스명은 `nemochat-sft-sealed03-holder.service`입니다. 에이전트 세션과 독립적으로 동작하며 MemoryMax768G / MemoryHigh640G / CPUQuota800%, 명시적 User/Group 없음, 재시작 없음 정책입니다. 캡처 단계는3600초로 제한되며 RAM 보유의 절대 종료 시각은2026-10-19 12:15 KST입니다. 부모가 조기 정리할 때는 `systemctl --user stop nemochat-sft-sealed03-holder.service`를 사용합니다. STARTED 증거가 있으면 원본 캡처를 몰래 반복하지 않습니다.

## 승인 조건

시작 직전 및 캡처 직전 MemAvailable1TiB 이상을 확인합니다. 원본 manifest와 과거 producer exit0, 정확한 소스/토크나이저 해시 연결을 인증합니다. 파일당 최대3개 관측만 허용합니다: 정렬된 O_DIRECT 캡처가 원래 SHA와 같으면 즉시 통과하고, 다르면 초기 RAM 후보+buffered+새 direct 관측으로 제한된 페이지 투표를 수행합니다. 최종 전체 SHA가 원래 기대값과 정확히 같아야 하며 기대값 재설정은 금지됩니다.

모든 FD는 seal15 및 O_RDONLY입니다. 인증된 동일 RAM 바이트에서 배열 shape/dtype, 전체 토큰 범위, 실제 mask값/target수, 대화 전체 집계, SFT18..29의 정당한 alias 분류, C int32 idx/bin 연결과 전체18..29 부재를 검증합니다. 카탈로그 도메인은 실제 NUL로 끝납니다(PROTOCOL.md 참조).

실제 로컬 digest-pinned NeMo 이미지가 SCM_RIGHTS로 FD를 받아 전체 stream 및 mmap SHA를 독립 검증합니다. CPU 전용/유한 실행, 네트워크 없음, 읽기전용 root/mount, 모든 capability 제거, no-new-privileges, 디바이스 없음, CPU/RAM/PID 제한을 적용하고 소유 컨테이너만 정확히 삭제한 뒤 부재를 재확인합니다. 같은 UID의 일반 호스트 프로세스는 거부하며, 소유 label/pinned image/running Engine init 또는 검증된 자손(torchrun 포함)만 허용합니다. 이미지 상속 label은 허용하지만 소유 label은 반드시 일치해야 합니다.

모든 조건 통과시에만 `results/midtraining/sft-after-kd-sealed03/READY.sealed.json`을 만듭니다. schema는 `nemochat-sealed-full03-v1`, status는 `accepted_sealed_snapshot_full03`, 입력모드는 `sealed_scm_rights`입니다. 카탈로그2개를 포함한25개 파일입니다. 살아 있는 정확한 holder identity가 필수이며 오래된 READY는 승인이 아닙니다. 해시/의미 조건 실패시 RAM을 모두 닫고 정확한 BLOCKED 증거를 기록합니다. 기존 mmap은 계속 차단되고 GPU 실행은 부모만 승인합니다. checkpoint integrity는 `not_checked_explicit_user_waiver`이며 호스트 수리나 가중치 검증을 주장하지 않습니다.

소스 동결 후 producer/client/consumer/tests/docs를 수정하지 마십시오. 결과에는 request/동결 해시, 실제 fixture와 보존된 초기 실패, 테스트 로그, 정확한 MainPID/argv/서비스 제한, 파일별 stat/관측/SHA/감사, CPU 컨테이너 증거, 진행상황/cgroup/holder heartbeat가 남습니다.
