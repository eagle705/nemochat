# Full03 데이터 승인 차단 진단

소유 범위는 이 디렉터리와 `results/midtraining/sft-after-kd-data-acceptance-01/`뿐입니다.
GPU 할당, 모델 Weight payload 읽기, prepared 데이터/기존 Manifest 수정,
SHA 기준 변경, page cache 제거, 기존 completion marker 생성은 수행하지 않습니다.

- `verify.py`: 같은 immutable chunk에서 SHA와 Token/Mask 통계를 계산하고,
  buffered/O_DIRECT 불일치 최초 페이지를 보존합니다. 정렬된 O_DIRECT 요청의
  EOF short-read를 명시적으로 처리하며 buffered fallback은 없습니다.
- `serial_check.py`: 독립 GNU `dd bs=4M | sha256sum`으로 SFT 7개 및 C의
  8개 `.bin`/`.idx` 쌍을 두 경로로 완전히 읽습니다. 프로세스 ExitCode와
  읽기 전후 inode/크기/mtime/ctime을 기록합니다.
- `dd_audit.py`: C의 8개 전체 Token 배열을 직접 읽고 같은 byte stream에서
  SHA, Reserved ID Count, Token Bounds를 계산합니다. SHA가 틀리면 Count가
  정상처럼 보여도 인증된 부재 증명으로 인정하지 않습니다.
- `provenance.py`: 실제 CPU Producer의 clean exit0, 기존 Handoff의 Source SHA,
  Counted Mask Audit 및 Tokenizer 자산/전체 Backend 동등성을 인증합니다.
- `range_evidence.py`: 최초 불일치 4096-byte 범위를 독립 dd로 두 번씩 다시 읽어
  buffered zero page와 direct JSON 내용 차이를 보존합니다.

초기 paired scan은 도구의 420초 제한으로 중단되었으며 기록을 보존했습니다.
초기 NumPy private header API 호환성 오류 때문에 해당 NPY 의미 통계는
무효입니다. Public version별 Header API로 수정하고 회귀 테스트 후 네 배열
전체를 `array-audit02/`에서 별도로 다시 조사합니다. 전체 파일 SHA 결과와
파서 통계의 유효성은 구분합니다. CPU affinity는 8개 Core, BLAS/OMP는 1개
Thread이며 bounded streaming 방식입니다. 모든 소유 Background 작업의 종료를
확인한 뒤 반환합니다.

새 SFT의 ID 18–29는 일괄 금지하지 않습니다. 승인된 공식 신규 Alias 의미에
따라 실제 발생을 측정·분류합니다. 기존 Alias에서의 의미 전환에는 이전에
학습한 C Corpus의 새롭고 인증된 부재 증명이 필요합니다. Embedding 순열이나
Vocabulary Resize는 없습니다. 기존 Counted Audit은 전체 Row/Mask를 셌지만
독립 Source-offset 재렌더링은 216개이며 전체 2차 재렌더링이 아닙니다.

기존 Publisher는 실패 파일명/SHA를 남기지 않았으므로 당시 최초 실패 파일은
로그만으로 복원할 수 없습니다. 새 검사에서 실패한 파일만 정확히 보고합니다.
독립 전체 Direct 검사도 실패하므로 Direct Transport를 수리 완료로 취급하지
않습니다. 현재 mmap 경로가 안전하지 않으면 READY를 생성하지 않습니다.
일부 재검사 성공은 기존 실패나 Host Fault를 지우지 않습니다. 인증되지 않은
관찰 Bytes로 신뢰할 수 있는 Row Digest를 만들지 않으며, 작은 Pilot 대체,
새 Raw 재렌더링, 데이터 재선택도 승인 없이 수행하지 않습니다.
