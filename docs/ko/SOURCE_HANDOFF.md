# 소스 구성과 로컬 커밋 인계

[English](../en/SOURCE_HANDOFF.md)

## 최신 수락 이력: 2026-10-19

최종 네이티브 SFT 소스는 **`iter_0001248`**이다. 1200이나 이전 905가
아니다. 905의 전체 네이티브 상태에서 343회 더 진행했으며, 대화 159,744개,
assistant 학습 타깃 117,409,886개, **패딩 포함** 처리 위치
2,617,245,696개를 기록했다. 서로 다른 카운터를 고유 데이터 토큰 수로
합쳐 설명하면 안 된다.

이미 게시된 **비공개**, 모델 전용 HF 저장소는
`eagle705/nemochat-10BA1B-BF16-SFT-1200`, 불변 리비전은
`893c00ca610f25bde4c6b46050a224ecc9388385`이다. 요청된 저장소 이름의
1200과 실제 소스 1248을 구별한다. 기존 영수증은 페이로드 53개,
19,601,967,508바이트, 가중치 샤드 12개, 작은 자산 41개를 기록한다.
이번 소스 정리는 원격 게시/수정 작업이 아니다.

고정된 400문항 진단(MMLU 200, ARC-Challenge 100, HellaSwag 100)의
SFT1248 정확도는 **22.5% / 35% / 48%**, heldout NLL은
**1.399376922740647**, PPL은 **4.052674050965295**이다. 전체 벤치마크나
수정 없는 공식 harness 실행이 아니다. 실제 greedy 생성 6개는 모두 EOS
없이 128토큰 한도에 도달했고, 반복/나쁜 한국어/불완전하거나 잘못된 JSON을
보였다. 생성 품질 개선을 주장하지 않는다. native/HF logits 동등성은
실행하지 않았고, native 페이로드 SHA/CRC 검사는 사용자의 명시적 면제다.

Git에 넣지 않는 기존 근거:
- `results/midtraining/hf-sft1200-publication-01/publication_receipt.json`
- `results/midtraining/hf-sft1200-publication-01/parent-final-verification.json`
- `results/midtraining/sft1248-benchmark-gpu-01/summary.json`
- 이번 정리의 목록/값을 제거한 보안 스캔/테스트/리뷰 diff:
  `results/maintenance/source-commit-20261019-01/`

과거 실행 기록은 새 학습/업로드 실행 허가가 아니다.

## 역할별 진입점

- 데이터: `sft/data/nemotron_sft.py`와
  `src/nemochat/data/nemotron_sft.py`. 공식 렌더링, 작성자 선택 assistant
  턴, heldout 예약, 무단 잘라내기 금지가 핵심이다.
- 설정: `sft/configs/nemotron_dataset_settings.yaml`와
  `src/nemochat/data/sft_dataset_settings.py`. 실제 로컬 discovery/토크나이저
  전환/예약 영수증을 엄격히 교차 검사하므로 소스만으로 운영 준비 완료를
  주장할 수 없다.
- 토크나이저: `sft/data/prepare_tokenizer.py`,
  `src/nemochat/tokenization/posttraining.py`,
  `sft/configs/tokenizer_super35.yaml`. ID 18–29의 예약 이름만 전환하며
  매핑/가중치/스케줄러를 바꾸지 않는다.
- KD 이력: `midtraining/README.md`, `midtraining/training/offline_kd/`,
  `midtraining/patches/offline_kd_pr/`. 일반 범용 trainer가 아니라 동결된
  변형 소스와 Bridge 패치다.
- SFT 이력: `sft/training/`, `sft/resume/runtime/`, `sft/archive/`. runtime02–05, sealed-memory 전달,
  continuation/recovery 소스는 경로를 갱신해 보존하며, 정확한 원본 바이트는 Git 이력/로컬 스냅샷에 남긴다. 이동한 소스는 GPU 재검증하지 않았다.
- 변환/게시: `pretraining/publication/`, `midtraining/publication/`, `sft/publication/`,
  `src/nemochat/hub/base_checkpoint.py`. 모델 전용 HF와 optimizer를 포함한
  네이티브 재개는 별개다. 게시는 별도 허가가 필요하다.
- 대시보드: `dashboard/server.py`, `dashboard/README.ko.md`. 기본 loopback,
  읽기 전용이며 실제 캡처/SDK는 선택적 외부 입력이다.
- 사전학습 래퍼: `pretraining/entrypoints/README.ko.md`. ignored 런타임 helper가
  필요한 과거 머신 종속 진입점이다.

루트 README의 기존 CPU 표는 초기 튜토리얼 범위를 설명한다. 위에서 명시한
실제 수락 이력만 최신 상태로 해석하며, 다른 CPU scaffolding까지 GPU 검증
완료로 바꾸어 읽지 않는다.

## 버전과 배포 경계

호스트 CPU extra는 **Transformers >=4.51,<5**를 유지한다. 수락된
native/export 작업은 별도 고정 NeMo 이미지의 **Transformers 5.12.1**을
사용했다. 이를 맞추려고 호스트 라이브러리를 올리지 않는다. 기존 YAML의
데이터/teacher 리비전과 실제 로컬 토크나이저 전환 manifest를 따른다.
활성 SFT 3개, 보류 128K 데이터, 정답 없는 RL seed를 구별하며 128K CP
네이티브 검증은 완료되지 않았다.

데이터/가중치/teacher cache/자격 증명/원시 로그/운영 영수증은 소스 커밋에서
제외한다. `LOGS.md`와 제외한 모든 untracked 원본을 로컬에 보존한다.
`uv.lock`의 메타데이터는 `pyproject.toml`과 맞지만 uv 사용 의도/작업 흐름은
확인되지 않았으므로 이번에는 stage하지 않는다. GPU 이미지 잠금 파일이 아니다.

KD CPU 메서드 비교/패치 재현에 필요한 최소 upstream 소스만 저작권과 전체
라이선스를 동반한다. 범위는
[`licenses/README.md`](../../midtraining/patches/offline_kd_pr/licenses/README.md)에
기록했다. 폰트의 OFL/출처 파일도 함께 보존한다. 프로젝트 전체나 데이터/
모델에 임의의 라이선스를 부여하지 않는다.

## 검증과 인계

일반 검사는 `CUDA_VISIBLE_DEVICES='' WANDB_MODE=disabled HF_HUB_OFFLINE=1
.venv/bin/python -m pytest tests dashboard/tests -q`로 실행한다.
정확한 과거 실행 종속 목록 `tests/historical_contracts.json`에는 함수 88개,
수집된 case 149개가 남았다. 전체 항목 재감사로 portable 함수 3개(case 7개),
즉 manifest 읽기 전 opt-in 거부, statx 전체 mock 정렬 오류, 커밋된 CPU gate
소스 존재 검사를 제외하여 기본 실행에 복구했다. 토크나이저 로더는 준비 단계의
자산 4개(`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`README.md`) 영수증을 모두 요구하고 Transformers import 전에 크기/SHA를
검사한다. vocab/EOS 확인만으로 호환 전환을 증명하지 않는다.
등록된 과거 테스트에는 `NEMOCHAT_HISTORICAL_TESTS=1`을 명시해야 하며, 원래의
assertion/소스 guard는 그대로 유지한다. skip을 과거 통합 검증 성공으로
세지 않는다. 누락된 운영 영수증을 private 데이터 복사나 가짜 수락 fixture로
메우지 않는다. 실제 수치/기존 HEAD 기준 실패/파일시스템 및 의존성 한계는
maintenance 보고서를 확인한다.

부모 에이전트가 staged allowlist/diff, 라이선스, 보안 스캔과 실제 테스트
결과를 독립 리뷰한 다음 로컬 커밋을 담당한다. push/PR/공개 배포/학습/
패키지 업그레이드/새 다운로드는 이번 범위가 아니다.

## 2026-10-19 단계별 정리

소스는 최상위 pretraining/midtraining/sft/rl로 분리합니다. midtraining에는 KD/계속 사전학습만 있으며 SFT는 별도 최상위 폴더입니다. 기존 데이터/체크포인트/실행 증거는 이동하거나 덮어쓰지 않습니다. [단계별 경로 명세](STAGE_LAYOUT.md)를 확인하세요. RL 학습기가 구현되었다는 뜻은 아닙니다.
