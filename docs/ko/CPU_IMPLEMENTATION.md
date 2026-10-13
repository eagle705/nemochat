# CPU에서 실행해 볼 수 있는 첫 구현

[영문 walkthrough](../en/CPU_IMPLEMENTATION.md), [실행 명령 README](../../README.md), [실제 검증 결과](../IMPLEMENTATION_VERIFICATION.md).

## 구현 스타일

튜토리얼 readability를 우선했다. Trainer 상속 계층/registry/backend factory 없이 **script → 명시적인 함수 → 결과 파일** 순서로 읽는다. `ModelDimensions`, `LossTotals`, `WSDConfig`가 작은 dataclass이며 나머지는 직접 함수와 dict를 사용한다. 코드·주석·docstring은 영어, 이 문서는 한국어 해설이다.

## 파일별로 읽기

1. `scripts/model_config.py`: A/B 차원을 보고 `models/dimensions.py`에서 총/활성 parameter 계산을 읽는다. GPU 없이 계산 가능. 실제 instantiated numel/FLOPs와는 다르다.
2. `scripts/prepare_data.py`: 선택한 ClimbMix shard만 받고, tokenize/dedup/quota를 거쳐 uint32 flat pack을 저장한다. 마지막 val shard·권리 조건·tokenizer revision을 검증한다. 전체 원천을 받지 않는다. 생성된 `.u32`는 아직 Megatron `.bin/.idx`가 아니다.
3. `eval/metrics.py`: CE/PPL/BPB와 CORE 수식을 직접 읽는다. Loss와 bytes를 각각 합한 뒤 나눈다.
4. `eval/token_bytes.py`: ByteLevel raw-byte alphabet을 역변환한다. partial UTF-8를 text로 decode해서 byte 수를 왜곡하지 않는다. Super 실제 tokenizer로 한글/emoji/code round-trip을 확인했다.
5. `scripts/base_eval.py`: 실제 HF checkpoint를 CPU에서 forward한다. BPB는 document-local chunk protocol, CORE는 continuation mean NLL/teacher-forced 정확도다. MCore 분산 adapter로 포장하지 않는다.
6. `scripts/train_cpu.py`: tiny HF control의 **실제** AdamW update/save/resume. Model safetensors, optimizer/RNG, data hash와 cursor를 보존한다. 13.8B NemoChat 학습은 아니다.
7. `scripts/niah.py`: 실제 tokenizer 기반 input → real completions endpoint → exact match/CSV → PNG/SVG. `serve_cpu.py`로 작은 모델을 loopback에서 실제 생성시켜 전 경로를 검증할 수 있다.
8. `scripts/lc_schedule.py`: long-only 후 long/4K 교대 계획과 cursor. Emitted step이지 완료된 GPU update가 아니다.
9. `models/bridge_provider.py`: 공식 pinned API를 바탕으로 A/B provider를 직접 만드는 코드. 현재 Bridge/TE GPU 환경에서는 실행하지 못했으므로 미검증이다.

Package 상대 경로는 모두 `src/nemochat/` 기준이다.

## 지금 완료한 것과 아닌 것

CPU unit tests 외에 tiny model inference, real ClimbMix 선택 shard, 실제 NIAH request/score/heatmap, optimizer update와 resumed/uninterrupted tensor equality를 실행했다. Fixture 점수를 final 모델 성능으로 주장하지 않는다. Tiny GPT-2 NIAH 실패는 평가 경로가 작동한다는 결과일 뿐이다.

아직 Curator GPU 정제/near dedup, official MCore indexed-format 변환, hybrid export mapping, vocab-parallel BPB, CP/Mamba correctness, 64K training controller, SFT/RLVR/MOPD는 구현 완료가 아니다. 존재하지 않는 SDK 기능 대신 성공하는 dummy로 대체하지 않았다.

첫 테스트용 tokenizer가 Unicode 일부를 표현하지 못했을 때 BPB의 raw-byte 검증이 오류를 냈다. 검증을 우회하지 않고 완전한 byte tokenizer를 가진 tiny GPT-2 checkpoint로 바꿔 실행했다.

## 실행 시 유의사항

- `.venv` CPU 환경과 미래 GPU container는 별도다.
- README의 `runs/my-*` 경로를 사용한다. 기존 outputs를 덮어쓰지 않는다.
- ClimbMix smoke는 최대 100MB의 **선택 shard 1개**를 내려받고 256-token pack만 남긴다. 임시 shard/cache는 삭제한다. 최종 subset보다 network 읽기량이 큰 것은 숨기지 않는다.
- CORE COPA 3개 smoke score는 full CORE가 아니다. Baseline centering과 output schema 검증용이다.
- NIAH prompt+generation 전체 budget을 사용하며 unsupported/error는 오답 0과 별도로 표시한다. 1 sample/cell은 pipeline smoke일 뿐 신뢰할 수 있는 capability 평가가 아니다.
- 최종 serving/MOPD는 이후 GPU 단계에서 구현한다. 현재 CPU HTTP server는 vLLM/Dynamo 대체물이 아니다.
- 코드 작성/테스트는 실행했지만 commit/push/외부 게시/weights 공개는 하지 않았다.

블로그 Part I에는 이 readable building blocks와 검증 범위를 기록하고, 실제 B300 학습 결과가 나온 뒤 quality/profile Figures를 추가한다. [1–4]/[5–6]/[7] 세 묶음은 그대로 유지한다.
