# 필터 적용 fresh-zero native prefix

원본 global iteration **86, 98, 121, 122, 126**을 8개 rank 모두에서 제외합니다. 총 40개 tar이며, collective 정렬을 위해 정상 판정된 34개도 함께 제외합니다. 원래 native GPTDataset을 `[468736,120,8]` 예산으로 먼저 만든 뒤 sample index wrapper로 1,138개 global batch를 선택합니다. physical↔logical 전체 대응표는 `prefix-manifest.json`에, 체크포인트에는 대응표 identity와 다음 logical/physical cursor를 기록합니다. 마지막 physical sentinel은 1143입니다.

native tar/zstd/torch codec, `_meta.json`, 원본 full cache identity, Adam/DDP/clip/router 및 3662/37/366 WSD 스케줄은 유지합니다. task-local reader subclass가 URL을 선택하고 decode 결과의 physical iteration을 logical iteration으로 연결합니다. 원본 파일 이름·내용·포장을 변경하지 않습니다. 모든 cache read는 O_DIRECT와 기존 shard SHA를 강제하며 retry/buffered fallback은 없습니다. 모든 microbatch를 원래 teacher의 physical-index digest와 비교합니다. 새로운 SHA 실패는 실제 SHA/길이를 남기고 중단합니다.

BASE19074 model-only로 시작하며 optimizer/RNG/cursor는 초기화합니다. native full-state checkpoint는 logical 64마다 및 최종 1138에서 저장합니다. checkpoint payload 해시 검증은 기존 명시적 면제를 유지합니다. 첫 LR=0 호출은 양수 LR 학습으로 세지 않습니다. 과거 6개 read 거부를 영구 파일 손상으로 해석하지 않습니다.

독립 user-systemd service가 공용 GPU 8개 lock을 유지합니다. repo와 입력은 read-only, 신규 output/scratch만 writable입니다. HOME/Inductor/Triton은 빈 디렉터리로 시작하며, freeze/Engine readback/rank receipt/그래프는 신규 실행에만 귀속됩니다.
