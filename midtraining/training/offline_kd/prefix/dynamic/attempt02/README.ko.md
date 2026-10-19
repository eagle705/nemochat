# Dynamic 전체 Rank Cache 제외 정책

사용자의 최신 승인에 따라 cache 입력 불일치는 해당 physical global batch 전체를 8개 Rank에서 동적으로 제외하고 다음 source로 진행한다. 기존 cache/소스/결과와 SHA 기대값은 수정하지 않는다. 원본 BASE19074에서 logical update 0으로 새로 시작한다.

각 Rank는 자기 shard를 O_DIRECT로 한 번 읽고 원래 SHA/size/stat 및 native codec metadata/shape/finite/index를 확인한다. 동일한 immutable bytes에서 decode한 tensor와 역사적 Teacher input digest를 확인한 16개 sample을 보존한다. 모든 Rank의 동의 전에는 forward/backward/optimizer/LR/expert-bias update가 없다. input mapping/digest 알고리즘 오류와 model/optimizer/collective 오류는 여전히 fatal이다. LM fallback, retry, SHA 재기준화, OS reset은 없다.

원래 full GPTDataset 예산 [468736,120,8], seed43, 전체 cache identity, native KD Loss와 ghost-tail 0.5 LM + 0.5 KD, native pretrain/optimizer/checkpoint를 유지한다. Scheduler는 WSD3662/warm37/decay366, LR1e-5..1e-6 그대로다. Logical optimizer counter와 physical scan/quarantine counter는 별도이다. 원본 0..1142를 한 번만 조사하며 역사적 [20,86,98,121,122,126]은 과거 거부로 기록한다. 최대 accepted1137이며 새 불일치가 생기면 줄어든다.

검증된 lookahead 1개로 불량 suffix도 정상 native 종료/최종 저장한다. Checkpoint16/64/128/.../final은 model+optimizer+RNG+native sampler와 atomic/fsync dynamic mapping transaction을 결합한다. Byte hashing 면제는 명시적으로 유지한다. Resume cursor는 scan 끝이 아니라 **다음 미소비 physical**을 가리키며 prefetched batch를 건너뛰지 않는다. 이번 실행은 fresh이며 별도 restore 요청 없이 resume CLI는 거부한다. CPU test의 fault injection은 실제 cache나 GPU 데이터를 변조하지 않는다. 유효 sample이 0이면 NoData/0 updates이며 학습 완료로 주장하지 않는다.
