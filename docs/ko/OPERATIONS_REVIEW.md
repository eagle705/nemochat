# 운영 확장 리뷰

Canonical: [영문 기록](../en/OPERATIONS_REVIEW.md).

상태: 리뷰 수정 후 local CPU/offline 회귀 gate 통과. HF remote 왕복·W&B online·GPU stage 검증은 별도 pending이다.

독립 리뷰로 발견한 예산 초과의 fit/slack 오류, retention보다 많은 기존 checkpoint의 peak storage 누락, 큰 integer token 목표의 update rounding, HF SDK가 제외하는 `.git`·`.cache/huggingface` namespace 문제를 부모가 수정했다. 회귀 테스트를 추가했으며 planning/HF/신규 회귀의 targeted 결과는 **90 passed**다.

Storage는 최초 footprint와 cleanup 전 다음 write를 포함하지만 완료된 write 뒤 pruning한다는 가정이다. 지연된 cleanup·동시 upload/copy의 전체 peak를 보장하지 않는다. Integer update count는 정확히 계산하고 시간/가격은 floating-point 계산으로 유지한다.

사용자가 공급한 대표 표본 파일을 `population × 평균`으로 계산하려면 **self-weighting 표본**이어야 한다. Unequal inclusion probability를 provenance에 적는 것만으로 weighted estimator가 되는 것은 아니다. 가중 표본 estimator는 구현되지 않았다. Reservoir는 선택한 파일 population 전체를 읽되 tokenize 대상만 bounded sample로 제한한다. Corpus 전체를 읽지 않는 것과 선택한 파일의 sublinear I/O는 다르다. 표본 shard 밖으로 대표성과 uncertainty를 확장하지 않는다.

W&B의 실제 offline SDK와 CPU exact-state 비교는 검증됐다. Schema/실패 처리 보강 완료. 부모 전체 회귀는 **358 passed, 1 skipped, 25 subtests passed**이며 실제 offline SDK 테스트도 포함한다. Dependency·compileall·wheel·문서 링크/동반 문서·JSON 검증 통과. CUDA RNG·다중 rank·checkpoint/run-ID 자동 일치는 미검증이다. Secret key 필터는 value 기반 DLP가 아니므로 credential이 포함된 URL/path/metadata를 전달하지 않는다.

실제 HF remote 왕복·W&B online은 미실행이다. Native bundle inventory 검증은 실제 checkpoint deserialize/optimizer resume 증거가 아니다. Synthetic transport fixture는 정상 model state로 가장하지 않는다.
