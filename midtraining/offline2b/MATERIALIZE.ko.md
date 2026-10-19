# 승인된 offline1920 native corpus 생성

`materialize.py`는 실제 CPU 생성기이며 기존 `prepare.py` capacity census는 변경하지 않습니다. 승인 예산은 **3662 optimizer calls**, sequence4096, GB128, MB1, processed positions **1,919,942,656**, 67/10/3/20 혼합입니다. 반복·추가 다운로드·혼합 변경은 금지합니다.

Native 0.5% reserve와 반올림·shift를 포함한 최소 고유 token 수:

| Source | Required tokens |
|---|---:|
| Code Concepts | 1,292,808,193 |
| Algorithmic | 192,966,657 |
| Formal Logic | 57,901,057 |
| ClimbMix Replay | 385,921,025 |

Capacity SHA는 `467608152ca9288b561d5111d9f39d33cd26d546ea05d876024755e13f9593db`입니다. Code 여유는 262,453 tokens뿐입니다. 기존 census에서 확인한 Algorithmic 원문 7개는 사용자 승인에 따라 **문서 전체를 제외**합니다. ID 치환은 하지 않으며, 새 train/validation의 IDs18..29는 실제 전체 검사에서 모두 0이어야 합니다.

## 실제 생성·검증 경로

- 실패한 과거 generation의 실패 상태를 유지합니다. 고정 selector 소스, 공식 raw13개, 새 전체 catalog, seed, tokenizer, 전체 heldout 계보를 인증합니다.
- Code/Algorithmic/Formal의 인증된 native prefix를 문서 단위로 복사하여 새 native index를 만듭니다. BPE는 남은 catalog tail에만 수행하고 source별 7개 cursor sample을 재검증합니다.
- Replay native ID는 그대로 보존합니다. decode/encode roundtrip 불일치는 문서 전체 제외, 알려진 upstream 마지막 clipped 문서는 항상 제외합니다. benchmark 및 전체 heldout/token 예약을 적용합니다.
- 기존+신규 grouped heldout **32,397개**, 원래 ClimbMix 원문 예약 **84,990개**를 보존합니다. 네 validation pair는 seed와 byte-identical입니다.
- 새로운 reader로 전체 index/count/shape/range/EOS, 전 source duplicate/heldout overlap을 재검사합니다. 실제 8개 bin/idx의 SHA와 IDs18..29도 전체 스트림으로 검사합니다. 모두 통과한 뒤에만 `manifest.json`과 SHA-bound `success.json`을 발행합니다.

출력: `datasets/midtraining/super35-kd-01/offline1920-v1/data/`.
근거: `results/midtraining/offline1920-corpus-01/`의 승인 request, `producer-review.json`, source freeze, Engine create/start/exit 기록입니다. 부모 request는 immutable producer 실행 후 도착했습니다. `producer-review.json`에서 예산과 제외 문서 7개의 전체 identity가 일치함을 검증하고, 부모 request SHA를 실행 중 producer request/source SHA 및 정확한 container ID에 연결합니다. manifest의 producer request SHA와 이어지는 명시적 hash chain이며 실행 소스는 변경하지 않습니다.
`progress.json`은 단계·source·tokens/documents·경과시간·출력 bytes를 나타내며 준비 완료 증거가 아닙니다.

## 실행과 부모 인계

`launch_cpu.py`는 terminal 자식 job이 아니라 독립 Docker Engine container를 생성합니다. 고정 승인 image, UID1001:GID985, CPU-only runc, network none, 12CPUs/64GiB(no swap), application 2시간/출력16GB 상한을 적용합니다. root/repo는 읽기 전용이며 새 generation과 지정 scratch만 쓰기 가능합니다. `engine-create.json`의 정확한 container/request identity를 사용하며 launcher를 중복 실행하지 않습니다.

```sh
.venv/bin/python -m pytest tests/test_offline2b_materialize.py tests/test_offline2b_corpus.py tests/test_midtraining_recover_corpus.py -q
python midtraining/offline2b/materialize.py  # preview만 수행
```

부모는 정확한 Engine exit0/OOM 아님과 `longrun_native.load_prepared_manifest()`의 success/manifest SHA 일치를 확인한 뒤 부모 소유 3662-call native controller를 해제합니다. 생성기는 학습을 시작하거나 model weight payload를 읽지 않습니다. 실패/progress 결과를 수동으로 성공 처리하지 않으며 모든 legacy/census 근거는 보존합니다.
