# CPU 우선 긴 문맥 유틸리티 구현

## 구현 범위

`src/nemochat/long_context/`는 일반 딕셔너리와 리스트를 사용하는 작은 함수들로 구성됩니다. **학습 스케줄을 계산하는 도구이지 실제 학습기나 NVIDIA NeMo/Megatron Bridge 통합이 아닙니다.** attention/RoPE 수정, GPU 실행, 모델의 64K 지원을 주장하지 않습니다. 합성 입력과 모의 네트워크 테스트는 모델 성능 측정이 아닙니다. 실제 엔드포인트 실행 결과만 측정 데이터로 취급합니다.

## 고정 유효 토큰 예산 스케줄

```bash
.venv/bin/python scripts/lc_schedule.py \
  --long-only-steps 2 --alternating-steps 4 \
  --effective-tokens 131072 --micro-batch-size 1 --world-size 1 \
  --cursor /tmp/lc-cursor.json --output /tmp/lc-steps.jsonl --count 3
# 같은 명령을 다시 실행하면 다음 3개 스텝을 생성합니다.
```

64K는 65,536 토큰, 4K는 4,096 토큰입니다. long-only 단계 이후 long/short 교대 단계는 long부터 시작합니다. 기본 길이 순서는 `65536, 65536, 65536, 4096, 65536, 4096`입니다. 각 스텝의 관계식은 다음과 같습니다.

```
effective_tokens = sequence_length * micro_batch_size * world_size * gradient_accumulation_steps
```

기본 설정의 누적 횟수는 long에서 2, short에서 32입니다. short에서는 시퀀스 수가 증가하지만 유효 토큰 예산은 바뀌지 않습니다. 시퀀스 길이는 패딩을 포함하는 전체 길이이며 loss mask를 적용한 실제 학습 토큰 수를 의미하지 않습니다. 정수가 아닌 값, 음수/0, 빈 계획, 나누어떨어지지 않는 예산, 너무 작은 예산은 반올림하지 않고 거부합니다.

API는 `make_plan`, `step_at`, `load_cursor`, `advance_cursor`, `save_cursor`입니다. 커서 JSON은 전체 계획, 버전, `next_step`을 저장합니다. 재시작 시 계획이 다르면 오류가 발생하며 임의로 초기화하지 않습니다. CLI는 현재 배치의 JSONL을 덮어쓴 다음 커서를 저장합니다. 각 파일은 원자적으로 교체하지만 **두 파일을 함께 처리하는 트랜잭션은 아닙니다.** 출력 저장과 커서 저장 사이에 프로세스가 중단되면 해당 배치가 재생될 수 있습니다. 커서마다 단일 작성자만 사용해야 하며 잠금은 제공하지 않습니다. 실제 학습기는 소비 완료 스텝과 자체 체크포인트를 별도로 관리해야 합니다.

## NIAH 결과 파일 보존

NIAH generate/run/score/plot 출력은 CSV·PNG·SVG·metadata까지 **새롭고 서로 겹치지 않는 경로**여야 한다. 기존 파일·깨진 symlink를 tokenizer loading/completion 호출 전에 거부하므로 재실행에는 새 prefix를 사용한다. Exclusive reservation과 no-clobber publication으로 다른 작성자의 결과를 보존한다. 모든 fixture·고유 ID를 첫 completion 전에 검증한다. 위 schedule/cursor의 의도적인 replacement는 변경하지 않는다. Reservation sidecar는 성공·실패 모두에서 한 번의 시도 기록(tombstone)으로 남으므로 retry마다 새 경로를 사용한다. 일부 artifact 공개 후 실패해도 이미 공개된 파일을 rollback 삭제하지 않는다. Private staging 이름은 다른 작성자가 변경하지 않는 소유권 계약이며, cleanup은 각 FD close·private unlink를 독립적으로 시도한다. 강제 종료나 cleanup 실패 시 staging 파일이 남을 수 있다. Bundle은 한 transaction이 아니므로 파일 하나만 보고 성공으로 판단하지 말고 exit status·전체 구성 확인이 필요하다. Incremental completion recovery는 추가하지 않았다.

## 실제 HF 토크나이저로 합성 입력 생성

```bash
.venv/bin/python scripts/niah.py generate \
  --tokenizer /path/to/tokenizer --local-files-only \
  --config configs/eval/niah_smoke.json --output /tmp/niah-fixtures.jsonl
# 선택 사항: --haystack /path/to/corpus.txt
# Config 값이 CLI보다 우선한다. 제공 config의 random_depths(현재 0)를
# 수정하거나 --config 없이 --random-depths 3을 사용한다.
```

설정 파일 값은 대응하는 CLI 값보다 우선합니다. smoke 설정은 2개 문맥 예산 × 3개 깊이 × 3개 샘플, 즉 **합성 입력 18개**를 생성하며 예측은 생성하지 않습니다. `generate_fixtures(tokenizer, context_lengths, depths, ...)`는 전달받은 토크나이저를 직접 사용합니다. 모델이 사용하는 로컬 스냅샷이나 고정된 HF revision을 권장합니다. 원격 코드 실행은 허용하지 않으며 오프라인에서는 `--local-files-only`를 사용합니다.

조립 방식은 다음과 같이 명확하게 정의됩니다.

1. haystack을 special token 없이 인코딩합니다.
2. 토큰을 반복/절단하고 `round(filler_count * depth)`에서 나누어 각 부분을 디코딩합니다. 디코딩의 공백 정리 기능은 끕니다.
3. 두 부분 사이에 시드로 생성한 임의 128비트 16진수 비밀 코드와 완전한 needle 문장을 넣고 앞뒤에 검색 지시문을 붙입니다.
4. **완성된 전체 프롬프트**를 special token까지 포함해 다시 인코딩합니다. `prompt_tokens + generation_tokens <= context_length`가 될 때까지 filler를 줄입니다. 정확히 최대 예산을 채우는 것은 보장하지 않습니다.
5. 원본 프롬프트와 최종 토큰의 디코딩 결과 모두에서 needle 문장이 그대로 유지되는지 검사합니다. 정규화로 needle이 변하는 토크나이저는 거부합니다.

이는 토큰 조각 기반 haystack 조립 방식이며 모든 filler 조각의 원래 토큰 ID가 재인코딩 후 같다는 보장은 아닙니다. 바이트 토큰 경계가 문자 중간에 걸려도 최종 전체 프롬프트를 실제 인코딩한 길이가 예산 판단 기준입니다.

깊이는 전체 프롬프트, 문자 수, 바이트 수의 비율이 아니라 **haystack 토큰 조각의 비율**입니다. `needle_offset_haystack_tokens`, `actual_haystack_depth`에서 반올림 결과를 확인할 수 있습니다. haystack이 비어 있으면 실제 깊이는 null입니다. 명시한 깊이 및 `random_depths`로 추가한 임의 깊이, 샘플별 비밀 코드는 시드로 재현됩니다. 그리드/순서/임의 깊이 개수를 변경하면 난수 순서도 바뀝니다. 반복된 일반 문장은 단순 smoke용이며 현실적인 성능 벤치마크용 말뭉치가 아닙니다.

각 행에는 프롬프트, 정답, needle, 전체 토큰 길이, 최종 토큰 ID의 SHA-256, 말뭉치 해시, 시드, 깊이, 생성 예약 토큰 수, 토크나이저 이름 및 CLI revision이 기록됩니다. `validate_fixture`는 요청 전에 전체 프롬프트를 다시 인코딩하여 토큰 해시/길이, 예산, needle 보존을 검증합니다. 이름과 revision만 비교하는 방식이 아닙니다.

## 실제 예측 요청

```bash
# 인증이 필요하면 비밀 관리 도구로 OPENAI_API_KEY 환경변수를 설정합니다.
.venv/bin/python scripts/niah.py run \
  --tokenizer /path/to/tokenizer --local-files-only \
  --input /tmp/niah-fixtures.jsonl --output /tmp/niah-predictions.jsonl \
  --endpoint http://localhost:8000/v1/completions --model actual-served-model \
  --timeout 60 --api-key-env OPENAI_API_KEY
```

URL은 일반 텍스트 **`/v1/completions` 전체 주소**입니다. chat 템플릿의 숨은 토큰 비용을 피하기 위해 chat completions를 사용하지 않습니다. `urllib.request`로 실제 프롬프트를 보내고 `max_tokens`, temperature 0, stream false를 지정합니다. 인증 키는 이름을 지정한 환경변수에서만 읽으며 URL에 넣은 인증 정보는 거부합니다. 인증 키가 있는 urllib 요청은 loopback IP 또는 `localhost`가 아닌 경우 **네트워크 호출 전에 HTTPS를 강제**합니다. 로컬 smoke 서버는 HTTP를 허용합니다. URL 인증 정보·잘못된 포트는 거부하며 redirect로 bearer key를 전달하지 않습니다. 인증된 loopback HTTP 요청은 환경 프록시를 비활성화하므로 `HTTP_PROXY`가 설정되고 `NO_PROXY`가 비어 있어도 평문 bearer key가 프록시로 유출되지 않습니다. 다른 urllib 요청은 기존 프록시 설정을 유지합니다. 키가 없으면 HTTP도 허용하지만 프롬프트 보호를 위해 원격 서비스에는 HTTPS를 권장합니다. 선택적인 hosted SDK 클라이언트의 transport 설정은 호출자가 책임지며 이 URL 검사의 적용 대상이 아닙니다. 신뢰할 수 없는 제공자에게 민감한 말뭉치를 보내지 마세요. temperature 0도 서버 실행의 완전한 결정성을 보장하지는 않습니다.

API의 `run_predictions(..., client=client)`는 선택적으로 `client.completions.create(..., timeout=...)`를 지원하는 hosted SDK 클라이언트를 받습니다. SDK 설치와 설정은 호출자의 책임입니다. chat 전용 모델은 이 프로토콜에서 지원하지 않습니다. 임의의 정답 복사나 더미 예측으로 대체하지 않습니다.

HTTP 실패, 잘못된 JSON/choices/text, 서버 오류 객체, 네트워크 실패, timeout은 명시적인 상태로 기록합니다. HTTP 400/404/413/422/501은 요청 `unsupported`, 나머지는 error로 분류합니다. unsupported라는 상태가 모델 자체의 능력 부족을 증명하는 것은 아닙니다. 오류 본문에는 민감한 정보가 있을 수 있으므로 읽거나 저장하지 않고 HTTP 상태 분류를 유지합니다. 성공 HTTP JSON 본문은 1,048,576바이트(`MAX_RESPONSE_BYTES`)로 제한합니다. 최대 cap + 1바이트만 읽고 한도를 초과하면 파싱 전에 거부하여 `status=error`로 기록하며 예측이나 채점 표본으로 포함하지 않습니다. 이 제한은 urllib에만 적용되며 호출자가 설정한 SDK 클라이언트에는 적용되지 않습니다.

서버 `usage.prompt_tokens`가 로컬에서 측정한 전체 프롬프트 토큰 수와 정확히 일치하는 문자열 응답만 `status=ok`가 됩니다. usage가 없으면 `tokenization_unverified`, 다르면 `tokenization_mismatch`이며 반환 텍스트를 보존하지만 정확도 분모에서는 제외합니다. **개수 일치가 서버 토큰 ID 일치까지 증명하지는 않습니다.** 운영자는 같은 토크나이저 및 BOS 정책을 확인해야 합니다. BOS 정책이 다르면 프로토콜을 맞출 때까지 제외됩니다. 로컬 예산을 검증할 뿐 서버의 실제 문맥 용량을 가정하지 않습니다.

요청은 순차적이며 요청별 timeout만 제공합니다. 자동 재시도, 예측 실행의 재개 엔진, 전체 실행 시간 제한은 없습니다. 결과를 메모리에 모은 뒤 출력하므로 중단되면 해당 실행의 미저장 결과를 잃을 수 있습니다.

## 엄격한 채점과 정직한 시각화

```bash
.venv/bin/python scripts/niah.py score --input /tmp/niah-predictions.jsonl \
  --output /tmp/niah-scores.jsonl --csv /tmp/niah-scores.csv
.venv/bin/python scripts/niah.py plot --input /tmp/niah-predictions.jsonl \
  --output-prefix /tmp/niah-measured
```

`exact_match`는 양쪽에 `" ".join(text.split())`만 적용합니다. **공백 정규화만 허용**하며 소문자 변환, 구두점 삭제, 부분 문자열 검색, 설명 제거는 하지 않습니다. 정답 앞에 “The code is”가 붙거나 뒤에 마침표가 붙으면 오답입니다. 생성 길이 제한으로 잘린 유효 응답도 반환된 내용 그대로 채점하며 복구하지 않습니다.

`score_predictions`는 문맥/깊이별 total, 유효 count, successes, excluded, 상태별 개수, accuracy, **95% Wilson 구간**을 계산합니다. 중복 ID, 서로 다른 모델/엔드포인트, 혼합된 `data_kind`는 거부합니다. `ok`가 아닌 요청은 분모에 들어가지 않습니다. 유효 응답이 없으면 정확도와 구간이 **null이며 0점이 아닙니다.** 원시 fixture에는 예측이 없으므로 정확도가 null입니다. 인위적인 예측을 별도로 만들 경우 반드시 `data_kind=fixture`로 표시해야 합니다. 적은 샘플의 불확실성은 크며 제외 비율이 높으면 선택 편향이 발생할 수 있습니다.

PNG/SVG heatmap은 입력 JSONL의 예측을 읽고 고정 **0..1** 색상 범위를 사용합니다. 유효/전체 샘플 수와 구간을 표시하고 timeout, unsupported, 토큰 불일치/오류를 다른 색과 텍스트로 구분합니다. 혼합 제외 상태는 모두 나열하며 관측하지 않은 셀은 not sampled로 표시합니다. 측정 그림에는 모델 이름이 표시됩니다. fixture 그림에는 **SYNTHETIC FIXTURE — NOT MODEL PERFORMANCE** 제목이 반드시 붙습니다.

`.metadata.json`에는 `data_kind`, measured/synthetic 플래그, 모델/엔드포인트, 채점 정의, 색상 범위, 셀별 표본 수와 불확실성이 저장됩니다. JSONL/CSV/PNG/SVG/metadata는 같은 디렉터리의 임시 파일과 `os.replace`로 각각 원자적 교체를 수행합니다. 여러 출력 파일 전체는 트랜잭션이 아닙니다. 기록된 provenance는 사용자가 제공한 메타데이터이며 실제 서버 호출에 대한 암호학적 증명이 아닙니다.

## Loopback CPU smoke 서버

`scripts/serve_cpu.py`는 단일 스레드의 loopback 전용 대조군 서버이며 운영용 서비스가 아닙니다. 수락한 각 소켓에는 헤더·본문 읽기와 응답 쓰기를 위한 5초 비활성 timeout(`REQUEST_TIMEOUT_SECONDS`)이 적용됩니다. 본문 읽기가 중단되면 쓰기 가능한 연결에 JSON HTTP 408을 반환하고, 불완전한 헤더는 표준 HTTP 핸들러가 연결을 닫습니다. timeout과 연결 끊김은 요청 traceback 없이 처리하여 이후 `/health` 및 completion 요청을 계속 받을 수 있습니다. 이는 전체 요청 시간 제한이나 모델 생성 timeout이 아니므로 조금씩 데이터를 보내는 연결이나 느린 생성은 여전히 단일 서버를 점유할 수 있습니다.

## 검증

```bash
.venv/bin/python -m pytest tests/test_long_context.py tests/test_cpu_server.py -q
```

작은 문자 토크나이저, 로컬에서 구성한 HF byte-level 토크나이저, 모의 네트워크 응답으로 토큰 예산, needle 왕복 보존, 시드 재현성, 스케줄 재개, HTTP 요청 형태, 오류 제외, 엄격한 채점, Wilson 구간, CLI, 그림 표시를 검증합니다. **기능 정확성 테스트이지 성능 측정은 아닙니다.** 외부 모델 실행이나 실제 학습 결과를 주장하지 않습니다.
