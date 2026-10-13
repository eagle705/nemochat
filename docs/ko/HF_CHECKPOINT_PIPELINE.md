# B300 주간 임대를 위한 Hugging Face 체크포인트 파이프라인

새 CLI 실행은 저장소 루트 기준 경로와 새 산출물 이름을 사용합니다. [런타임 디렉터리 규약](RUNTIME_LAYOUT.md)에 따라 입력 데이터는 `datasets/`, checkpoint는 `checkpoints/`, 보고서·trace·plot은 `results/`, 텍스트/W&B 로그는 `logs/`에 둡니다. 필요한 부모 폴더만 만들며 기존 fixture/config와 과거 증거 경로는 보존합니다.

## 범위와 두 종류의 산출물

이 구현은 **쓰기 완료 후 고정한 체크포인트**의 포장·전송·바이트 검증을
담당합니다. Megatron 체크포인트 생성, DCP→Transformers 변환, 학습 재개
검증을 대신하지 않습니다. 자세한 API 근거와 영문 명세는
[영문 문서](../en/HF_CHECKPOINT_PIPELINE.md)를 참고하세요.

- `torch_dist_training`: native DCP metadata, 모든 모델/optimizer shard,
  scheduler, rank별 RNG, data/token cursor, tokenizer, 학습 manifest,
  환경 명세를 포함한 **신뢰된 전체 재개 번들**입니다.
- `hf_model`: 별도로 변환한 model-only `config.json`, safetensors,
  선택적인 tokenizer 파일입니다. 실제 HF loader 호환성은 별도 시험해야 합니다.

전체 학습 번들은 `from_pretrained` 호환 모델이 **아닙니다**. 기존 학습
디렉터리를 HF 모델 루트에 통째로 복사하지 마세요. model-only 포장은
`.pt`, `.pth`, `.bin`, `.distcp`, `.metadata`와 알려진 학습 state 이름을
거부합니다. WSM model-only 결과는 optimizer 재개 체크포인트가 아닙니다.

## 오프라인 포장

모든 rank가 optimizer update를 완료하고 저장 작업을 끝낸 뒤 source를
동결하세요. 포장기는 복사본의 무결성을 검증하지만 동시 writer를 잠그지
않습니다. 역할 선언 예시는
[training_components.example.json](../../configs/artifacts/training_components.example.json)에
있습니다. 실제 upstream 저장 경로로 교체하고 모든 rank의 상태를 포함하세요.
공통 파일에 state가 함께 저장되면 여러 역할이 같은 파일을 가리킬 수 있습니다.

필수 역할은 `model`, `optimizer`, `scheduler`, `rng`, `cursor`, `tokenizer`,
`training_manifest`, `environment`입니다. model에는 `.metadata`와 `.distcp`
파일이 있어야 하며 scheduler/cursor/training_manifest/environment에는
비어 있지 않은 JSON 객체 명세가 필요합니다. 실제 native binary 상태는
추가 파일로 보존하세요. 선언은 생산자의 의미 계약이지 optimizer 내용이나
정확한 재개를 검증했다는 증거는 아닙니다. 전체 source 파일 목록은 역할
목록과 별개로 모두 hash에 포함됩니다.

```bash
python scripts/hf_checkpoint.py pack \
  --source checkpoints/pretrain/week01/frozen-step-1000 --output checkpoints/packages/pretrain-week01-step1000 \
  --kind torch_dist_training --stage pretrain --run week01 --checkpoint step_1000 \
  --components checkpoints/pretrain/week01/training_components.json
python scripts/hf_checkpoint.py verify \
  --source checkpoints/packages/pretrain-week01-step1000 --kind torch_dist_training
python scripts/hf_checkpoint.py upload --source checkpoints/packages/pretrain-week01-step1000 \
  --repo-id OWNER/nemochat-checkpoints --dry-run
python scripts/hf_checkpoint.py restore-local \
  --source checkpoints/packages/pretrain-week01-step1000 --output checkpoints/restored/local-check \
  --kind torch_dist_training --trusted-inputs
```

위 작업은 네트워크나 인증 정보 없이 실행할 수 있습니다. `--dry-run`은
Hub 접근/공개 여부를 검사하지 않으며 **원격 업로드가 아닙니다**.
`checkpoint_inventory.json`은 모든 shard/index/tokenizer/state 파일의
경로·크기·SHA256을 기록합니다. 누락·추가·변경 파일, unsafe path,
wildcard 문자, 중복 JSON key, symlink, special file, inventory 밖을
참조하는 shard index를 거부합니다. 별도로 준비한 깨끗한 모델 export는
`--kind hf_model`로 포장하고 `--components`를 주지 않습니다.

## 비공개 Hub 전송과 불변 다운로드

별도의 승인된 작업으로 **기존 private model repository**를 준비하세요.
이 파이프라인은 repository를 자동 생성하거나 공개 상태로 바꾸지 않습니다.
`HF_TOKEN`은 임대 플랫폼 secret manager 등을 통해 환경 변수로만 공급하세요.
코드에 토큰 CLI 옵션, login 또는 토큰 저장 기능이 없습니다. shell trace,
SDK debug log, 전체 `os.environ` dump, manifest 내 자격 증명은 금지하세요.
CLI는 SDK 예외에 credential이 포함될 수 있어 상세 예외를 출력하지 않습니다.

```bash
python scripts/hf_checkpoint.py upload --source checkpoints/packages/pretrain-week01-step1000 \
  --repo-id OWNER/nemochat-checkpoints
```

저장 경로는 다음처럼 stage/run/checkpoint/kind별로 분리됩니다.

```text
stages/pretrain/runs/week01/checkpoints/step_1000/torch_dist_training/
```

업로드는 repo privacy를 검사하고 기존 경로 덮어쓰기를 거부하며, 관찰한
parent commit을 전달해 동시 writer 충돌을 방지합니다. 반환된 **전체 commit
SHA**에서 manifest hash, 정확한 원격 파일 목록, 모든 shard의 실제 bytes를
다시 다운로드해 검증한 뒤에만 `repo_id`, `revision`, `path`, `kind`,
`manifest_sha256` 성공 receipt를 발행합니다. receipt는 임대 호스트와 독립된
저장소에 보관하세요. read-back 실패 시 remote commit은 존재할 수 있지만
검증된 성공 receipt는 없습니다. 같은 경로를 덮어쓰는 재시도 대신 불변
commit을 조사하거나 새 체크포인트 이름을 사용하세요.

다음 임대에서는 receipt의 실제 값을 사용합니다.

```bash
python scripts/hf_checkpoint.py download --repo-id OWNER/nemochat-checkpoints \
  --revision FULL_40_CHARACTER_COMMIT_SHA \
  --path stages/pretrain/runs/week01/checkpoints/step_1000/torch_dist_training \
  --output checkpoints/restored/week01-step1000 --kind torch_dist_training \
  --manifest-sha256 MANIFEST_SHA256_FROM_RECEIPT --trusted-inputs
```

`main`, tag, 짧은 SHA, 종류 불일치는 거부합니다. `snapshot_download`는
`local_dir`와 정확한 파일 allow-pattern만 사용합니다. manifest를 먼저 받은
뒤 선언된 파일만 받으며 같은 repo의 다른 checkpoint는 받지 않습니다.
Hub download metadata는 최종 payload 밖에 남습니다. 검증 완료 후 같은
filesystem에서 Linux `renameat2(RENAME_NOREPLACE)`로 새 디렉터리를 원자적으로
공개합니다. 기존 목적지·symlink ancestor·경로 traversal은 거부합니다.
이는 fsync/power-loss 내구성 트랜잭션은 아니며 Linux renameat2가 없는 환경은
지원하지 않습니다. 학습 번들에는 나중에 loader가 실행할 수 있는 pickle
상태가 있으므로 **네트워크 호출 전** `--trusted-inputs` 승인이 필요합니다.
전송 유틸리티 자체는 torch/pickle를 역직렬화하지 않습니다.

공개 repo는 기본적으로 거부합니다. 예외는 `--rights-reviewed`와
`--repo-policy reviewed-policy.json`을 함께 전달해야 하며, 정책의
`rights_reviewed`가 true이고 `allow_public_repo`가 정확히 같은 repo여야 합니다.
다운로드에도 같은 정책이 적용됩니다. 이 정책은 사람의 권리 검토 확인이지
자동 라이선스 승인 시스템이 아닙니다.
[기본 정책](../../configs/artifacts/repo_policy.private.json)은 공개를 허용하지 않습니다.

## 공식 API 조사와 버전 차이

urllib로 [최신 main 업로드 문서](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload),
[0.36.0 문서](https://huggingface.co/docs/huggingface_hub/v0.36.0/en/guides/upload),
[다운로드 문서](https://huggingface.co/docs/huggingface_hub/guides/download)를
읽고 실제 `.venv`의 `huggingface_hub==0.36.2` signature/source를 조사했습니다.
시스템 Python에는 Hub 패키지가 없어 오프라인 기능은 독립적으로 실행됩니다.

- 최신 main 문서는 `upload_folder`의 streamed multi-commit 및 자동 재사용,
  다른 호스트에서의 재시작을 설명합니다.
- 설치된 0.36.2 `upload_folder`는 `create_commit`을 사용하고 `path_in_repo`,
  `parent_commit`, `CommitInfo.oid`를 제공합니다. 구현은 이 API를 사용합니다.
  이 구버전에서 durable task-level 재개 또는 다른 호스트로 upload process
  continuation이 보장된다고 주장하지 않습니다.
- 0.36.2 `upload_large_folder`는 local task cache와 여러 commit으로 재개하지만
  `path_in_repo`, `parent_commit`, 사용자 commit message, 반환 commit SHA가
  없습니다(`None` 반환). 자동 repo 생성의 public 기본값도 주의해야 합니다.
  무한 재시도는 영구 auth 오류를 숨길 수 있습니다. **현재 pipeline에는
  활성화하지 않았습니다.** 통합하려면 repo 상대 경로 tree, cache 제외,
  단일 writer, 최종 불변 SHA 해석, 동일 privacy/full read-back 검증이 필요합니다.
- 존재하지 않는 `resume_upload`/`multi_commits` 인자를 사용하지 않습니다.
  SDK 업그레이드 전 signature와 반환 의미를 재검증하세요. 최신 문서만으로
  구버전 설치 API 동작을 추정하면 안 됩니다.

## 임대 종료 전 필수 gate

1. 학습 종료 직전에 시작하지 말고 업로드 **및 전체 read-back** 시간을 확보합니다.
   source·포장 복사본·download temp·Xet cache를 포함한 disk/Hub quota도 확인합니다.
2. 완료된 update 경계의 전체 번들을 로컬 검증하고 private repo에 업로드합니다.
3. 마지막 검증된 SHA/manifest hash receipt와 코드·환경·학습 명세를 off-host에
   보관합니다. progress bar나 manifest 업로드만으로는 완료가 아닙니다.
4. 임대 반환 전에 SHA를 새 목적지로 독립 복원하여 bytes를 확인하고 가능하면
   실제 trainer resume까지 실행합니다. 다음 검증 완료 전 이전 checkpoint는 유지합니다.
5. 새 B300 임대에서 PyTorch/Megatron/CUDA, TP/PP/DP/EP, 데이터 revision,
   tokenizer 및 global update/cursor/RNG 계약을 맞춥니다. 동일 hash는 다른
   분산 topology 호환성 또는 tensor-exact 재개를 보장하지 않습니다.
6. 데이터 shard는 source에 넣지 않으면 번들에 없습니다. pinned 데이터 revision을
   별도로 복구하세요. Hub history는 retention 정책이나 영구 quota 보장이 아닙니다.

SHA inventory는 손상 검출이지 생산자 인증·권리 승인·악성 manifest 방어가
아닙니다. repo 접근 제어와 생산자를 신뢰해야 하며 Hub CI나 다운로드 성공은
훈련 코드 실행을 신뢰할 근거가 아닙니다.

## 실제 검증 범위

`python -m pytest -q tests/test_hf_checkpoint.py`는 local pack/restore,
모든 shard hash, 경로/index/symlink/덮어쓰기 거부, 인증 정보, privacy,
revision 및 mock API 실패/read-back을 검사합니다. 별도로 실제 filesystem에서
CLI pack→verify→dry-run→restore-local 흐름을 **명시적으로 synthetic인 opaque
학습 번들**로 실행했습니다. 이는 전송·무결성 검증이지 실제 torch 재개,
원격 업로드/다운로드, Hub auth/CI, B300 실행, 모델 loader 호환성, 모델 품질,
여러 주에 걸친 off-host 임대 재개를 검증한 결과가 아닙니다.
