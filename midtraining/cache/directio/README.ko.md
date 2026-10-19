# Native codec 보존 O_DIRECT 읽기 전송 — 부모 검토용

**이 패키지와 CPU gate는 GPU 실행을 허가하지 않습니다. CPU 결과가 BLOCKED이면 GPU release도 차단해야 합니다.**

## 변경 범위

원본 cache/teacher/BASE/SDK/continuation03은 수정하지 않습니다. 새 `directio/` 소스, 단위 테스트와 새 acceptance 결과만 추가합니다. local closed-tar allowlist에 등록된 정확한 경로만 `O_RDONLY|O_DIRECT|O_NOFOLLOW`로 열며, statx의 실제 DIO 정렬을 확인합니다. 최대 1 MiB 정렬 버퍼로 읽고, EOF의 비블록 배수 크기와 정렬된 short read를 처리합니다. pre/post fstat 및 경로 identity, 정확한 크기와 예상 SHA가 모두 맞을 때만 immutable bytes → BytesIO를 반환합니다. buffered 재읽기나 fallback은 없습니다.

native `open_logit_file`의 원본 AST와 전체 소스 SHA가 일치할 때만 process-local 바인딩을 바꿉니다. streaming tar / `_meta.json` 검증 / zstd / torch / 17-bit index 복원 / Loss 및 외부 API는 바꾸지 않습니다. Direct fd와 mmap은 worker에서 읽는 동안만 존재합니다. at-fork barrier는 읽기 중 fork가 fd/mapping을 상속하지 않게 합니다.

## CPU 실행 범위와 한계

정확히 고정된 NeMo 이미지에서 network=none, DeviceRequests=[], NVIDIA_VISIBLE_DEVICES=void로 실행합니다. PyTorch DataLoader worker 1개와 pin_memory=True 설정을 실제 사용하되, accelerator가 없는 CPU 환경에서는 PyTorch가 pinning을 자동으로 비활성화합니다. CUDA를 가짜로 구현하지 않습니다.

첫 번째 CPU Engine은 일반 `megatron.core` import가 Transformer Engine의 `libcuda.so.1`을 요구해 실패했습니다. 실패 기록을 보존합니다. 두 번째 시도는 실제 SDK `msc_utils.py`를 직접 import하고, 실제 runtime 및 전체 native `utils_logits.py`를 private package에서 import하여 무관한 GPU package initializer만 우회합니다. AST로 복제한 decoder나 MSC 함수 stub이 아닙니다. 다만 이것은 GPU training entrypoint 전체 import/실행 증거가 아닙니다.

원본 9144개 shard의 예상 SHA는 기존 full audit에서 가져오며, DP7/70만 이미 인증된 재생성 tar/direct 비교의 올바른 SHA를 사용합니다. 원본 파일 자체를 바꾸거나 overlay를 사용하지 않습니다. 모든 rank × iteration을 프로그램으로 중복/누락 검사하고, 예상 SHA, metadata 전체, zstd, ZIP CRC, 실제 native decoder, 16 microbatches, fp16 `[4096,1,32]`, uint16+bool 17-bit 저장과 ID 범위 `0..131071`, finite 값을 검사합니다. 기대 SHA와 다른 파일은 decoder로 넘기지 않고 blocking file로 남깁니다. 예상 SHA를 결과에 맞춰 바꾸면 안 됩니다.

## 다음 GPU generation: 부모만 준비/실행

영문 README의 최소 통합 절차를 따릅니다. 먼저 전체 COMPLETE/clean gate와 정확한 Engine Exit 0을 부모가 직접 검토해야 합니다. 새 scratch SDK/staged 소스, 새 output/HOME/compiler cache를 만들고, 복사한 launcher 소스만 수정합니다. CACHE는 symlink overlay가 아닌 원본 allowlist 디렉터리로 설정합니다. native reader 생성 전에 `integration.activate(..., opt_in=True)`를 적용합니다. 기존 소스나 설치된 SDK를 직접 고치지 않습니다.

원본 완료 BASE의 model-only step 0 시작, optimizer/RNG/cursor reset, full sampler 468736, WSD 전체 3662 / warmup 37 / decay 366, 완료 prefix 1143 / 599261184 positions를 유지합니다. metadata/cacheID, teacher, Top32/ghost-tail/alpha .5도 보존합니다. checkpoint payload SHA 면제는 cache 검증과 별개입니다. 이 작업에서는 weight payload를 읽거나 hash하지 않습니다.

현재 제공물은 검토 가능한 통합 API와 계획이며, 실행/예약된 GPU supervisor가 아닙니다. 실제 source-only diff 검토, GPU readiness, 새로운 launch, all-rank 샘플/optimizer update/finite gradient/save/Engine 종료 검증은 부모 책임입니다.

## READPATH 완화이지 하드웨어 수리 증거가 아님

O_DIRECT는 관측된 buffered read 경로를 우회할 뿐입니다. 올바른 direct read나 GPU ECC 0만으로 host 정상 또는 kernel/storage/hardware 원인을 확정할 수 없습니다. fadvise, dropcache, OS 상태 변경, reset/reboot, input 교체, teacher 재생성은 하지 않습니다. 새 direct SHA 불일치는 원인 미확정 blocker로 보존하고 GPU를 release하지 않습니다.
