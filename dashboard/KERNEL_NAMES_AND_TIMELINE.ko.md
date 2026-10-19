# 원시 커널명 해석과 iteration 실행 타임라인

원시 이름은 그대로 기본 식별자로 유지하고 설명을 옆에 붙입니다. 신뢰도를 **공개 근거 확인 / 명칭 기반 추정 / 미확인**으로 구분합니다. [영문·공식 근거 링크·실행 기록](KERNEL_NAMES_AND_TIMELINE.md).

## NCCL 이름

`ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)`

- `ncclDevKernel`: NCCL이 생성한 GPU 커널 진입점.
- `AllGather`: 각 rank의 입력을 rank 순서로 모아 모든 rank에 전달하는 연산의 특수화 표기.
- `RING`: rank들을 논리적 링으로 연결해 이웃 rank끼리 데이터 조각을 전달하는 알고리즘. 물리 케이블이 원형이라는 뜻은 아닙니다.
- `LL`: **Low Latency**. 지연 최적화 프로토콜 이름입니다. Ring은 알고리즘, LL은 프로토콜로 서로 다른 선택 축입니다. NVIDIA 설명상 Simple은 대역폭 위주, LL128은 지연·대역폭 절충입니다. 이 실행이 실제 낮은 지연을 달성했다는 뜻은 아닙니다.
- `ncclDevKernelArgsStorage<...>`의 `4096`: **커널 인자 저장 공간의 바이트 용량**. 텐서 원소 수·학습 토큰 수·통신 payload 크기가 아닙니다. `(unsigned long)`도 C++ 템플릿 인자의 타입 표기이지 데이터 dtype이 아닙니다.

**중요한 정정:** 확인한 NCCL v2.26.2-1 생성기는 여러 AllGather/Broadcast 변형을 RING_LL 특수화 진입점에 연결합니다. 실제 work 함수 ID가 다르면 device 함수 테이블로 디스패치합니다. 따라서 이름에 RING_LL이 있다고 **실제 work도 반드시 Ring+LL로 실행됐다**고 확정할 수 없습니다. 캡처 라이브러리 버전 일치도 아직 입증하지 않았습니다. 프로토콜 변경 전 실제 작업/버전을 확인해야 합니다.

## NVJET 이름

`nvjet_sm103_tss_128x128_64x9_2x2_2cta_h_badd_NTT`

- `nvjet`: NVIDIA 직원이 확인한 내부 커널 백엔드 이름. Jetson 전용이라는 뜻은 아닙니다.
- `sm103`: PTX의 `sm_103` / SM10.3 타깃 관례에 연결한 **추정**. 해당 NVJET 토큰의 공개 정의나 실제 실행 GPU 입증은 아닙니다.
- `128x128`: 타일 후보 표기로 **추정**. 실제 입력 M/N/K가 아닙니다.
- `2cta`: 두 CTA(thread block) 협력 개념과의 연결을 **추정**. 2개 커널 실행·물리 클러스터 크기2·특정 명령 사용을 입증하지 않습니다.
- `tss`, `64x9`, `2x2`, `h`, `badd`, `NTT`: 정확한 내부 의미 **미확인**. stage 수, transpose 순서, FP16, bias/add, cluster 형상으로 임의 확장하지 않습니다.

이 실제 SQLite에서 같은 이름의 전체 캡처 launch 두 개를 별도로 확인했습니다: grid128x1x1 / block256x1x1 / **physical cluster4x1x1**, registers255/thread, dynamic shared memory229760bytes. 이름 해석과 launch 메타데이터는 별개입니다. 특히 `2cta=cluster2`, `2x2=physical cluster2x2`라고 볼 수 없습니다. 이는 전체 캡처의 개별 실행 관측이며 측정 iteration 병목/입력 shape 증거가 아닙니다.

## 타임라인 사용

Nsight 패널 → 측정 업데이트 → Rank0 → Update3/4 → HOST 구간을 ‘업데이트 전체’로 두면 선택 iteration 전체가 보입니다.

- HOST NVTX 단계, CPU thread별 연관 API, GPU device/context/stream별 kernel을 같은 상대 시간 축에 표시합니다.
- **GPU 시작순 / CPU API enqueue 순**을 별도로 선택합니다. 서로 다른 stream의 GPU timestamp 순서는 의존 순서가 아닙니다.
- 전체 /100ms /10ms /1ms 창, 시간 이동, lane 페이지,100행 단위 목록을 제공합니다.
- bar 또는 목록을 클릭하면 원시 이름, 실제 ns/ms, CPU/GPU 시간차, stream/context, 정확한 globalTid 문자열, correlationId, phase가 나옵니다.
- 작은 이벤트도 목록에서 보존합니다. 최소1px 표시는 가시성용이며 실제 실행 길이가 아닙니다. 비어 있는 부분을 GPU idle/critical path로 확정하지 않습니다.
- memcpy/memset, 다른 프로세스, 비연관 CPU API, OS 스케줄링, 의존성 그래프, native Python 프레임은 포함하지 않습니다. **전체 Nsight Systems 대체품은 아닙니다.**

## 실제 검증

- 부모가 원본 SQLite를 다시 읽어 새로운 `results/dashboard_timeline/parent-4092da896b57/inspector.json`을 생성했습니다.8.885초/1331517bytes, 기존 자료 보존.
- 측정 update3/4의7025/6834이벤트 모두 보존, truncation 없음. 실제6streams/2CPUthreads와 API 이름을 표시합니다.
- 관련232tests +32subtests 통과, Node/Python/diff 검사 통과. 실제 Chrome에서 정렬·페이지·클릭·zoom·시간 이동·전체 원시 이름·토큰 해석·데스크톱/모바일 overflow를 확인했습니다.
- 새 GPU 학습/프로파일 캡처/attach/collector를 실행하지 않았고 trainer는 변경하지 않았습니다. 분석 SQL은 별도 오프라인 작업이며 브라우저·매 갱신마다 실행하지 않습니다.
- 캐시된 타임라인을 포함한 현재 snapshot은 약4.05MB입니다. 관측한 수집 비용은 약423ms, 캐시 GET은 약44ms로 학습 slowdown 검증은 아닙니다. 반복 전송의 bandwidth 비용은 남아 있으며 lazy transport는 향후 최적화 후보입니다.
- 현재 자료는 독립 native rank0진단입니다. 실제 선택 Bridge pilot·전체 rank·다른 학습 단계나 정확한 Python 호출 프레임까지 자동 수집했다고 주장하지 않습니다.
