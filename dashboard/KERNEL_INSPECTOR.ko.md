# 커널 → 연산 → Python 소스 인스펙터

[English](KERNEL_INSPECTOR.md). 기본 화면은 학습 모니터이며, 여기에 기존 Nsight Bridge처럼 연산·소스·근거를 드릴다운하는 인스펙터를 붙입니다. 별도 GPU profiler나 native Nsight GUI plugin을 구현했다고 주장하지 않습니다.

## 무엇을 우선 조사할지 보여주기

학습 단계·캡처·측정 업데이트·구간을 선택하면 **그 구간에 실제로 연결된 커널**의 시간·횟수 순위를 표시합니다. 의미가 확인되는 연산 계열은 쉬운 설명을 붙이고, 저장된 소스 근거가 있으면 실제 코드와 행 번호를 보여줍니다.

전체 capture 순위와 학습 update 순위를 분리합니다. 전체 통계에 초기화·체크포인트 로딩이 포함되면 Fill/AllGather가 크게 보일 수 있어 학습 최적화 대상을 잘못 고를 수 있습니다.

우선 조사 카드에는 **관측 값·범위 → 개선 가설 → 다음 실험**을 제시합니다. CUDA 누적 시간만으로 critical path, 전체 노드 불균형, 최적화 효과를 확정하지 않습니다. 변경 채택에는 정확성·메모리·비프로파일링 전체 업데이트 A/B가 필요합니다.

## 소스 근거를 구분합니다

1. **커널 명칭 기반 참고 설명**
   - 통신·reduction·copy/cast·TopK·permutation·행렬·scan 등의 연산 계열 설명입니다.
   - 실제 Python caller, shape, model layer, 병목을 입증하지 않습니다. 불명확한 vendor symbol은 미분류로 둡니다.
2. **실제 CUDA launch 연결**
   - correlationId와 process ID로 커널과 CUDA API 호출을 연결합니다.
   - 같은 process의 API가 NVTX 구간 안에 있다는 사실은 구간 연관입니다. 정확한 thread nesting이나 Python stack frame과 같지 않습니다.
3. **검증된 host 구간 소스**
   - 저장된 파일 hash를 확인하고 `model(...)`, `loss.backward()/finalize_model_grads(...)`, `opt.step()`을 감싸는 실제 NVTX 구간 코드를 보여줍니다.
   - 해당 host wrapper 코드의 검증이며 내부 커널의 정확한 Python 호출 프레임은 아닙니다.
4. **기존 Bridge의 검증된 host 연산**
   - plugin의 파일 hash·snippet hash·source ID·행 범위 검증을 재사용해 실제 저장된 함수 코드를 보여줍니다.
   - 교육용 sparse-MoE control의 명시적 연산 설명이며 native Nemotron/MCore 커널 매핑이 아닙니다.
5. **실제 Python call stack / compiler 매핑**
   - native 호출 위치 확정에는 실제 캡처된 frame/launch/compiler 정보와 source 버전 근거가 필요합니다.
   - 그럴듯한 라이브러리 파일명, 현재 checkout, 생성한 예제 코드로 대체하지 않습니다.

소스 카드에는 종류·검증 상태·경로·함수·행 범위·실제 저장된 코드·hash를 표시합니다. 브라우저가 임의 source 파일을 읽도록 하지 않고, 상대 경로·symlink·크기·hash/identity 오류를 검증합니다. hash만으로 producer가 악의적으로 파일과 manifest를 함께 바꾼 경우까지 인증할 수는 없습니다.

## 현재 캡처의 한계와 추가 데이터

현재 native software trace는 **별도의 native MCore 진단 실행**입니다. 선택한 Bridge pilot과 동일 실행의 capture라고 간주하지 않습니다. 분석 가능한 범위는 rank 0의 측정 update 3/4이며 다른 rank 완료 기록은 전체 노드 trace가 아닙니다.

부모가 SQLite를 읽기 전용으로 확인한 결과 **CUDA runtime 기록 3,569,543개 중 callchainId가 있는 기록은 0개**이고 export된 stack/symbol 테이블도 없습니다. 수집하지 않은 native Python 호출 프레임을 지금 파일 가공만으로 복원할 수는 없습니다. 대신 실제 host 구간과 저장된 harness 코드로 조사 경로를 보여줄 수 있습니다.

기존 capture는 CPU sampling을 껐습니다. 설치된 **Nsight Systems 2026.5.1** help에는 Python/CUDA backtrace 옵션과 tracing/sampling 전제 조건이 있습니다. 정확한 native 호출 위치를 추가하려면 승인된 짧은 진단 capture가 필요하며, 실행 중인 trainer에 자동으로 profiler를 붙이지 않습니다. help에서 옵션이 보인다는 사실만으로 이 호스트의 stack 수집 성공이나 software tracing 호환성을 검증했다고 하지 않습니다. 오버헤드·권한도 별도로 확인해야 합니다.

## 학습 간섭을 줄이는 처리 구조

```
기존 SQLite + 선택한 NVTX 구간
       → 별도 읽기 전용 분석
       → 제한된 inspector.json
       → live 대시보드가 캐시해서 표시

기존 Bridge operation source 참조
       → 저장된 snapshot/hash/ID 검증
       → 실제 코드 카드
```

무거운 분석을 매 브라우저 연결이나 매 수집 주기마다 실행하지 않습니다. 소스 snapshot을 실행하지 않으며 trainer callback·monkeypatch·필수 writer·GPU 실행·신규 capture·collector를 이 기능 때문에 추가하지 않습니다. 기존 plugin은 선택적 수집/annotation을 맡고, 대시보드는 결과를 소비합니다.

입력 계약은 `INSPECTOR_CONTRACT.md`, 지원 명령은 `capture_analysis.py --help`를 기준으로 합니다. 실제 분석·소스·브라우저 검증은 `INSPECTOR_VERIFICATION.md`에 기록했습니다. 현재 native 어댑터는 저장된 rank0측정 update3/4와 명시적인 phase 이름을 지원하며, 다른 naming/selection은 별도 어댑터가 필요합니다. 출력은 새 경로에 만들고 실패 근거를 보존합니다. 권고를 단정적으로 보이게 하려고 미확인 매핑을 확정으로 바꾸지 않습니다.
