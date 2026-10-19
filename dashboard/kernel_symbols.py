"""Bounded raw CUDA symbol references; never caller or launch attribution.

Sources are fixed primary references, not URLs supplied by capture artifacts.
Pinned NCCL source is a reference version, not a claim about the installed build.
"""

import html
import re
import unicodedata

MAX_SYMBOL_CHARS = 8192
MAX_PARTS = 64
MAX_NESTING = 16

NCCL_SOURCE = 'https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/device/generate.py'
NCCL_STORAGE = 'https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/include/device.h'
NCCL_COLLECTIVES = 'https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2262/user-guide/docs/usage/collectives.html'
NCCL_ENV = 'https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2262/user-guide/docs/env.html'

NCCL_TUNING = 'https://developer.nvidia.com/blog/understanding-nccl-tuning-to-accelerate-gpu-to-gpu-communication/'
NCCL_COMMON = 'https://github.com/NVIDIA/nccl/blob/v2.26.2-1/src/device/common.h'
NVJET_REFERENCE = 'https://forums.developer.nvidia.com/t/purpose-of-these-kernels/328273/2'
PTX_REFERENCE = 'https://docs.nvidia.com/cuda/archive/13.1.1/parallel-thread-execution/index.html'
TILE_REFERENCE = 'https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html'

_NVJET_TOKENS = {
    'nvjet': ('NVIDIA 내부 백엔드', 'NVIDIA 직원의 공개 답변은 nvjet을 NVIDIA 내부 커널 백엔드로 설명합니다. 각 접미사의 의미까지 문서화한 답변은 아닙니다.', 'documented', [NVJET_REFERENCE]),
    'sm103': ('아키텍처 표기 추정', 'PTX는 sm_103 타깃을 문서화합니다. sm103이 이를 가리킨다는 것은 이름 관례에 따른 추정이며 이 NVJET 토큰의 정확한 의미나 실행 GPU를 확정하지 않습니다.', 'inferred', [PTX_REFERENCE]),
    '128x128': ('타일 후보 표기', 'NVIDIA는 일반 cuBLAS 타일 예로 128x128을 문서화합니다. 이 NVJET 토큰도 타일 후보라는 추정이며 정확한 타일 배치나 실제 입력 M/N/K를 확정하지 않습니다.', 'inferred', [TILE_REFERENCE]),
    '2cta': ('CTA 그룹 후보 표기', 'PTX는 cta_group::2의 두 CTA 그룹 동작을 문서화합니다. 이 NVJET 토큰과의 연결은 추정이며 실제 블록 수·클러스터 구성·명령 사용을 확정하지 않습니다.', 'inferred', [PTX_REFERENCE]),
}

_BASE_WARNING = '커널 심볼 참조 해석이며 Python 호출자나 입력 텐서 shape·dtype의 증거가 아닙니다.'
_NCCL_TOKENS = {
    'ncclDevKernel': ('NCCL 장치 커널', 'NCCL 소스가 생성하는 장치 커널 이름 접두사입니다. 참조 구현은 이름의 특수화와 다른 작업을 함수 테이블로 디스패치할 수도 있습니다.', [NCCL_SOURCE, NCCL_COMMON]),
    'AllGather': ('전체 수집', '모든 rank의 데이터를 rank 인덱스 순서로 모아 모든 rank에 전달하는 AllGather 연산 참조입니다.', [NCCL_COLLECTIVES]),
    'RING': ('링 알고리즘', 'rank들을 논리적 링으로 연결해 이웃 rank 사이에서 데이터 조각을 전달하는 NCCL Ring 알고리즘 이름입니다. 물리 배선이 원형이라는 뜻은 아니며 LL 등의 프로토콜과 별도 선택 축입니다.', [NCCL_SOURCE, NCCL_ENV, NCCL_TUNING]),
    'LL128': ('LL128 프로토콜', 'LL128은 LL과 Simple 사이의 지연·대역폭 절충을 설명하는 NCCL 통신 프로토콜입니다. 이름의 128은 입력 텐서 길이가 아닙니다.', [NCCL_ENV, NCCL_SOURCE, NCCL_TUNING]),
    'SIMPLE': ('Simple 프로토콜', 'NCCL의 대역폭 최적화 Simple 통신 프로토콜 표기입니다. RING 알고리즘과 별도 선택 축이며 이 실행의 성능을 보증하지 않습니다.', [NCCL_ENV, NCCL_SOURCE, NCCL_TUNING]),
    'Simple': ('Simple 프로토콜', 'NCCL 문서의 대역폭 최적화 Simple 통신 프로토콜 표기입니다. 입력 텐서 크기나 이 실행의 성능 증거가 아닙니다.', [NCCL_ENV, NCCL_TUNING]),
    'LL': ('LL · Low Latency 프로토콜', 'Low Latency: 지연을 줄이는 방향으로 설계된 NCCL 통신 프로토콜 이름입니다. RING 알고리즘이나 텐서 길이를 뜻하지 않으며 이 실행의 낮은 지연을 보증하지 않습니다.', [NCCL_ENV, NCCL_TUNING]),
}


def _part(token, label='미확인 토큰', meaning='공개 참조에서 이 토큰의 정확한 의미를 확인하지 못했습니다.', confidence='unknown', sources=()):
    return {'token': token, 'label': label, 'meaning': meaning,
            'confidence': confidence, 'sources': list(sources)}


def _parsing_copy(raw):
    """Decode entities once, tracking raw source ranges for every character.

    No Unicode normalization, case folding, demangling, or recursive unescape.
    The returned copy is private to parsing; output tokens use raw slices.
    """
    chars, ranges = [], []
    cursor = 0
    for match in re.finditer(r'&(?:#[0-9]{1,16}|#[xX][0-9A-Fa-f]{1,16}|[A-Za-z][A-Za-z0-9]{0,31});', raw):
        for index in range(cursor, match.start()):
            chars.append(raw[index])
            ranges.append((index, index + 1))
        decoded = html.unescape(match[0])
        # Preserve an invalid entity that unescape would silently remove.
        for char in decoded or match[0]:
            chars.append(char)
            ranges.append(match.span())
        cursor = match.end()
    for index in range(cursor, len(raw)):
        chars.append(raw[index])
        ranges.append((index, index + 1))
    return ''.join(chars), ranges


def _raw_slice(raw, ranges, start, end):
    return raw[ranges[start][0]:ranges[end - 1][1]] if start < end else ''


def _well_formed(name):
    if any(unicodedata.category(char).startswith('C') for char in name):
        return False
    stack = []
    closing = {')': '(', '>': '<', ']': '['}
    for index, char in enumerate(name):
        if char in '(<[':
            stack.append(char)
            if len(stack) > MAX_NESTING:
                return False
        elif char in closing:
            if not stack or stack.pop() != closing[char]:
                return False
            if name.startswith('ncclDevKernel_') and char == ')' and not stack and index != len(name) - 1:
                return False
    if stack:
        return False
    if name.startswith(('ncclDevKernel_', 'nvjet_')):
        head = name.split('(', 1)[0] if name.startswith('ncclDevKernel_') else name
        return re.fullmatch(r'[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*', head) is not None
    return True


def decode_symbol(name: str) -> dict:
    """Return annotations alongside the unchanged caller-owned raw name."""
    result = {'family': 'other', 'evidence': 'mixed_symbol_reference',
              'parts': [], 'warnings': [_BASE_WARNING]}
    if type(name) is not str or not name or len(name) > MAX_SYMBOL_CHARS:
        result['warnings'].append('문자열 입력 또는 길이 한도(1~8192자)를 벗어나 해석하지 않습니다. 원본 이름은 변경하지 않습니다.')
        return result
    raw_name = name
    name, ranges = _parsing_copy(raw_name)
    if name != raw_name:
        result['warnings'].append('HTML 엔티티는 별도 파싱 사본에서 한 번만 해석합니다. 표시 토큰과 원본 식별자는 변경하지 않습니다.')
    if not _well_formed(name):
        result['warnings'].append('심볼 형식이 잘못되었거나 중첩 한도(16)를 벗어나 해석하지 않습니다. 원본 이름은 변경하지 않습니다.')
        return result
    head = name.split('(', 1)[0]
    count = head.count('_') + 1
    if name.startswith('ncclDevKernel_'):
        count += 2 if '(' in name else 0
    if name.startswith(('ncclDevKernel_', 'nvjet_')) and count > MAX_PARTS:
        result['warnings'].append('토큰 수 한도(64)를 벗어나 부분 해석하지 않습니다. 원본 이름은 변경하지 않습니다.')
        return result
    if name.startswith('ncclDevKernel_'):
        result['family'] = 'NCCL'
        result['warnings'].extend([
            'NCCL v2.26.2-1 참조이며 캡처 라이브러리 버전 일치를 보증하지 않습니다.',
            '이 참조 소스의 특수화 커널은 함수 테이블로 다른 작업을 디스패치할 수 있어 RING·LL 이름만으로 실제 실행 알고리즘·프로토콜을 확정할 수 없습니다.',
        ])
        for match in re.finditer(r'[^_]+', name.split('(', 1)[0]):
            token = _raw_slice(raw_name, ranges, *match.span())
            entry = _NCCL_TOKENS.get(match[0])
            result['parts'].append(_part(token, *entry[:2], 'documented', entry[2]) if entry else _part(token))
        if '(' in name:
            argument_start = name.index('(') + 1
            argument = name[argument_start:-1]
            raw_argument = _raw_slice(raw_name, ranges, argument_start, len(name) - 1)
            storage = re.fullmatch(r'ncclDevKernelArgsStorage<\s*(?:\(unsigned long\)\s*)?([0-9]{1,16})(?:[uU](?:[lL]{1,2})?|[lL]{1,2}[uU]?)?\s*>', argument)
            if storage:
                capacity_token = _raw_slice(raw_name, ranges, argument_start + storage.start(1), argument_start + storage.end(1))
                result['parts'].append(_part(raw_argument, '커널 인자 저장 타입', 'capacity는 커널 인자 저장 공간의 바이트 용량이며 입력 텐서 길이가 아닙니다.', 'documented', [NCCL_STORAGE]))
                result['parts'].append(_part(capacity_token, '인자 저장 용량', f'{storage[1]} 바이트의 커널 인자 저장 capacity입니다. 통신 텐서의 원소 수가 아니며 해당 빌드에서 이 크기를 사용한다는 보증도 아닙니다.', 'documented', [NCCL_STORAGE]))
            elif argument:
                result['parts'].append(_part(raw_argument))
    elif name.startswith('nvjet_'):
        result['family'] = 'nvjet'
        result['warnings'].append('공개 자료는 이 전체 NVJET 심볼의 명명 규칙을 정의하지 않습니다. 일반 PTX·cuBLAS 문서와의 연결은 추정이며 숫자는 실제 입력 크기 증거가 아닙니다.')
        result['warnings'].append('2cta·2x2 표기는 실제 launch 횟수나 grid/block/클러스터 차원의 증거가 아닙니다. 기록된 launch 메타데이터는 이름 해석과 별도로 확인해야 합니다.')
        for match in re.finditer(r'[^_]+', name):
            token = _raw_slice(raw_name, ranges, *match.span())
            entry = _NVJET_TOKENS.get(match[0])
            result['parts'].append(_part(token, *entry) if entry else _part(token, 'NVJET 불투명 토큰', '이 NVJET 내부 토큰의 정확한 의미는 공개 참조에서 미확인입니다. dtype·레이아웃·연산·튜닝 인자로 확장하지 않습니다.'))
    else:
        result['warnings'].append('지원하지 않는 심볼 계열 또는 버전 표기입니다. 부분 문자열만으로 의미를 추정하지 않습니다.')
        result['parts'].append(_part(raw_name))
    return result
