"""Offline, read-only measured NVTX/runtime/kernel analysis. Never launches a profiler."""
from collections import defaultdict
from contextlib import closing
import sqlite3
import time

MASK = 0xFFFFFFFFFF000000
MAX_ROWS = 1_000_000
MAX_RUNTIME_ROWS = 10_000_000
QUERY_SECONDS = 20
TIMELINE_FIELDS = ['row_id','name_index','gpu_start_ns','gpu_end_ns','device_id',
    'context_id','stream_id','correlation_id','api_start_ns','api_end_ns',
    'thread_index','phase_index','api_name_index']
TIMELINE_MAX_EVENTS = 16384
TIMELINE_MAX_NAMES = 4096
TIMELINE_MAX_API_NAMES = 512
TIMELINE_MAX_THREADS = 256


def _timeline_integer(value, signed=False):
    if type(value) is not int or not -(2**53 - 1) <= value < 2**53 or (not signed and value < 0):
        raise ValueError('Invalid JavaScript-safe timeline integer')
    return value


def _timeline(event, kernels, phase_events):
    origin = event['start']
    phases = [dict(id=p['name'].rsplit('/',1)[1],
        start_ns=_timeline_integer(p['start']-origin),end_ns=_timeline_integer(p['end']-origin))
        for p in phase_events]
    names, api_names, threads, events = [], [], [], []
    dictionaries = [{}, {}, {}]
    span = _timeline_integer(event['end']-origin)
    for row in sorted(kernels,key=lambda r:(r['start'],r['kernel_row'])):
        gpu_start = _timeline_integer(row['start']-origin,True)
        gpu_end = _timeline_integer(row['end']-origin,True)
        api_start = _timeline_integer(row['api_start']-origin)
        api_end = _timeline_integer(row['api_end']-origin)
        if gpu_end<gpu_start or not 0<=api_start<=api_end<=event['end']-origin:
            raise ValueError('Invalid timeline GPU/API duration or containment')
        span = max(span,gpu_end)
        tid = row['globalTid']
        if type(tid) is not int or not 0 <= tid < 2**64:
            raise ValueError('Invalid timeline globalTid')
        values = [row['name'],row['api_name'],str(tid)]
        lists = [names,api_names,threads]
        caps = [TIMELINE_MAX_NAMES,TIMELINE_MAX_API_NAMES,TIMELINE_MAX_THREADS]
        if len(events) >= TIMELINE_MAX_EVENTS or any(v is not None and v not in d and len(d)>=cap
                for v,d,cap in zip(values,dictionaries,caps)):
            continue
        indices=[]
        for value,dictionary,items in zip(values,dictionaries,lists):
            if value is None:
                indices.append(None)
                continue
            if not isinstance(value,str) or len(value)>8192 or '\x00' in value:
                raise ValueError('Invalid timeline dictionary name')
            if value not in dictionary:
                dictionary[value]=len(items)
                items.append(value)
            indices.append(dictionary[value])
        memberships=[i for i,p in enumerate(phases) if p['start_ns']<=api_start<=api_end<=p['end_ns']]
        nullable = lambda key: None if row[key] is None else _timeline_integer(row[key])
        events.append([_timeline_integer(row['kernel_row']),indices[0],gpu_start,gpu_end,
            _timeline_integer(row['deviceId']),nullable('context_id'),nullable('stream_id'),
            _timeline_integer(row['correlationId']),api_start,api_end,indices[2],
            memberships[0] if len(memberships)==1 else None,indices[1]])
    return dict(schema_version=1,available=True,origin='selected_host_nvtx_start',
        host_duration_ns=_timeline_integer(event['end']-origin),span_duration_ns=span,
        order='gpu_start_ns,row_id',total_events=len(kernels),returned_events=len(events),
        truncated=len(events)!=len(kernels),names=names,api_names=api_names,threads=threads,
        phases=phases,fields=list(TIMELINE_FIELDS),events=events)


def union_ns(intervals):
    total = 0
    left = right = None
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if left is None:
            left, right = a, b
        elif a > right:
            total += right - left
            left, right = a, b
        else:
            right = max(right, b)
    return total + (right - left if left is not None else 0)


def semantic(name):
    """Symbol-family reference only: neither captured Python source nor causality."""
    value = name.lower()
    if 'nccl' in value:
        meanings = [
            ('allgather', 'AllGather: 각 rank의 입력을 rank 순서로 모아 모든 rank에 전달합니다'),
            ('allreduce', 'AllReduce: rank들의 값을 합/최대 등으로 축약하고 모든 rank가 결과를 받습니다'),
            ('reducescatter', 'ReduceScatter: rank 값을 축약한 뒤 결과 조각을 rank별로 나눕니다'),
            ('alltoall', 'AllToAll: 각 rank가 다른 rank들에 서로 다른 조각을 교환합니다'),
            ('sendrecv', 'SendRecv: peer rank 사이 송수신을 수행합니다; 상위 collective/모델 역할은 미확인'),
            ('broadcast', 'Broadcast: 한 root rank의 값을 다른 rank들에 배포합니다'),
        ]
        meaning = next((message for token, message in meanings if token in value), 'NCCL: GPU 간 통신 심볼; 구체적인 연산은 미분류')
        return 'NCCL', meaning + ' · 명칭 기반 참고 설명; Python caller/shape/대기 원인은 미확인', 'symbol_reference'
    if 'l2norm' in value:
        return 'reduction', 'L2 norm: 텐서 목록의 원소 제곱합을 축약해 노름을 계산하는 계열; gradient norm에 쓰였는지와 Python caller는 미확인', 'symbol_reference'
    if 'scaleptrfunctor' in value or 'multi_tensor_scale' in value:
        return 'fill_copy_cast', 'Tensor scaling: 여러 텐서의 원소에 배율을 곱하는 계열; gradient clipping 용도와 Python caller는 미확인', 'symbol_reference'
    families = [
        ('chunk_scan', 'mamba_chunk_scan', 'Mamba 계열 chunk scan 심볼; 정확한 호출 소스는 미확인'),
        ('topk', 'TopK', '상위 K 값/인덱스 선택 계열; 실제 모델 역할은 미확인'),
        ('top_k', 'TopK', '상위 K 값/인덱스 선택 계열; 실제 모델 역할은 미확인'),
        ('permute', 'permute', '데이터 순열/재배치 계열; 정확한 호출 소스는 미확인'),
        ('reduce', 'reduction', '축약 연산 계열; 정확한 호출 소스는 미확인'),
        ('reduction', 'reduction', '축약 연산 계열; 정확한 호출 소스는 미확인'),
        ('fill', 'fill_copy_cast', '값 채우기 계열; 초기화와 학습 중 용도를 이름만으로 구분할 수 없음'),
        ('copy', 'fill_copy_cast', '데이터 복사 계열; 정확한 호출 소스는 미확인'),
        ('cast', 'fill_copy_cast', '자료형 변환 계열; 정확한 호출 소스는 미확인'),
        ('gemm', 'GEMM', '행렬 곱 계열; 정확한 호출 소스는 미확인'),
        ('cutlass', 'opaque_vendor', 'vendor 템플릿 커널; 이름만으로 정확한 연산을 확정하지 않음'),
        ('cublas', 'opaque_vendor', 'vendor 라이브러리 커널; 정확한 호출 소스는 미확인'),
    ]
    for token, category, meaning in families:
        if token in value:
            return category, meaning, 'symbol_reference'
    return 'unknown', '미분류 원시 심볼; 연산 의미와 소스는 미확인', 'unclassified'


def _rows(db, sql, args=()):
    if QUERY_SECONDS <= 0:
        raise sqlite3.OperationalError('interrupted: query budget expired')
    deadline = time.monotonic() + QUERY_SECONDS
    db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    rows = db.execute(sql + f' LIMIT {MAX_ROWS + 1}', args).fetchall()
    if len(rows) > MAX_ROWS:
        raise ValueError('Query row limit exceeded')
    return rows


def _range(db, name):
    rows = _rows(db, '''SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) name
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE COALESCE(n.text,s.value)=?''', (name,))
    if len(rows) != 1:
        raise ValueError('Missing or ambiguous NVTX range')
    row = dict(rows[0])
    if any(type(row[k]) is not int for k in ('start','end','globalTid')) or not 0 <= row['start'] < row['end']:
        raise ValueError('Invalid NVTX range')
    return row


def _scope(db, event, tables, phase_events=None):
    pid = event['globalTid'] & MASK
    args = (pid, event['start'], event['end'])
    kernel_columns = {r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
    runtime_columns = {r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME)')}
    context = 'k.contextId' if 'contextId' in kernel_columns else 'NULL'
    stream = 'k.streamId' if 'streamId' in kernel_columns else 'NULL'
    api_name = 'a.value' if 'nameId' in runtime_columns else 'NULL'
    api_join = 'LEFT JOIN StringIds a ON a.id=rt.nameId' if 'nameId' in runtime_columns else ''
    kernels = _rows(db, f'''SELECT k.rowid kernel_row,k.start,k.end,k.deviceId,k.correlationId,
        s.value name,rt.globalTid,rt.start api_start,rt.end api_end,
        {context} context_id,{stream} stream_id,{api_name} api_name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME rt ON k.correlationId=rt.correlationId
        AND k.globalPid=(rt.globalTid & 0xFFFFFFFFFF000000)
        LEFT JOIN StringIds s ON s.id=k.demangledName {api_join}
        WHERE (rt.globalTid & 0xFFFFFFFFFF000000)=? AND rt.start>=? AND rt.end<=?''', args)
    if len({r['kernel_row'] for r in kernels}) != len(kernels):
        raise ValueError('Ambiguous runtime/kernel correlation')
    ids = sorted({r['correlationId'] for r in kernels})
    if len(ids)>32766:
        raise ValueError('Correlation identifier query bound exceeded')
    if ids:
        # Reject process-local correlation reuse even if the competing API lies
        # outside this host scope. Temporal filtering must not hide ambiguity.
        placeholders=','.join('?' for _ in ids)
        duplicates=_rows(db,f'''SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE (globalTid & 0xFFFFFFFFFF000000)=? AND correlationId IN ({placeholders})
            GROUP BY correlationId HAVING count(*)>1''',(pid,*ids))
        if duplicates:
            raise ValueError('Ambiguous process-local runtime correlation identifier')
    grouped = defaultdict(list)
    for row in kernels:
        if not isinstance(row['name'], str) or len(row['name']) > 8192 or row['end'] < row['start']:
            raise ValueError('Invalid kernel name or duration')
        grouped[row['name']].append(row)
    total = sum(r['end'] - r['start'] for r in kernels)
    aggregates = []
    for name, rows in grouped.items():
        duration = sum(r['end'] - r['start'] for r in rows)
        category, meaning, evidence = semantic(name)
        aggregates.append(dict(name=name, category=category, meaning=meaning, meaning_evidence=evidence,
            count=len(rows), total_ms=duration / 1e6, average_us=duration / len(rows) / 1e3,
            percent=duration / total * 100 if total else 0,
            worker_launches=sum(r['globalTid'] != event['globalTid'] for r in rows),
            source_attribution='unavailable', source=None))
    aggregates.sort(key=lambda r: (-r['total_ms'], r['name']))
    apis = _rows(db, '''SELECT start,end,globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE (globalTid & 0xFFFFFFFFFF000000)=? AND start>=? AND end<=?''', args)
    by_device = defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
        if table not in tables:
            continue
        activities = _rows(db, f'''SELECT start,end,deviceId FROM {table}
            WHERE globalPid=? AND end>? AND start<?''', args)
        for row in activities:
            by_device[row['deviceId']].append((max(event['start'],row['start']),min(event['end'],row['end'])))
    result = dict(name=event['name'],host_ms=(event['end']-event['start'])/1e6,
        kernel_count=len(kernels),summed_kernel_ms=total/1e6,activity_union_ms=None,
        same_process_contained_api_count=len(apis),
        worker_thread_correlated_kernel_count=sum(r['globalTid']!=event['globalTid'] for r in kernels),
        activity_by_device=[dict(device_id=d,global_pid=pid,activity_union_ms=union_ns(v)/1e6)
            for d,v in sorted(by_device.items())],kernels=aggregates[:64],
        displayed_kernel_count=sum(k['count'] for k in aggregates[:64]),
        unique_kernel_names=len(aggregates))
    if phase_events is not None:
        result['timeline'] = _timeline(event,kernels,phase_events)
    return result


def _investigation_action(kernel):
    name = kernel['name'].lower()
    if 'allgather' in name:
        focus = '다른 rank의 준비/대기 시간과 통신 group·bucket 크기를 확인; 파라미터 gather인지 스택으로 검증하고 지원되는 gather overlap/bucket 설정 하나만 비교'
    elif 'sendrecv' in name:
        focus = 'peer 송수신의 반대편 rank와 launch 의존성을 대조; EP dispatch/combine 등 상위 역할을 스택으로 확인하고 지원되는 통신 overlap 설정 하나만 비교'
    elif kernel['category'] == 'fill_copy_cast':
        focus = '반복 복사·cast의 실제 tensor 크기와 버퍼 수명을 확인; 중복 변환 제거나 batching 후보 하나를 선택하고 optimizer/checkpoint 의미를 보존'
    elif kernel['category'] == 'reduction':
        focus = '축약 대상 크기·호출 빈도와 동기화를 확인; 묶음 축약/fusion 후보 하나를 선택하되 수치·gradient 의미를 보존'
    elif kernel['category'] == 'GEMM':
        focus = '실제 M/N/K·dtype·padding과 라이브러리 알고리즘을 확보; shape/precision이 같은 GEMM 선택 또는 padding 후보 하나만 비교'
    elif kernel['category'] in ('permute', 'TopK', 'mamba_chunk_scan'):
        focus = '실제 입력 크기와 커널 호출 스택을 확보; 재배치/선택/scan의 메모리 이동과 반복 launch를 분리해 batching/fusion 후보 하나만 비교'
    else:
        focus = 'Python/CUDA 호출 스택과 launch/stream 의존성을 먼저 확보; 미분류 심볼만으로 변경 대상을 정하지 않음'
    return focus + '; 정확도·메모리·비계측 전체 업데이트 A/B로 검증 (속도 향상 미확정).'


def _candidates(updates):
    cards = []
    for update in updates:
        for phase in update['phases']:
            for kernel in phase['kernels'][:2]:
                cards.append(dict(title=f"{phase['id']}: {kernel['category']} 조사 후보",
                    rank=update['rank'], update=update['update'], phase=phase['id'], category=kernel['category'],
                    evidence=f"선택 update {update['update']}, {phase['id']}: {kernel['name']}; {kernel['count']} calls, {kernel['total_ms']:.6f} ms summed ({kernel['percent']:.3f}% of phase sum); host {phase['host_ms']:.6f} ms, {phase['same_process_contained_api_count']} APIs",
                    interpretation='합산 시간은 중첩 가능; 호스트 시간은 대기/계측 포함. critical path, 전체 노드 병목 또는 speedup 확정 아님.',
                    action=_investigation_action(kernel),
                    confidence='candidate',source_confidence='unavailable',observed_summed_kernel_ms=kernel['total_ms']))
    cards.sort(key=lambda c: -c['observed_summed_kernel_ms'])
    for i, card in enumerate(cards[:12], 1):
        card['priority'] = i
    return cards[:12]


def validate_selection(selection):
    if not isinstance(selection, dict) or selection.get('status') != 'coverage-verified':
        raise ValueError('Selection must be coverage-verified')
    updates = selection.get('updates')
    if not isinstance(updates, list) or not 1 <= len(updates) <= 8:
        raise ValueError('Invalid selected update count')
    seen = set()
    for update in updates:
        if not isinstance(update, dict) or type(update.get('rank')) is not int or update['rank'] != 0 or type(update.get('update')) is not int or update['update'] not in (3,4):
            raise ValueError('Only measured rank0 updates 3/4 supported')
        key = (update['rank'],update['update'])
        if key in seen or update.get('name') != f'native/rank_{key[0]}/measured/update_{key[1]}':
            raise ValueError('Duplicate or inconsistent update selector')
        seen.add(key)
        phases = update.get('phases')
        if not isinstance(phases,list) or not 1 <= len(phases) <= 16:
            raise ValueError('Invalid phases')
        names = set()
        for phase in phases:
            name = phase.get('name') if isinstance(phase,dict) else None
            valid = {f'native/rank_{key[0]}/update_{key[1]}/phase/{p}' for p in ('data','forward','backward+sync','optimizer')}
            if name not in valid or name in names:
                raise ValueError('Invalid or duplicate phase selector')
            names.add(name)


def analyze_capture(root, path, selection):
    validate_selection(selection)
    if path.stat().st_size > 2 * 1024**3:
        raise ValueError('SQLite exceeds size bound')
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=.1)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('PRAGMA trusted_schema=OFF')
        db.execute('BEGIN')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {r[1] for r in db.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME)')}
        callchain = 'sum(callchainId IS NOT NULL)' if 'callchainId' in columns else '0'
        deadline = time.monotonic() + QUERY_SECONDS
        db.set_progress_handler(lambda: int(time.monotonic() > deadline),1000)
        count, nonnull = db.execute(f'SELECT count(*),{callchain} FROM CUPTI_ACTIVITY_KIND_RUNTIME').fetchone()
        if count > MAX_RUNTIME_ROWS:
            raise ValueError('runtime diagnostic row limit exceeded')
        updates = []
        for selected in selection['updates']:
            event = _range(db,selected['name'])
            phase_events = [_range(db,p['name']) for p in selected['phases']]
            for phase_event in phase_events:
                if phase_event['globalTid']!=event['globalTid'] or not event['start']<=phase_event['start']<phase_event['end']<=event['end']:
                    raise ValueError('Phase outside selected update')
            update = _scope(db,event,tables,phase_events)
            update.update(rank=selected['rank'],update=selected['update'],phases=[])
            for phase_event in phase_events:
                phase = _scope(db,phase_event,tables)
                phase.update(id=phase_event['name'].rsplit('/',1)[1],source=None)
                update['phases'].append(phase)
            updates.append(update)
    return dict(schema_version=1,available=True,scope='rank0 selected updates only, instrumented independent native diagnostic',
        evidence_level='process_nvtx_runtime_correlation',
        callstacks=dict(runtime_rows=count,runtime_rows_with_callchain=nonnull or 0,python_source_attribution=False),
        updates=updates,candidates=_candidates(updates),warnings=[
            'No captured native kernel Python frames; verified host scopes are separate evidence.',
            'Same-process cross-thread API containment is temporal association, not exact NVTX nesting.',
            'Kernel sums may overlap. Per-process/device activity unions are clipped to host scope, not utilization.',
            'Independent native diagnostic, not the selected Bridge pilot; only selected updates, no startup/warmup ranking.'])


def main(argv=None):
    import argparse
    import json
    import os
    from pathlib import Path
    try:
        from .profiling import _path, _read_bytes, MAX_JSON_BYTES
        from .inspector import sqlite_identity
        from .source_evidence import verified_native_phase_source
    except ImportError:
        from profiling import _path, _read_bytes, MAX_JSON_BYTES
        from inspector import sqlite_identity
        from source_evidence import verified_native_phase_source
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    for key in ('sqlite','analysis','native-source','source-provenance','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args(argv)
    root=args.root.resolve(strict=True)
    relative=Path(args.output)
    if relative.is_absolute() or '..' in relative.parts or '\\' in args.output or '\x00' in args.output or not relative.parts:
        raise ValueError('Output must be repository-relative')
    output=root/relative
    current=root
    for part in relative.parts:
        current=current/part
        if current.is_symlink():
            raise ValueError('Symlink output rejected')
    if output.exists():
        raise FileExistsError('Output already exists')
    capture=_path(root,args.sqlite)
    selection=json.loads(_read_bytes(_path(root,args.analysis)))
    native=_path(root,args.native_source)
    provenance=json.loads(_read_bytes(_path(root,args.source_provenance)))
    before=sqlite_identity(root,capture)
    result=analyze_capture(root,capture,selection)
    if sqlite_identity(root,capture)!=before:
        raise ValueError('SQLite changed during analysis')
    result.update(sqlite=before,source=args.output,native_source=args.native_source,
                  source_provenance=args.source_provenance)
    for update in result['updates']:
        for phase in update['phases']:
            phase['source']=verified_native_phase_source(root,native,provenance.get('native_snapshot_sha256'),phase['id'])
    for card in result['candidates']:
        phase=next(p for u in result['updates'] if u['update']==card['update'] for p in u['phases'] if p['id']==card['phase'])
        card['source_confidence']='host_scope_verified' if phase['source']['status']=='host_scope_verified' else 'unavailable'
    encoded=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')+b'\n'
    if len(encoded)>MAX_JSON_BYTES:
        raise ValueError('Inspector output exceeds bounded JSON reader size')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        stream.write(encoded)
    print(json.dumps({'output':str(output),'updates':[(u['update'],u['kernel_count'],u['summed_kernel_ms']) for u in result['updates']],
                      'callstacks':result['callstacks']}))
    return result


if __name__=='__main__':
    main()
