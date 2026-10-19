"""Typed, bounded inspector import. No SQL analysis or arbitrary source route."""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

_DIGESTS = OrderedDict()
_CACHE = OrderedDict()


def _helpers():
    try:
        from .profiling import _path, _read_bytes, _fingerprint
    except ImportError:
        from profiling import _path, _read_bytes, _fingerprint
    return _path, _read_bytes, _fingerprint


def sqlite_identity(root, path):
    _, _, fingerprint = _helpers()
    before = fingerprint(path)
    if any(v is not None for v in before[1:]):
        raise ValueError('Offline inspector requires SQLite without active sidecars')
    info = path.stat()
    if info.st_size > 2 * 1024**3:
        raise ValueError('SQLite size limit')
    key = (str(path),before)
    if key not in _DIGESTS:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if before != fingerprint(path):
            raise ValueError('SQLite changed during hash')
        _DIGESTS[key] = digest.hexdigest()
        while len(_DIGESTS)>8:
            _DIGESTS.popitem(last=False)
    return dict(path=str(path.relative_to(root)),sha256=_DIGESTS[key],size=info.st_size,mtime_ns=info.st_mtime_ns)


def _number(value, integer=False):
    if type(value) not in ((int,) if integer else (int,float)) or not math.isfinite(value) or not 0<=value<2**63:
        raise ValueError('Invalid numeric value')
    return value


def _text(value, limit=8192):
    if not isinstance(value,str) or len(value)>limit or '\x00' in value:
        raise ValueError('Invalid bounded text')
    return value


def _list(value, limit, minimum=0):
    if not isinstance(value,list) or not minimum<=len(value)<=limit:
        raise ValueError('Invalid record count')
    return value


def _object(value):
    if not isinstance(value,dict):
        raise ValueError('Invalid object')
    return value


def _source(root,paths,phase):
    rejected = dict(status='rejected',kind='host_scope',path=None,function=None,start_line=None,end_line=None,
                    file_sha256=None,snippet_sha256=None,source_id=None,snippet='',warning='Native host source requires explicit configured source/provenance; not a kernel frame.')
    if not paths.get('nsight_native_source') or not paths.get('nsight_source_provenance'):
        return rejected
    path,read,_ = _helpers()
    try:
        source_path=path(root,paths['nsight_native_source'])
        provenance=json.loads(read(path(root,paths['nsight_source_provenance'])))
        expected=provenance['native_snapshot_sha256']
        try:
            from .source_evidence import verified_native_phase_source
        except ImportError:
            from source_evidence import verified_native_phase_source
        return verified_native_phase_source(root,source_path,expected,phase)
    except (OSError,ValueError,KeyError,TypeError,UnicodeError,RecursionError):
        return rejected


def _scope(raw):
    raw = _object(raw)
    safe = dict(name=_text(raw['name'],512))
    for field in ('host_ms','summed_kernel_ms'):
        safe[field]=_number(raw[field])
    for field in ('kernel_count','same_process_contained_api_count','worker_thread_correlated_kernel_count','displayed_kernel_count','unique_kernel_names'):
        safe[field]=_number(raw[field],True)
    safe['activity_union_ms']=None
    safe['activity_by_device']=[]
    for device in _list(raw.get('activity_by_device'),128):
        device=_object(device)
        safe['activity_by_device'].append(dict(device_id=_number(device['device_id'],True),
            global_pid=_number(device['global_pid'],True),activity_union_ms=_number(device['activity_union_ms'])))
    safe['kernels']=[]
    for kernel in _list(raw['kernels'],64):
        kernel=_object(kernel)
        name=_text(kernel['name'])
        try:
            from .capture_analysis import semantic
        except ImportError:
            from capture_analysis import semantic
        category,meaning,evidence=semantic(name)
        try:
            from .kernel_symbols import decode_symbol
        except ImportError:
            from kernel_symbols import decode_symbol
        row=dict(name=name,category=category,meaning=meaning,meaning_evidence=evidence,source_attribution='unavailable',source=None,symbol_analysis=decode_symbol(name))
        for field in ('count','worker_launches'):
            row[field]=_number(kernel[field],True)
        for field in ('total_ms','average_us','percent'):
            row[field]=_number(kernel[field])
        if row['count']==0 or row['worker_launches']>row['count'] or row['percent']>100.000001:
            raise ValueError('Invalid kernel aggregate')
        safe['kernels'].append(row)
    if sum(k['count'] for k in safe['kernels'])!=safe['displayed_kernel_count'] or safe['displayed_kernel_count']>safe['kernel_count']:
        raise ValueError('Kernel count mismatch')
    return safe


def _timeline(raw, scope):
    try:
        from .capture_analysis import TIMELINE_FIELDS, _timeline_integer, MAX_ROWS
    except ImportError:
        from capture_analysis import TIMELINE_FIELDS, _timeline_integer, MAX_ROWS
    raw = _object(raw)
    if type(raw.get('schema_version')) is not int or raw['schema_version']!=1 or raw.get('available') is not True:
        raise ValueError('Invalid timeline schema')
    if raw.get('origin')!='selected_host_nvtx_start' or raw.get('order')!='gpu_start_ns,row_id' or raw.get('fields')!=TIMELINE_FIELDS:
        raise ValueError('Invalid timeline layout')
    safe = dict(schema_version=1,available=True,origin='selected_host_nvtx_start',
        order='gpu_start_ns,row_id',fields=list(TIMELINE_FIELDS))
    for field in ('host_duration_ns','span_duration_ns','total_events','returned_events'):
        safe[field]=_timeline_integer(raw[field])
    host=safe['host_duration_ns']
    if host==0 or not math.isclose(host/1e6,scope['host_ms'],rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Timeline host duration mismatch')
    if type(raw['truncated']) is not bool or safe['total_events']!=scope['kernel_count'] or safe['total_events']>MAX_ROWS:
        raise ValueError('Invalid timeline totals')
    safe['truncated']=raw['truncated']
    for field,cap in (('names',4096),('api_names',512),('threads',256)):
        safe[field]=[_text(v,20 if field=='threads' else 8192) for v in _list(raw[field],cap)]
        if len(set(safe[field]))!=len(safe[field]):
            raise ValueError('Duplicate timeline dictionary entry')
    for tid in safe['threads']:
        if not re.fullmatch(r'0|[1-9][0-9]{0,19}',tid) or int(tid)>=2**64:
            raise ValueError('Invalid exact timeline thread identifier')
    safe['phases']=[]
    for phase in _list(raw['phases'],16):
        phase=_object(phase)
        safe['phases'].append(dict(id=_text(phase['id'],512),
            start_ns=_timeline_integer(phase['start_ns']),end_ns=_timeline_integer(phase['end_ns'])))
    if [p['id'] for p in safe['phases']]!=[p['id'] for p in scope['phases']]:
        raise ValueError('Timeline phase scope mismatch')
    for p,s in zip(safe['phases'],scope['phases']):
        if not 0<=p['start_ns']<p['end_ns']<=host or not math.isclose(
                (p['end_ns']-p['start_ns'])/1e6,s['host_ms'],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Invalid timeline phase interval')
    safe['events']=[]
    rows=set(); previous=None; used={1:set(),10:set(),12:set()}
    phase_counts=[0]*len(safe['phases'])
    span=host
    for event in _list(raw['events'],16384):
        event=_list(event,13,13)
        clean=[]
        for i,value in enumerate(event):
            clean.append(None if value is None and i in (5,6,11,12) else _timeline_integer(value,i in (2,3)))
        row,name,start,end,device,context,stream,correlation,api_start,api_end,thread,phase,api=clean
        for i,key in ((1,'names'),(10,'threads'),(11,'phases'),(12,'api_names')):
            if clean[i] is not None and clean[i]>=len(safe[key]):
                raise ValueError('Timeline dictionary index out of bounds')
        if row==0 or row in rows or (previous is not None and (start,row)<=previous):
            raise ValueError('Timeline row identity or chronological order invalid')
        if end<start or not 0<=api_start<=api_end<=host:
            raise ValueError('Timeline GPU/API timing invalid')
        memberships=[i for i,p in enumerate(safe['phases']) if p['start_ns']<=api_start<=api_end<=p['end_ns']]
        if phase!=(memberships[0] if len(memberships)==1 else None):
            raise ValueError('Timeline CPU API phase membership invalid')
        for i in memberships:
            phase_counts[i]+=1
        for i in used:
            if clean[i] is not None: used[i].add(clean[i])
        rows.add(row); previous=(start,row); span=max(span,end)
        safe['events'].append(clean)
    returned=len(safe['events'])
    if returned!=safe['returned_events'] or returned>safe['total_events'] or safe['truncated']!=(returned<safe['total_events']):
        raise ValueError('Timeline returned/truncated count mismatch')
    if safe['span_duration_ns']<span or (not safe['truncated'] and safe['span_duration_ns']!=span):
        raise ValueError('Timeline span mismatch')
    for i,key in ((1,'names'),(10,'threads'),(12,'api_names')):
        if used[i]!=set(range(len(safe[key]))):
            raise ValueError('Unused timeline dictionary entries')
    for count,phase in zip(phase_counts,scope['phases']):
        if count>phase['kernel_count'] or (not safe['truncated'] and count!=phase['kernel_count']):
            raise ValueError('Timeline phase kernel count mismatch')
    return safe


def _sanitize(root,paths,document,identity):
    raw=_object(document)
    if type(raw.get('schema_version')) is not int or raw['schema_version']!=1 or raw.get('available') is not True:
        raise ValueError('Invalid schema')
    if raw.get('evidence_level')!='process_nvtx_runtime_correlation' or raw.get('sqlite')!=identity:
        raise ValueError('SQLite identity mismatch')
    calls=_object(raw['callstacks'])
    runtime=_number(calls['runtime_rows'],True)
    nonnull=_number(calls['runtime_rows_with_callchain'],True)
    if nonnull>runtime or calls.get('python_source_attribution') is not False:
        raise ValueError('Unsupported source attribution')
    result=dict(schema_version=1,available=True,source=paths['nsight_analysis'],sqlite=identity,
        scope='rank0 selected updates only, instrumented independent native diagnostic',
        evidence_level='process_nvtx_runtime_correlation',callstacks=dict(runtime_rows=runtime,
            runtime_rows_with_callchain=nonnull,python_source_attribution=False),updates=[],candidates=[],warnings=[])
    seen=set()
    for update in _list(raw['updates'],8,1):
        update=_object(update)
        rank=_number(update['rank'],True); number=_number(update['update'],True)
        if rank!=0 or number not in (3,4) or number in seen or update['name']!=f'native/rank_0/measured/update_{number}':
            raise ValueError('Invalid measured update')
        seen.add(number)
        safe=_scope(update)
        safe.update(rank=rank,update=number,phases=[])
        phase_seen=set()
        for phase in _list(update['phases'],16,1):
            phase=_object(phase)
            label=phase['id']
            if label not in ('data','forward','backward+sync','optimizer') or label in phase_seen or phase['name']!=f'native/rank_0/update_{number}/phase/{label}':
                raise ValueError('Invalid phase scope')
            phase_seen.add(label)
            scope=_scope(phase)
            scope.update(id=label,source=_source(root,paths,label))
            if scope['host_ms']>safe['host_ms'] or scope['kernel_count']>safe['kernel_count']:
                raise ValueError('Phase exceeds update')
            safe['phases'].append(scope)
        safe['timeline']=_timeline(update['timeline'],safe) if 'timeline' in update else dict(schema_version=1,available=False)
        result['updates'].append(safe)
    # Validate producer record caps, but regenerate meanings/cards from selected
    # sanitized metrics instead of trusting free-text hypotheses or snippets.
    _list(raw['candidates'],12)
    try:
        from .capture_analysis import _candidates
    except ImportError:
        from capture_analysis import _candidates
    result['candidates']=_candidates(result['updates'])
    for card in result['candidates']:
        phase=next(p for u in result['updates'] if u['update']==card['update'] for p in u['phases'] if p['id']==card['phase'])
        card['source_confidence']=phase['source']['status'] if phase['source']['status']=='host_scope_verified' else 'unavailable'
    result['warnings']=['Offline selected-scope analysis imported; SQLite digest checked, producer not authenticated.',
        'No native kernel Python frame attribution. Host source evidence is separate.',
        'Instrumented independent manual MCore diagnostic, not actual selected Bridge trainer.',
        'Summed GPU durations may overlap; clipped per-process/device unions are not utilization or critical path.']
    return result


def load_inspector(root,paths):
    root=Path(root).resolve()
    empty=dict(schema_version=1,available=False,source=None,updates=[],candidates=[],warnings=[])
    if paths.get('nsight_analysis') is None:
        return empty
    try:
        path,read,fingerprint=_helpers()
        artifact=path(root,paths['nsight_analysis'])
        capture=path(root,paths['nsight_sqlite'])
        configured=[]
        for key in ('nsight_native_source','nsight_source_provenance'):
            if paths.get(key):
                configured.append((key,paths[key],fingerprint(path(root,paths[key]))))
        cache_key=(str(root),str(artifact),fingerprint(artifact),str(capture),fingerprint(capture),tuple(configured))
        if cache_key in _CACHE:
            return deepcopy(_CACHE[cache_key])
        identity=sqlite_identity(root,capture)
        result=_sanitize(root,paths,json.loads(read(artifact)),identity)
        _CACHE[cache_key]=deepcopy(result)
        while len(_CACHE)>8:
            _CACHE.popitem(last=False)
        return result
    except (OSError,ValueError,KeyError,TypeError,UnicodeError,RecursionError,AttributeError):
        empty['warnings']=['Configured inspector rejected: missing, unsafe, malformed, over limit, source configuration unsafe, or SQLite fingerprint mismatch.']
        return empty
