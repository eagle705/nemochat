"""Offline inspector tests use real SQLite, explicitly synthetic durations."""
import sys
from pathlib import Path
import json
import sqlite3
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fixture(root):
    path = root / 'capture.sqlite'
    pid = 7 << 24
    with sqlite3.connect(path) as db:
        db.executescript('''
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY,value TEXT);
        INSERT INTO StringIds VALUES(1,'ncclKernel_AllReduce'),(2,'Fill'),(3,'chunk_scan_fwd');
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,callchainId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,deviceId INTEGER,demangledName INTEGER,correlationId INTEGER);
        ''')
        db.executemany('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,NULL)', [
            (100,1000,pid+1,'native/rank_0/measured/update_3'),
            (100,1000,pid+1,'native/rank_0/update_3/phase/forward'),
            (0,90,pid+1,'native/rank_0/warmup/update_2')])
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,NULL)', [
            (110,120,pid+1,1),(130,140,pid+2,2),(10,20,pid+1,3),
            (110,120,(8<<24)+1,1),(990,1010,pid+1,4)])
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)', [
            (150,350,pid,0,1,1),(250,550,pid,0,1,2),(10,9000,pid,0,2,3),
            (150,9000,8<<24,0,2,1),(900,1100,pid,0,3,4)])
    selection = {'status':'coverage-verified','updates':[{'rank':0,'update':3,
        'name':'native/rank_0/measured/update_3','phases':[{'name':'native/rank_0/update_3/phase/forward'}]}]}
    return path, selection


def test_measured_scope_worker_launches_and_clipped_union(tmp_path):
    import capture_analysis as analysis
    path, selection = fixture(tmp_path)
    before = path.read_bytes()
    result = analysis.analyze_capture(tmp_path, path, selection)
    update = result['updates'][0]
    assert update['kernel_count'] == 2
    assert update['summed_kernel_ms'] == pytest.approx(.0005)
    assert update['kernels'][0]['worker_launches'] == 1
    assert update['kernels'][0]['average_us'] == pytest.approx(.25)
    assert update['kernels'][0]['percent'] == 100
    assert update['kernels'][0]['source'] is None
    assert update['phases'][0]['kernel_count'] == 2
    # All overlapping recorded activities, including launches outside the scope,
    # are clipped separately; this is not the correlated duration sum.
    assert update['activity_by_device'][0]['activity_union_ms'] == pytest.approx(.0009)
    assert result['callstacks'] == {'runtime_rows':5,'runtime_rows_with_callchain':0,'python_source_attribution':False}
    assert all('Fill' not in c['evidence'] for c in result['candidates'])
    assert path.read_bytes() == before


@pytest.mark.parametrize('mutation', ['runtime','range','phase','selection','warmup','rows'])
def test_ambiguity_selection_and_query_limits_fail_closed(tmp_path, monkeypatch, mutation):
    import capture_analysis as analysis
    path, selection = fixture(tmp_path)
    with sqlite3.connect(path) as db:
        if mutation == 'runtime':
            db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1 AND globalTid=?', ((7<<24)+1,))
        elif mutation == 'range':
            db.execute('INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE text LIKE "%measured%"')
        elif mutation == 'phase':
            db.execute('UPDATE NVTX_EVENTS SET end=1001 WHERE text LIKE "%phase%"')
    if mutation == 'selection':
        selection['updates'][0]['update'] = 4
    elif mutation == 'warmup':
        selection['updates'][0]['name'] = 'native/rank_0/warmup/update_2'
    elif mutation == 'rows':
        monkeypatch.setattr(analysis, 'MAX_ROWS', 1)
    with pytest.raises(ValueError):
        analysis.analyze_capture(tmp_path, path, selection)


def test_callchain_enabled_rows_are_not_discarded(tmp_path):
    import capture_analysis as analysis
    path, selection = fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET callchainId=1 WHERE correlationId=2')
    result = analysis.analyze_capture(tmp_path,path,selection)
    assert result['updates'][0]['kernel_count'] == 2
    assert result['callstacks']['runtime_rows_with_callchain'] == 1


def test_cli_fresh_output_and_safe_cached_import(tmp_path):
    import capture_analysis as analysis
    import inspector
    from profiling import collect_profiling
    path, selection = fixture(tmp_path)
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    source = b'def train():\n    with nvphase("forward"):\n        run_model()\n'
    (tmp_path/'native.py').write_bytes(source)
    import hashlib
    digest = hashlib.sha256(source).hexdigest()
    (tmp_path/'provenance.json').write_text(json.dumps({'native_snapshot_sha256':digest}))
    args = ['--root',str(tmp_path),'--sqlite','capture.sqlite','--analysis','selection.json',
            '--native-source','native.py','--source-provenance','provenance.json','--output','fresh/inspector.json']
    analysis.main(args)
    with pytest.raises((ValueError,FileExistsError)):
        analysis.main(args)
    paths = dict(nsight_sqlite='capture.sqlite',nsight_analysis='fresh/inspector.json',
                 nsight_native_source='native.py',nsight_source_provenance='provenance.json')
    result = collect_profiling(tmp_path,{'profiling':paths})['nsight']['inspector']
    assert result['available']
    assert result['updates'][0]['phases'][0]['source']['status']=='host_scope_verified'
    assert result['updates'][0]['phases'][0]['source']['snippet'].startswith('    with nvphase')
    # Source without explicit local configuration is not followed or trusted.
    unpinned = inspector.load_inspector(tmp_path,{k:v for k,v in paths.items() if 'native_source' not in k})
    assert unpinned['updates'][0]['phases'][0]['source']['status']=='rejected'
    assert 'run_model' not in json.dumps(unpinned)
    document = json.loads((tmp_path/'fresh/inspector.json').read_text())
    document['secret']='private payload'
    document['updates'][0]['kernels'][0]['source']={'snippet':'private payload'}
    (tmp_path/'fresh/inspector.json').write_text(json.dumps(document))
    sanitized = inspector.load_inspector(tmp_path,paths)
    assert 'private payload' not in json.dumps(sanitized)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO StringIds VALUES(99,'changed')")
    assert not inspector.load_inspector(tmp_path,paths)['available']


@pytest.mark.parametrize('field,value',[('schema_version',True),('updates',[]),('candidates',[{}]*13),('callstacks',{'runtime_rows':True})])
def test_corrupt_artifacts_fail_closed_without_payload_leak(tmp_path,field,value):
    import capture_analysis as analysis
    import inspector
    path, selection=fixture(tmp_path)
    document=analysis.analyze_capture(tmp_path,path,selection)
    document['sqlite']=inspector.sqlite_identity(tmp_path,path)
    document[field]=value
    document['private']='do not leak'
    (tmp_path/'bad.json').write_text(json.dumps(document))
    result=inspector.load_inspector(tmp_path,{'nsight_sqlite':'capture.sqlite','nsight_analysis':'bad.json'})
    assert not result['available']
    assert 'do not leak' not in json.dumps(result)


def test_bridge_saved_source_verification_integrates_without_trusting_snippet(tmp_path):
    import hashlib
    from profiling import collect_profiling
    source=b'def example():\n    return 1\n'
    file_hash=hashlib.sha256(source).hexdigest()
    fields=dict(path='original.py',start_line=1,end_line=2,file_sha256=file_hash,snippet_sha256=file_hash)
    source_id=hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()
    (tmp_path/'sources').mkdir()
    (tmp_path/'sources'/f'{file_hash}.py').write_bytes(source)
    record=dict(role='example',source={**fields,'source_id':source_id,'snapshot':f'sources/{file_hash}.py','snippet':'secret fake snippet'})
    (tmp_path/'operations.jsonl').write_text(json.dumps(record)+'\n')
    result=collect_profiling(tmp_path,{'profiling':{'bridge_operations':'operations.jsonl'}})
    evidence=result['bridge']['operations'][0]['source_evidence']
    assert evidence['status']=='snapshot_verified'
    assert evidence['snippet']==source.decode()
    assert 'secret fake snippet' not in json.dumps(result)


def test_inspector_optimization_cards_never_rank_startup(tmp_path):
    import capture_analysis as analysis
    from profiling import _optimizations
    path,selection=fixture(tmp_path)
    measured=analysis.analyze_capture(tmp_path,path,selection)
    group={'nsight':{'available':True,'source':'capture.sqlite','summary':{'kernel_count':999},
                    'kernels':[{'name':'STARTUP_FILL','count':999,'total_ms':999}], 'inspector':measured},
           'lens':{'spans':[]}}
    cards=_optimizations('pretrain',group)
    assert cards
    assert 'STARTUP_FILL' not in json.dumps(cards)
    assert any('ncclKernel' in c['evidence'] for c in cards)


def test_duplicate_runtime_id_outside_scope_is_still_ambiguous(tmp_path):
    import capture_analysis as analysis
    path,selection=fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,2,?,1,NULL)',((7<<24)+9,))
    with pytest.raises(ValueError,match='[Aa]mbiguous'):
        analysis.analyze_capture(tmp_path,path,selection)


def test_timeout_and_runtime_diagnostic_ceiling(tmp_path,monkeypatch):
    import capture_analysis as analysis
    path,selection=fixture(tmp_path)
    monkeypatch.setattr(analysis,'MAX_RUNTIME_ROWS',1)
    with pytest.raises(ValueError,match='runtime'):
        analysis.analyze_capture(tmp_path,path,selection)
    monkeypatch.setattr(analysis,'MAX_RUNTIME_ROWS',10_000_000)
    monkeypatch.setattr(analysis,'QUERY_SECONDS',-1)
    with pytest.raises(sqlite3.OperationalError,match='interrupted'):
        analysis.analyze_capture(tmp_path,path,selection)


def test_source_hash_mismatch_never_retains_manifest_snippet(tmp_path):
    import capture_analysis as analysis
    import inspector
    path,selection=fixture(tmp_path)
    document=analysis.analyze_capture(tmp_path,path,selection)
    document['sqlite']=inspector.sqlite_identity(tmp_path,path)
    document['updates'][0]['phases'][0]['source']={'status':'host_scope_verified','snippet':'private unverified code'}
    (tmp_path/'inspector.json').write_text(json.dumps(document))
    (tmp_path/'native.py').write_text('with nvphase("forward"):\n    private_call()\n')
    (tmp_path/'provenance.json').write_text(json.dumps({'native_snapshot_sha256':'0'*64}))
    result=inspector.load_inspector(tmp_path,dict(nsight_sqlite='capture.sqlite',nsight_analysis='inspector.json',nsight_native_source='native.py',nsight_source_provenance='provenance.json'))
    assert result['available']
    assert result['updates'][0]['phases'][0]['source']['status']=='rejected'
    assert 'private_' not in json.dumps(result)
