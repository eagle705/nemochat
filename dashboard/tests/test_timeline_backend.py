"""Selected timeline: CPU-only synthetic SQLite and strict importer checks."""
from copy import deepcopy
import json
import sqlite3
import pytest
from test_capture_analysis import fixture
import capture_analysis as analysis
import inspector

FIELDS = ['row_id','name_index','gpu_start_ns','gpu_end_ns','device_id','context_id','stream_id','correlation_id','api_start_ns','api_end_ns','thread_index','phase_index','api_name_index']


def test_timeline_records_actual_gpu_api_worker_lanes_and_untrimmed_span(tmp_path):
    path, selection = fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.executescript('ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN contextId INTEGER; ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN streamId INTEGER; ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN nameId INTEGER;')
        db.execute("INSERT INTO StringIds VALUES(4,'cudaLaunchKernel')")
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET contextId=9,streamId=17')
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET nameId=4')
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=1200 WHERE correlationId=2 AND globalPid=?',(7<<24,))
    update = analysis.analyze_capture(tmp_path,path,selection)['updates'][0]
    timeline = update['timeline']
    assert timeline['fields'] == FIELDS
    assert timeline['names'] == ['ncclKernel_AllReduce']
    assert timeline['api_names'] == ['cudaLaunchKernel']
    assert timeline['threads'] == [str((7<<24)+1),str((7<<24)+2)]
    assert timeline['events'] == [[1,0,50,250,0,9,17,1,10,20,0,0,0],[2,0,150,1100,0,9,17,2,30,40,1,0,0]]
    assert timeline['host_duration_ns'] == 900
    assert timeline['span_duration_ns'] == 1100
    assert timeline['total_events'] == timeline['returned_events'] == 2
    assert timeline['truncated'] is False
    assert timeline['phases'] == [dict(id='forward',start_ns=0,end_ns=900)]
    assert 'timeline' not in update['phases'][0]


def document(tmp_path):
    path, selection = fixture(tmp_path)
    raw = analysis.analyze_capture(tmp_path,path,selection)
    identity = inspector.sqlite_identity(tmp_path,path)
    raw['sqlite'] = identity
    return raw, identity


def test_importer_projects_timeline_and_legacy_absence_without_assertions(tmp_path):
    raw, identity = document(tmp_path)
    timeline = raw['updates'][0]['timeline']
    timeline['private'] = 'not allowed'
    timeline['phases'][0]['source'] = 'not allowed'
    safe = inspector._sanitize(tmp_path,dict(nsight_analysis='artifact.json'),raw,identity)
    assert safe['updates'][0]['timeline']['events'] == timeline['events']
    assert safe['updates'][0]['timeline']['available'] is True
    assert 'not allowed' not in json.dumps(safe)
    assert all(e[5] is None and e[6] is None and e[12] is None for e in timeline['events'])
    del raw['updates'][0]['timeline']
    safe = inspector._sanitize(tmp_path,dict(nsight_analysis='artifact.json'),raw,identity)
    assert safe['updates'][0]['timeline']['available'] is False


@pytest.mark.parametrize('mutation', [
    'bool_event','float_event','unsafe_id','negative_id','bad_name_index','bad_thread_index',
    'bad_phase_index','bad_api_index','reverse_gpu','reverse_api','api_outside','sort',
    'duplicate_row','count','total','truncated','truncated_type','host','span','phase_id',
    'phase_time','phase_membership','threads_decimal','threads_duplicate','names_duplicate',
    'event_object','dictionary_object','phase_count','event_short','null_timeline'])
def test_timeline_importer_rejects_untyped_inconsistent_or_unsafe_records(tmp_path,mutation):
    raw, identity = document(tmp_path)
    t = raw['updates'][0]['timeline']
    e = t['events'][0]
    edits = {'bool_event':(0,True),'float_event':(2,50.0),'unsafe_id':(5,2**53),
        'negative_id':(6,-1),'bad_name_index':(1,9),'bad_thread_index':(10,9),
        'bad_phase_index':(11,9),'bad_api_index':(12,0),'reverse_gpu':(3,49),
        'reverse_api':(9,9),'api_outside':(9,901),'phase_membership':(11,None),
        'event_object':(5,{'secret':'private'})}
    if mutation in edits:
        key,value=edits[mutation]; e[key]=value
    elif mutation=='sort': t['events'].reverse()
    elif mutation=='duplicate_row': t['events'][1][0]=e[0]
    elif mutation=='count': t['returned_events']=1
    elif mutation=='total': t['total_events']=3
    elif mutation=='truncated': t['truncated']=True
    elif mutation=='truncated_type': t['truncated']=1
    elif mutation=='host': t['host_duration_ns']=901
    elif mutation=='span': t['span_duration_ns']=899
    elif mutation=='phase_id': t['phases'][0]['id']='private'
    elif mutation=='phase_time': t['phases'][0]['end_ns']=901
    elif mutation=='threads_decimal': t['threads'][0]='001'
    elif mutation=='threads_duplicate': t['threads'][1]=t['threads'][0]
    elif mutation=='names_duplicate': t['names'].append(t['names'][0])
    elif mutation=='dictionary_object': t['api_names'].append({'secret':'private'})
    elif mutation=='phase_count': raw['updates'][0]['phases'][0]['kernel_count']=3
    elif mutation=='event_short': e.pop()
    elif mutation=='null_timeline': raw['updates'][0]['timeline']=None
    with pytest.raises((ValueError,TypeError,KeyError)):
        inspector._sanitize(tmp_path,dict(nsight_analysis='artifact.json'),raw,identity)


def test_producer_refuses_reversed_recorded_api_duration(tmp_path):
    path, selection = fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET end=start-1 WHERE correlationId=2')
    with pytest.raises(ValueError,match='API'):
        analysis.analyze_capture(tmp_path,path,selection)


def test_cli_compact_artifact_and_size_limit_before_output_creation(tmp_path,monkeypatch):
    import hashlib
    import profiling
    path,selection=fixture(tmp_path)
    (tmp_path/'selection.json').write_text(json.dumps(selection))
    source=b'with nvphase("forward"):\n    run_model()\n'
    (tmp_path/'native.py').write_bytes(source)
    (tmp_path/'provenance.json').write_text(json.dumps({'native_snapshot_sha256':hashlib.sha256(source).hexdigest()}))
    args=['--root',str(tmp_path),'--sqlite','capture.sqlite','--analysis','selection.json',
        '--native-source','native.py','--source-provenance','provenance.json','--output','new/inspector.json']
    analysis.main(args)
    output=(tmp_path/'new/inspector.json').read_bytes()
    assert b'"events":[[1,0,50,250' in output
    assert len(output)<profiling.MAX_JSON_BYTES
    args[-1]='too-big/inspector.json'
    # Input files still fit this bound; the output with source/aggregates does not.
    monkeypatch.setattr(profiling,'MAX_JSON_BYTES',2048)
    with pytest.raises(ValueError,match='size|bound'):
        analysis.main(args)
    assert not (tmp_path/'too-big/inspector.json').exists()


@pytest.mark.parametrize('cap',['TIMELINE_MAX_EVENTS','TIMELINE_MAX_NAMES','TIMELINE_MAX_API_NAMES','TIMELINE_MAX_THREADS'])
def test_timeline_caps_are_explicit_and_preserve_full_observed_span(tmp_path,monkeypatch,cap):
    path,selection=fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME ADD COLUMN nameId INTEGER')
        db.execute("INSERT INTO StringIds VALUES(11,'syntheticLaunchA'),(12,'syntheticLaunchB')")
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET nameId=10+correlationId')
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=1200,demangledName=3 WHERE correlationId=2')
    monkeypatch.setattr(analysis,cap,1)
    raw=analysis.analyze_capture(tmp_path,path,selection)
    timeline=raw['updates'][0]['timeline']
    assert timeline['total_events']==2 and timeline['returned_events']==1
    assert timeline['truncated'] is True
    assert timeline['span_duration_ns']==1100
    assert len(timeline['names'])==len(timeline['api_names'])==len(timeline['threads'])==1
    identity=inspector.sqlite_identity(tmp_path,path); raw['sqlite']=identity
    safe=inspector._sanitize(tmp_path,dict(nsight_analysis='artifact.json'),raw,identity)
    assert safe['updates'][0]['timeline']==timeline


@pytest.mark.parametrize('overlap',[True,False])
def test_cpu_api_phase_membership_is_null_when_ambiguous_or_uncontained(tmp_path,overlap):
    path,selection=fixture(tmp_path)
    with sqlite3.connect(path) as db:
        if overlap:
            db.execute('INSERT INTO NVTX_EVENTS VALUES(100,1000,?,?,NULL)',((7<<24)+1,'native/rank_0/update_3/phase/backward+sync'))
            selection['updates'][0]['phases'].append(dict(name='native/rank_0/update_3/phase/backward+sync'))
        else:
            db.execute('UPDATE NVTX_EVENTS SET start=125,end=135 WHERE text LIKE "%phase%"')
    raw=analysis.analyze_capture(tmp_path,path,selection)
    assert all(e[11] is None for e in raw['updates'][0]['timeline']['events'])
    identity=inspector.sqlite_identity(tmp_path,path); raw['sqlite']=identity
    assert inspector._sanitize(tmp_path,dict(nsight_analysis='artifact.json'),raw,identity)['available']


def test_gpu_start_order_ties_use_rowid_and_keep_subnanobar_events(tmp_path):
    path,selection=fixture(tmp_path)
    with sqlite3.connect(path) as db:
        db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=99,end=100 WHERE globalPid=? AND correlationId IN (1,2)',(7<<24,))
    timeline=analysis.analyze_capture(tmp_path,path,selection)['updates'][0]['timeline']
    assert [(e[0],e[2],e[3]) for e in timeline['events']]==[(1,-1,0),(2,-1,0)]
