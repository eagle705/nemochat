"""Behavioral tests with explicitly synthetic local artifact fixtures."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import pytest
from functools import lru_cache


def test_sqlite_stat_cache_invalidates_and_does_not_share_mutable_results(tmp_path, monkeypatch):
    path = tmp_path / 'capture.sqlite'
    make_sqlite(path)
    module = adapter()
    original = module._sqlite
    calls = []
    def tracked(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(module, '_sqlite', tracked)
    first = collect(tmp_path, nsight_sqlite='capture.sqlite')
    first['nsight']['kernels'].clear()
    second = collect(tmp_path, nsight_sqlite='capture.sqlite')
    assert len(second['nsight']['kernels']) == 2
    assert len(calls) == 1
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0, 1000, 1, 100, 2, 2)')
    assert collect(tmp_path, nsight_sqlite='capture.sqlite')['nsight']['summary']['kernel_count'] == 4
    assert len(calls) == 2


def test_sqlite_wal_change_is_observed_without_immutable(tmp_path, monkeypatch):
    path = tmp_path / 'capture.sqlite'
    make_sqlite(path)
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA wal_autocheckpoint=0')
    try:
        assert collect(tmp_path, nsight_sqlite='capture.sqlite')['nsight']['summary']['kernel_count'] == 3
        before = path.stat()
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0, 1000, 1, 100, 2, 2)')
        db.commit()
        assert path.stat().st_mtime_ns == before.st_mtime_ns
        assert collect(tmp_path, nsight_sqlite='capture.sqlite')['nsight']['summary']['kernel_count'] == 4
    finally:
        db.close()


@pytest.mark.parametrize("bad", ["../outside.jsonl", "/etc/passwd", "nested/../spans.jsonl", "link/spans.jsonl", "missing.jsonl", "."])
def test_rejects_missing_paths_traversal_and_symlink_components(tmp_path, bad):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "spans.jsonl").write_text("sensitive payload")
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    result = collect(tmp_path, lens_spans=bad)
    assert result["lens"]["available"] is False
    assert result["lens"]["source"] is None
    assert result["lens"]["warnings"]
    assert "sensitive payload" not in json.dumps(result)



def make_sqlite(path):
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO StringIds VALUES(1, 'void raw_kernel<float>(int)'), (2, 'second_kernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER,
                deviceId INTEGER, globalPid INTEGER, demangledName INTEGER, shortName INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (0, 2000000, 0, 99, 1, 2),
                (1000000, 4000000, 0, 99, 1, 2),
                (0, 1000000, 1, 100, 2, 2);
        ''')


def test_sqlite_uses_real_nsight_schema_raw_names_and_capture_scope(tmp_path):
    path = tmp_path / 'capture.sqlite'
    make_sqlite(path)
    before = path.read_bytes()
    result = collect(tmp_path, nsight_sqlite='capture.sqlite')['nsight']
    assert result['available'] is True
    assert result['source'] == 'capture.sqlite'
    assert result['kernels'] == [
        {'name': 'void raw_kernel<float>(int)', 'count': 2, 'total_ms': 5.0, 'average_us': 2500.0, 'percent': pytest.approx(100 * 5 / 6)},
        {'name': 'second_kernel', 'count': 1, 'total_ms': 1.0, 'average_us': 1000.0, 'percent': pytest.approx(100 / 6)},
    ]
    assert result['summary']['kernel_count'] == 3
    assert result['summary']['scope'] == 'whole capture; all recorded processes/devices; includes startup and warmup'
    assert result['summary']['coverage'] == [
        {'global_pid': 99, 'device_id': 0, 'kernel_count': 2},
        {'global_pid': 100, 'device_id': 1, 'kernel_count': 1},
    ]
    assert result['summary']['summed_kernel_ms_not_wall'] == 6.0
    assert 'utilization' not in result['summary']
    assert any('overlap' in w for w in result['warnings'])
    assert any('source attribution' in w for w in result['warnings'])
    assert path.read_bytes() == before



def test_actual_rank_zero_capture_fits_bounded_query_budget():
    root = MODULE.parents[1]
    path = 'results/e2e/20261013-1539/profile_retry/rank0-01-trace/capture.sqlite'
    if not (root / path).is_file():
        pytest.skip('Optional real rank-zero capture is not present')
    result = collect(root, nsight_sqlite=path)['nsight']
    assert result['available'], result['warnings']
    assert result['summary']['kernel_count'] == 631489
    assert len(result['kernels']) == 268
    assert sum(row['count'] for row in result['kernels']) == 631489



def test_lens_console_span_schema_and_privacy_allowlist(tmp_path):
    span = {'name': 'host forward', 'context': {'trace_id': '0x' + '1' * 32, 'span_id': '0x' + '2' * 16},
            'parent_id': '0x' + '3' * 16, 'start_time': '2026-10-13T04:20:12.000000Z',
            'end_time': '2026-10-13T04:20:12.003000Z',
            'attributes': {'nv.dl.rank': 0, 'bridge.role': 'routing', 'prompt': 'DO NOT EXPOSE',
                           'source.code': 'DO NOT EXPOSE', 'token': 'DO NOT EXPOSE', 'bridge.id': {'payload': 'DO NOT EXPOSE'}},
            'resource': {'attributes': {'process.pid': 42, 'secret': 'DO NOT EXPOSE'}},
            'events': [{'exception.message': 'DO NOT EXPOSE'}]}
    (tmp_path / 'spans.jsonl').write_text(json.dumps(span) + '\n' + '{partial')
    result = collect(tmp_path, lens_spans='spans.jsonl')['lens']
    assert result['available'] is True
    assert result['spans'] == [{'name': 'host forward', 'duration_ms': pytest.approx(3),
                               'trace_id': span['context']['trace_id'], 'span_id': span['context']['span_id'],
                               'parent_span_id': span['parent_id'],
                               'attributes': {'nv.dl.rank': 0, 'bridge.role': 'routing', 'process.pid': 42}}]
    assert 'DO NOT EXPOSE' not in json.dumps(result)
    assert any('host' in w and 'CUDA' in w for w in result['warnings'])
    assert any('malformed' in w for w in result['warnings'])



def test_bridge_operations_are_allowlisted_host_scope_not_source_contents(tmp_path):
    record = {'bridge_id': 'control:1', 'role': 'routing', 'title': 'Router scores', 'explanation': 'Host router scope',
              'status': 'ok', 'evidence': 'explicit_python_scope', 'nvtx_emitted': True, 'lens_enabled': True,
              'host_start_monotonic_ns': 1000000, 'host_end_monotonic_ns': 3000000,
              'source': {'path': 'control.py', 'source_id': 'a' * 64, 'file_sha256': 'b' * 64,
                         'snippet_sha256': 'c' * 64, 'start_line': 1, 'end_line': 3,
                         'snapshot': '/etc/passwd', 'code': 'DO NOT EXPOSE'},
              'payload': 'DO NOT EXPOSE', 'nvtx_error': {'message': 'DO NOT EXPOSE'}}
    (tmp_path / 'operations.jsonl').write_text(json.dumps(record))
    result = collect(tmp_path, bridge_operations='operations.jsonl')['bridge']
    assert result['available'] is True
    operation = result['operations'][0]
    assert operation['role'] == 'routing'
    assert operation['duration_ms'] == 2
    assert operation['source'] == {k: v for k, v in record['source'].items() if k not in ('snapshot', 'code')}
    assert 'DO NOT EXPOSE' not in json.dumps(result)
    assert '/etc/passwd' not in json.dumps(result)
    assert any('independent of native training' in w for w in result['warnings'])
    assert any('not proof of capture' in w for w in result['warnings'])



def test_verified_summary_preserves_selected_vs_whole_capture_counts(tmp_path):
    make_sqlite(tmp_path / 'capture.sqlite')
    summary = {'status': 'coverage-verified', 'capture_counts': {'CUPTI_ACTIVITY_KIND_KERNEL': 3},
               'trace_ranks': [0], 'whole_node_trace_coverage': False,
               'selected_update_ranges': 1,
               'updates': [{'rank': 0, 'update': 3, 'associated_correlated_kernel_count': 2,
                            'cpu_nvtx_ms': 10, 'summed_kernel_ms_not_wall': 5,
                            'captured_process_activity_by_device': [{'device': 0, 'clipped_activity_union_ms': 4}],
                            'payload': 'DO NOT EXPOSE'}], 'diagnostics': {'secret': 'DO NOT EXPOSE'}}
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    result = collect(tmp_path, nsight_sqlite='capture.sqlite', nsight_summary='summary.json')['nsight']
    assert result['summary']['kernel_count'] == 3
    selected = result['summary']['selected_ranges']
    assert selected['verification'] == 'whole_capture_kernel_count_checked; selected_correlation_not_recomputed'
    assert selected['trace_ranks'] == [0]
    assert selected['updates'][0]['associated_correlated_kernel_count'] == 2
    assert selected['updates'][0]['cpu_nvtx_ms'] == 10
    assert 'DO NOT EXPOSE' not in json.dumps(result)
    assert any('union' in w and 'utilization' in w for w in result['warnings'])
    summary['capture_counts']['CUPTI_ACTIVITY_KIND_KERNEL'] = 8
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    rejected = collect(tmp_path, nsight_sqlite='capture.sqlite', nsight_summary='summary.json')['nsight']
    assert 'selected_ranges' not in rejected['summary']
    assert any('summary' in w.lower() for w in rejected['warnings'])



def test_stage_profiles_have_explicit_missing_stages_and_no_control_propagation(tmp_path):
    make_sqlite(tmp_path / 'capture.sqlite')
    (tmp_path / 'operations.jsonl').write_text(json.dumps({'role': 'routing', 'bridge_id': 'control'}))
    result = collect(tmp_path, nsight_sqlite='capture.sqlite', bridge_operations='operations.jsonl')
    assert set(result['stages']) == {'pretrain', 'mid', 'sft', 'long', 'rlvr', 'mopd'}
    assert result['stages']['pretrain']['nsight']['available']
    assert result['bridge']['available']
    assert not result['stages']['pretrain']['bridge']['available']
    for stage in ('mid', 'sft', 'long', 'rlvr', 'mopd'):
        assert not result['stages'][stage]['nsight']['available']
        assert not result['stages'][stage]['bridge']['available']
    assert result['stages']['mid']['optimizations'] == []
    result = collect(tmp_path, default_stage='sft', stages={'sft': {'nsight_sqlite': 'capture.sqlite'}})
    assert result['nsight']['available']
    assert result['stages']['sft']['nsight']['source'] == 'capture.sqlite'
    assert not result['stages']['pretrain']['nsight']['available']



def test_optimizations_are_scoped_evidence_candidates_not_name_attribution(tmp_path):
    make_sqlite(tmp_path / 'capture.sqlite')
    result = collect(tmp_path, nsight_sqlite='capture.sqlite')
    suggestions = result['stages']['pretrain']['optimizations']
    assert suggestions
    assert result['optimizations'] == suggestions
    for suggestion in suggestions:
        assert set(suggestion) == {'stage', 'severity', 'title', 'evidence', 'hypothesis', 'next_experiment', 'confidence', 'source'}
        assert suggestion['stage'] == 'pretrain'
        assert suggestion['confidence'] in ('observed', 'candidate')
        assert suggestion['source'] == 'capture.sqlite'
        assert suggestion['evidence'] and suggestion['next_experiment']
        assert '정확' in suggestion['next_experiment'] and 'A/B' in suggestion['next_experiment']
    assert any(s['confidence'] == 'observed' for s in suggestions)
    assert any(s['confidence'] == 'candidate' for s in suggestions)
    assert len(suggestions) <= 4
    assert collect(tmp_path)['optimizations'] == []



@pytest.mark.parametrize('contents', [
    '{"name":"host","attributes":{"nv.dl.rank":' + '9' * 400 + '}}',
    '{"name":"host","payload":' + '[' * 10000 + '0' + ']' * 10000 + '}',
], ids=['huge_integer', 'deep_nesting'])
def test_malformed_or_extreme_json_does_not_crash_or_leak(tmp_path, contents):
    (tmp_path / 'spans.jsonl').write_text(contents)
    result = collect(tmp_path, lens_spans='spans.jsonl')
    json.dumps(result, allow_nan=False)
    assert result['lens']['warnings']


def test_record_file_query_and_display_bounds(tmp_path, monkeypatch):
    module = adapter()
    records = [json.dumps({'name': 'host', 'attributes': {}})] * 4
    (tmp_path / 'spans.jsonl').write_text('\n'.join(records))
    monkeypatch.setattr(module, 'MAX_JSON_RECORDS', 2)
    lens = collect(tmp_path, lens_spans='spans.jsonl')['lens']
    assert len(lens['spans']) == 2
    assert any('limit' in warning for warning in lens['warnings'])
    monkeypatch.setattr(module, 'MAX_JSON_BYTES', 10)
    assert not collect(tmp_path, lens_spans='spans.jsonl')['lens']['available']
    make_sqlite(tmp_path / 'capture.sqlite')
    monkeypatch.setattr(module, 'MAX_KERNEL_ROWS', 2)
    nsight = collect(tmp_path, nsight_sqlite='capture.sqlite')['nsight']
    assert not nsight['available']
    assert nsight['warnings']


def test_read_only_uri_progress_timeout_and_symlink_sidecar_rejection(tmp_path, monkeypatch):
    module = adapter()
    path = tmp_path / 'capture.sqlite'
    make_sqlite(path)
    connect = module.sqlite3.connect
    calls = []
    def tracked(database, **kwargs):
        calls.append(database)
        assert '?mode=ro' in database and 'immutable' not in database
        assert kwargs['uri'] is True
        return connect(database, **kwargs)
    monkeypatch.setattr(module.sqlite3, 'connect', tracked)
    assert collect(tmp_path, nsight_sqlite='capture.sqlite')['nsight']['available']
    assert calls
    module._CACHE.clear()
    monkeypatch.setattr(module, 'SQLITE_TIMEOUT_SECONDS', -1)
    # An actual large SQL scan reaches the progress handler.
    with connect(path) as db:
        db.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0,1000,0,99,1,2)', [()] * 3000)
    timed_out = collect(tmp_path, nsight_sqlite='capture.sqlite')['nsight']
    assert not timed_out['available']
    assert any('timed out' in w for w in timed_out['warnings'])
    Path(str(path) + '-wal').symlink_to(tmp_path / 'sensitive')
    assert not collect(tmp_path, nsight_sqlite='capture.sqlite')['nsight']['available']



def test_default_unobserved_stage_does_not_relabel_legacy_pretrain(tmp_path):
    make_sqlite(tmp_path / 'capture.sqlite')
    result = collect(tmp_path, nsight_sqlite='capture.sqlite', default_stage='sft')
    assert not result['nsight']['available']
    assert result['stages']['pretrain']['nsight']['available']
    assert result['optimizations'] == []



def test_long_context_stage_alias_is_explicitly_normalized(tmp_path):
    make_sqlite(tmp_path / 'capture.sqlite')
    result = collect(tmp_path, default_stage='long_context', stages={'long_context': {'nsight_sqlite': 'capture.sqlite'}})
    assert result['stages']['long']['nsight']['available']
    assert result['nsight']['source'] == 'capture.sqlite'
    assert not result['stages']['pretrain']['nsight']['available']
    assert all(card['stage'] == 'long' for card in result['optimizations'])



def test_actual_summary_exposes_rank_zero_selected_scope_in_candidates():
    root = MODULE.parents[1]
    base = 'results/e2e/20261013-1539/'
    paths = {'nsight_sqlite': base + 'profile_retry/rank0-01-trace/capture.sqlite',
             'nsight_summary': base + 'profile-summary.json'}
    if not all((root / path).is_file() for path in paths.values()):
        pytest.skip('Optional native capture/summary not present')
    result = collect(root, **paths)
    selected = result['nsight']['summary']['selected_ranges']
    assert [u['associated_correlated_kernel_count'] for u in selected['updates']] == [7025, 6834]
    assert selected['trace_ranks'] == [0] and selected['whole_node_trace_coverage'] is False
    assert any('rank [0]' in card['evidence'] for card in result['optimizations'])
    assert all('정확' in card['next_experiment'] and 'A/B' in card['next_experiment'] for card in result['optimizations'])



def test_explicit_stage_lens_advice_keeps_control_boundary_and_ab_gate(tmp_path):
    (tmp_path / 'spans.jsonl').write_text(json.dumps({'name': 'control host',
        'start_time': '2026-10-13T04:20:12Z', 'end_time': '2026-10-13T04:20:13Z'}))
    result = collect(tmp_path, stages={'pretrain': {'lens_spans': 'spans.jsonl'}})
    advice = result['optimizations'][0]
    assert advice['confidence'] == 'observed'
    assert '별도 교육용' in advice['hypothesis']
    assert '정확' in advice['next_experiment'] and 'A/B' in advice['next_experiment']



def test_summary_extreme_numeric_metadata_is_omitted_without_crashing(tmp_path):
    make_sqlite(tmp_path / 'capture.sqlite')
    (tmp_path / 'summary.json').write_text(json.dumps({'status': 'coverage-verified',
        'capture_counts': {'CUPTI_ACTIVITY_KIND_KERNEL': 3}, 'updates': [{'rank': 10 ** 400}]}))
    result = collect(tmp_path, nsight_sqlite='capture.sqlite', nsight_summary='summary.json')
    assert result['nsight']['summary']['selected_ranges']['updates'] == [{}]
    json.dumps(result, allow_nan=False)

MODULE = Path(__file__).resolve().parents[1] / "profiling.py"


@lru_cache(maxsize=1)
def adapter():
    spec = importlib.util.spec_from_file_location("dashboard_profiling_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect(root, **paths):
    if not MODULE.exists():
        return None  # A missing adapter is a behavioral failure, not import noise.
    return adapter().collect_profiling(root, {"profiling": paths})


def test_unconfigured_profiles_are_honestly_unavailable(tmp_path):
    result = collect(tmp_path)
    assert isinstance(result, dict)
    assert set(result) == {"lens", "nsight", "bridge", "stages", "optimizations"}
    for name, records in [("lens", "spans"), ("nsight", "kernels"), ("bridge", "operations")]:
        assert result[name] == {"available": False, "source": None, records: [], "warnings": [], **({"summary": {}} if name == "nsight" else {})}
