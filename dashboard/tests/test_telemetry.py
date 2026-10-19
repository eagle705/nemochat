"""Synthetic fixtures only: these tests are not training evidence."""
import json
from dashboard import collectors
from dashboard import server
import subprocess
import sys
from pathlib import Path
import pytest


def test_strict_v1_contract_schema_and_identity_validation():
    from dashboard import telemetry
    telemetry.validate_manifest(manifest())
    telemetry.validate_record(record(), manifest())
    schema = json.loads((Path(telemetry.__file__).with_name('telemetry.schema.json')).read_text())
    assert schema['$defs']['train_step']['additionalProperties'] is False
    mutations = [('step', True), ('step', -1), ('rank', 8), ('rank', False), ('world_size', 0), ('run_id', 'other'), ('stage', 'sft'), ('timestamp', 'not-a-date'), ('timestamp', '2026-02-30T00:00:00Z'), ('schema_version', True), ('event', 'benchmark'), ('password', 'SECRET')]
    for field, value in mutations:
        row = record()
        row[field] = value
        with pytest.raises(ValueError):
            telemetry.validate_record(row, manifest())
    for container, field, values in [('counters', 'scope', ['rank_local', None]), ('counters', 'processed', [-1, True, float('nan'), float('inf')]), ('metrics', 'loss', [True, float('nan'), float('inf'), '2']), ('timing', 'step_seconds', [-1, True, float('inf')])]:
        for value in values:
            row = record()
            row[container][field] = value
            with pytest.raises(ValueError):
                telemetry.validate_record(row, manifest())
    for field, value in [('run_id', 'bad/id'), ('stage', 'unknown'), ('total_steps', True), ('target_tokens', -1), ('env', {'PASSWORD': 'SECRET'})]:
        obj = manifest()
        obj[field] = value
        with pytest.raises(ValueError):
            telemetry.validate_manifest(obj)
    row = record()
    row['metrics']['extra'] = {'reward_mean': -1, 'teacher_kl': None}
    telemetry.validate_record(row, manifest())
    row['metrics']['extra']['api_key'] = 1
    with pytest.raises(ValueError):
        telemetry.validate_record(row, manifest())


def manifest():
    return {'schema_version': 1, 'run_id': 'pilot', 'name': 'Synthetic pilot', 'stage': 'pretrain', 'model': {'name': 'fixture'}, 'data': {'name': 'fixture', 'scope': 'synthetic test only'}, 'total_steps': 100, 'target_tokens': 10000}


def record(step=10, tokens=1000):
    return {'schema_version': 1, 'event': 'train_step', 'run_id': 'pilot', 'stage': 'pretrain', 'step': step, 'timestamp': '2026-10-13T00:00:00Z', 'rank': 7, 'world_size': 8, 'metrics_scope': 'rank_local', 'counters': {'scope': 'global_cumulative', 'processed': tokens, 'loss_bearing': tokens-10, 'source_read': tokens+10}, 'metrics': {'loss': 2., 'learning_rate': .001, 'grad_norm': None}, 'timing': {'step_seconds': 1.5}}


def setup_run(tmp_path):
    base = tmp_path / 'pilot'
    base.mkdir()
    (base / 'run.json').write_text(json.dumps(manifest()))
    (base / 'metrics.jsonl').write_text(json.dumps(record())+'\n')
    return base, {'id': 'pilot', 'stage': 'pretrain', 'path': 'pilot', 'format': 'canonical'}


@pytest.mark.parametrize('change', ['duplicate', 'backward', 'counter', 'writer', 'world', 'scope', 'identity', 'secret', 'malformed', 'rotation', 'manifest'])
def test_fail_closed_retains_last_good_without_secrets(tmp_path, change):
    base, spec = setup_run(tmp_path)
    assert collectors.collect_run(tmp_path, spec, {})['tokens']['processed'] == 1000
    row = record(11, 1100)
    if change == 'duplicate': row = record()
    if change == 'backward': row['step'] = 9
    if change == 'counter': row['counters']['processed'] = 999
    if change == 'writer': row['rank'] = 0
    if change == 'world': row['world_size'] = 16
    if change == 'scope': row['metrics_scope'] = 'global'
    if change == 'identity': row['run_id'] = 'other'
    if change == 'secret': row['password'] = 'TOP_SECRET_VALUE'
    if change == 'manifest':
        obj = manifest(); obj['data']['password'] = 'TOP_SECRET_VALUE'
        (base/'run.json').write_text(json.dumps(obj))
    elif change == 'rotation':
        (base/'metrics.jsonl').write_text('{}\n')
    else:
        with (base/'metrics.jsonl').open('a') as out:
            out.write(('not JSON TOP_SECRET_VALUE' if change == 'malformed' else json.dumps(row))+'\n')
    run = collectors.collect_run(tmp_path, spec, {})
    assert run['tokens']['processed'] == 1000
    assert run['warnings']
    assert 'TOP_SECRET_VALUE' not in json.dumps(run)
    assert len(run['history']) == 1


def test_partial_line_retries_until_complete_and_stat_cache(tmp_path, monkeypatch):
    base, spec = setup_run(tmp_path)
    text = json.dumps(record(11, 1100))
    with (base/'metrics.jsonl').open('a') as out: out.write(text[:30])
    run = collectors.collect_run(tmp_path, spec, {})
    assert run['tokens']['processed'] == 1000
    assert any('partial' in w.lower() for w in run['warnings'])
    with (base/'metrics.jsonl').open('a') as out: out.write(text[30:]+'\n')
    assert collectors.collect_run(tmp_path, spec, {})['tokens']['processed'] == 1100
    def forbidden(*args, **kwargs): raise AssertionError('unchanged telemetry should not reopen')
    monkeypatch.setattr(Path, 'open', forbidden)
    assert collectors.collect_run(tmp_path, spec, {})['tokens']['processed'] == 1100


def test_unavailable_canonical_files_can_appear_without_restart(tmp_path):
    base = tmp_path/'pilot'; base.mkdir()
    spec = {'id': 'pilot', 'path': 'pilot', 'format': 'canonical'}
    assert collectors.collect_run(tmp_path, spec, {})['tokens']['processed'] is None
    (base/'run.json').write_text(json.dumps(manifest()))
    (base/'metrics.jsonl').write_text(json.dumps(record())+'\n')
    assert collectors.collect_run(tmp_path, spec, {})['tokens']['processed'] == 1000


def test_cli_validation_read_only_and_rejects_complete_invalid_record(tmp_path):
    base, spec = setup_run(tmp_path)
    command = [sys.executable, str(Path(collectors.__file__).with_name('telemetry.py')), 'validate', '--manifest', str(base/'run.json'), '--metrics', str(base/'metrics.jsonl')]
    before = {p.name: p.read_bytes() for p in base.iterdir()}
    good = subprocess.run(command, capture_output=True, text=True)
    assert good.returncode == 0
    assert json.loads(good.stdout)['records_validated'] == 1
    assert before == {p.name: p.read_bytes() for p in base.iterdir()}
    with (base/'metrics.jsonl').open('a') as out: out.write('TOP_SECRET_BAD_JSON\n')
    bad = subprocess.run(command, capture_output=True, text=True)
    assert bad.returncode == 1
    assert 'TOP_SECRET' not in bad.stdout + bad.stderr


def test_config_accepts_canonical_and_run_identity_rejects_mismatch(tmp_path):
    base, spec = setup_run(tmp_path)
    assert server.validate_config(tmp_path, {'runs': [spec]})['runs'][0]['format'] == 'canonical'
    spec['id'] = 'other'
    run = collectors.collect_run(tmp_path, spec, {})
    assert run['tokens']['processed'] is None
    assert run['warnings']


@pytest.mark.parametrize('adapter', ['bridge', 'native'])
def test_legacy_provenance_sources_and_unweighted_loss_scope(tmp_path, adapter):
    base = tmp_path/'legacy'; base.mkdir()
    fname = 'rank-1-updates.jsonl' if adapter == 'bridge' else 'rank-1-phases.jsonl'
    (base/fname).write_text(json.dumps({'event': 'optimizer-update-completed', 'update': 1, 'local_loss': 2, 'processed_tokens': 100})+'\n')
    run = collectors.collect_run(tmp_path, {'id': 'legacy', 'path': 'legacy', 'format': adapter, 'rank_count': 2}, {})
    provenance = run['provenance']
    assert provenance['adapter'] == adapter
    assert provenance['schema_version'] is None
    assert provenance['metrics_scope'] == 'unweighted_mean_of_rank_local'
    assert 'global' not in provenance['metrics_scope']
    source = next(s for s in provenance['sources'] if s['path'] == 'legacy/'+fname)
    assert source['available'] and source['updated_at']
    assert any(not s['available'] for s in provenance['sources'])
    assert all(not s['path'].startswith('/') for s in provenance['sources'])
    assert any('unweighted' in w.lower() for w in run['warnings'])


def test_canonical_bounded_reads_history_and_global_totals_survive_window(tmp_path, monkeypatch):
    from dashboard import telemetry
    base, spec = setup_run(tmp_path)
    monkeypatch.setattr(telemetry, 'MAX_READ', 1500)
    (base/'metrics.jsonl').write_text(''.join(json.dumps(record(i, i*100))+'\n' for i in range(1, 521)))
    first = collectors.collect_run(tmp_path, spec, {})
    assert first['progress']['step'] < 520
    for _ in range(300):
        run = collectors.collect_run(tmp_path, spec, {})
        if run['progress']['step'] == 520: break
    assert run['tokens']['processed'] == 52000
    assert len(run['history']) == 256
    assert run['provenance']['sources'][1]['truncated']


def test_canonical_real_append_global_cumulative_no_rank_multiplication(tmp_path):
    base, spec = setup_run(tmp_path)
    run = collectors.collect_run(tmp_path, spec, {})
    assert run['tokens']['processed'] == 1000
    assert run['metrics']['loss'] == 2
    assert run['provenance']['metrics_scope'] == 'rank_local'
    assert run['provenance']['counter_scope'] == 'global_cumulative'
    assert any('rank_local' in w for w in run['warnings'])
    with (base/'metrics.jsonl').open('a') as out:
        out.write(json.dumps(record(11, 1100))+'\n')
    run = collectors.collect_run(tmp_path, spec, {})
    assert run['tokens']['processed'] == 1100
    assert [h['step'] for h in run['history']] == [10, 11]
    assert {s['path'] for s in run['provenance']['sources']} == {'pilot/run.json', 'pilot/metrics.jsonl'}
    assert all(s['available'] and s['updated_at'] for s in run['provenance']['sources'])
