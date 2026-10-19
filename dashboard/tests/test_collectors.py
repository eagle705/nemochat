"""Behavioral tests use explicit, synthetic local artifacts, never GPUs."""
import importlib.util
import json
from pathlib import Path


def module():
    path = Path(__file__).parents[1] / 'collectors.py'
    assert path.exists(), 'collector implementation is missing'
    spec = importlib.util.spec_from_file_location('dashboard_collectors_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def put(root, path, obj):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))
    return p


def test_missing_run_has_null_facts_and_unobserved_stages(tmp_path):
    c = module()
    snapshot = c.collect_snapshot(tmp_path, {'runs': [{'id': 'pilot', 'path': 'results/pilot', 'stage': 'pretrain'}], 'gpu_enabled': False})
    assert snapshot['schema_version'] == 1
    run = snapshot['runs'][0]
    assert run['status'] == 'unknown'
    assert run['tokens']['processed'] is None
    assert run['metrics']['loss'] is None
    assert snapshot['host']['memory']['total_bytes'] > 0
    assert next(s for s in snapshot['stages'] if s['id'] == 'sft')['status'] == 'not_observed'


def test_bridge_live_updates_keep_global_counters_separate(tmp_path):
    c = module()
    put(tmp_path, 'results/pilot/requested-config.json', {'train': {'train_iters': 2, 'global_batch_size': 8}, 'model': {'bf16': True, 'seq_length': 128, '_target_': 'HybridModelProvider'}, 'env_vars': {'API_KEY': 'never-expose'}})
    put(tmp_path, 'results/pilot/data-verification.json', {'counts': {'tokens': 2000000}})
    for rank in range(8):
        p = tmp_path / f'results/pilot/rank-{rank}-updates.jsonl'
        p.write_text(json.dumps({'rank': rank, 'update': 1, 'local_loss': 10 + rank, 'lr': .001, 'grad_norm': 4, 'processed_tokens': 128, 'loss_bearing_tokens': 127, 'read_source_tokens': 129, 'seconds': 3}) + '\n')
    cfg = {'runs': [{'id': 'pilot', 'path': 'results/pilot', 'format': 'bridge', 'rank_count': 8}], 'gpu_enabled': False}
    run = c.collect_snapshot(tmp_path, cfg)['runs'][0]
    assert run['status'] == 'running'
    assert run['tokens'] == {'processed': 1024, 'loss_bearing': 1016, 'source_read': 1032, 'target': 2048}
    assert run['metrics']['loss'] == 13.5
    assert run['data']['prepared_tokens'] == 2000000
    assert 'never-expose' not in json.dumps(run)
    assert len(run['history']) == 1
    put(tmp_path, 'results/pilot/verified-summary.json', {'status': 'passed', 'processed_tokens': 2048, 'loss_tokens': 2032, 'source_prefix_tokens': 2064, 'optimizer_updates': 2})
    run = c.collect_snapshot(tmp_path, cfg)['runs'][0]
    assert run['status'] == 'completed'
    assert run['tokens']['processed'] == 2048


def test_native_resumed_receipts_do_not_sum_cumulative_tokens(tmp_path):
    c = module()
    for rank in range(2):
        put(tmp_path, f'results/native/rank-{rank}.json', {'rank': rank, 'status': 'completed-native-updates-not-exact-resume-approval', 'global_loss_bearing_tokens': 512, 'history': [{'update': 1, 'loss': 5, 'lr': .01, 'input_token_count': 128, 'loss_bearing_tokens': 128, 'global_loss_bearing_tokens_cumulative': 256, 'seconds': 2}, {'update': 2, 'loss': 4, 'lr': .001, 'input_token_count': 128, 'loss_bearing_tokens': 128, 'global_loss_bearing_tokens_cumulative': 512, 'seconds': 3}]})
    put(tmp_path, 'results/native/resolved_config.json', {'plan': {'stop': 2, 'contract': {'sequence': 128, 'loss_bearing_tokens_per_update': 256, 'selection': 'real document prefixes', 'variant': 'A'}}})
    cfg = {'runs': [{'id': 'native', 'path': 'results/native', 'format': 'native', 'rank_count': 2, 'terminal_receipt': 'results/receipt.json'}], 'gpu_enabled': False}
    run = c.collect_snapshot(tmp_path, cfg)['runs'][0]
    assert run['status'] == 'completed'
    assert run['tokens']['loss_bearing'] == 512
    assert run['tokens']['processed'] == 512
    assert run['history'][-1]['loss_bearing_tokens'] == 512
    assert run['data']['scope'] == 'real document prefixes'
    put(tmp_path, 'results/receipt.json', {'status': 'passed', 'producer_run_success': False, 'producer': {'state': {'Status': 'exited', 'ExitCode': 137}}, 'checkpoint_integrity_verified': True})
    run = c.collect_snapshot(tmp_path, cfg)['runs'][0]
    assert run['status'] == 'failed', 'checkpoint validity must not turn a failed producer into success'
    assert any('checkpoint' in w.lower() for w in run['warnings'])


def test_bounded_incremental_cache_partial_lines_rotation_and_no_directory_scan(tmp_path, monkeypatch):
    c = module()
    p = put(tmp_path, 'results/pilot/requested-config.json', {})
    log = p.parent / 'rank-0-updates.jsonl'
    row = {'rank': 0, 'update': 1, 'loss_bearing_tokens': 128, 'processed_tokens': 128}
    log.write_text(json.dumps(row) + '\n' + '{"rank":0,"update":2')
    cfg = {'runs': [{'id': 'pilot', 'path': 'results/pilot', 'format': 'bridge'}], 'gpu_enabled': False}
    first = c.collect_snapshot(tmp_path, cfg)
    assert first['runs'][0]['tokens']['processed'] == 128
    def forbidden(*args, **kwargs):
        raise AssertionError('unchanged artifacts must not be reopened/scanned')
    with monkeypatch.context() as m:
        m.setattr(Path, 'open', forbidden)
        m.setattr(Path, 'rglob', forbidden)
        # Host /proc reads are separate from the stat-cached run collector.
        second = c.collect_run(tmp_path, cfg['runs'][0], cfg)
        assert second['history'] == first['runs'][0]['history']
    with log.open('a') as stream:
        stream.write(',"processed_tokens":128,"loss_bearing_tokens":128}\n')
    assert c.collect_run(tmp_path, cfg['runs'][0], cfg)['tokens']['processed'] == 256
    log.write_text(json.dumps({'update': 4, 'processed_tokens': 16, 'loss_bearing_tokens': 16}) + '\n')
    assert c.collect_run(tmp_path, cfg['runs'][0], cfg)['tokens']['processed'] == 16
    metadata_only = put(tmp_path, 'results/meta/requested-config.json', {'env_vars': {'PASSWORD': 'hide-me'}})
    run = c.collect_run(tmp_path, {'id': 'meta', 'path': 'results/meta'}, cfg)
    assert run['tokens']['processed'] is None, 'missing update evidence must not show zero tokens'
    assert 'hide-me' not in json.dumps(run)


def test_host_samples_cpu_and_timeout_bounded_nvidia_smi(tmp_path, monkeypatch):
    c = module()
    executable = tmp_path / 'nvidia-smi'
    executable.write_text('#!/bin/sh\nprintf "0, Test GPU, GPU-test, 43, 100, 200, 51, 10.5\\n"\n')
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path))
    host = c.collect_host({'nvidia_smi_timeout_seconds': .1})
    assert host['gpus'][0]['utilization_percent'] == 43
    assert host['gpus'][0]['memory_used_mb'] == 100
    second = c.collect_host({'gpu_enabled': False})
    assert second['cpu']['utilization_percent'] is not None
    executable.write_text('#!/bin/sh\n/bin/sleep 2\n')
    import time
    start = time.monotonic()
    host = c.collect_host({'nvidia_smi_timeout_seconds': .1})
    assert time.monotonic() - start < 1
    assert host['gpus'] == []
    assert host['warnings']


def test_stale_activity_not_refreshed_by_config_seven_stages_and_benchmark_allowlist(tmp_path):
    c = module()
    import os, time
    put(tmp_path, 'results/pilot/requested-config.json', {'model': {'num_layers': 52, 'hidden_size': 2048}})
    log = tmp_path / 'results/pilot/rank-0-updates.jsonl'
    log.write_text(json.dumps({'update': 1, 'loss': 9, 'processed_tokens': 128}) + '\n')
    os.utime(log, (time.time()-500, time.time()-500))
    put(tmp_path, 'results/benchmark.json', {'scores': {'ce': 8.5}, 'api_key': 'SECRET'})
    spec = {'id': 'pilot', 'path': 'results/pilot', 'stage': 'long_context', 'benchmarks': [{'path': 'results/benchmark.json', 'metrics': [{'name': 'heldout-ce', 'field': 'scores.ce', 'scope': 'small frozen heldout'}]}]}
    snapshot = c.collect_snapshot(tmp_path, {'runs': [spec], 'gpu_enabled': False})
    run = snapshot['runs'][0]
    assert run['status'] == 'stale'
    assert run['stage'] == 'long'
    assert len(snapshot['stages']) == 7
    assert run['benchmarks'] == [{'name': 'heldout-ce', 'value': 8.5, 'scope': 'small frozen heldout'}]
    assert run['model']['num_layers'] == 52
    assert 'SECRET' not in json.dumps(snapshot)
    # Symlink escape is rejected even inside an otherwise allowlisted run.
    (log.parent / 'verified-summary.json').symlink_to(tmp_path.parent / 'outside.json')
    run = c.collect_run(tmp_path, spec, {})
    assert any('rejected' in w for w in run['warnings'])


def test_diagnostic_warmup_and_measured_counters_are_separate(tmp_path):
    c = module()
    history = [{'update': 2, 'kind': 'warmup', 'loss_bearing_tokens': 128, 'input_token_count': 128, 'source_prefix_tokens': 129, 'seconds_profiled_not_throughput': 100}, {'update': 3, 'kind': 'measured', 'loss_bearing_tokens': 128, 'input_token_count': 128, 'source_prefix_tokens': 129, 'seconds_profiled_not_throughput': 2}]
    put(tmp_path, 'results/diag/result.json', {'status': 'passed', 'contract': {'schedule': {'total_steps': 4}, 'loss_bearing_tokens_per_update': 256}, 'profile_contract': {'global_updates': [2, 3]}, 'ranks': [{'rank': r, 'status': 'passed', 'history': history} for r in range(2)]})
    run = c.collect_run(tmp_path, {'id': 'diag', 'path': 'results/diag', 'format': 'native', 'rank_count': 2}, {})
    assert run['tokens']['loss_bearing'] == 512
    assert run['tokens']['warmup_loss_bearing'] == 256
    assert run['tokens']['measured_loss_bearing'] == 256
    assert run['tokens']['target'] == 512
    assert run['history'][0]['kind'] == 'warmup'
    assert run['timing']['tokens_per_second'] is None


def test_huge_partial_lines_and_long_history_stay_bounded_without_fake_totals(tmp_path):
    c = module()
    log = tmp_path / 'log.jsonl'
    log.write_text('x' * c.MAX_TAIL)
    c.read_artifact(log, True)
    with log.open('a') as stream: stream.write('x' * c.MAX_TAIL)
    _, _, truncated = c.read_artifact(log, True)
    assert truncated
    assert len(c._CACHE[(str(log), True)]['pending']) <= c.MAX_TAIL
    put(tmp_path, 'results/long/requested-config.json', {})
    log = tmp_path / 'results/long/rank-0-updates.jsonl'
    log.write_text(''.join(json.dumps({'update': i, 'processed_tokens': 128, 'loss_bearing_tokens': 128}) + '\n' for i in range(600)))
    run = c.collect_run(tmp_path, {'id': 'long', 'path': 'results/long'}, {'history_limit': 20})
    assert len(run['history']) == 20
    assert run['tokens']['processed'] is None
    assert run['history'][-1]['processed_tokens'] is None
    assert run['warnings']
