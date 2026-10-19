"""Synthetic stage receipts; never read raw rollout or checkpoint payloads."""
import importlib.util
import json
from pathlib import Path


def collector():
    path = Path(__file__).parents[1] / 'collectors.py'
    spec = importlib.util.spec_from_file_location('stage_collectors_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(root, name, document):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


def stage_spec(stage='sft', **extra):
    return dict(id='pilot', path='results/pilot', stage=stage, format='bridge', rank_count=2, **extra)


def test_sft_global_delta_counters_and_weighted_loss_not_rank_multiplied(tmp_path):
    c = collector()
    for rank, local_loss, weight in ((0, 2, 1), (1, 6, 3)):
        put(tmp_path, f'results/pilot/rank-{rank}-train-complete.json', {
            'rank': rank, 'updates': 2,
            'history': [{'rank': rank, 'update': step, 'loss': local_loss,
                         'loss_tokens': weight, 'processed_tokens': 8,
                         'global_loss': 5, 'global_loss_tokens': 4,
                         'global_processed_tokens': 16, 'lr': .001}
                        for step in (1, 2)]})
    run = c.collect_run(tmp_path, stage_spec(), {})
    assert run['metrics']['loss'] == 5
    assert run['tokens']['processed'] == 32
    assert run['tokens']['loss_bearing'] == 8
    assert run['history'][0]['processed_tokens'] == 16
    assert run['history'][1]['loss_bearing_tokens'] == 8
    assert run['status'] == 'completed'
    assert run['provenance']['metrics_scope'] == 'global_loss_token_weighted'
    assert run['provenance']['counter_scope'] == 'global_deltas_accumulated_once_per_update'


def test_mid_failed_exact_producer_receipt_overrides_completed_history(tmp_path):
    c = collector()
    for rank in range(2):
        put(tmp_path, f'results/pilot/rank-{rank}-train-complete.json', {
            'rank': rank, 'updates': 1, 'history': [{'update': 1, 'global_loss': 3,
            'global_loss_tokens': 4, 'global_processed_tokens': 4}]})
    put(tmp_path, 'results/pilot/container-state.json', {
        'state': {'Status': 'exited', 'ExitCode': 1}, 'config': {'Password': 'DO_NOT_SEND'}})
    run = c.collect_run(tmp_path, stage_spec('mid'), {})
    assert run['status'] == 'failed'
    assert 'DO_NOT_SEND' not in json.dumps(run)
    put(tmp_path, 'results/pilot/container-state.json', {'state': {'Status': 'exited', 'ExitCode': 0}})
    assert c.collect_run(tmp_path, stage_spec('mid'), {})['status'] == 'completed'


def test_sft_zero_exit_without_rank_completion_does_not_claim_success(tmp_path):
    c = collector()
    put(tmp_path, 'results/pilot/container-state.json', {'state': {'Status': 'exited', 'ExitCode': 0}})
    run = c.collect_run(tmp_path, stage_spec(), {})
    assert run['status'] != 'completed'
    assert any('rank train-complete receipts missing' in warning for warning in run['warnings'])


def test_rlvr_receipt_only_is_blocked_not_completed_and_never_reads_payloads(tmp_path, monkeypatch):
    c = collector()
    put(tmp_path, 'results/pilot/parent-verification.json', {
        'status': 'verified_rollout_only_no_signal', 'native_training_completed': False,
        'optimizer_updates': 0, 'scheduler_samples': 0, 'unique_rollouts': 4,
        'sampling_calls_per_rank': 64, 'all_rank_sampling_calls': 512,
        'ranks': 8, 'rewards': [0, 0, 0, 0], 'prompt': 'SECRET_PROMPT',
        'engine_state': {'Status': 'exited', 'ExitCode': 0}})
    put(tmp_path, 'results/pilot/rank-0-terminal.json', {
        'status': 'intentional_native_abort_rollout_only', 'native_training_completed': False,
        'counters': {'rollout_rounds': 1, 'rollouts': 4, 'optimizer_updates': 0, 'scheduler_samples': 0},
        'code': 'SECRET_CODE'})
    real_open = Path.open
    def safe_open(path, *args, **kwargs):
        assert path.name in ('parent-verification.json', 'rank-0-terminal.json'), path
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', safe_open)
    run = c.collect_run(tmp_path, dict(id='pilot', path='results/pilot', stage='rlvr', format='rlvr'), {})
    assert run['status'] == 'observed'
    assert run['training_status'] == 'blocked'
    assert run['training_scope'] == 'rollout_only'
    assert run['progress']['step'] == 0
    assert run['metrics']['reward_mean'] == 0
    assert run['rollout_summary']['unique_rollouts'] == 4
    assert run['rollout_summary']['all_rank_sampling_calls'] == 512
    assert run['rollout_summary']['rollout_rounds'] == 1
    assert run['rollout_summary']['scheduler_samples'] == 0
    assert run['tokens']['processed'] is None
    assert 'SECRET' not in json.dumps(run)
    assert {s['path'] for s in run['provenance']['sources']} == {
        'results/pilot/parent-verification.json', 'results/pilot/rank-0-terminal.json'}


def test_host_timeline_require_exact_update_identity_and_do_not_export_args(tmp_path):
    c = collector()
    put(tmp_path, 'results/pilot/requested-config.json', {})
    phase = tmp_path / 'results/pilot/rank-0-phases.jsonl'
    rows = [dict(event=event, op='outer.forward', counter=1, rank=0,
                 update=3, host_timestamp_ns=1000000000 + delta,
                 host_monotonic_ns=500000000 + delta, args=['SECRET_PROMPT'])
            for event, delta in (('before', 0), ('after', 2000000))]
    rows.append(dict(event='before', op='outer.forward', counter=2, rank=0,
                     host_timestamp_ns=1003000000, host_monotonic_ns=503000000))
    phase.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    run = c.collect_run(tmp_path, stage_spec(), {})
    assert len(run['host_timeline']) == 1
    event = run['host_timeline'][0]
    assert event['update'] == 3
    assert event['duration_ms'] == 2
    assert event['scope'] == 'host_operation_not_gpu_kernel'
    assert event['rank'] == 0
    assert 'SECRET_PROMPT' not in json.dumps(run)


def test_stage_receipts_do_not_inherit_legacy_summary_success_or_wrong_rank(tmp_path):
    c = collector()
    put(tmp_path, 'results/pilot/verified-summary.json', {'status': 'passed', 'optimizer_updates': 1})
    assert c.collect_run(tmp_path, stage_spec(), {})['status'] != 'completed'
    for rank in range(2):
        put(tmp_path, f'results/pilot/rank-{rank}-train-complete.json', {
            'rank': 0, 'updates': 1, 'history': [{'update': 1, 'global_loss': 3,
            'global_loss_tokens': 4, 'global_processed_tokens': 4}]})
    assert c.collect_run(tmp_path, stage_spec(), {})['status'] != 'completed'


def test_host_timeline_preserves_real_operation_identity_without_fabricating_update(tmp_path):
    c = collector()
    put(tmp_path, 'results/pilot/requested-config.json', {})
    phase = tmp_path / 'results/pilot/rank-0-phases.jsonl'
    phase.write_text(''.join(json.dumps(dict(event=event, op='MoELayer.forward', counter=72, rank=0,
        host_timestamp_ns=1000000000 + delta, host_monotonic_ns=500000000 + delta)) + '\n'
        for event, delta in (('before', 0), ('after', 2000000))))
    run = c.collect_run(tmp_path, stage_spec(), {})
    assert len(run['host_timeline']) == 1
    event = run['host_timeline'][0]
    assert event['operation_id'] == 72
    assert event['update'] is None
    assert event['iteration_identity'] == 'unknown'
    assert event['duration_ms'] == 2


def test_rlvr_contradictory_or_failed_receipts_do_not_claim_verified_zero_updates(tmp_path):
    c = collector()
    verification = {'status': 'verified_rollout_only_no_signal', 'native_training_completed': False,
                    'optimizer_updates': 1, 'scheduler_samples': 0, 'unique_rollouts': 4,
                    'rewards': [0, 0, 0, 0], 'engine_state': {'Status': 'exited', 'ExitCode': 0}}
    put(tmp_path, 'results/pilot/parent-verification.json', verification)
    put(tmp_path, 'results/pilot/rank-0-terminal.json', {'native_training_completed': False,
        'status': 'intentional_native_abort_rollout_only',
        'counters': {'rollouts': 4, 'optimizer_updates': 0, 'scheduler_samples': 0}})
    spec = dict(id='pilot', path='results/pilot', stage='rlvr', format='rlvr')
    run = c.collect_run(tmp_path, spec, {})
    assert run['status'] == 'unknown'
    assert any('inconsistent' in warning for warning in run['warnings'])
    verification.update(status='failed', engine_state={'Status': 'exited', 'ExitCode': 1})
    put(tmp_path, 'results/pilot/parent-verification.json', verification)
    assert c.collect_run(tmp_path, spec, {})['status'] == 'failed'


def test_complete_stage_receipts_survive_bounded_or_unsupported_update_logs(tmp_path):
    c = collector()
    for rank in range(2):
        put(tmp_path, f'results/pilot/rank-{rank}-train-complete.json', {'rank': rank, 'updates': 1,
            'history': [{'update': 1, 'global_loss': 5, 'global_loss_tokens': 4, 'global_processed_tokens': 16}]})
        (tmp_path / f'results/pilot/rank-{rank}-updates.jsonl').write_text('x' * (c.MAX_TAIL + 1000))
    run = c.collect_run(tmp_path, stage_spec(), {})
    assert run['tokens']['processed'] == 16
    assert run['history'][0]['loss_bearing_tokens'] == 4
    assert run['status'] == 'completed'
    assert any('bounded' in warning.lower() for warning in run['warnings'])


def test_host_pairs_include_thread_identity_under_nested_calls(tmp_path):
    c = collector()
    put(tmp_path, 'results/pilot/requested-config.json', {})
    rows = [dict(event=event, op='outer.forward', counter=9, rank=0, thread_id=thread,
                 host_timestamp_ns=1000000000 + delta, host_monotonic_ns=500000000 + delta)
            for event, thread, delta in (('before', 1, 0), ('before', 2, 1000000),
                                         ('after', 1, 3000000), ('after', 2, 5000000))]
    (tmp_path / 'results/pilot/rank-0-phases.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    events = c.collect_run(tmp_path, stage_spec(), {})['host_timeline']
    assert len(events) == 2
    assert [(event['thread_id'], event['duration_ms']) for event in events] == [(1, 3), (2, 4)]
