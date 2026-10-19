"""Explicit MID/SFT and RLVR read-only receipt adapters.

No trainer imports, filesystem discovery, or raw rollout payload reads.
"""


def aggregate_stage_history(rank_rows, expected, limit, number):
    """Repeated producer global fields are per-update deltas, not rank deltas."""
    groups = {}
    for rank, rows in rank_rows.items():
        for row in rows:
            step = number(row.get('update'))
            if step is not None:
                groups.setdefault(step, {})[rank] = row
    totals = dict(processed=0 if groups else None,
                  loss_bearing=0 if groups else None, source_read=None)
    history = []
    for step, by_rank in sorted(groups.items()):
        rows = list(by_rank.values())
        complete = len(rows) == expected
        def global_value(key):
            values = [number(row.get(key)) for row in rows]
            return values[0] if complete and values and values[0] is not None and all(v == values[0] for v in values) else None
        for counter, field in (('processed', 'global_processed_tokens'), ('loss_bearing', 'global_loss_tokens')):
            value = global_value(field)
            totals[counter] = totals[counter] + value if value is not None and totals[counter] is not None else None
        first = rows[0]
        history.append(dict(step=step, kind=None, loss=global_value('global_loss'),
                            learning_rate=number(first.get('lr')), grad_norm=number(first.get('grad_norm')),
                            processed_tokens=totals['processed'], loss_bearing_tokens=totals['loss_bearing'],
                            source_read_tokens=None, elapsed_seconds=None))
    return history[-limit:], totals


def finish_bridge_stage(run, receipts, expected, terminal, number, limit):
    """Train completion is separate from whole-stage completion."""
    complete = len(receipts) == expected and all(
        receipt.get('rank') == rank and type(receipt.get('updates')) is int and receipt['updates'] > 0
        and isinstance(receipt.get('history'), list)
        and receipt['updates'] == run['progress']['step']
        and any(isinstance(row, dict) and row.get('update') == receipt['updates'] for row in receipt['history'])
        for rank, receipt in enumerate(receipts))
    if complete and not any(source['truncated'] for source in run['provenance']['sources']
                            if source['kind'].endswith('-train-complete')):
        # Exact small terminal histories remain authoritative when live logs
        # were unsupported or their bounded tails omitted earlier deltas.
        history, totals = aggregate_stage_history(
            {rank: receipt['history'] for rank, receipt in enumerate(receipts)}, expected, limit, number)
        run['history'] = history
        run['tokens'].update(totals)
        if history:
            run['metrics'].update(loss=history[-1]['loss'], learning_rate=history[-1]['learning_rate'], grad_norm=history[-1]['grad_norm'])
    if run['status'] == 'completed' and not complete:
        run['status'] = 'observed'
        run['warnings'].append('Legacy summary does not replace exact per-rank train-complete receipts')
    state = terminal.get('state', terminal.get('State', terminal.get('producer', {}).get('state', {})))
    failed = terminal.get('producer_run_success') is False or terminal.get('timed_out') or state.get('OOMKilled') or (
        state.get('Status') == 'exited' and state.get('ExitCode') != 0)
    if failed:
        run['status'] = 'failed'
    elif complete:
        run['status'] = 'completed'
    elif state.get('Status') == 'exited':
        run['status'] = 'observed' if run['status'] == 'unknown' else 'stale'
        run['warnings'].append('Producer exited but rank train-complete receipts missing; training completion unavailable')
    run['provenance']['metrics_scope'] = 'global_loss_token_weighted'
    run['provenance']['counter_scope'] = 'global_deltas_accumulated_once_per_update'
    run['warnings'] = [w for w in run['warnings'] if not w.startswith('Legacy loss')]
    run['warnings'].append('Bounded stage pilot only; train-complete is not whole-stage completion')
    run['timing']['scope'] = 'No per-update duration timestamps; throughput and GPU kernel chronology unavailable'


HOST_OPERATIONS = frozenset({
    'megatron.bridge.training.train.train_step', 'outer.forward', 'outer.backward',
    'MoELayer.forward', 'MoEAlltoAllTokenDispatcher.dispatch_preprocess',
    'MoEAlltoAllTokenDispatcher.preprocess', 'MoEAlltoAllTokenDispatcher.token_dispatch',
    'MoEAlltoAllTokenDispatcher.token_combine', 'MoEAlltoAllTokenDispatcher.combine_postprocess',
})


def import_host_phases(root, spec, run, observed_artifact):
    """Pair bounded host calls; unknown training update stays explicitly null."""
    from pathlib import Path
    run['host_timeline'] = []
    source = str(Path(spec['path']) / 'rank-0-phases.jsonl')
    try:
        rows, timestamp, cut = observed_artifact(root, source, run, 'host_phases', True)
    except (ValueError, TypeError):
        run['warnings'].append('Host phase path rejected')
        return
    if timestamp is None:
        return
    pending = {}
    for row in rows:
        update = row.get('update', row.get('iteration'))
        if update is not None and (type(update) is not int or update < 1):
            continue
        if (type(row.get('counter')) is not int
                or type(row.get('rank')) is not int or row['rank'] != 0
                or row.get('op') not in HOST_OPERATIONS
                or type(row.get('host_timestamp_ns')) is not int
                or type(row.get('host_monotonic_ns')) is not int):
            continue
        if row.get('thread_id') is not None and type(row['thread_id']) is not int:
            continue
        key = (update, row['op'], row['counter'], row['rank'], row.get('thread_id'))
        if row.get('event') == 'before':
            pending[key] = row
        elif row.get('event') == 'after' and key in pending:
            before = pending.pop(key)
            delta = row['host_monotonic_ns'] - before['host_monotonic_ns']
            if delta >= 0 and row['host_timestamp_ns'] >= before['host_timestamp_ns']:
                run['host_timeline'].append(dict(
                    name=row['op'], rank=0, thread_id=row.get('thread_id'), update=update, operation_id=row['counter'],
                    iteration_identity='producer_update' if update is not None else 'unknown',
                    start_ns=before['host_timestamp_ns'], end_ns=row['host_timestamp_ns'],
                    duration_ms=delta / 1000000, source=source, scope='host_operation_not_gpu_kernel'))
    run['host_timeline'] = run['host_timeline'][-128:]
    if cut:
        run['warnings'].append('Host phase history bounded; omitted events do not invalidate training receipt counters')
    if not run['host_timeline']:
        run['warnings'].append('Host operations exist but exact update identity/pairs unavailable; no GPU kernel timestamps inferred')


def collect_rlvr(root, spec, config, empty_run, observed_artifact, number):
    """Import safe verification counters only, never rollouts/records/config."""
    from pathlib import Path
    run = empty_run(spec)
    if spec.get('stage') != 'rlvr':
        run['warnings'].append('RLVR adapter requires explicitly configured rlvr stage')
        return run
    documents = []
    for name in ('parent-verification.json', 'rank-0-terminal.json'):
        try:
            document, _, cut = observed_artifact(root, str(Path(spec['path']) / name), run, Path(name).stem)
        except (ValueError, TypeError):
            run['warnings'].append('RLVR receipt path rejected')
            return run
        documents.append(document)
        if cut:
            run['warnings'].append('RLVR receipt malformed or exceeds bounded read')
    verified, terminal = documents
    run['updated_at'] = max((s['updated_at'] for s in run['provenance']['sources'] if s['updated_at']), default=None)
    run['provenance'].update(metrics_scope='verified_unique_rollout_rewards', counter_scope='explicit_global_receipt_no_rank_multiplication')
    run['data'].update(name=spec.get('data_name'), kind='verified rollout diagnostic', scope='rollout_only; not RLVR policy learning or SFT-to-RLVR lineage')
    run['timing']['scope'] = 'Rollout verification receipts only; no optimizer throughput or GPU timeline'
    run['rollout_summary'] = {field: number(verified.get(field)) for field in (
        'unique_rollouts', 'sampling_calls_per_rank', 'all_rank_sampling_calls', 'optimizer_updates', 'scheduler_samples')}
    run['rollout_summary']['rollout_rounds'] = number(terminal.get('counters', {}).get('rollout_rounds'))
    run['progress']['step'] = number(verified.get('optimizer_updates'))
    rewards = verified.get('rewards', [])
    values = [number(value) for value in rewards[:512]] if isinstance(rewards, list) else []
    run['metrics']['reward_mean'] = sum(values) / len(values) if values and all(v is not None for v in values) else None
    state = verified.get('engine_state', {})
    if verified.get('status') in ('failed', 'error') or state.get('OOMKilled') or (
            state.get('Status') == 'exited' and state.get('ExitCode') != 0):
        run['status'] = 'failed'
        run['warnings'].append('Exact RLVR producer receipt failed; no policy training completion inferred')
    elif verified.get('status') == 'verified_rollout_only_no_signal':
        counters = terminal.get('counters', {})
        consistent = verified.get('native_training_completed') is False and terminal.get('native_training_completed') is False and (
            terminal.get('status') == 'intentional_native_abort_rollout_only') and all(
                number(verified.get(field)) == 0 and number(counters.get(field)) == 0
                for field in ('optimizer_updates', 'scheduler_samples')) and (
                number(verified.get('unique_rollouts')) == number(counters.get('rollouts')))
        if not consistent:
            run['warnings'].append('RLVR verification/terminal receipts inconsistent or incomplete; rollout_only outcome not asserted')
            return run
        run['status'] = 'observed'
        run['training_status'] = 'blocked'
        run['training_scope'] = 'rollout_only'
        run['native_training_completed'] = False
        run['warnings'].append('RLVR blocked / rollout_only: no reward signal; policy training not completed; optimizer_updates and scheduler_samples are zero')
    return run
