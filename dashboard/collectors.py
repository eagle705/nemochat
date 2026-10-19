"""Read-only, bounded local artifact collection. No training dependencies."""
import datetime as dt
import json
import math
from collections import OrderedDict
import os
from pathlib import Path
import socket
import subprocess
import csv
import time


def utc(timestamp=None):
    return dt.datetime.fromtimestamp(time.time() if timestamp is None else timestamp, dt.timezone.utc).isoformat()


def empty_run(spec):
    return {'id': spec['id'], 'name': spec.get('name', spec['id']), 'stage': {'lc': 'long', 'long_context': 'long'}.get(spec.get('stage'), spec.get('stage', 'pretrain')),
            'status': 'unknown', 'source': spec['path'], 'updated_at': None,
            'model': dict(name=None, parameters_total=None, parameters_active=None, precision=None),
            'data': dict(name=None, kind=None, scope=None),
            'tokens': dict(processed=None, loss_bearing=None, source_read=None, target=None),
            'progress': dict(step=None, total_steps=None),
            'timing': dict(elapsed_seconds=None, tokens_per_second=None, eta_seconds=None, scope=None),
            'metrics': dict(loss=None, learning_rate=None, grad_norm=None),
            'history': [], 'benchmarks': [], 'warnings': [],
            'provenance': dict(adapter=spec.get('format', 'bridge'), schema_version=1 if spec.get('format') == 'canonical' else None, sources=[], metrics_scope=None if spec.get('format') == 'canonical' else 'unweighted_mean_of_rank_local', counter_scope='global_cumulative' if spec.get('format') == 'canonical' else 'legacy_rank_deltas_or_explicit_global_cumulative_receipts')}


_CPU_PREVIOUS = None


def collect_host(config):
    global _CPU_PREVIOUS
    utilization = None
    warnings = []
    memory = dict(total_bytes=None, used_bytes=None)
    model = None
    try:
        values = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
        memory = dict(total_bytes=values['MemTotal'], used_bytes=values['MemTotal'] - values['MemAvailable'])
        model = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), None)
        ticks = [int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
        current = (sum(ticks), ticks[3] + ticks[4])
        if _CPU_PREVIOUS:
            total_delta = current[0] - _CPU_PREVIOUS[0]
            idle_delta = current[1] - _CPU_PREVIOUS[1]
            utilization = max(0, min(100, 100*(total_delta-idle_delta)/total_delta)) if total_delta > 0 else 0
        _CPU_PREVIOUS = current
    except (OSError, ValueError, KeyError):
        warnings.append('Host CPU/memory unavailable')
    gpus = []
    if config.get('gpu_enabled', True):
        fields = ['index', 'name', 'uuid', 'utilization_percent', 'memory_used_mb', 'memory_total_mb', 'temperature_c', 'power_w']
        query = 'index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw'
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=' + query, '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=min(3, max(.1, float(config.get('nvidia_smi_timeout_seconds', 1.5)))), check=True)
            for row in list(csv.reader(result.stdout[:65536].splitlines()))[:64]:
                if len(row) != len(fields):
                    continue
                gpu = {}
                for key, value in zip(fields, row):
                    value = value.strip()
                    if key in ('name', 'uuid'):
                        gpu[key] = value
                    else:
                        try:
                            gpu[key] = number(float(value))
                        except ValueError:
                            gpu[key] = None
                gpus.append(gpu)
        except (OSError, subprocess.SubprocessError):
            warnings.append('nvidia-smi unavailable, failed or timed out; no CUDA context created')
    return dict(hostname=socket.gethostname(), cpu=dict(model=model, logical_count=os.cpu_count(), utilization_percent=utilization), memory=memory, gpus=gpus, sampled_at=utc(), warnings=warnings)


# Cache lifetime is the one collector process, not an HTTP request.
_CACHE = OrderedDict()
MAX_JSON = 2 * 1024 * 1024
MAX_TAIL = 256 * 1024
MAX_HISTORY = 512


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def secure_path(root, relative):
    root = Path(root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or any(p.startswith('.') for p in rel.parts):
        raise ValueError('Only explicit relative artifact paths are permitted')
    result = (root / rel).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Artifact path escapes repository')
    return result


def read_artifact(path, jsonl=False):
    """Stat-cached JSON and append-only bounded JSONL tails; never glob."""
    try:
        stat = path.stat()
    except OSError:
        return [] if jsonl else {}, None, False
    key = (str(path), jsonl)
    signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    previous = _CACHE.get(key)
    if previous and previous['signature'] == signature:
        _CACHE.move_to_end(key)
        return previous['value'], stat.st_mtime, previous['truncated']
    truncated = False
    try:
        if jsonl:
            append = previous and previous['signature'][0] == stat.st_ino and stat.st_size >= previous['offset'] and previous['signature'][2] < stat.st_size
            start = previous['offset'] if append else max(0, stat.st_size - MAX_TAIL)
            if stat.st_size - start > MAX_TAIL:
                start = stat.st_size - MAX_TAIL
                append = False
            truncated = bool(start and not append) or bool(append and previous['truncated'])
            with path.open('rb') as stream:
                stream.seek(start)
                raw = stream.read(MAX_TAIL)
            prefix = previous['pending'] if append else b''
            raw = prefix + raw
            if start and not append:
                raw = raw.split(b'\n', 1)[-1]
            lines = raw.split(b'\n')
            pending = lines.pop()
            if len(pending) > MAX_TAIL:
                pending = pending[-MAX_TAIL:]
                truncated = True
            value = list(previous['value']) if append else []
            for line in lines:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        value.append(item)
                except (ValueError, UnicodeError):
                    continue
            if len(value) > MAX_HISTORY:
                value = value[-MAX_HISTORY:]
                truncated = True
        else:
            if stat.st_size > MAX_JSON:
                return {}, stat.st_mtime, True
            value = json.loads(path.read_bytes())
            if not isinstance(value, dict):
                value = {}
            pending = b''
        _CACHE[key] = dict(signature=signature, value=value, offset=stat.st_size, pending=pending, truncated=truncated)
        _CACHE.move_to_end(key)
        while len(_CACHE) > 2048:
            _CACHE.popitem(last=False)
        return value, stat.st_mtime, truncated
    except (OSError, ValueError, UnicodeError):
        return [] if jsonl else {}, stat.st_mtime, True


def aggregate_history(rank_rows, expected, limit):
    """One row per rank/update; global counters only from complete evidence."""
    groups = {}
    for rank, rows in rank_rows.items():
        for row in rows:
            step = number(row.get('update', row.get('step')))
            if step is not None:
                groups.setdefault(step, {})[rank] = row
    history = []
    cumulative = dict(processed=0 if groups else None, loss_bearing=0 if groups else None, source_read=0 if groups else None)
    elapsed = 0
    for step, ranks in sorted(groups.items()):
        rows = list(ranks.values())
        complete = len(rows) == expected
        for counter, aliases in [('processed', ('processed_tokens', 'input_token_count')), ('loss_bearing', ('loss_bearing_tokens',)), ('source_read', ('read_source_tokens', 'source_prefix_tokens'))]:
            values = [next((number(r[a]) for a in aliases if number(r.get(a)) is not None), None) for r in rows]
            explicit = next((number(r.get('global_loss_bearing_tokens_cumulative')) for r in rows if number(r.get('global_loss_bearing_tokens_cumulative')) is not None), None) if counter == 'loss_bearing' else None
            if explicit is not None:
                cumulative[counter] = explicit
            elif complete and all(v is not None for v in values) and cumulative[counter] is not None:
                cumulative[counter] += sum(values)
            else:
                cumulative[counter] = None
        seconds = [number(r.get('seconds', r.get('seconds_profiled_not_throughput'))) for r in rows]
        elapsed = elapsed + max(seconds) if elapsed is not None and all(v is not None for v in seconds) else None
        losses = [number(r.get('loss', r.get('local_loss'))) for r in rows]
        first = rows[0]
        history.append(dict(step=step, kind=first.get('kind'), loss=sum(losses)/len(losses) if complete and all(v is not None for v in losses) else None,
                            learning_rate=number(first.get('lr')), grad_norm=number(first.get('grad_norm')),
                            processed_tokens=cumulative['processed'], loss_bearing_tokens=cumulative['loss_bearing'],
                            source_read_tokens=cumulative['source_read'], elapsed_seconds=elapsed))
    return history[-limit:], cumulative


def observed_artifact(root, relative, run, kind, jsonl=False):
    path = secure_path(root, relative)
    value, timestamp, cut = read_artifact(path, jsonl)
    run['provenance']['sources'].append(dict(path=relative, kind=kind, available=timestamp is not None, updated_at=utc(timestamp) if timestamp is not None else None, truncated=cut))
    return value, timestamp, cut


def collect_canonical(root, spec, config):
    try:
        from dashboard.telemetry import read_canonical
    except ImportError:
        from telemetry import read_canonical
    run = empty_run(spec)
    run['provenance'] = dict(adapter='canonical', schema_version=1, sources=[], metrics_scope=None, counter_scope='global_cumulative')
    try:
        paths = [secure_path(root, str(Path(spec['path']) / name)) for name in ('run.json', 'metrics.jsonl')]
    except (ValueError, TypeError):
        run['warnings'].append('Canonical artifact path rejected')
        return run
    for name, path in zip(('manifest', 'train_step'), paths):
        try:
            stat = path.stat()
            timestamp = utc(stat.st_mtime)
        except OSError:
            timestamp = None
        run['provenance']['sources'].append(dict(path=str(path.relative_to(Path(root).resolve())), kind=name, available=timestamp is not None, updated_at=timestamp, truncated=False))
    try:
        manifest, rows, warnings, cut = read_canonical(*paths)
        if manifest['run_id'] != spec['id'] or manifest['stage'] != {'lc': 'long', 'long_context': 'long'}.get(spec.get('stage'), spec.get('stage', 'pretrain')):
            raise ValueError('Canonical configured identity mismatch')
    except (OSError, ValueError, TypeError):
        run['warnings'].append('Canonical telemetry rejected or unavailable')
        return run
    run['name'] = manifest['name']
    run['stage'] = manifest['stage']
    run['model'].update(manifest.get('model', {}))
    run['data'].update(manifest.get('data', {}))
    run['progress']['total_steps'] = manifest.get('total_steps')
    run['tokens']['target'] = manifest.get('target_tokens')
    run['warnings'].extend(warnings)
    run['provenance']['sources'][1]['truncated'] = cut
    for row in rows[-min(MAX_HISTORY, max(1, int(config.get('history_limit', 256)))):]:
        run['history'].append(dict(step=row['step'], timestamp=row['timestamp'], rank=row['rank'], metrics_scope=row['metrics_scope'], **row['metrics'], processed_tokens=row['counters']['processed'], loss_bearing_tokens=row['counters']['loss_bearing'], source_read_tokens=row['counters']['source_read'], elapsed_seconds=None, step_seconds=row['timing']['step_seconds']))
    if rows:
        row = rows[-1]
        run['progress']['step'] = row['step']
        run['tokens'].update({k: row['counters'][k] for k in ('processed', 'loss_bearing', 'source_read')})
        run['metrics'].update(row['metrics'])
        run['provenance']['metrics_scope'] = row['metrics_scope']
        if row['metrics_scope'] == 'rank_local':
            run['warnings'].append('rank_local metrics from designated writer; not a global reduced loss')
        run['status'] = 'observed'
    run['status'] = manifest.get('status', run['status'])
    run['updated_at'] = max((s['updated_at'] for s in run['provenance']['sources'] if s['updated_at']), default=None)
    run['timing']['scope'] = 'reported single step duration; no cumulative wall time or throughput inferred'
    return run


def collect_run(root, spec, config):
    if spec.get('format') == 'rlvr':
        try:
            from dashboard.stage_adapters import collect_rlvr
        except ImportError:
            from stage_adapters import collect_rlvr
        return collect_rlvr(root, spec, config, empty_run, observed_artifact, number)
    if spec.get('format') == 'canonical':
        return collect_canonical(root, spec, config)
    run = empty_run(spec)
    try:
        base = secure_path(root, spec['path'])
    except (ValueError, TypeError):
        run['warnings'].append('Run path rejected')
        return run
    if not base.is_dir():
        run['warnings'].append('Run directory unavailable')
        return run
    expected = min(64, max(1, int(spec.get('rank_count', 1))))
    times = []
    activity_times = []
    truncated = False
    def read(name, jsonl=False):
        nonlocal truncated
        try:
            path = secure_path(root, str(Path(spec['path']) / name))
        except ValueError:
            run['warnings'].append('Artifact symlink/path rejected')
            return [] if jsonl else {}
        value, timestamp, cut = observed_artifact(root, str(Path(spec['path']) / name), run, 'rank_updates' if jsonl and name.endswith('-updates.jsonl') else 'rank_phases' if jsonl else Path(name).stem, jsonl)
        if timestamp is not None:
            times.append(timestamp)
            if jsonl and value:
                activity_times.append(timestamp)
        truncated |= cut
        return value
    requested = read('requested-config.json')
    summary = read('verified-summary.json')
    data = read('data-verification.json')
    rank_rows = {rank: read(f'rank-{rank}-updates.jsonl', True) for rank in range(expected)}
    native_config = {}
    native_result = {}
    receipts = []
    terminal = {}
    if spec.get('format') == 'native':
        native_config = read('resolved_config.json')
        native_result = read('result.json')
        result_ranks = {r.get('rank'): r for r in native_result.get('ranks', []) if isinstance(r, dict)}
        rank_rows = {}
        for rank in range(expected):
            receipt = read(f'rank-{rank}.json') or result_ranks.get(rank, {})
            receipts.append(receipt)
            phases = read(f'rank-{rank}-phases.jsonl', True)
            live = [r for r in phases if r.get('event') == 'optimizer-update-completed']
            rank_rows[rank] = live + receipt.get('history', [])
        if spec.get('terminal_receipt'):
            terminal, timestamp, cut = observed_artifact(root, spec['terminal_receipt'], run, 'terminal_receipt')
            truncated |= cut
            if timestamp is not None:
                times.append(timestamp)
    stage_bridge = spec.get('format') == 'bridge' and spec.get('stage') in ('mid', 'sft')
    if stage_bridge:
        try:
            from dashboard.stage_adapters import aggregate_stage_history, finish_bridge_stage, import_host_phases
        except ImportError:
            from stage_adapters import aggregate_stage_history, finish_bridge_stage, import_host_phases
        terminal = read('container-state.json')
        if spec.get('terminal_receipt'):
            terminal, timestamp, cut = observed_artifact(root, spec['terminal_receipt'], run, 'terminal_receipt')
            truncated |= cut
            if timestamp is not None:
                times.append(timestamp)
        receipts = [read(f'rank-{rank}-train-complete.json') for rank in range(expected)]
        for rank, receipt in enumerate(receipts):
            rank_rows[rank] = rank_rows[rank] + receipt.get('history', [])
        history, counters = aggregate_stage_history(rank_rows, expected, min(MAX_HISTORY, max(1, int(config.get('history_limit', 256)))), number)
    else:
        history, counters = aggregate_history(rank_rows, expected, min(MAX_HISTORY, max(1, int(config.get('history_limit', 256)))))
    model = requested.get('model', {})
    train = requested.get('train', {})
    run['model'].update(name=model.get('hf_model_id') or model.get('_target_'), precision='bf16' if model.get('bf16') else None)
    for field in ('num_layers', 'hidden_size', 'num_moe_experts', 'moe_router_topk', 'seq_length', 'parameters_total', 'parameters_active'):
        run['model'][field] = number(model.get(field))
    if spec.get('model_census'):
        try:
            census, _, _ = observed_artifact(root, spec['model_census'], run, 'model_census')
            total = number(census.get('logical_numel_from_ep_deduplication'))
            if census.get('status') == 'passed' and total is not None and total > 0:
                run['model']['parameters_total'] = total
                run['model']['census_source'] = spec['model_census']
                run['warnings'].append('Parameter total from explicitly configured prior model census; not measured again in this run: ' + spec['model_census'])
        except (ValueError, TypeError, OSError):
            run['warnings'].append('Model census unavailable or path rejected')
    run['data'].update(name=spec.get('data_name'), kind='indexed documents' if data else None, scope=spec.get('scope', 'bounded pilot; not stage completion'), prepared_tokens=number(data.get('counts', {}).get('tokens')))
    total = number(train.get('train_iters'))
    run['progress']['total_steps'] = total
    batch, sequence = number(train.get('global_batch_size')), number(model.get('seq_length'))
    run['tokens']['target'] = total*batch*sequence if all(v is not None for v in (total,batch,sequence)) else None
    run['history'] = history
    run['tokens'].update(counters)
    if history:
        last = history[-1]
        run['progress']['step'] = last['step']
        run['metrics'].update(loss=last['loss'], learning_rate=last['learning_rate'], grad_norm=last['grad_norm'])
        run['timing'].update(elapsed_seconds=last['elapsed_seconds'], scope='max rank update durations, cumulative observed history; not end-to-end wall time')
    for field, alias in [('processed', 'processed_tokens'), ('loss_bearing', 'loss_tokens'), ('source_read', 'source_prefix_tokens')]:
        if number(summary.get(alias)) is not None:
            run['tokens'][field] = summary[alias]
    run['timing']['container_elapsed_seconds'] = number(summary.get('training_container_seconds'))
    if number(summary.get('optimizer_updates')) is not None:
        run['progress']['step'] = summary['optimizer_updates']
    if summary.get('status') == 'passed':
        run['status'] = 'completed'
    elif summary.get('status') in ('failed', 'error'):
        run['status'] = 'failed'
    elif history:
        run['status'] = 'running' if activity_times and time.time() - max(activity_times) <= config.get('stale_seconds', 120) else 'stale'
    elif times:
        run['status'] = 'observed'
    if spec.get('format') == 'native':
        plan = native_config.get('plan', {})
        contract = plan.get('contract') or native_result.get('contract', {})
        stop = number(plan.get('stop', native_config.get('argv', {}).get('stop')))
        horizon = number(contract.get('schedule', {}).get('total_steps'))
        run['progress']['total_steps'] = stop if stop is not None else horizon
        per_step = number(contract.get('loss_bearing_tokens_per_update'))
        run['tokens']['target'] = run['progress']['total_steps'] * per_step if run['progress']['total_steps'] is not None and per_step is not None else None
        run['model'].update(name=spec.get('model_name') or contract.get('variant'), precision=spec.get('precision'))
        run['data'].update(kind='document prefixes', scope=contract.get('selection'), prepared_tokens=number(plan.get('accounting', {}).get('prepared_train_tokens')))
        globals_ = [number(r.get('global_loss_bearing_tokens')) for r in receipts]
        if globals_ and all(v is not None and v == globals_[0] for v in globals_):
            run['tokens']['loss_bearing'] = globals_[0]
        statuses = [r.get('status', '') for r in receipts]
        if all(s == 'passed' or s.startswith('completed-native-updates') for s in statuses):
            run['status'] = 'completed'
        if any(s in ('failed', 'error') for s in statuses) or native_result.get('status') in ('failed', 'error'):
            run['status'] = 'failed'
        state = terminal.get('state', terminal.get('State', terminal.get('producer', {}).get('state', {})))
        if terminal.get('producer_run_success') is False or state.get('OOMKilled') or terminal.get('timed_out') or state.get('Status') == 'exited' and state.get('ExitCode') != 0:
            run['status'] = 'failed'
        elif state.get('Status') == 'exited' and state.get('ExitCode') == 0:
            run['status'] = 'completed'
        if terminal.get('checkpoint_integrity_verified'):
            run['warnings'].append('Checkpoint integrity receipt is separate from producer success and resume equivalence')
        if native_result.get('profile_contract'):
            profile_contract = native_result['profile_contract']
            updates = profile_contract.get('global_updates', [])
            if updates and per_step is not None:
                run['tokens']['target'] = len(set(updates)) * per_step
            for kind in ('warmup', 'measured'):
                selected = {rank: [r for r in rows if r.get('kind') == kind] for rank, rows in rank_rows.items()}
                _, category = aggregate_history(selected, expected, MAX_HISTORY)
                run['tokens'][kind + '_loss_bearing'] = category['loss_bearing']
            run['timing']['scope'] = 'diagnostic CPU update durations including warmup; NOT throughput benchmark or GPU utilization'
            run['warnings'].append('Warmup and measured diagnostic updates retained separately; no throughput inferred')
        for phase in ('heldout_before', 'heldout_after'):
            value = number(receipts[0].get(phase, {}).get('global_ce')) if receipts else None
            if value is not None:
                run['benchmarks'].append(dict(name=phase + '_ce', value=value, scope='fixed small heldout native diagnostic; not broad model quality'))
    if times:
        run['updated_at'] = utc(max(times))
    if truncated:
        run['warnings'].append('Artifact read/history bounded or malformed; totals may be incomplete')
        for row in run['history']:
            row['processed_tokens'] = None
            row['source_read_tokens'] = None
            # Explicit native cumulative receipts can still supply a reliable
            # run total, but a truncated delta history is not a global curve.
            row['loss_bearing_tokens'] = None
            row['elapsed_seconds'] = None
        run['timing']['elapsed_seconds'] = None
        for field in ('processed', 'loss_bearing', 'source_read'):
            if number(summary.get({'processed':'processed_tokens','loss_bearing':'loss_tokens','source_read':'source_prefix_tokens'}[field])) is None:
                run['tokens'][field] = None
    for artifact in spec.get('benchmarks', [])[:16]:
        try:
            document, _, cut = observed_artifact(root, artifact['path'], run, 'benchmark')
            if cut:
                run['warnings'].append('Benchmark artifact unavailable or exceeds read bound')
            for metric in artifact.get('metrics', [])[:32]:
                value = document
                for part in metric['field'].split('.')[:8]:
                    value = value.get(part) if isinstance(value, dict) else None
                value = number(value)
                if value is not None:
                    run['benchmarks'].append(dict(name=metric['name'], value=value, scope=metric.get('scope', 'imported local artifact; not dashboard-executed benchmark')))
        except (ValueError, KeyError, TypeError):
            run['warnings'].append('Benchmark path or numeric field rejected')
    run['warnings'].append('Legacy loss is an unweighted mean of rank-local losses when all ranks are present; not a globally target-weighted reduced loss')
    run['warnings'].append('Run-local evidence only; overlapping/resumed runs are never summed')
    if stage_bridge:
        finish_bridge_stage(run, receipts, expected, terminal, number, min(MAX_HISTORY, max(1, int(config.get('history_limit', 256)))))
        import_host_phases(root, spec, run, observed_artifact)
    return run


def collect_snapshot(root: Path, config: dict) -> dict:
    started = time.monotonic()
    runs = [collect_run(root, spec, config) for spec in config.get('runs', [])[:32]]
    stages = [dict(id=s, name=n, status='observed' if any(r['stage'] == s and r['status'] != 'unknown' for r in runs) else 'not_observed', run_ids=[r['id'] for r in runs if r['stage'] == s]) for s, n in [('data', 'Data preparation'), ('pretrain', 'Pretraining'), ('mid', 'Mid-training'), ('long', 'Long context'), ('sft', 'SFT'), ('rlvr', 'RLVR'), ('mopd', 'MOPD')]]
    return dict(schema_version=1, generated_at=utc(), collection=dict(interval_seconds=max(2, config.get('interval_seconds', 5)), duration_ms=(time.monotonic() - started)*1000, warnings=[]), host=collect_host(config), runs=runs, stages=stages, insights=[], profiling={k: dict(available=False, source=None, warnings=[], **{field: []}) for k, field in [('lens','spans'),('nsight','kernels'),('bridge','operations')]})
