"""Canonical v1 read-only ingestion; no trainer integration or auto-emission."""
import json
import datetime as dt
import math
import re
from pathlib import Path
from collections import OrderedDict

SCHEMA = json.loads(Path(__file__).with_name('telemetry.schema.json').read_text())


def _validate(value, schema):
    if '$ref' in schema:
        schema = SCHEMA['$defs'][schema['$ref'].rsplit('/', 1)[-1]]
    kinds = schema.get('type', [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    valid = {'object': isinstance(value, dict), 'string': isinstance(value, str),
             'integer': type(value) is int, 'number': type(value) in (int, float) and math.isfinite(value), 'null': value is None}
    if kinds and not any(valid[kind] for kind in kinds):
        raise ValueError('Invalid telemetry type')
    if 'const' in schema and value != schema['const'] or 'enum' in schema and value not in schema['enum']:
        raise ValueError('Invalid telemetry literal')
    if isinstance(value, dict):
        props = schema.get('properties', {})
        if set(schema.get('required', [])) - set(value) or schema.get('additionalProperties') is False and set(value) - set(props):
            raise ValueError('Invalid telemetry fields')
        for key, item in value.items():
            _validate(item, props[key])
    if isinstance(value, str):
        if not schema.get('minLength', 0) <= len(value) <= schema.get('maxLength', 4096) or 'pattern' in schema and not re.fullmatch(schema['pattern'], value):
            raise ValueError('Invalid telemetry string')
        if schema.get('format') == 'date-time':
            try:
                parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
                if parsed.tzinfo is None:
                    raise ValueError
            except ValueError:
                raise ValueError('Invalid telemetry timestamp') from None
    if type(value) in (int, float):
        if not math.isfinite(value) or value < schema.get('minimum', -math.inf) or value > schema.get('maximum', math.inf):
            raise ValueError('Invalid telemetry number')


def validate_manifest(value):
    _validate(value, SCHEMA['$defs']['manifest'])
    return value


def validate_record(value, manifest):
    _validate(value, SCHEMA['$defs']['train_step'])
    if value['run_id'] != manifest['run_id'] or value['stage'] != manifest['stage'] or value['rank'] >= value['world_size']:
        raise ValueError('Telemetry identity or rank mismatch')
    return value

MAX_MANIFEST = 65536
MAX_READ = 262144
MAX_LINE = 65536
MAX_HISTORY = 512
_CACHE = OrderedDict()


def _decode(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicate telemetry key')
            obj[key] = value
        return obj
    return json.loads(raw, object_pairs_hook=unique)


def validate_sequence(row, previous):
    if previous is None:
        return
    if row['step'] <= previous['step']:
        raise ValueError('Duplicate or nonmonotonic step')
    for field in ('rank', 'world_size', 'metrics_scope'):
        if row[field] != previous[field]:
            raise ValueError('Designated writer or scope changed')
    if dt.datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')) < dt.datetime.fromisoformat(previous['timestamp'].replace('Z', '+00:00')):
        raise ValueError('Nonmonotonic timestamp')
    if any(row['counters'][key] < previous['counters'][key] for key in ('processed', 'loss_bearing', 'source_read')):
        raise ValueError('Nonmonotonic global cumulative counter')


def read_canonical(manifest_path, metrics_path):
    """Read at most 64 KiB manifest and 256 KiB log per poll. Never tail-skip.

    Invalid complete records or rotation latch rejection for this cache entry;
    no later records are accepted. Partial final lines retry on next append.
    Cache eviction/restart revalidates the stream from the beginning.
    """
    key = (str(manifest_path), str(metrics_path))
    state = _CACHE.get(key)
    if state is None:
        state = dict(offset=0, pending=b'', rows=[], manifest=None, manifest_signature=None,
                     signature=None, rejected=False, history_cut=False)
        _CACHE[key] = state
    _CACHE.move_to_end(key)
    while len(_CACHE) > 128:
        _CACHE.popitem(last=False)
    warnings = []
    try:
        stat = manifest_path.stat()
        signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if signature != state['manifest_signature']:
            if stat.st_size > MAX_MANIFEST:
                raise ValueError('Manifest exceeds read bound')
            with manifest_path.open('rb') as stream:
                manifest = validate_manifest(_decode(stream.read(MAX_MANIFEST + 1)))
            if state['manifest'] and any(manifest[k] != state['manifest'][k] for k in ('run_id', 'stage')):
                raise ValueError('Manifest identity changed')
            state['manifest'] = manifest
            state['manifest_signature'] = signature
        stat = metrics_path.stat()
        signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        previous = state['signature']
        if previous and (stat.st_ino != previous[0] or stat.st_size < previous[2] or stat.st_size == previous[2] and stat.st_mtime_ns != previous[1]):
            raise ValueError('Metrics stream rotated or rewritten')
        if not state['rejected'] and state['offset'] < stat.st_size:
            with metrics_path.open('rb') as stream:
                stream.seek(state['offset'])
                raw = stream.read(MAX_READ)
            state['offset'] += len(raw)
            lines = (state['pending'] + raw).split(b'\n')
            state['pending'] = lines.pop()
            for line in lines:
                if len(line) > MAX_LINE:
                    raise ValueError('Telemetry line exceeds bound')
                row = validate_record(_decode(line), state['manifest'])
                validate_sequence(row, state['rows'][-1] if state['rows'] else None)
                state['rows'].append(row)
                if len(state['rows']) > MAX_HISTORY:
                    state['rows'].pop(0)
                    state['history_cut'] = True
            if len(state['pending']) > MAX_LINE:
                raise ValueError('Telemetry pending line exceeds bound')
        state['signature'] = signature
    except OSError:
        warnings.append('Canonical source unavailable; last-good validated prefix retained')
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        state['rejected'] = True
        state['pending'] = b''
    if state['rejected']:
        warnings.append('Canonical telemetry rejected; last-good validated prefix retained; later records blocked')
    if state['pending']:
        warnings.append('Canonical trailing partial JSON line skipped until complete')
    if state['history_cut'] or state['signature'] and state['offset'] < state['signature'][2]:
        warnings.append('Canonical read/history bounded; cumulative counters are reported directly, not summed')
    if state['manifest'] is None:
        raise ValueError('Canonical manifest unavailable or invalid')
    import copy
    return copy.deepcopy(state['manifest']), copy.deepcopy(state['rows']), warnings, bool(state['history_cut'] or state['pending'] or state['rejected'] or state['signature'] and state['offset'] < state['signature'][2])


def validate_files(manifest_path, metrics_path):
    """Stream CLI validation with bounded line memory; never writes files."""
    with Path(manifest_path).open('rb') as stream:
        raw = stream.read(MAX_MANIFEST + 1)
    if len(raw) > MAX_MANIFEST:
        raise ValueError('Manifest exceeds bound')
    manifest = validate_manifest(_decode(raw))
    previous = None
    count = 0
    warnings = []
    with Path(metrics_path).open('rb') as stream:
        while True:
            line = stream.readline(MAX_LINE + 2)
            if not line:
                break
            if len(line.rstrip(b'\n')) > MAX_LINE:
                raise ValueError('Telemetry line exceeds bound')
            if not line.endswith(b'\n'):
                warnings.append('Trailing partial JSON line skipped until complete')
                break
            row = validate_record(_decode(line), manifest)
            validate_sequence(row, previous)
            previous = row
            count += 1
    return dict(valid=True, schema_version=1, records_validated=count, warnings=warnings)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Read-only validation of canonical files')
    validate.add_argument('--manifest', required=True)
    validate.add_argument('--metrics', required=True)
    args = parser.parse_args()
    try:
        result = validate_files(args.manifest, args.metrics)
    except (OSError, ValueError, UnicodeError, RecursionError, OverflowError):
        print(json.dumps(dict(valid=False, schema_version=1, error='Canonical telemetry validation failed; input values withheld')))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
