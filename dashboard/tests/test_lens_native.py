"""Read-only Lens interface tests; fixtures are not native training evidence."""
import importlib.util
import json
from pathlib import Path
import pytest
import os
import subprocess


PROBE = r'''
import json, sys, socket
from pathlib import Path
# A host-only local interface probe, not training or a benchmark.
def denied(*args, **kwargs):
    raise RuntimeError('Network forbidden in interface probe')
socket.socket.connect = denied
from nemo.lens import NemoLensConfig, SpanRegistry, managed_span, setup_telemetry
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from google.protobuf.json_format import MessageToDict
from importlib.metadata import version
out = Path(sys.argv[1])
class LocalExporter(SpanExporter):
    def __init__(self):
        self.spans = []
    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS
    def shutdown(self):
        pass
exporter = LocalExporter()
SpanRegistry.register('dashboard_validation', {'validation'}, {'default': {'validation'}})
handle = setup_telemetry(NemoLensConfig(enabled=True, exporter='console',
    service_name='dashboard.validation.host-only', run_id='interface-probe',
    metrics_enabled=False, logs_enabled=False),
    resource_attributes={'nv.dl.rank': 0, 'nv.dl.run.uuid': 'explicit-probe-uuid'},
    span_exporter=exporter)
with managed_span('validation', 'dashboard.validation.root'):
    with managed_span('validation', 'dashboard.validation.child', **{'dl.iteration': 7}):
        pass
handle.shutdown()
assert len(exporter.spans) == 2
(out / 'otlp.json').write_text(json.dumps(MessageToDict(encode_spans(exporter.spans)), indent=2))
(out / 'console.jsonl').write_text('\n'.join(json.dumps(json.loads(s.to_json())) for s in exporter.spans))
receipt = {'scope': 'host-only SDK interface, not native training or benchmark',
           'lens_version': version('nemo-lens'), 'otel_version': version('opentelemetry-sdk'),
           'spans': [{'name': s.name, 'trace_id': format(s.context.trace_id, '032x'),
                     'span_id': format(s.context.span_id, '016x'),
                     'parent_span_id': format(s.parent.span_id, '016x') if s.parent else None,
                     'start_ns': s.start_time, 'end_ns': s.end_time} for s in exporter.spans]}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
'''


def test_real_lens_sdk_fresh_local_subprocess(tmp_path):
    interpreter = ROOT / 'tutorials/nemo_lens/.venv/bin/python'
    if not interpreter.exists():
        pytest.skip('Optional isolated Lens interpreter missing; SDK not verified')
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('OTEL_', 'NEMO_LENS_', 'WANDB_', 'SLURM_', 'KUBERNETES_')) and k not in
        {'PYTHONPATH', 'ENABLE_SINGLE_TRACEID', 'DEPLOYMENT_ENV', 'ENVIRONMENT'}}
    env['CUDA_VISIBLE_DEVICES'] = ''
    completed = subprocess.run([str(interpreter), '-I', '-c', PROBE, str(tmp_path)],
                               env=env, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    otlp = module.collect_profiling(tmp_path, {'profiling': {'lens_spans': 'otlp.json'}})['lens']
    console = module.collect_profiling(tmp_path, {'profiling': {'lens_spans': 'console.jsonl'}})['lens']
    assert len(otlp['spans']) == len(console['spans']) == 2
    for actual, expected in zip(otlp['spans'], receipt['spans']):
        for key in ('name', 'trace_id', 'span_id', 'parent_span_id'):
            assert actual[key] == expected[key]
        assert actual['duration_ms'] == (expected['end_ns'] - expected['start_ns']) / 1e6
        assert actual['attributes']['nv.dl.run.uuid'] == 'explicit-probe-uuid'
        assert actual['attributes']['nemo.run.id'] == 'interface-probe'
        assert actual['attributes']['nv.dl.rank'] == 0
    assert otlp['spans'][0]['attributes']['dl.iteration'] == 7
    assert otlp['spans'][0]['parent_span_id'] == otlp['spans'][1]['span_id']
    assert [s['attributes'] for s in otlp['spans']] == [s['attributes'] for s in console['spans']]
    assert not any('educational' in w for w in otlp['warnings'] + console['warnings'])

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('lens_native_adapter', ROOT / 'dashboard/profiling.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def collect(tmp_path, record, *, pretty=False):
    (tmp_path / 'spans.jsonl').write_text(json.dumps(record, indent=2 if pretty else None))
    return module.collect_profiling(tmp_path, {'profiling': {'lens_spans': 'spans.jsonl'}})['lens']


def request(**changes):
    span = {'name': 'megatron.train_step', 'traceId': '11' * 16, 'spanId': '22' * 8,
            'parentSpanId': '33' * 8, 'startTimeUnixNano': '1791850000000000001',
            'endTimeUnixNano': '1791850000000000018',
            'attributes': [{'key': 'megatron.iteration', 'value': {'intValue': '7'}},
                           {'key': 'prompt', 'value': {'stringValue': 'PRIVATE'}}]}
    span.update(changes)
    return {'resourceSpans': [{'resource': {'attributes': [
        {'key': 'nv.dl.run.uuid', 'value': {'stringValue': 'run-explicit'}},
        {'key': 'nv.dl.rank', 'value': {'intValue': '0'}}]},
        'scopeSpans': [{'scope': {'name': 'megatron'}, 'spans': [span]}]}]}


@pytest.mark.parametrize('pretty', [False, True])
def test_otlp_preserves_contract_identity_and_integer_ns(tmp_path, pretty):
    result = collect(tmp_path, request(), pretty=pretty)
    assert result['available'], result['warnings']
    assert result['spans'] == [{'name': 'megatron.train_step', 'trace_id': '11' * 16,
        'span_id': '22' * 8, 'parent_span_id': '33' * 8, 'duration_ms': 17 / 1e6,
        'attributes': {'nv.dl.run.uuid': 'run-explicit', 'nv.dl.rank': 0, 'megatron.iteration': 7}}]
    assert 'PRIVATE' not in json.dumps(result)
    assert not any('educational' in w for w in result['warnings'])
    assert any('coverage' in w for w in result['warnings'])
    assert set(result) == {'available', 'source', 'spans', 'warnings'}


@pytest.mark.parametrize('value', [
    {'boolValue': True}, {'intValue': True}, {'doubleValue': float('nan')},
    {'doubleValue': float('inf')}, {'stringValue': 'x' * 513},
    {'arrayValue': {'values': []}}, {'intValue': '9' * 100},
    {'intValue': '1', 'boolValue': False}, {'boolValue': 'false'},
])
def test_rejects_bad_numeric_semantics_without_json_passthrough(tmp_path, value):
    data = request()
    data['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'] = [
        {'key': 'dl.iteration', 'value': value}]
    result = collect(tmp_path, data)
    assert 'dl.iteration' not in result['spans'][0]['attributes']
    assert any('attribute' in w for w in result['warnings'])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('bad', [True, 17, 'garbage', '0' * 32, '0x' + '1' * 32])
def test_invalid_otlp_id_warns_without_echoing(tmp_path, bad):
    result = collect(tmp_path, request(traceId=bad))
    assert result['spans'][0]['trace_id'] is None
    assert any('identity' in w for w in result['warnings'])


def test_explicit_resource_identity_wins_span_spoof_and_semconv(tmp_path):
    data = request()
    data['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'] += [
        {'key': 'nv.dl.run.uuid', 'value': {'stringValue': 'spoof'}},
        {'key': 'dl.data_parallel.rank', 'value': {'intValue': '2'}},
        {'key': 'dl.loss', 'value': {'doubleValue': 0.25}}]
    attrs = collect(tmp_path, data)['spans'][0]['attributes']
    assert attrs['nv.dl.run.uuid'] == 'run-explicit'
    assert attrs['dl.data_parallel.rank'] == 2
    assert attrs['dl.loss'] == 0.25


@pytest.mark.parametrize('key,value', [('nv.dl.run.uuid', True), ('nv.dl.rank', 0.5),
                                      ('dl.iteration', 1.5), ('bridge.id', True)])
def test_semantic_identity_types_are_strict(tmp_path, key, value):
    data = request()
    data['resourceSpans'][0]['resource']['attributes'] = [
        {'key': key, 'value': {'boolValue' if type(value) is bool else 'doubleValue': value}}]
    data['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'] = []
    result = collect(tmp_path, data)
    assert key not in result['spans'][0]['attributes']
    assert any('attribute' in w for w in result['warnings'])


@pytest.mark.parametrize('path', ['../outside', '/etc/passwd', 'x/../spans.jsonl'])
def test_lens_paths_remain_read_only_and_confined(tmp_path, path):
    result = module.collect_profiling(tmp_path, {'profiling': {'lens_spans': path}})['lens']
    assert not result['available']
    assert result['warnings']


def test_oversize_otlp_line_and_file_warn(tmp_path, monkeypatch):
    # A pretty single OTLP document is file-bounded; JSONL also has a per-line cap.
    monkeypatch.setattr(module, 'MAX_JSON_BYTES', 16)
    result = collect(tmp_path, request())
    assert not result['available'] and result['warnings']



def test_native_advice_does_not_classify_names_as_bridge(tmp_path):
    collect(tmp_path, request())
    result = module.collect_profiling(tmp_path, {'profiling': {'stages': {
        'pretrain': {'lens_spans': 'spans.jsonl'}}}})
    assert '브리지 span은' not in result['optimizations'][0]['hypothesis']


@pytest.mark.parametrize('bad', [True, float('inf'), 'NaN', '-1', '18446744073709551616', None])
def test_invalid_otlp_nanoseconds_are_unavailable_and_warn(tmp_path, bad):
    result = collect(tmp_path, request(startTimeUnixNano=bad))
    assert result['spans'][0]['duration_ms'] is None
    assert any('duration unavailable' in w for w in result['warnings'])
    json.dumps(result, allow_nan=False)


def test_batch_jsonl_partial_bounds_and_bridge_classification(tmp_path, monkeypatch):
    bridge = request(name='native-sounding-name')
    bridge['resourceSpans'][0]['scopeSpans'][0]['spans'][0]['attributes'] = [
        {'key': 'bridge.id', 'value': {'stringValue': 'control:1'}}]
    (tmp_path / 'spans.jsonl').write_text(json.dumps(request()) + '\n' + json.dumps(bridge) + '\n{partial')
    result = module.collect_profiling(tmp_path, {'profiling': {'lens_spans': 'spans.jsonl'}})['lens']
    assert len(result['spans']) == 2
    assert any('educational' in w for w in result['warnings'])
    assert any('malformed' in w for w in result['warnings'])
    monkeypatch.setattr(module, 'MAX_JSON_RECORDS', 1)
    result = collect(tmp_path, {'resourceSpans': [{'scopeSpans': [{'spans': [
        request()['resourceSpans'][0]['scopeSpans'][0]['spans'][0]] * 3}]}]})
    assert len(result['spans']) == 1
    assert any('limit' in w for w in result['warnings'])


def test_existing_actual_console_control_21_span_identity():
    path = ROOT / 'results/gpu_validation/20261013-124945/bridge_cuda/lens-spans.jsonl'
    if not path.exists():
        pytest.skip('Optional existing local 21-span bridge artifact absent')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    result = module.collect_profiling(ROOT, {'profiling': {'lens_spans': str(path.relative_to(ROOT))}})['lens']
    assert len(records) == len(result['spans']) == 21
    assert [(s['trace_id'], s['span_id'], s['parent_span_id']) for s in result['spans']] == [
        (r['context']['trace_id'], r['context']['span_id'], r['parent_id']) for r in records]
    assert any('educational' in w for w in result['warnings'])
