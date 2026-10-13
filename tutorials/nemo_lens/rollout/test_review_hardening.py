"""Local fault injection and malformed evidence regressions (no downloads)."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run
from diagnose import analyze, ns, span_id
from verify import validate
import verify_jaeger


def fixture():
    """Handwritten valid evidence, deliberately not an executed measurement."""
    spans = []
    trace_id = '1' * 32
    resource = {'service.name': run.SERVICE, 'nemo.run.id': 'fixture',
                'tutorial.device': 'cpu', 'demo.workflow': 'scripted'}
    def add(name, parent, start, end, attrs, kind='INTERNAL'):
        sid = format(len(spans) + 1, '016x')
        stamp = lambda t: '2026-01-01T00:00:00.' + format(t, '09d') + 'Z'
        spans.append({'name': 'demo.rl.' + name, 'context': {'span_id': '0x' + sid, 'trace_id': '0x' + trace_id},
                      'parent_id': '0x' + parent if parent else None, 'start_time': stamp(start),
                      'end_time': stamp(end), 'attributes': attrs, 'status': {'status_code': 'UNSET'},
                      'kind': 'SpanKind.' + kind, 'resource': {'attributes': resource.copy()}})
        return sid
    root = add('batch', None, 0, 100, {'demo.sample_size': 2, 'demo.scenario': 'baseline'})
    rows = []
    for tid in range(2):
        attrs = {'demo.trajectory.id': tid}
        trajectory = add('trajectory', root, 1, 90, attrs.copy())
        add('caller_queue', trajectory, 2, 10, attrs.copy())
        add('generation', trajectory, 11, 20, {**attrs, 'demo.generated_tokens': 8})
        client = add('tool.client', trajectory, 21, 70, {**attrs, 'http.response.status_code': 200}, 'CLIENT')
        add('tool.server', client, 22, 60, {**attrs, 'demo.injected_delay_ms': 0.0}, 'SERVER')
        add('reward', trajectory, 71, 80, {**attrs, 'demo.reward.correct': True})
        rows.append({'trajectory_id': tid, 'span_id': trajectory, 'generated_tokens': 8, 'correct': True})
    result = {'status': 'executed', 'mode': 'file', 'scenario': 'baseline', 'trace_id': trace_id,
              'run_id': 'fixture', 'service': run.SERVICE, 'model': 'sshleifer/tiny-gpt2',
              'revision': run.REVISION, 'expected_spans': 13, 'trajectories': rows, 'ui_url': 'fixture'}
    remote = []
    for s in spans:
        tags = [{'key': k, 'type': {bool: 'bool', int: 'int64', float: 'float64', str: 'string'}[type(v)],
                 'value': v} for k, v in s['attributes'].items()]
        if s['kind'] != 'SpanKind.INTERNAL':
            tags.append({'key': 'span.kind', 'value': s['kind'].split('.')[-1].lower()})
        remote.append({'traceID': trace_id, 'spanID': span_id(s), 'operationName': s['name'], 'processID': 'p',
                       'startTime': ns(s['start_time']) // 1000,
                       'duration': (ns(s['end_time']) - ns(s['start_time'])) // 1000,
                       'tags': tags, 'references': [{'refType': 'CHILD_OF', 'traceID': trace_id,
                                                   'spanID': s['parent_id'][2:]}] if s['parent_id'] else []})
    trace = {'traceID': trace_id, 'spans': remote, 'processes': {'p': {'serviceName': run.SERVICE,
             'tags': [{'key': k, 'value': v} for k, v in resource.items() if k != 'service.name']}}}
    return trace, result, spans


class EvidenceTests(unittest.TestCase):
    def test_valid_fixture(self):
        t, r, s = fixture()
        self.assertEqual(validate(t, r, s)['span_count'], 13)

    def test_result_rejections(self):
        for mutation in ('failed', 'count', 'false', 'tokens', 'duplicate_tid', 'duplicate_sid', 'wrong_sid'):
            with self.subTest(mutation=mutation):
                t, r, s = fixture()
                if mutation == 'failed': r['status'] = 'failed'
                if mutation == 'count': r['expected_spans'] = 999
                if mutation == 'false': r['trajectories'][0]['correct'] = False
                if mutation == 'tokens': r['trajectories'][0]['generated_tokens'] = 7
                if mutation == 'duplicate_tid': r['trajectories'][1]['trajectory_id'] = 0
                if mutation == 'duplicate_sid': r['trajectories'][1]['span_id'] = r['trajectories'][0]['span_id']
                if mutation == 'wrong_sid': r['trajectories'][0]['span_id'] = 'f' * 16
                with self.assertRaises((ValueError, KeyError, TypeError)): validate(t, r, s)

    def test_structural_rejections(self):
        for mutation in ('duplicate', 'orphan', 'zero_batch', 'reverse_server', 'server_outside',
                         'duplicate_tid', 'missing_tid', 'missing_stage', 'crossed_stage', 'bad_time', 'bad_id', 'error'):
            with self.subTest(mutation=mutation):
                t, r, s = fixture()
                if mutation == 'duplicate': s.append(copy.deepcopy(s[1]))
                if mutation == 'orphan': s[1]['parent_id'] = '0x' + 'f' * 16
                if mutation == 'zero_batch': s[0]['end_time'] = s[0]['start_time']
                if mutation == 'reverse_server': s[5]['end_time'] = s[0]['start_time']
                if mutation == 'server_outside': s[5]['end_time'] = s[0]['end_time']
                if mutation == 'duplicate_tid': s[7]['attributes']['demo.trajectory.id'] = 0
                if mutation == 'missing_tid': del s[5]['attributes']['demo.trajectory.id']
                if mutation == 'missing_stage': s.pop(2)
                if mutation == 'crossed_stage': s[2]['attributes']['demo.trajectory.id'] = 1
                if mutation == 'bad_time': s[5]['start_time'] += 'Z'
                if mutation == 'bad_id': s[5]['context']['span_id'] = '0x0'
                if mutation == 'error': s[5]['status']['status_code'] = 'ERROR'
                with self.assertRaises((ValueError, KeyError, TypeError)): analyze(s)

    def test_remote_and_local_mismatches(self):
        for mutation in ('operation', 'status', 'tags', 'hierarchy', 'resource', 'generation', 'reward'):
            with self.subTest(mutation=mutation):
                t, r, s = fixture()
                if mutation == 'operation': s[2]['name'], s[3]['name'] = s[3]['name'], s[2]['name']
                if mutation == 'status': t['spans'][0]['tags'].append({'key': 'otel.status_code', 'value': 'ERROR'})
                if mutation == 'tags': t['spans'][1]['tags'] = []
                if mutation == 'hierarchy': t['spans'][2]['references'][0]['spanID'] = span_id(s[0])
                if mutation == 'resource': s[2]['resource']['attributes']['nemo.run.id'] = 'other'
                if mutation == 'generation': s[3]['attributes']['demo.generated_tokens'] = 7
                if mutation == 'reward': s[6]['attributes']['demo.reward.correct'] = False
                with self.assertRaises((ValueError, KeyError, TypeError)): validate(t, r, s)

    def test_integral_double_roundtrip_and_strict_tag_types(self):
        t, r, s = fixture()
        tag = next(x for x in t['spans'][5]['tags'] if x['key'] == 'demo.injected_delay_ms')
        tag['value'] = 0  # Actual Jaeger JSON represents integral float64 this way.
        self.assertEqual(validate(t, r, s)['span_count'], 13)
        for kind, value in (('int64', 0), ('float64', True), ('float64', '0'),
                            ('float64', 1), ('float64', float('nan'))):
            with self.subTest(kind=kind, value=value):
                tag.update(type=kind, value=value)
                with self.assertRaises(ValueError): validate(t, r, s)
        for index, key, kind, value in ((1, 'demo.trajectory.id', 'int64', False),
                                       (1, 'demo.trajectory.id', 'float64', 0.0),
                                       (6, 'demo.reward.correct', 'int64', 1)):
            with self.subTest(key=key, kind=kind):
                t, r, s = fixture()
                next(x for x in t['spans'][index]['tags'] if x['key'] == key).update(type=kind, value=value)
                with self.assertRaises(ValueError): validate(t, r, s)

    def test_remote_timestamp_rejections(self):
        for field in ('startTime', 'duration'):
            for value in (None, -1, True, 1.0, '1'):
                with self.subTest(field=field, value=value):
                    t, r, s = fixture()
                    if value is None: del t['spans'][5][field]
                    else: t['spans'][5][field] = value
                    with self.assertRaises((ValueError, KeyError)): validate(t, r, s)
        for field, delta in (('startTime', -2), ('startTime', 2), ('duration', 3)):
            with self.subTest(field=field, delta=delta):
                t, r, s = fixture()
                t['spans'][5][field] += delta
                with self.assertRaises(ValueError): validate(t, r, s)

    def test_remote_parent_quantization_bound(self):
        t, r, s = fixture()
        # All handwritten spans are sub-microsecond: floor produces zero durations.
        # Independently-corresponding child end may exceed parent end by 1us only.
        t['spans'][5]['duration'] = 1
        self.assertEqual(validate(t, r, s)['span_count'], 13)
        t['spans'][5]['duration'] = 2
        with self.assertRaisesRegex(ValueError, 'outside remote parent'): validate(t, r, s)

    def test_pinned_sdk_microprecision_correspondence(self):
        t, r, s = fixture()
        # Captured client timing from actual trace 574b44243a5678ea094edb71c6cf2b71.
        # Use a fully contained synthetic chain around it, not altered real evidence.
        for span in s:
            span['start_time'] = '2026-10-13T01:46:46.352372Z'
            span['end_time'] = '2026-10-13T01:46:46.605552Z'
        for span in s:
            if span['name'].rsplit('.', 1)[-1] in ('caller_queue', 'generation', 'reward'):
                stamp = s[0]['start_time'] if span['name'].endswith(('caller_queue', 'generation')) else s[0]['end_time']
                span['start_time'] = span['end_time'] = stamp
        for remote, local in zip(t['spans'], s):
            remote['startTime'] = ns(local['start_time']) // 1000 - 1
            remote['duration'] = max(0, (ns(local['end_time']) - ns(local['start_time'])) // 1000 - 1)
        t['spans'][4].update(startTime=1791856006352371, duration=253179)
        self.assertEqual(validate(t, r, s)['span_count'], 13)
        t['spans'][4]['startTime'] -= 1
        with self.assertRaises(ValueError): validate(t, r, s)

    def test_diagnose_rejects_boolean_child_identity(self):
        for index in (8, 9, 10, 11, 12):
            with self.subTest(index=index):
                _, _, s = fixture()
                s[index]['attributes']['demo.trajectory.id'] = True
                with self.assertRaises(ValueError): analyze(s)

    def test_timestamp_precision(self):
        self.assertEqual(ns('1970-01-01T00:00:00.000000001Z'), 1)
        for stamp in ('2026-01-01T00:00:00.1234567890Z', '2026-01-01T00:00:00.1+00:00',
                      '2026-01-01T00:00:00.-1Z', '2026-02-30T00:00:00.1Z'):
            with self.assertRaises(ValueError): ns(stamp)

    def test_bounded_jaeger_read(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'x' * 33
        with patch.object(verify_jaeger, 'MAX_RESPONSE_BYTES', 32), patch.object(verify_jaeger.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = response
            with self.assertRaisesRegex(ValueError, 'too large'): verify_jaeger.get_json('http://127.0.0.1:1', '/api/services')
            response.read.assert_called_once_with(33)
            response.read.return_value = b'{"data": []}'
            self.assertEqual(verify_jaeger.get_json('http://127.0.0.1:1', '/api/services'), {'data': []})


class LifecycleTests(unittest.TestCase):
    def exercise(self, fault, body_error=None):
        import nemo.lens
        from opentelemetry.sdk.trace.export import SpanExportResult
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = SimpleNamespace(output=output, mode='otlp' if fault in ('remote', 'constructor') else 'file',
                                   endpoint='http://127.0.0.1:14318/v1/traces', scenario='baseline',
                                   trajectories=2, timeout=1, fail_id=-1)
            stream = Mock()
            if fault in ('write', 'flush'): getattr(stream, fault).side_effect = OSError('private stream details')
            remote = Mock()
            remote.export.side_effect = OSError('private remote details') if fault == 'remote' else None
            remote.export.return_value = SpanExportResult.SUCCESS
            calls = []
            server = Mock(server_port=1)
            thread = Mock()
            server.shutdown.side_effect = lambda: calls.append('shutdown')
            if fault == 'cleanup':
                def shutdown():
                    calls.append('shutdown')
                    raise OSError('private cleanup details')
                server.shutdown.side_effect = shutdown
            thread.join.side_effect = lambda: calls.append('join')
            server.server_close.side_effect = lambda: calls.append('server_close')
            fake_span = Mock()
            fake_span.to_json.return_value = '{}'
            def setup(*a, **kw):
                exporter = kw['span_exporter']
                def close():
                    calls.append('telemetry')
                    exporter.export([fake_span] * (12 if fault == 'coverage' else 13))
                    exporter.shutdown()
                return SimpleNamespace(shutdown=close)
            async def batch(*a, **kw):
                if body_error is not None: raise body_error
                return '1' * 32, []
            real_open = Path.open
            def opened(path, *a, **kw):
                if path.name == 'spans.jsonl': return stream
                if path.name == 'failure.json' and fault == 'failure_record': raise OSError('private receipt details')
                return real_open(path, *a, **kw)
            constructor = Mock(return_value=remote, side_effect=OSError('constructor') if fault == 'constructor' else None)
            with patch.object(nemo.lens.SpanRegistry, 'register'), \
                 patch.object(Path, 'open', opened), patch.object(nemo.lens, 'setup_telemetry', setup), \
                 patch.object(run, 'load_model', return_value=lambda: 8), \
                 patch.object(run, 'start_tool_server', return_value=(server, thread)), \
                 patch.object(run, 'rollout_batch', batch), \
                 patch('opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter', constructor):
                with self.assertRaises(BaseException) as caught: run.run(args)
            self.assertFalse((output / 'result.json').exists())
            self.assertTrue((output / 'failure.json').exists() or fault == 'failure_record')
            stream.close.assert_called_once()
            if fault != 'constructor': self.assertEqual(calls, ['shutdown', 'join', 'server_close', 'telemetry'])
            if body_error is not None:
                self.assertIs(caught.exception, body_error)
                self.assertNotIn('private', ' '.join(getattr(body_error, '__notes__', [])))

    def test_export_failures_and_coverage(self):
        for fault in ('write', 'flush', 'remote', 'coverage', 'constructor'):
            with self.subTest(fault=fault): self.exercise(fault)

    def test_cleanup_fails_successful_body(self):
        self.exercise('cleanup')

    def test_preserves_body_and_attempts_all_cleanup(self):
        self.exercise('cleanup', run.ToolFailure('exact body'))

    def test_failure_record_never_replaces_body(self):
        self.exercise('failure_record', run.ToolFailure('exact body'))


if __name__ == '__main__':
    unittest.main()
