"""Local Chromium regressions. Opt in: NEMOCHAT_LENS_BROWSER_TESTS=1 pytest this file.

The fixture HTML reproduces selectors inspected on real Jaeger, not Jaeger itself.
An actual Jaeger trace is exercised separately by the verifier CLI.
"""
import json
import os

import pytest

from test_lens_ui_cli import SERVICE, TRACE_ID, server, trace, ui

pytestmark = pytest.mark.skipif(os.environ.get('NEMOCHAT_LENS_BROWSER_TESTS') != '1',
                                reason='isolated Playwright browser tests require explicit opt-in')


def html(identity=TRACE_ID, service=SERVICE, operations=None, extra=''):
    if operations is None:
        operations = ['train.forward', 'train.backward']
    rows = ''.join(f'<div class="span-row"><span class="span-svc-name">{service}</span>'
                   f'<small class="endpoint-name">{name}</small></div>' for name in operations)
    return f'''<html><body><label>Service (1)</label>
      <header class="TracePageHeader"><span class="TraceName">{service}: train.forward</span>
      <small class="TraceIDLength">{identity[:7]}</small>
      <button aria-label="Copy to clipboard" onclick="navigator.clipboard.writeText('{identity}')">copy</button>
      </header><div class="TraceTimelineViewer">{rows}</div>{extra}</body></html>'''


def responder(page, data=None):
    def respond(path):
        if path == '/api/services':
            body = json.dumps({'data': [SERVICE]})
        elif path.startswith('/api/traces'):
            body = json.dumps({'data': [trace() if data is None else data]})
        else:
            return page(path)
        return 200, {'Content-Type': 'application/json'}, body
    return respond


def run(url, output):
    ui.main(['--url', url, '--output', str(output), '--trace-id', TRACE_ID])


@pytest.mark.parametrize('kind', ['cross-origin', 'same-origin'])
def test_browser_redirect_never_contacts_destination(tmp_path, kind):
    with server(lambda _: (200, {}, html())) as (destination, hits):
        source_hits = []
        def page(path):
            source_hits.append(path)
            target = destination + '/leak' if kind == 'cross-origin' else '/leak'
            return 302, {'Location': target}, ''
        with server(responder(page)) as (source, _):
            with pytest.raises(Exception):
                run(source, tmp_path / 'out')
        assert not hits
        assert '/leak' not in source_hits
    assert not (tmp_path / 'out' / 'verification.json').exists()


@pytest.mark.parametrize('kind', ['image', 'script', 'fetch', 'frame', 'websocket'])
def test_external_request_attempt_fails_without_contact(tmp_path, kind):
    with server(lambda _: (200, {}, 'unexpected')) as (destination, hits):
        extra = {
            'image': f'<img src="{destination}/leak">',
            'script': f'<script src="{destination}/leak"></script>',
            'fetch': f'<script>fetch("{destination}/leak").catch(()=>{{}})</script>',
            'frame': f'<iframe src="{destination}/leak"></iframe>',
            'websocket': f'<script>new WebSocket("{destination.replace("http:", "ws:")}/leak")</script>',
        }[kind]
        with server(responder(lambda _: (200, {'Content-Type': 'text/html'}, html(extra=extra)))) as (source, _):
            with pytest.raises(RuntimeError, match='Blocked browser request'):
                run(source, tmp_path / 'out')
        assert hits == []
    assert not (tmp_path / 'out' / 'verification.json').exists()


@pytest.mark.parametrize('kind', ['short-id', 'full-id', 'service', 'missing-operation',
                                 'unrelated-operation', 'error-page', 'js-error', 'final-url'])
def test_invalid_rendering_has_no_receipt(tmp_path, kind):
    kwargs = {
        'short-id': {'identity': 'deadbeef'},
        'full-id': {'identity': TRACE_ID[:7] + 'ffff'},
        'service': {'service': 'nemochat-lens-other'},
        'missing-operation': {'operations': ['train.forward']},
        'unrelated-operation': {'operations': ['train.forward'], 'extra': '<p>train.backward failed</p>'},
        'error-page': {'extra': '<h2>Error loading trace: trace not found</h2>'},
        'js-error': {'extra': '<script>throw new Error("fixture JavaScript error")</script>'},
        'final-url': {'extra': '<script>history.replaceState({}, "", "/unrelated")</script>'},
    }[kind]
    with server(responder(lambda _: (200, {'Content-Type': 'text/html'}, html(**kwargs)))) as (source, _):
        with pytest.raises(RuntimeError):
            run(source, tmp_path / 'out')
    assert not (tmp_path / 'out' / 'verification.json').exists()


@pytest.mark.parametrize('kind', ['api-redirect', 'nonhex', 'mismatch', 'wrong-service'])
def test_api_failure_has_no_receipt(tmp_path, kind):
    data = trace()
    if kind == 'nonhex':
        data['traceID'] = '../other'
    elif kind == 'mismatch':
        data['traceID'] = 'deadbeef'
    elif kind == 'wrong-service':
        data['processes']['p']['serviceName'] = 'nemochat-lens-other'
    with server(lambda _: (200, {}, 'unexpected')) as (destination, hits):
        respond = responder(lambda _: (200, {'Content-Type': 'text/html'}, html()), data)
        def dispatch(path):
            if kind == 'api-redirect' and path.startswith('/api/traces'):
                return 302, {'Location': destination + '/leak'}, ''
            return respond(path)
        with server(dispatch) as (source, _):
            with pytest.raises(ValueError):
                run(source, tmp_path / 'out')
        assert not hits
    assert not (tmp_path / 'out' / 'verification.json').exists()


def test_valid_local_fixture_fresh_receipt(tmp_path):
    with server(responder(lambda _: (200, {'Content-Type': 'text/html'}, html()))) as (source, _):
        output = tmp_path / 'fresh'
        run(source, output)
        receipt = json.loads((output / 'verification.json').read_text())
        assert receipt['trace_id'] == TRACE_ID
        assert receipt['rendered_operations'] == ['train.backward', 'train.forward']
        assert not receipt['page_errors']
        for filename in ['search.png', 'trace.png']:
            assert (output / filename).read_bytes().startswith(b'\x89PNG')
        before = (output / 'verification.json').read_bytes()
        with pytest.raises(SystemExit):
            run(source, output)
        assert (output / 'verification.json').read_bytes() == before
