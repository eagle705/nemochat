"""CPU-only CLI/API contracts; browser rendering is explicitly opt-in."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import subprocess
import sys
from threading import Thread
from urllib.request import ProxyHandler, build_opener

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_lens_ui.py'
spec = importlib.util.spec_from_file_location('verify_lens_ui', SCRIPT)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
TRACE_ID = 'abc123456789'
SERVICE = 'nemochat-lens-cpu'


def trace():
    return {'traceID': TRACE_ID, 'processes': {'p': {'serviceName': SERVICE}},
            'spans': [{'operationName': name, 'processID': 'p'}
                      for name in ['train.forward', 'train.backward']]}


@contextmanager
def server(respond):
    """All transport regression destinations are local, recorded HTTP servers."""
    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            status, headers, body = respond(self.path)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{httpd.server_port}', hits
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_help_without_browser_dependency():
    result = subprocess.run([sys.executable, str(SCRIPT), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--trace-id' in result.stdout


@pytest.mark.parametrize('url', [
    'https://example.com', 'http://example.com',
    'http://user:password@127.0.0.1:16687',
    'http://127.0.0.1:16687?token=example',
    'http://127.0.0.1:16687/other', 'file:///tmp/page',
    'http://127.0.0.1:bad',
])
def test_nonlocal_or_credentialed_urls_rejected_before_browser_import(tmp_path, url):
    output = tmp_path / 'never-created'
    result = subprocess.run([sys.executable, str(SCRIPT), '--url', url, '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'credential-free loopback HTTP' in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('identity', ['nothex', '../abc', 'a' * 33])
def test_invalid_requested_identity_before_browser_import(tmp_path, identity):
    with pytest.raises(SystemExit):
        ui.main(['--trace-id', identity, '--output', str(tmp_path / 'fresh')])
    assert not (tmp_path / 'fresh').exists()


def test_output_never_reused(tmp_path):
    receipt = tmp_path / 'verification.json'
    receipt.write_text('historical receipt')
    with pytest.raises(SystemExit):
        ui.main(['--output', str(tmp_path)])
    assert receipt.read_text() == 'historical receipt'


def test_missing_optional_dependency_is_friendly(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, 'playwright.sync_api', None)
    with pytest.raises(SystemExit):
        ui.main(['--output', str(tmp_path / 'fresh')])
    assert 'separate browser verification environment' in capsys.readouterr().err
    assert not (tmp_path / 'fresh').exists()


def test_api_redirect_destination_not_contacted():
    with server(lambda _: (200, {}, 'unexpected')) as (destination, hits):
        with server(lambda _: (302, {'Location': destination + '/leak'}, '')) as (source, _):
            opener = build_opener(ProxyHandler({}), ui.NoRedirect())
            with pytest.raises(ValueError, match='redirects are not allowed'):
                opener.open(source + '/api/services')
        assert hits == []


@pytest.mark.parametrize('mutation, message', [
    ('nonhex', 'hexadecimal'), ('mismatch', 'requested trace'), ('service', 'intended Lens'),
])
def test_api_identity_validation(mutation, message):
    data = trace()
    if mutation == 'nonhex':
        data['traceID'] = '../other'
    elif mutation == 'mismatch':
        data['traceID'] = 'deadbeef'
    else:
        data['processes']['p']['serviceName'] = 'nemochat-lens-unintended'
    with pytest.raises(ValueError, match=message):
        ui.validate_trace(data, TRACE_ID, SERVICE)


def test_api_response_is_bounded_before_decode(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # No browser is launched: reject at the API read seam, before JSON decode.
    monkeypatch.setitem(sys.modules, 'playwright.sync_api',
                        SimpleNamespace(Error=RuntimeError, sync_playwright=lambda: None))
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'x' * 33
    opener = MagicMock()
    opener.open.return_value = response
    monkeypatch.setattr(ui, 'build_opener', lambda *args: opener)
    monkeypatch.setattr(ui, 'MAX_RESPONSE_BYTES', 32)
    output = tmp_path / 'oversized'
    with pytest.raises(ValueError, match='response too large'):
        ui.main(['--output', str(output)])
    response.read.assert_called_once_with(33)
    assert not (output / 'verification.json').exists()
    response.read.return_value = b'{"data": []}'
    with pytest.raises(RuntimeError, match='service is not stored'):
        ui.main(['--output', str(tmp_path / 'valid-small')])


def test_api_identity_and_operation_set():
    assert ui.validate_trace(trace(), TRACE_ID, SERVICE) == (
        TRACE_ID, ['train.backward', 'train.forward'])
