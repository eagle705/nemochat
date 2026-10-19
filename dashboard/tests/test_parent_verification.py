"""Regression coverage for issues observed in the real browser."""
from test_backend import module
import http.client
import threading


def test_local_korean_font_is_served_as_font_and_nothing_else(tmp_path):
    s = module()
    static = tmp_path / 'static'
    static.mkdir()
    payload = b'font-fixture-not-real-font'
    (static / 'NemoChatKorean.ttf').write_bytes(payload)
    (static / 'secret.ttf').write_bytes(b'private')
    service = s.SnapshotService(tmp_path, {}, collect_fn=lambda r,c: {'collection': {'warnings': []}}, profiling_fn=lambda r,c: {})
    service.start()
    server = s.make_server(tmp_path, {}, port=0, service=service, static_dir=static)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
        conn.request('GET', '/NemoChatKorean.ttf')
        response = conn.getresponse()
        assert response.status == 200, 'Korean font must be served locally'
        assert response.getheader('Content-Type').split(';')[0] == 'font/ttf'
        assert response.read() == payload
        conn.close()
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
        conn.request('GET', '/secret.ttf')
        assert conn.getresponse().status == 404
        conn.close()
    finally:
        service.stop(); server.shutdown(); server.server_close(); thread.join(2)


def test_explicit_prior_parameter_census_is_scoped_and_not_inferred(tmp_path):
    import json
    from test_collectors import module as collectors_module
    c = collectors_module()
    (tmp_path / 'results' / 'pilot').mkdir(parents=True)
    (tmp_path / 'results' / 'census.json').write_text(json.dumps({
        'status': 'passed', 'logical_numel_from_ep_deduplication': 12345,
        'scope': 'prior explicit native model census', 'API_KEY': 'never expose'}))
    (tmp_path / 'results' / 'pilot' / 'verified-summary.json').write_text(json.dumps({'status': 'passed', 'training_container_seconds': 42.5}))
    spec = {'id': 'pilot', 'path': 'results/pilot', 'stage': 'pretrain', 'model_census': 'results/census.json'}
    run = c.collect_run(tmp_path, spec, {})
    assert run['timing']['container_elapsed_seconds'] == 42.5
    assert run['model']['parameters_total'] == 12345
    assert run['model']['parameters_active'] is None
    assert run['model']['census_source'] == 'results/census.json'
    assert any('prior' in warning for warning in run['warnings'])
    assert 'never expose' not in json.dumps(run)
    spec['model_census'] = '../secret.json'
    run = c.collect_run(tmp_path, spec, {})
    assert run['model']['parameters_total'] is None
