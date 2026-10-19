"""Real loopback HTTP tests; no GPU workloads or remote calls."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import http.client


def test_service_enrolls_new_stage_from_approved_root_without_restart(tmp_path):
    s=module();(tmp_path/'runs').mkdir()
    cfg={'interval_seconds':2,'runs':[],'run_roots':[{'path':'runs','format':'bridge','rank_count':8,'prefixes':{'bridge-sft-':'sft'}}]}
    service=s.SnapshotService(tmp_path,cfg,collect_fn=lambda r,c:{'schema_version':1,'collection':{'warnings':[]},'runs':c['runs']},profiling_fn=lambda r,c:{})
    service.start()
    try:
        assert json.loads(service.read()[0])['runs']==[]
        p=tmp_path/'runs'/'bridge-sft-new';p.mkdir();(p/'requested-config.json').write_text('{}')
        import time
        deadline=time.monotonic()+4
        while not json.loads(service.read()[0])['runs'] and time.monotonic()<deadline:time.sleep(.05)
        x=json.loads(service.read()[0]);assert x['runs'][0]['stage']=='sft'
        assert x['collection']['discovery']['discovered_runs']==1
    finally:service.stop()


def test_local_timeline_script_is_served_and_loaded_before_main_ui(tmp_path):
    s = module()
    service = s.SnapshotService(tmp_path, {}, collect_fn=lambda r,c: {}, profiling_fn=lambda r,c: {})
    server = s.make_server(tmp_path, {}, port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
    try:
        conn.request('GET', '/timeline.js')
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader('Content-Type').startswith('text/javascript')
        assert b'prepareTimeline' in response.read()
        conn.request('GET', '/')
        response = conn.getresponse()
        html = response.read().decode()
        assert html.index('/timeline.js') < html.index('/app.js')
    finally:
        conn.close()
        server.shutdown(); server.server_close(); thread.join(2)


def module():
    path = Path(__file__).parents[1] / 'server.py'
    assert path.exists(), 'backend implementation is missing'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('dashboard_server_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_polling_serves_one_independent_collector_cache(tmp_path):
    s = module()
    calls = []
    def collect(root, config):
        calls.append(threading.current_thread().name)
        return {'schema_version': 1, 'generated_at': 'test', 'collection': {'warnings': []}, 'runs': []}
    service = s.SnapshotService(tmp_path, {'interval_seconds': 2}, collect_fn=collect, profiling_fn=lambda r,c: {})
    assert not calls, 'imports/construction must not start collection'
    service.start()
    server = s.make_server(tmp_path, {}, port=0, service=service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for _ in range(3):
            conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
            conn.request('GET', '/api/snapshot')
            response = conn.getresponse()
            assert response.status == 200
            assert json.loads(response.read())['schema_version'] == 1
            conn.close()
        assert len(calls) == 1
        assert calls[0] == 'dashboard-collector'
        assert service.thread.is_alive()
    finally:
        server.shutdown(); server.server_close(); service.stop(); thread.join(2)


def test_sse_matches_polling_bounded_clients_and_secure_static_paths(tmp_path):
    s = module()
    static = tmp_path / 'static'
    static.mkdir()
    (static / 'index.html').write_text('<h1>local frontend</h1>')
    (tmp_path / 'secret.json').write_text('must-not-be-served')
    service = s.SnapshotService(tmp_path, {'interval_seconds': 2}, collect_fn=lambda r,c: {'schema_version': 1, 'collection': {'warnings': []}}, profiling_fn=lambda r,c: {})
    service.start()
    server = s.make_server(tmp_path, {'max_sse_clients': 1, 'write_timeout_seconds': .2}, port=0, service=service, static_dir=static)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    clients = []
    def get(path, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
        clients.append(conn); conn.request('GET', path, headers=headers or {})
        return conn.getresponse()
    try:
        first = get('/api/events')
        assert first.status == 200
        assert first.readline().strip() == b'event: snapshot'
        event = json.loads(first.readline().split(b'data: ', 1)[1])
        assert event == json.loads(get('/api/snapshot').read())
        assert get('/api/events').status == 503
        assert get('/').read() == b'<h1>local frontend</h1>'
        for path in ['/../secret.json', '/%2e%2e/secret.json', '/api/snapshot?path=secret.json', '/config.json', '/CONTRACT.md']:
            assert get(path).status == 404
        assert get('/api/snapshot', {'Host': 'evil.example'}).status == 403
        assert server.write_timeout == .2
        assert server.sse_count == 1
    finally:
        for conn in clients: conn.close()
        service.stop(); server.shutdown(); server.server_close(); thread.join(2)


def test_explicit_config_reload_adds_run_without_client_and_rejects_bad_paths(tmp_path):
    s = module()
    config_path = tmp_path / 'dashboard.json'
    config_path.write_text(json.dumps({'interval_seconds': 2, 'runs': []}))
    calls = []
    def collect(root, config):
        calls.append([r['id'] for r in config.get('runs', [])])
        return {'collection': {'warnings': []}, 'runs': config.get('runs', [])}
    service = s.SnapshotService(tmp_path, s.load_config(tmp_path, 'dashboard.json'), collect_fn=collect, profiling_fn=lambda r,c: {}, config_path=config_path)
    service.start()
    try:
        config_path.write_text(json.dumps({'interval_seconds': 2, 'runs': [{'id': 'next-stage', 'path': 'results/next', 'stage': 'long_context', 'format': 'bridge'}]}))
        import time
        deadline = time.monotonic() + 4
        while calls[-1] != ['next-stage'] and time.monotonic() < deadline: time.sleep(.05)
        assert calls[-1] == ['next-stage']
        assert json.loads(service.read()[0])['runs'][0]['stage'] == 'long'
        config_path.write_text(json.dumps({'runs': [{'id': 'bad', 'path': '../credentials'}]}))
        deadline = time.monotonic() + 4
        while not json.loads(service.read()[0])['collection']['warnings'] and time.monotonic() < deadline: time.sleep(.05)
        payload = json.loads(service.read()[0])
        assert payload['runs'][0]['id'] == 'next-stage'
        assert payload['collection']['warnings']
        assert 'credentials' not in json.dumps(payload)
    finally:
        service.stop()


def test_slow_sse_reader_is_released_by_socket_write_timeout(tmp_path):
    s = module()
    service = s.SnapshotService(tmp_path, {}, collect_fn=lambda r,c: {'collection': {'warnings': []}, 'large': 'x' * (4*1024*1024)}, profiling_fn=lambda r,c: {})
    service.start()
    server = s.make_server(tmp_path, {'max_sse_clients': 1, 'write_timeout_seconds': .1}, port=0, service=service)
    import socket, time
    original = server.get_request
    def bounded_socket():
        sock, address = original()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        return sock, address
    server.get_request = bounded_socket
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    client = socket.socket(); client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    try:
        client.connect(('127.0.0.1', server.server_port))
        client.sendall(f'GET /api/events HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\n\r\n'.encode())
        deadline = time.monotonic() + 1
        while server.sse_count == 0 and time.monotonic() < deadline: time.sleep(.005)
        assert server.sse_count == 1
        deadline = time.monotonic() + 1
        while server.sse_count and time.monotonic() < deadline: time.sleep(.01)
        assert server.sse_count == 0
        assert server.sse_slots.acquire(blocking=False)
        server.sse_slots.release()
    finally:
        client.close(); service.stop(); server.shutdown(); server.server_close(); thread.join(2)


def test_cli_documents_reload_and_explicit_remote_opt_in():
    import subprocess
    path = Path(__file__).parents[1] / 'server.py'
    result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True, check=True)
    assert '--reload-config' in result.stdout
    assert '--allow-remote' in result.stdout
    result = subprocess.run([sys.executable, str(path), '--host', '0.0.0.0'], capture_output=True, text=True)
    assert result.returncode != 0
    assert '--allow-remote' in result.stderr


def test_initial_collector_failure_is_schema_valid_and_never_leaks_exception(tmp_path):
    s = module()
    def fail(root, config):
        raise ValueError('PASSWORD=should-never-leak')
    service = s.SnapshotService(tmp_path, {}, collect_fn=fail, profiling_fn=lambda r,c: {})
    service.start()
    try:
        payload = json.loads(service.read()[0])
        assert payload['schema_version'] == 1
        assert payload['runs'] == []
        assert payload['collection']['warnings']
        assert 'should-never-leak' not in json.dumps(payload)
    finally: service.stop()
