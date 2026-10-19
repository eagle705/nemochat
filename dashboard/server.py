"""Local read-only dashboard. One collector thread; clients only read cache."""
import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import re

try:
    from .collectors import collect_snapshot, secure_path
    from .discovery import expand_run_roots, validate_run_roots
except ImportError:
    from collectors import collect_snapshot, secure_path
    from discovery import expand_run_roots, validate_run_roots


def validate_config(root, config):
    if not isinstance(config, dict) or not isinstance(config.get('runs', []), list) or len(config.get('runs', [])) > 32:
        raise ValueError('Invalid dashboard configuration')
    config = copy.deepcopy(config)
    ids = set()
    stages = {'data', 'pretrain', 'mid', 'long', 'sft', 'rlvr', 'mopd'}
    for run in config.get('runs', []):
        if not isinstance(run, dict) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run.get('id', '')) or run['id'] in ids:
            raise ValueError('Invalid or duplicate run identifier')
        ids.add(run['id'])
        secure_path(root, run['path'])
        run['stage'] = {'lc': 'long', 'long_context': 'long'}.get(run.get('stage'), run.get('stage', 'pretrain'))
        if run['stage'] not in stages or run.get('format', 'bridge') not in ('bridge', 'native', 'canonical', 'rlvr'):
            raise ValueError('Unsupported run stage or artifact adapter')
        if not 1 <= int(run.get('rank_count', 1)) <= 64:
            raise ValueError('Rank count must be between 1 and 64')
        if run.get('model_census'):
            secure_path(root, run['model_census'])
        if run.get('terminal_receipt'):
            secure_path(root, run['terminal_receipt'])
    profile = config.get('profiling', {})
    for entry in [profile] + list(profile.get('stages', {}).values()):
        for field in ('lens_spans', 'nsight_sqlite', 'nsight_summary', 'bridge_operations', 'nsight_analysis', 'nsight_native_source', 'nsight_source_provenance'):
            if entry.get(field):
                secure_path(root, entry[field])
    for key, low, high, default in [('interval_seconds', 2, 300, 5), ('history_limit', 1, 512, 256), ('stale_seconds', 2, 86400, 120), ('nvidia_smi_timeout_seconds', .1, 3, 1.5)]:
        value = float(config.get(key, default))
        if not low <= value <= high:
            raise ValueError('Configuration numeric limit outside allowed range')
    validate_run_roots(root,config)
    return config


def load_config(root, relative):
    path = secure_path(root, relative)
    if path.suffix != '.json' or path.stat().st_size > 256 * 1024:
        raise ValueError('Expected a bounded dashboard JSON configuration')
    return validate_config(root, json.loads(path.read_bytes()))


def collect_profiles(root, config):
    try:
        try:
            from .profiling import collect_profiling
        except ImportError:
            from profiling import collect_profiling
        return collect_profiling(root, config)
    except ImportError:
        return {kind: dict(available=False, source=None, warnings=['Profiling module unavailable'], **{field: []}) for kind, field in [('lens', 'spans'), ('nsight', 'kernels'), ('bridge', 'operations')]}


class SnapshotService:
    def __init__(self, root, config, collect_fn=None, profiling_fn=None, config_path=None):
        self.root = Path(root).resolve()
        self.config = validate_config(self.root, config)
        self.config_path = Path(config_path).resolve() if config_path else None
        self.config_signature = None
        self.reload_warning = None
        self.collect_fn = collect_fn or collect_snapshot
        self.profiling_fn = profiling_fn or collect_profiles
        self.interval = max(2, float(config.get('interval_seconds', 5)))
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.thread = None
        self.payload = json.dumps(dict(schema_version=1, generated_at=None,
            collection=dict(interval_seconds=self.interval, duration_ms=None, warnings=[]),
            host=dict(hostname=None, cpu=dict(model=None, logical_count=None, utilization_percent=None), memory=dict(total_bytes=None, used_bytes=None), gpus=[], sampled_at=None, warnings=['No successful host sample']),
            runs=[], stages=[], insights=[], profiling=dict(lens=dict(available=False, source=None, spans=[], warnings=[]), nsight=dict(available=False, source=None, kernels=[], summary={}, warnings=[]), bridge=dict(available=False, source=None, operations=[], warnings=[]), stages={}, optimizations=[]))).encode()
        self.version = 0

    def start(self):
        with self.condition:
            if self.thread and self.thread.is_alive():
                return
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._loop, name='dashboard-collector', daemon=True)
            self.thread.start()
        if not self.ready.wait(15):
            raise RuntimeError('Initial dashboard collection exceeded startup timeout')

    def reload_config(self):
        if not self.config_path:
            return
        try:
            stat = self.config_path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if signature == self.config_signature:
                return
            self.config_signature = signature
            relative = str(self.config_path.relative_to(self.root))
            self.config = load_config(self.root, relative)
            self.interval = float(self.config.get('interval_seconds', 5))
            self.reload_warning = None
        except (OSError, ValueError, TypeError, KeyError):
            self.reload_warning = 'Configuration reload rejected; previous explicit allowlist retained'

    def _loop(self):
        while not self.stop_event.is_set():
            start = time.monotonic()
            self.reload_config()
            try:
                effective, discovery = expand_run_roots(self.root,self.config)
                effective = validate_config(self.root,effective)
                snapshot = self.collect_fn(self.root, effective)
                snapshot['collection']['discovery'] = discovery
                snapshot['collection'].setdefault('warnings',[]).extend(discovery['warnings'])
                if self.reload_warning:
                    snapshot['collection'].setdefault('warnings', []).append(self.reload_warning)
                snapshot['profiling'] = self.profiling_fn(self.root, effective)
                snapshot['collection']['duration_ms'] = (time.monotonic() - start) * 1000
                payload = json.dumps(snapshot, allow_nan=False, ensure_ascii=False, separators=(',', ':')).encode()
                with self.condition:
                    self.payload = payload
                    self.version += 1
                    self.condition.notify_all()
            except Exception:
                # Never publish exception text: source artifacts/config can contain secrets.
                with self.condition:
                    previous = json.loads(self.payload)
                    warnings = previous.setdefault('collection', {}).setdefault('warnings', [])
                    message = 'Collection failed; last cached snapshot retained'
                    if message not in warnings:
                        warnings.append(message)
                    previous['collection']['warnings'] = warnings[-20:]
                    self.payload = json.dumps(previous).encode()
                    self.version += 1
                    self.condition.notify_all()
            self.ready.set()
            self.stop_event.wait(self.interval)

    def read(self):
        with self.condition:
            return self.payload, self.version

    def stop(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread:
            self.thread.join(5)


class DashboardHandler(BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(self.server.write_timeout)
        super().setup()

    def log_message(self, *args):
        pass

    def end_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        # Reject browser DNS-rebinding and cross-origin fetches. No CORS.
        host = self.headers.get('Host', '').split(':', 1)[0]
        if host not in self.server.allowed_hosts:
            self.send_error(403)
            return
        origin = self.headers.get('Origin')
        if origin and origin not in ('http://' + self.headers.get('Host', ''), 'https://' + self.headers.get('Host', '')):
            self.send_error(403)
            return
        if self.path in ('/api/events', '/api/stream'):
            self.events()
            return
        if self.path == '/api/snapshot':
            payload, _ = self.server.service.read()
            content_type = 'application/json; charset=utf-8'
        elif self.path in ('/', '/index.html', '/style.css', '/app.js', '/timeline.js', '/NemoChatKorean.ttf'):
            name = 'index.html' if self.path == '/' else self.path[1:]
            file = (self.server.static_dir / name).resolve()
            limit = 3 * 1024 * 1024 if name == 'NemoChatKorean.ttf' else 2 * 1024 * 1024
            if not file.is_relative_to(self.server.static_dir) or not file.is_file() or file.stat().st_size > limit:
                self.send_error(404)
                return
            payload = file.read_bytes()
            content_type = {'index.html': 'text/html', 'style.css': 'text/css', 'app.js': 'text/javascript', 'timeline.js': 'text/javascript', 'NemoChatKorean.ttf': 'font/ttf'}[name]
            if name != 'NemoChatKorean.ttf':
                content_type += '; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (OSError, TimeoutError):
            pass

    def events(self):
        if not self.server.sse_slots.acquire(blocking=False):
            self.send_error(503, 'SSE client limit reached; use snapshot polling')
            return
        with self.server.sse_lock:
            self.server.sse_count += 1
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
            self.send_header('Connection', 'close')
            self.send_header('X-Accel-Buffering', 'no')
            self.end_headers()
            version = -1
            service = self.server.service
            while not service.stop_event.is_set():
                payload, current = service.read()
                if version != current:
                    self.wfile.write(b'event: snapshot\ndata: ' + payload + b'\n\n')
                    version = current
                else:
                    self.wfile.write(b': heartbeat\n\n')
                self.wfile.flush()
                with service.condition:
                    service.condition.wait_for(lambda: service.version != version or service.stop_event.is_set(), timeout=2)
        except (OSError, TimeoutError):
            pass
        finally:
            with self.server.sse_lock:
                self.server.sse_count -= 1
            self.server.sse_slots.release()


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def process_request(self, request, client_address):
        # Bound request threads too, including clients that never send headers.
        if not self.request_slots.acquire(blocking=False):
            try:
                request.settimeout(self.write_timeout)
                request.sendall(b'HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n')
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.request_slots.release()


def make_server(root, config, host='127.0.0.1', port=8787, service=None, static_dir=None):
    server = DashboardServer((host, port), DashboardHandler)
    server.service = service or SnapshotService(root, config)
    server.static_dir = Path(static_dir or Path(__file__).parent / 'static').resolve()
    server.write_timeout = min(5, max(.1, float(config.get('write_timeout_seconds', 2))))
    server.sse_slots = threading.BoundedSemaphore(min(16, max(1, int(config.get('max_sse_clients', 4)))))
    server.request_slots = threading.BoundedSemaphore(32)
    server.sse_lock = threading.Lock()
    server.sse_count = 0
    server.allowed_hosts = {'127.0.0.1', 'localhost', host}
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--config', default='dashboard/config.json')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--interval', type=float)
    parser.add_argument('--reload-config', action='store_true', help='Stat-check explicit config each collection; valid allowlist edits apply without restart')
    parser.add_argument('--allow-remote', action='store_true', help='Explicitly opt into non-loopback access; no authentication/TLS is provided')
    args = parser.parse_args()
    if args.host not in ('127.0.0.1', 'localhost') and not args.allow_remote:
        parser.error('Non-loopback binding requires --allow-remote; prefer SSH forwarding')
    root = Path(args.root).resolve()
    config = load_config(root, args.config)
    if args.interval is not None:
        config['interval_seconds'] = min(300, max(2, args.interval))
    service = SnapshotService(root, config, config_path=secure_path(root, args.config) if args.reload_config else None)
    server = make_server(root, config, args.host, args.port, service=service)
    server.service.start()
    print(f'Dashboard: http://{args.host}:{server.server_port} (read-only local artifacts)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.service.stop()


if __name__ == '__main__':
    main()
