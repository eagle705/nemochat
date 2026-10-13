"""Verify a real local Jaeger UI, with a fail-closed browser origin boundary."""
import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener


MAX_RESPONSE_BYTES = 4 * 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('UI API redirects are not allowed')


def validate_trace(trace, requested, service):
    trace_id = trace['traceID']
    if not isinstance(trace_id, str) or not re.fullmatch(r'[0-9a-fA-F]{1,32}', trace_id):
        raise ValueError('Invalid API trace identity: expected hexadecimal traceID')
    if requested and trace_id.lower() != requested.lower():
        raise ValueError('API trace identity does not match requested trace')
    if not trace['spans'] or any(
        trace['processes'][span['processID']]['serviceName'] != service
        for span in trace['spans']
    ):
        raise ValueError('Trace does not belong to intended Lens service')
    return trace_id, sorted({span['operationName'] for span in trace['spans']})


def render(pw, url, root, trace_id, names, service):
    browser = pw.chromium.launch(headless=True, args=['--no-sandbox', '--no-proxy-server'])
    try:
        context = browser.new_context(viewport={'width': 1440, 'height': 1000},
                                      service_workers='block')
        if not hasattr(context, 'route_web_socket'):
            raise RuntimeError('Upgrade Playwright: route_web_socket is required for safe verification')
        context.grant_permissions(['clipboard-read', 'clipboard-write'], origin=url)
        blocked, errors = [], []
        origin = urlparse(url)

        def route_request(route):
            target = urlparse(route.request.url)
            if (target.scheme, target.hostname, target.port or 80) != (
                    origin.scheme, origin.hostname, origin.port or 80) or target.username or target.password:
                blocked.append('outside approved origin')
                route.abort()
                return
            # continue_() does NOT intercept subsequent HTTP redirect hops. Fetch
            # with redirects disabled, then fulfill only non-redirect responses.
            response = route.fetch(max_redirects=0)
            if 300 <= response.status < 400:
                blocked.append('HTTP redirect')
                route.abort()
            else:
                route.fulfill(response=response)

        def block_socket(socket):
            blocked.append('WebSocket')
            # Routing does not connect to the server unless connect_to_server()
            # is called. Do not close synchronously before the opening handshake:
            # that deadlocks the installed Playwright sync callback. check()
            # fails the run and browser.close() tears down the blocked socket.

        context.route('**/*', route_request)
        context.route_web_socket('**/*', block_socket)
        context.on('page', lambda page: page.on('pageerror', lambda e: errors.append(str(e))))
        page = context.new_page()
        page.set_default_timeout(15000)

        def check(expected):
            if blocked:
                raise RuntimeError('Blocked browser request: ' + repr(blocked))
            if errors:
                raise RuntimeError('Browser JavaScript errors: ' + repr(errors))
            if page.url != expected or len(context.pages) != 1:
                raise RuntimeError('Unexpected final browser URL or popup')
            text = page.locator('body').inner_text()
            if re.search(r'trace not found|404 not found|internal server error|failed to fetch|'
                         r'error loading|something went wrong|unable to load|trace.*not available', text, re.I):
                raise RuntimeError('Rendered UI contains an error page')
            return text

        for path in ['/search', '/trace/' + trace_id]:
            expected = url + path
            page.goto(expected, wait_until='networkidle', timeout=45000)
            check(expected)
            if path == '/search':
                page.get_by_text(re.compile(r'^Service \(')).first.wait_for()
                check(expected)
                page.screenshot(path=str(root / 'search.png'), full_page=True)
                continue
            # Selectors inspected on the actual local Jaeger DOM. Header trace ID
            # is shortened; its clipboard button supplies the complete identity.
            header = page.locator('.TracePageHeader')
            header.locator('.TraceIDLength').wait_for()
            if header.locator('.TraceIDLength').inner_text().strip() != trace_id[:7]:
                raise RuntimeError('Rendered trace identity does not match')
            header.get_by_role('button', name='Copy to clipboard', exact=True).click()
            copied = page.evaluate('navigator.clipboard.readText()')
            if copied.lower() != trace_id.lower():
                raise RuntimeError('Rendered full trace identity does not match')
            if not header.locator('.TraceName').inner_text().startswith(service + ': '):
                raise RuntimeError('Rendered service identity does not match')
            timeline = page.locator('.TraceTimelineViewer')
            for name in names:
                operation = timeline.locator('.span-row .endpoint-name').filter(has_text=re.compile('^' + re.escape(name) + '$'))
                if not operation.count() or not operation.first.is_visible():
                    raise RuntimeError('Missing rendered operation: ' + name)
            services = timeline.locator('.span-row .span-svc-name').all_text_contents()
            if not services or any(s.strip() != service for s in services):
                raise RuntimeError('Rendered span service identity does not match')
            text = check(expected)
            page.screenshot(path=str(root / 'trace.png'), full_page=False)
            (root / 'trace_ui.txt').write_text(text)
            check(expected)
        context.close()
        if blocked or errors:
            raise RuntimeError('Browser verification failed: ' + repr(blocked + errors))
        return errors
    finally:
        browser.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:16687')
    p.add_argument('--output', required=True)
    p.add_argument('--trace-id')
    p.add_argument('--service', default='nemochat-lens-cpu')
    args = p.parse_args(argv)
    parsed = urlparse(args.url)
    try:
        port = parsed.port
    except ValueError:
        p.error('Only credential-free loopback HTTP UI verification is allowed')
    if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost'}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {'', '/'} or port == 0):
        p.error('Only credential-free loopback HTTP UI verification is allowed')
    args.url = args.url.rstrip('/')
    if args.trace_id and not re.fullmatch(r'[0-9a-fA-F]{1,32}', args.trace_id):
        p.error('Invalid trace identity: expected hexadecimal trace ID')
    if not args.service.startswith('nemochat-lens'):
        p.error('Expected a nemochat-lens service')
    root = Path(args.output)
    if root.exists() or root.is_symlink():
        p.error('Output must be a fresh, nonexistent directory')
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError:
        p.error('Install playwright and Chromium in the separate browser verification environment')
    root.mkdir(parents=True, exist_ok=False)
    opener = build_opener(ProxyHandler({}), NoRedirect())

    def get(path):
        with opener.open(args.url + path, timeout=15) as response:
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            if len(payload) > MAX_RESPONSE_BYTES:
                raise ValueError('UI API response too large')
            return json.loads(payload)['data']

    services = get('/api/services')
    if args.service not in services:
        raise RuntimeError('Intended Lens service is not stored')
    if args.trace_id:
        trace = get('/api/traces/' + args.trace_id)[0]
    else:
        traces = get('/api/traces?' + urlencode({'service': args.service, 'limit': 10}))
        if not traces:
            raise RuntimeError('No actual stored trace is available for UI verification')
        trace = max(traces, key=lambda t: len(t['spans']))
    trace_id, names = validate_trace(trace, args.trace_id, args.service)
    try:
        with sync_playwright() as pw:
            errors = render(pw, args.url, root, trace_id, names, args.service)
    except Error as exc:
        if 'Executable doesn\'t exist' in str(exc):
            p.error('Install Chromium with python -m playwright install chromium in the browser environment')
        raise
    result = {'url': args.url, 'services': services, 'service': args.service, 'trace_id': trace_id,
              'stored_span_count': len(trace['spans']), 'rendered_operations': names,
              'page_errors': errors, 'screenshots': [str(root / 'search.png'), str(root / 'trace.png')],
              'scope': 'Actual Jaeger UI rendered in Chromium; not native Nsight GUI verification'}
    with (root / 'verification.json').open('x') as receipt:
        receipt.write(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
