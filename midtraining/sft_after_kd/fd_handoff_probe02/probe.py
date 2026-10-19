"""Single bounded acquisition, diagnostic only. No production release."""
import array
import ast
import fcntl
import hashlib
import http.client
import json
import mmap
import os
from pathlib import Path
import resource
import shutil
import signal
import socket
import struct
import tempfile
import time
import traceback
import urllib.parse
import uuid

ROOT = Path('/raid/yoon/nemochat')
OUT = ROOT / 'results/midtraining/sft-after-kd-fd-probe-02'
IMAGE = 'nvcr.io/nvidia/nemo@sha256:fdd6e9c7929b76c8624ddfea939dda345beaf67695d9c3994994c18f9e79b9bf'
EXPECTED = '19a28db000dabf1bae97cc85c2abbf6c38ac2837f84b08baf71686724182aaf4'
MANIFEST = '78ab81fbbd9c0b13c55392b7d87c9e7035ad75778abad0dd3795d64121125c21'
SIZE = 2955198728
CHUNK = 4 * 1024 * 1024
CONSUMER = Path(__file__).with_name('consumer.py').read_text()
compile(CONSUMER, 'consumer.py', 'exec')

class UnixHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect('/var/run/docker.sock')

def api(method, path, body=None, raw=False, missing=False):
    connection = UnixHTTP('localhost', timeout=15)
    try:
        connection.request(method, '/v1.52' + path, None if body is None else json.dumps(body).encode(), {'Content-Type': 'application/json'})
        response = connection.getresponse()
        payload = response.read()
        if missing and response.status == 404:
            return None
        if response.status >= 300:
            raise RuntimeError((method, path, response.status, payload[:1500]))
        return payload if raw else (json.loads(payload) if payload else None)
    finally:
        connection.close()

def stat(path):
    value = os.stat(path)
    return {key: getattr(value, key) for key in ['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns']}

def digest(fd, size):
    h = hashlib.sha256()
    for offset in range(0, size, CHUNK):
        block = os.pread(fd, min(CHUNK, size-offset), offset)
        assert len(block) == min(CHUNK, size-offset)
        h.update(block)
    return h.hexdigest()

def seal(fd):
    fcntl.fcntl(fd, 1033, 15)
    assert fcntl.fcntl(fd, 1034) == 15
    denied = {}
    for key, action in [('write', lambda: os.pwrite(fd, b'x', 0)), ('grow', lambda: os.ftruncate(fd, os.fstat(fd).st_size+1)), ('shrink', lambda: os.ftruncate(fd, 0))]:
        try:
            action()
        except OSError as error:
            denied[key] = error.errno
        else:
            raise AssertionError(key)
    return {'mask': 15, 'mutation_denials': denied}

def validate_descriptor(fd, identity):
    assert fcntl.fcntl(fd, 1034) == 15, 'seals'
    assert fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY, 'access_mode'
    assert os.fstat(fd).st_size == identity['bytes'], 'size'

def identity(size, sha, shape, name, manifest):
    return {'role': 'heldout-token-diagnostic', 'name': name, 'bytes': size, 'sha256': sha, 'manifest_sha256': manifest, 'shape': shape, 'dtype': 'uint32'}

def consume(fd, ident, label, report, negative='', bypass=False):
    directory = tempfile.mkdtemp(prefix='fd02-', dir=os.environ['TMPDIR'])
    assert len((directory + '/s').encode()) < 108
    os.chmod(directory, 0o700)
    assert os.stat(directory).st_uid == 1001
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    peer = None
    cid = None
    name = 'nemochat-fd-probe02-' + label + '-' + uuid.uuid4().hex[:10]
    labels = {'nemochat.fd-probe': '02', 'probe.phase': label, 'probe.instance': name}
    report.update(name=name, labels=labels, identity=ident)
    try:
        listener.bind(directory + '/s')
        listener.listen(1)
        listener.settimeout(45)
        payload = {'Image': IMAGE, 'User': '1001:1001', 'Entrypoint': ['python3', '-c', CONSUMER], 'Cmd': [], 'WorkingDir': '/', 'Env': ['IDENTITY='+json.dumps(ident, sort_keys=True), 'EXPECT_REJECT='+negative, 'BAD_REQUEST='+str(int(label == 'bad-client-header')), 'NVIDIA_VISIBLE_DEVICES=void', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'PYTHONDONTWRITEBYTECODE=1'], 'Labels': labels, 'NetworkDisabled': True, 'HostConfig': {'Binds': [directory+':/handoff:ro'], 'NetworkMode': 'none', 'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges:true'], 'ReadonlyRootfs': True, 'Memory': 8*1024**3, 'MemorySwap': 8*1024**3, 'NanoCpus': 4*10**9, 'PidsLimit': 64, 'DeviceRequests': [], 'Devices': [], 'AutoRemove': False, 'Runtime': 'runc'}}
        report['create_attempted'] = True
        cid = api('POST', '/containers/create?name='+name, payload)['Id']
        report['container_id'] = cid
        inspected = api('GET', '/containers/'+cid+'/json')
        hc = inspected['HostConfig']
        assert hc['Memory'] == 8*1024**3 and hc['MemorySwap'] == 8*1024**3 and hc['NanoCpus'] == 4*10**9
        assert hc['NetworkMode'] == 'none' and hc['ReadonlyRootfs'] and hc['CapDrop'] == ['ALL']
        assert hc['SecurityOpt'] == ['no-new-privileges:true'] and not hc['Devices'] and not hc['DeviceRequests'] and not hc['Privileged']
        assert inspected['Config']['User'] == '1001:1001'
        assert len(inspected['Mounts']) == 1 and inspected['Mounts'][0]['Destination'] == '/handoff' and not inspected['Mounts'][0]['RW']
        report['sandbox'] = {key: hc[key] for key in ['Memory', 'MemorySwap', 'NanoCpus', 'NetworkMode', 'ReadonlyRootfs', 'CapDrop', 'SecurityOpt', 'Devices', 'DeviceRequests', 'Runtime']}
        report['mounts'] = inspected['Mounts']
        report['image_id'] = inspected['Image']
        api('POST', '/containers/'+cid+'/start')
        peer, _ = listener.accept()
        peer.settimeout(15)
        pid, uid, gid = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        state = api('GET', '/containers/'+cid+'/json')['State']
        report['peer'] = {'pid': pid, 'uid': uid, 'gid': gid, 'docker_state_pid': state['Pid']}
        assert uid == 1001 and pid == state['Pid'] and state['Running']
        request = json.loads(peer.recv(16384))
        if request != ident:
            report['host_header_denied'] = True
            peer.send(json.dumps({'denied': 'header_identity'}).encode())
        else:
            if not bypass:
                validate_descriptor(fd, ident)
            header = dict(ident)
            if label == 'bad-host-header':
                header['sha256'] = '0' * 64
            peer.sendmsg([json.dumps(header, sort_keys=True).encode()], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))])
            report['descriptor_sent'] = True
        deadline = time.monotonic() + 120
        while True:
            state = api('GET', '/containers/'+cid+'/json')['State']
            if not state['Running']:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('consumer deadline')
            time.sleep(.1)
        logs = api('GET', '/containers/'+cid+'/logs?stdout=1&stderr=1', raw=True)
        text = b''
        while logs:
            assert len(logs) >= 8
            length = int.from_bytes(logs[4:8], 'big')
            text += logs[8:8+length]
            logs = logs[8+length:]
        report['logs'] = text.decode(errors='replace')
        report['exit_code'] = state['ExitCode']
        assert state['ExitCode'] == 0, report['logs']
        report['result'] = json.loads(report['logs'].strip().splitlines()[-1])
        assert report['result']['status'] == ('EXPECTED_REJECTION' if negative else 'PASS')
        return report['result']
    finally:
        if peer is not None:
            peer.close()
        listener.close()
        # Resolve exact name on create-response interruption; never broad cleanup.
        if cid is None and report.get('create_attempted'):
            owned = api('GET', '/containers/'+name+'/json', missing=True)
            if owned:
                assert all(owned['Config']['Labels'].get(k) == v for k, v in labels.items())
                cid = owned['Id']
        if cid:
            owned = api('GET', '/containers/'+cid+'/json', missing=True)
            if owned:
                assert all(owned['Config']['Labels'].get(k) == v for k, v in labels.items())
                api('DELETE', '/containers/'+cid+'?force=1')
            assert api('GET', '/containers/'+cid+'/json', missing=True) is None
            report['cleanup_verified_absent'] = True
        shutil.rmtree(directory)
        report['sockets_closed_directory_removed'] = True

def fixture():
    # Genuine .npy v1.0, generated per format rather than fabricated consumer output.
    header = repr({'descr': '<u4', 'fortran_order': False, 'shape': (3, 4)}).encode()
    header += b' ' * ((64-(10+len(header)+1)%64)%64) + b'\n'
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<12I', *range(12))

def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'PROBE.json').exists(), 'single-run receipt exists; do not reacquire'
    report = {'schema': 'nemochat-fd-handoff-probe02', 'verdict': 'BLOCKED', 'release_allowed': False, 'SFT_accepted': False, 'original_mmap_inputs_blocked': True, 'tests': {}, 'source_views': [], 'source_write_operations': 0, 'production_requirements': ['Sealed exact-identity snapshots for all seven files', 'C-corpus provenance/semantic/reserved-ID proofs and source freeze', 'FD-aware native consumer adapter and per-row authentication', 'Reviewed memory/cgroup/lifetime budget; no production implementation authorized'], 'scope': 'One-file byte handoff only; not corpus, native-model, GPU, host-health or reserved-ID proof.'}
    fds = []
    mapped = []
    def timeout(*_):
        raise TimeoutError('590-second diagnostic deadline')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(590)
    resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    try:
        report['pinned_image'] = api('GET', '/images/'+urllib.parse.quote(IMAGE, safe='')+'/json')['Id']
        report['consumer_sha256'] = hashlib.sha256(CONSUMER.encode()).hexdigest()
        tiny_bytes = fixture()
        tiny = os.memfd_create('fd02-tiny', os.MFD_ALLOW_SEALING); fds.append(tiny)
        assert os.write(tiny, tiny_bytes) == len(tiny_bytes)
        report['tiny_seals'] = seal(tiny)
        tiny_ro = os.open('/proc/self/fd/'+str(tiny), os.O_RDONLY); fds.append(tiny_ro)
        ident = identity(len(tiny_bytes), hashlib.sha256(tiny_bytes).hexdigest(), [3, 4], 'tiny.npy', hashlib.sha256(b'genuine-fixture-v1').hexdigest())
        unsealed = os.memfd_create('fd02-unsealed', os.MFD_ALLOW_SEALING); fds.append(unsealed)
        os.write(unsealed, tiny_bytes)
        unsealed_ro = os.open('/proc/self/fd/'+str(unsealed), os.O_RDONLY); fds.append(unsealed_ro)
        for label, fd, negative, bypass in [('tiny-valid', tiny_ro, '', False), ('bad-client-header', tiny_ro, 'header_identity', False), ('bad-host-header', tiny_ro, 'header_identity', False), ('bad-access-mode', tiny, 'access_mode', True), ('bad-seals', unsealed_ro, 'seals', True)]:
            report['tests'][label] = {}
            consume(fd, ident, label, report['tests'][label], negative, bypass)
        report['tests']['host_descriptor_rejections'] = {}
        for label, fd, reason in [('rw', tiny, 'access_mode'), ('unsealed', unsealed_ro, 'seals')]:
            try:
                validate_descriptor(fd, ident)
            except AssertionError as error:
                assert str(error) == reason
                report['tests']['host_descriptor_rejections'][label] = reason
            else:
                raise AssertionError('host descriptor accepted')
        for fd in fds[:]:
            os.close(fd); fds.remove(fd)
        report['tiny_all_passed'] = True
        manifest = ROOT/'datasets/nemotron_sft/nemochat_baby/prepared-full-03/data/manifest.json'
        mb = manifest.read_bytes()
        assert hashlib.sha256(mb).hexdigest() == MANIFEST
        assert json.loads(mb)['files']['heldout-tokens.npy'] == {'bytes': SIZE, 'sha256': EXPECTED}
        source = manifest.parent/'heldout-tokens.npy'
        report.update(source=str(source), manifest_sha256=MANIFEST, expected_sha256=EXPECTED, expected_bytes=SIZE, source_before=stat(source))
        assert report['source_before']['st_size'] == SIZE
        sourcefds = [os.open(source, flags) for flags in [os.O_RDONLY, os.O_RDONLY|os.O_DIRECT, os.O_RDONLY|os.O_DIRECT]]
        fds.extend(sourcefds)
        candidate = os.memfd_create('fd02-heldout-candidate', os.MFD_ALLOW_SEALING); fds.append(candidate)
        os.ftruncate(candidate, SIZE)
        mapped = [mmap.mmap(-1, CHUNK), mmap.mmap(-1, CHUNK)]
        hashes = [hashlib.sha256() for _ in range(3)]
        counts = {'pages': 0, 'unanimous': 0, 'majority': 0, 'tie': 0, 'changed_from_buffered': 0}
        for offset in range(0, SIZE, CHUNK):
            wanted = min(CHUNK, SIZE-offset)
            views = [os.pread(sourcefds[0], wanted, offset)]
            for i in range(2):
                nr = os.preadv(sourcefds[i+1], [mapped[i]], offset)
                assert nr == wanted
                views.append(mapped[i][:nr])
            assert all(len(v) == wanted for v in views)
            for h, v in zip(hashes, views):
                h.update(v)
            selected = bytearray(views[0])
            for p in range(0, wanted, 4096):
                a, b, c = [v[p:p+4096] for v in views]
                if a == b == c:
                    page, kind = a, 'unanimous'
                elif a == b or a == c:
                    page, kind = a, 'majority'
                elif b == c:
                    page, kind = b, 'majority'
                else:
                    page, kind = a, 'tie'
                counts['pages'] += 1; counts[kind] += 1
                if page != a:
                    counts['changed_from_buffered'] += 1
                    selected[p:p+len(page)] = page
            assert os.pwrite(candidate, selected, offset) == wanted
        report['single_real_acquisition_completed'] = True
        report['source_bytes_read'] = SIZE*3
        report['page_votes'] = counts
        report['source_views'] = [{'kind': k, 'sha256': h.hexdigest(), 'matches_original_expected': h.hexdigest() == EXPECTED} for k,h in zip(['buffered_pread', 'O_DIRECT_1', 'O_DIRECT_2'], hashes)]
        for m in mapped:
            m.close()
        mapped = []
        for fd in sourcefds:
            os.close(fd); fds.remove(fd)
        report['source_after_acquisition'] = stat(source)
        assert report['source_before'] == report['source_after_acquisition']
        report['candidate_seals'] = seal(candidate)
        ro = os.open('/proc/self/fd/'+str(candidate), os.O_RDONLY); fds.append(ro)
        report['candidate_sha256'] = digest(ro, SIZE)
        assert report['candidate_sha256'] == EXPECTED, 'Candidate not immutable original SHA; BLOCKED, no retries'
        with mmap.mmap(ro, 0, access=mmap.ACCESS_READ) as mm:
            h = hashlib.sha256()
            for offset in range(0, SIZE, CHUNK):
                h.update(mm[offset:offset+CHUNK])
            report['host_readonly_mmap_sha256'] = h.hexdigest()
        assert report['host_readonly_mmap_sha256'] == EXPECTED
        report['real_consumer'] = {}
        consume(ro, identity(SIZE, EXPECTED, [45090, 16385], 'heldout-tokens.npy', MANIFEST), 'real-heldout', report['real_consumer'])
        report['host_postconsumer_sha256'] = digest(ro, SIZE)
        report['host_postconsumer_seals'] = fcntl.fcntl(ro, 1034)
        assert report['host_postconsumer_sha256'] == EXPECTED and report['host_postconsumer_seals'] == 15
        report['source_after'] = stat(source)
        report['source_metadata_unchanged'] = report['source_before'] == report['source_after']
        assert report['source_metadata_unchanged']
        report['verdict'] = 'VALIDATED_SINGLE_FILE_FD_HANDOFF_ONLY'
    except BaseException as error:
        report['error'] = repr(error)
        report['traceback'] = traceback.format_exc()
    finally:
        signal.alarm(0)
        for m in mapped:
            m.close()
        tracked = list(fds)
        for fd in fds:
            os.close(fd)
        closed = []
        for fd in tracked:
            try:
                os.fstat(fd)
            except OSError:
                closed.append(fd)
        report['holder_fds_closed_verified'] = len(closed) == len(tracked)
        report['all_containers_removed_verified'] = all(t.get('cleanup_verified_absent') for t in list(report['tests'].values()) + [report.get('real_consumer', {})] if 'container_id' in t)
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_holder_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['budget'] = {'seconds': 600, 'holder_address_space_bytes': 8*1024**3, 'container_memory_bytes': 8*1024**3, 'cpus': 4, 'candidate_shared_pages_not_rss_included': True, 'one_candidate_only': True}
        (OUT/'PROBE.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'verdict': report['verdict'], 'error': report.get('error'), 'elapsed_seconds': report['elapsed_seconds'], 'receipt': str(OUT/'PROBE.json')}), flush=True)

if __name__ == '__main__':
    main()
