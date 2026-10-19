"""CPU-only diagnostic consumer: authenticated metadata then one sealed RO FD."""
import array
import fcntl
import hashlib
import json
import mmap
import os
import socket
import sys

CHUNK = 4 * 1024 * 1024

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def main():
    expected = json.loads(os.environ['IDENTITY'])
    rejection = os.environ.get('EXPECT_REJECT', '')
    fds = []
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        sock.settimeout(90)
        sock.connect('/handoff/s')
        request = dict(expected)
        if os.environ.get('BAD_REQUEST') == '1':
            request['role'] = 'wrong-role'
        sock.send(json.dumps(request, sort_keys=True).encode())
        data, ancillary, flags, _ = sock.recvmsg(16384, socket.CMSG_SPACE(16 * array.array('i').itemsize), socket.MSG_CMSG_CLOEXEC)
        for level, kind, raw in ancillary:
            require(level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS, 'ancillary_kind')
            integers = array.array('i')
            integers.frombytes(raw)
            fds.extend(integers)
        require(not flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC), 'truncation')
        header = json.loads(data)
        require(header == expected, 'header_identity')
        require(len(fds) == 1, 'fd_count')
        fd = fds[0]
        seals = fcntl.fcntl(fd, 1034)
        mode = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
        require(seals == 15, 'seals')
        require(mode == os.O_RDONLY, 'access_mode')
        require(os.fstat(fd).st_size == expected['bytes'], 'size')
        digest = hashlib.sha256()
        os.lseek(fd, 0, os.SEEK_SET)
        while True:
            block = os.read(fd, CHUNK)
            if not block:
                break
            digest.update(block)
        read_sha = digest.hexdigest()
        require(read_sha == expected['sha256'], 'read_sha')
        with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as mapping:
            digest = hashlib.sha256()
            for offset in range(0, expected['bytes'], CHUNK):
                digest.update(mapping[offset:offset + CHUNK])
            mmap_sha = digest.hexdigest()
            require(mmap_sha == expected['sha256'], 'mmap_sha')
            require(memoryview(mapping).readonly, 'mmap_readonly')
            try:
                mapping[0] = 0
            except (TypeError, ValueError):
                mmap_mutation_denied = True
            else:
                raise ValueError('mmap_mutation_allowed')
        import numpy as np
        a = np.load('/proc/self/fd/' + str(fd), mmap_mode='r', allow_pickle=False)
        require(list(a.shape) == expected['shape'] and str(a.dtype) == 'uint32', 'shape_dtype')
        require(not a.flags.writeable, 'array_writeable')
        low, high = int(a.min()), int(a.max())
        require(low >= 0 and high < 131072, 'token_bounds')
        try:
            a.flat[0] = 0
        except (ValueError, TypeError):
            array_mutation_denied = True
        else:
            raise ValueError('array_mutation_allowed')
        a._mmap.close()
        denied = {}
        for name, action in [('write', lambda: os.pwrite(fd, b'x', 0)), ('grow', lambda: os.ftruncate(fd, expected['bytes'] + 1)), ('shrink', lambda: os.ftruncate(fd, 0))]:
            try:
                action()
            except OSError as error:
                denied[name] = error.errno
            else:
                raise ValueError(name + '_allowed')
        require(not rejection, 'negative_test_unexpected_pass')
        print(json.dumps({'status': 'PASS', 'identity': expected, 'uid': os.getuid(), 'seals': seals, 'access_mode': mode, 'read_sha256': read_sha, 'mmap_sha256': mmap_sha, 'shape': list(a.shape), 'dtype': str(a.dtype), 'token_min': low, 'token_max': high, 'token_bounds_ok': True, 'mmap_readonly': True, 'array_writeable': False, 'mmap_mutation_denied': mmap_mutation_denied, 'array_mutation_denied': array_mutation_denied, 'mutation_denials': denied, 'numpy_version': np.__version__}), flush=True)
    except ValueError as error:
        if str(error) != rejection:
            raise
        print(json.dumps({'status': 'EXPECTED_REJECTION', 'reason': str(error), 'received_fds': len(fds)}), flush=True)
    finally:
        for fd in fds:
            os.close(fd)
        sock.close()

if __name__ == '__main__':
    main()
