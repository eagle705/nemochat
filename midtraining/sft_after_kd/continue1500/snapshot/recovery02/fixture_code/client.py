"""Frozen sealed03 client. Never opens original data paths.

open_snapshot validates descriptor/header identity; catalogs additionally receive
full SHA validation. The separate CPU consumer full-hashes all seven originals.
"""
import array
import fcntl
import hashlib
import io
import json
import mmap
import os
import socket
import struct
import numpy as np

CHUNK = 8 * 1024**2
DOMAIN = b'nemochat-sft-row-v1\0'

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def row_digest(index, tokrow, maskrow):
    require(isinstance(index, (int, np.integer)) and 0 <= index < 2**64, 'row_index')
    tokrow = np.asarray(tokrow)
    maskrow = np.asarray(maskrow)
    require(tokrow.ndim == maskrow.ndim == 1 and tokrow.dtype == np.dtype('<u4') and maskrow.dtype == np.dtype('u1'), 'row_dtype')
    require(tokrow.flags.c_contiguous and maskrow.flags.c_contiguous, 'row_contiguous')
    h = hashlib.sha256(DOMAIN + struct.pack('<Q', int(index)))
    h.update(memoryview(tokrow)); h.update(memoryview(maskrow))
    return h.digest()

def digest_fd(fd, size):
    h = hashlib.sha256()
    for offset in range(0, size, CHUNK):
        data = os.pread(fd, min(CHUNK, size-offset), offset)
        require(len(data) == min(CHUNK, size-offset), 'short_read')
        h.update(data)
    return h.hexdigest()

def request_header(ready, file_id):
    return {k: ready[k] for k in ('schema', 'status', 'snapshot_id', 'sft_manifest_sha256', 'c_manifest_sha256', 'holder_identity', 'producer_source_freeze_sha256')} | {'file': ready['files'][file_id]}

def validate_descriptor(fd, meta):
    require(fcntl.fcntl(fd, 1034) == meta['seals'] == 15, 'seals')
    require(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY and meta['access_mode'] == 'O_RDONLY', 'access_mode')
    require(os.fstat(fd).st_size == meta['bytes'], 'size')

class OpenedSnapshotFile:
    def __init__(self, fd, meta):
        self.fd, self.array, self.mapping = fd, None, None
        validate_descriptor(fd, meta)
        if meta['shape'] is not None:
            self.mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            if meta['file_id'].startswith('rowhash:'):
                offset, shape, dtype = 0, tuple(meta['shape']), np.dtype('u1')
                require(len(shape) == 2 and shape[1] == 32 and meta['dtype'] == 'uint8', 'catalog_header')
                require(digest_fd(fd, meta['bytes']) == meta['sha256'], 'catalog_sha256')
            else:
                b = io.BytesIO(self.mapping[:65536])
                version = np.lib.format.read_magic(b)
                require(version in ((1,0), (2,0)), 'npy_version')
                reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(b); offset = b.tell()
                require(not fortran and list(shape) == meta['shape'] and str(dtype) == meta['dtype'], 'shape_dtype')
                require(not dtype.hasobject and dtype in (np.dtype('<u4'), np.dtype('u1')), 'dtype')
            require(offset + int(np.prod(shape, dtype=np.int64)) * dtype.itemsize == meta['bytes'], 'array_size')
            self.array = np.ndarray(shape, dtype=dtype, buffer=self.mapping, offset=offset)
            require(not self.array.flags.writeable, 'array_writeable')
    def close(self):
        self.array = None
        if self.mapping is not None:
            self.mapping.close(); self.mapping = None
        if self.fd is not None:
            os.close(self.fd); self.fd = None

def open_snapshot(ready, file_ids):
    require(ready['schema'] == 'nemochat-sealed-full03-v1' and ready['status'] == 'accepted_sealed_snapshot_full03', 'ready_schema_status')
    require(isinstance(file_ids, list) and len(file_ids) == len(set(file_ids)), 'file_ids')
    opened, pending = {}, []
    try:
        for file_id in file_ids:
            meta = ready['files'][file_id]
            require(meta['file_id'] == file_id, 'file_identity')
            header = request_header(ready, file_id)
            with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as sock:
                sock.settimeout(60)
                sock.connect(os.environ.get('NEMOCHAT_SEALED_SOCKET_PATH', ready['socket_path']))
                sock.sendall(json.dumps(header, sort_keys=True).encode())
                data, ancillary, flags, _ = sock.recvmsg(65536, socket.CMSG_SPACE(64), socket.MSG_CMSG_CLOEXEC)
                for level, kind, raw in ancillary:
                    require(level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS, 'ancillary_kind')
                    ints = array.array('i'); ints.frombytes(raw); pending.extend(ints)
                require(not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC), 'truncation')
                require(json.loads(data) == header, 'header_identity')
                require(len(pending) == 1, 'fd_count')
                fd = pending[0]
                obj = OpenedSnapshotFile(fd, meta)
                opened[file_id] = obj; pending.clear()
        return opened
    except BaseException:
        for obj in opened.values(): obj.close()
        for fd in pending: os.close(fd)
        raise
