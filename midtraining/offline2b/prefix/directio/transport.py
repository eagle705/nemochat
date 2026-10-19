"""Fail-closed local closed-tar transport. No buffered payload reads or fallback."""
import ctypes
from dataclasses import dataclass
import hashlib
import io
import mmap
import os
from pathlib import Path
import stat
import struct
import threading

# A fork waits until no aligned mapping / direct fd is live. Workers open locally.
_IO_LOCK = threading.Lock()
os.register_at_fork(before=_IO_LOCK.acquire, after_in_parent=_IO_LOCK.release,
                    after_in_child=_IO_LOCK.release)
CHUNK = 1024 * 1024


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


@dataclass(frozen=True)
class Expectation:
    path: str
    sha256: str
    stat: tuple


def expectation(path, sha256):
    p = Path(path)
    return Expectation(str(p), sha256, identity(p.lstat()))


def dio_alignment(fd):
    """Linux statx layout: stx_dio_mem_align / stx_dio_offset_align."""
    buf = ctypes.create_string_buffer(256)
    libc = ctypes.CDLL(None, use_errno=True)
    rc = libc.statx(fd, ctypes.c_char_p(b''), 0x1000, 0x2000, ctypes.byref(buf))
    mask = struct.unpack_from('I', buf.raw, 0)[0]
    mem, off = struct.unpack_from('II', buf.raw, 152)
    if rc != 0 or not mask & 0x2000 or not mem or not off:
        raise RuntimeError('statx DIO alignment unavailable; no fallback')
    if any(x > 4096 or 4096 % x for x in (mem, off)):
        raise RuntimeError(f'unsupported DIO alignment: {mem}/{off}')
    return mem, off


class DirectTransport:
    def __init__(self, entries):
        import re
        from types import MappingProxyType
        approved = {}
        for e in entries:
            p = Path(e.path)
            if (not p.is_absolute() or str(p) != e.path or p.resolve() != p
                    or not re.fullmatch(r'cp0_dp[0-7]__\d+\.tar', p.name)
                    or not re.fullmatch(r'[0-9a-f]{64}', e.sha256)
                    or len(e.stat) != 5 or not 0 < e.stat[2] <= 64 * 1024**2
                    or not stat.S_ISREG(p.lstat().st_mode)
                    or identity(p.lstat()) != e.stat or e.path in approved):
                raise ValueError('invalid/duplicate closed-tar allowlist entry')
            approved[e.path] = e
        if not approved:
            raise ValueError('empty allowlist')
        self.entries = MappingProxyType(approved)

    def read_bytes(self, path):
        e = self.entries[path]
        with _IO_LOCK:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
            try:
                pre = os.fstat(fd)
                if not stat.S_ISREG(pre.st_mode) or identity(pre) != e.stat:
                    raise RuntimeError('pre-read stat mismatch')
                _, align = dio_alignment(fd)
                parts = []
                offset = 0
                with mmap.mmap(-1, CHUNK) as buf:
                    while offset < pre.st_size:
                        want = min(CHUNK, ((pre.st_size-offset+4095)//4096)*4096)
                        view = memoryview(buf)[:want]
                        try:
                            n = os.preadv(fd, [view], offset)
                        finally:
                            view.release()
                        if n <= 0 or n > min(want, pre.st_size-offset):
                            raise RuntimeError('invalid/zero short read')
                        parts.append(buf[:n])
                        offset += n
                        if offset < pre.st_size and n % align:
                            raise RuntimeError('unaligned short read')
                if identity(os.fstat(fd)) != e.stat or identity(os.lstat(path)) != e.stat:
                    raise RuntimeError('post-read stat/path mismatch')
                data = b''.join(parts)
                if len(data) != pre.st_size or hashlib.sha256(data).hexdigest() != e.sha256:
                    raise RuntimeError('size/SHA mismatch')
                return data
            finally:
                os.close(fd)

    def open(self, path, mode='rb', **kwargs):
        if mode != 'rb' or kwargs not in ({}, {'prefetch_file': True}):
            raise ValueError('only native local rb/prefetch_file=True is approved')
        return io.BytesIO(self.read_bytes(path))
