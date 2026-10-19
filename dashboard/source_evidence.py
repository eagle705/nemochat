"""Bounded saved-source verification, never execution or kernel attribution.

Bridge identity/hash rules mirror lens_nsight_bridge.source.extract_verified.
This standalone implementation intentionally imports no plugin or GPU SDK.
"""
import ast
import hashlib
import json
import os
import stat
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path, PurePosixPath

_MAX_FILE = 512 * 1024
_MAX_SNIPPET = 12 * 1024
_MAX_LINES = 60
_WARNING = ('Explicit educational host function only; not a native kernel Python '
            'stack frame or verified kernel attribution. Hashes do not authenticate producers.')


_CACHE = OrderedDict()  # <=32 files, <=16MiB of saved bytes.


@lru_cache(maxsize=64)
def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _result(kind, status='rejected', warning='Source evidence rejected.'):
    return dict(status=status, kind=kind, path=None, function=None,
                start_line=None, end_line=None, file_sha256=None,
                snippet_sha256=None, source_id=None, snippet='', warning=warning)


def _path(root, name):
    root = Path(root).resolve(strict=True)
    # Public callers may supply an absolute configured path under root.
    if isinstance(name, Path) and name.is_absolute():
        name = name.relative_to(root).as_posix()
    elif isinstance(name, Path):
        name = name.as_posix()
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or PurePosixPath(name).is_absolute()
            or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('path')
    current = root
    for part in name.split('/'):
        current = current / part
        if current.is_symlink():
            raise ValueError('symlink')
    if not current.resolve().is_relative_to(root):
        raise ValueError('containment')
    return current


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_uncached(path, expected_stat):
    # Descriptor-relative traversal prevents symlink replacement during lookup.
    directory = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    descriptor = None
    try:
        for part in path.parts[1:-1]:
            next_directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                     dir_fd=directory)
            os.close(directory)
            directory = next_directory
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=directory)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _signature(before) != expected_stat:
            raise ValueError('file changed')
        with os.fdopen(descriptor, 'rb') as stream:
            descriptor = None
            data = stream.read(_MAX_FILE + 1)
            after = os.fstat(stream.fileno())
        if len(data) > _MAX_FILE or _signature(after) != expected_stat:
            raise ValueError('file changed or oversized')
        return data
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _read(path):
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_FILE:
        raise ValueError('size or type')
    signature = _signature(info)
    key = (str(path), signature)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    data = _read_uncached(path, signature)
    _CACHE[key] = data
    while len(_CACHE) > 32:
        _CACHE.popitem(last=False)
    return data


def _digest(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('hash')
    return value


def _snippet(data, first, last):
    lines = data.decode('utf-8').splitlines(keepends=True)
    if (type(first) is not int or type(last) is not int
            or not 1 <= first <= last <= len(lines) or last-first+1 > _MAX_LINES):
        raise ValueError('lines')
    text = ''.join(lines[first-1:last])
    if len(text.encode('utf-8')) > _MAX_SNIPPET:
        raise ValueError('snippet size')
    return text


def _function(data, first, last):
    try:
        tree = ast.parse(data.decode('utf-8'))
        matches = [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.lineno <= first <= last <= n.end_lineno]
        return min(matches, key=lambda n: n.end_lineno-n.lineno).name if matches else '<module>'
    except (SyntaxError, ValueError, RecursionError):
        return None


def verified_native_phase_source(root: Path, source_path: Path,
                                 expected_sha256: str, phase: str) -> dict:
    """Exact saved nvphase scope including host calls; never a kernel backtrace."""
    result = _result('host_scope')
    try:
        if phase not in ('data', 'forward', 'backward+sync', 'optimizer'):
            raise ValueError('unknown phase')
        expected_sha256 = _digest(expected_sha256)
        path = _path(root, source_path)
        if path.suffix != '.py' or any(part.startswith('.env') for part in path.parts):
            raise ValueError('not source')
        data = _read(path)
        if _hash(data) != expected_sha256:
            raise ValueError('file hash')
        tree = ast.parse(data.decode('utf-8'))
        scopes = [n for n in ast.walk(tree) if isinstance(n, ast.With)
                  and any(isinstance(i.context_expr, ast.Call)
                          and isinstance(i.context_expr.func, ast.Name)
                          and i.context_expr.func.id == 'nvphase'
                          and i.context_expr.args
                          and isinstance(i.context_expr.args[0], ast.Constant)
                          and type(i.context_expr.args[0].value) is str
                          and i.context_expr.args[0].value == phase for i in n.items)]
        if len(scopes) != 1:
            raise ValueError('missing or ambiguous marker')
        scope = scopes[0]
        text = _snippet(data, scope.lineno, scope.end_lineno)
        result.update(status='host_scope_verified', path=path.relative_to(Path(root).resolve()).as_posix(),
                      function=_function(data, scope.lineno, scope.end_lineno),
                      start_line=scope.lineno, end_line=scope.end_lineno,
                      file_sha256=expected_sha256, snippet_sha256=_hash(text.encode()),
                      snippet=text, warning='Verified saved host scope only; NOT a kernel backtrace or native kernel Python stack frame. Hashes do not authenticate producers.')
    except (OSError, ValueError, TypeError, KeyError, AttributeError, SyntaxError, RecursionError):
        pass
    return result


def verified_bridge_source(root: Path, operations_path: Path, record: dict) -> dict:
    """Verify captured host functions, not their relationship to GPU kernels."""
    result = _result('explicit_bridge_operation')
    try:
        operations = _path(root, operations_path)
        if not isinstance(record, dict):
            raise ValueError('record')
        saved_records = [json.loads(line) for line in _read(operations).decode('utf-8').splitlines() if line.strip()]
        # Bind evidence to the explicitly configured saved operation, not a caller fabrication.
        if not any(json.dumps(saved, sort_keys=True) == json.dumps(record, sort_keys=True)
                   for saved in saved_records):
            raise ValueError('operation identity')
        ref = record['source']
        if not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not isinstance(ref.get('snapshot'), str):
            raise ValueError('reference')
        _path(operations.parent, ref['path'])
        file_hash = _digest(ref['file_sha256'])
        snippet_hash = _digest(ref['snippet_sha256'])
        source_id = _digest(ref['source_id'])
        if ref['snapshot'] != 'sources/' + file_hash + '.py':
            raise ValueError('snapshot identity')
        data = _read(_path(operations.parent, ref['snapshot']))
        fields = {k: ref[k] for k in ('path', 'start_line', 'end_line',
                                    'file_sha256', 'snippet_sha256')}
        if _hash(data) != file_hash or _hash(json.dumps(fields, sort_keys=True).encode()) != source_id:
            raise ValueError('identity')
        text = _snippet(data, ref['start_line'], ref['end_line'])
        if _hash(text.encode()) != snippet_hash:
            raise ValueError('snippet hash')
        result.update(status='snapshot_verified', path=ref['path'],
                      function=_function(data, ref['start_line'], ref['end_line']),
                      start_line=ref['start_line'], end_line=ref['end_line'],
                      file_sha256=file_hash, snippet_sha256=snippet_hash,
                      source_id=source_id, snippet=text, warning=_WARNING)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        pass
    return result
