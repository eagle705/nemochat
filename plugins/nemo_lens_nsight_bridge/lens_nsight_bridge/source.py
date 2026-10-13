"""Content-addressed Python snapshots. Hashes detect drift, not authenticity."""
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Unsafe source path")
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(p in ("..", ".") for p in name.split("/")):
        raise ValueError("Unsafe source path")
    root = Path(root).resolve()
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("Source path escapes root")
    return path


def identity(ref):
    fields = {k: ref[k] for k in ("path", "start_line", "end_line", "file_sha256", "snippet_sha256")}
    return digest(json.dumps(fields, sort_keys=True).encode())


def snapshot_function(fn, source_root, output):
    root = Path(source_root).resolve()
    source_file = inspect.getsourcefile(fn)
    if source_file is None or not hasattr(fn, "__code__"):
        raise ValueError("Only actual Python functions with available source are supported")
    path = Path(source_file).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Instrumented source outside declared source root")
    relative = path.relative_to(root).as_posix()
    data = safe_path(root, relative).read_bytes()
    # Verify the loaded code object, not merely today's checkout. Compile only;
    # never execute the source snapshot. This MVP accepts undecorated functions.
    from types import CodeType
    def codes(code):
        yield code
        for item in code.co_consts:
            if isinstance(item, CodeType):
                yield from codes(item)
    compiled = compile(data, fn.__code__.co_filename, "exec", dont_inherit=True)
    if not any(code == fn.__code__ for code in codes(compiled)):
        raise ValueError("Loaded function differs from source snapshot")
    lines, first = inspect.getsourcelines(fn)
    last = first + len(lines) - 1
    # inspect reads the live checkout; require agreement with bytes captured now.
    snippet = "".join(data.decode("utf-8").splitlines(keepends=True)[first-1:last])
    if snippet != "".join(lines):
        raise ValueError("Source changed while snapshotting")
    file_hash = digest(data)
    ref = {"path": relative, "start_line": first, "end_line": last,
           "file_sha256": file_hash, "snippet_sha256": digest(snippet.encode()),
           "snapshot": "sources/" + file_hash + ".py"}
    ref["source_id"] = identity(ref)
    dest = safe_path(output, ref["snapshot"])
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        if dest.read_bytes() != data:
            raise ValueError("Existing snapshot mismatch")
    else:
        with dest.open("xb") as fh:
            fh.write(data)
    return ref


def extract_verified(output, ref, current_root=None):
    # Validate both names even when only the snapshot is requested.
    safe_path(output, ref["path"])
    snapshot = safe_path(output, ref["snapshot"])
    if ref["snapshot"] != "sources/" + ref["file_sha256"] + ".py":
        raise ValueError("Snapshot identity mismatch")
    data = snapshot.read_bytes()
    if digest(data) != ref["file_sha256"] or identity(ref) != ref["source_id"]:
        raise ValueError("Source snapshot hash/identity mismatch")
    first, last = ref["start_line"], ref["end_line"]
    lines = data.decode("utf-8").splitlines(keepends=True)
    if type(first) is not int or type(last) is not int or not 1 <= first <= last <= len(lines):
        raise ValueError("Invalid source line range")
    snippet = "".join(lines[first-1:last])
    if digest(snippet.encode()) != ref["snippet_sha256"]:
        raise ValueError("Snippet hash mismatch")
    if current_root is not None:
        if digest(safe_path(current_root, ref["path"]).read_bytes()) != ref["file_sha256"]:
            raise ValueError("Current checkout differs from profiled snapshot")
    return snippet
