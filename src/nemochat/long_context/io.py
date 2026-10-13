"""Small atomic writers. A cursor/output path must have a single writer."""
from contextlib import AbstractContextManager
import csv
import io
import json
import os
from pathlib import Path
import tempfile


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode("utf-8") if isinstance(content, str) else content
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FreshOutputs(AbstractContextManager):
    """Reserve fresh NIAH destinations before expensive IO; publish without replace.

    Exclusive sidecars coordinate cooperating writers. Hard-link publication also
    refuses a destination created by a non-cooperating writer during computation.
    Each attempt requires unique destinations: reservation sidecars persist on
    success and failure and are never automatically reclaimed. Published artifacts
    persist even after later publication failure; bundles are not all-or-nothing.
    Private staging names must not be modified by other writers. Cleanup attempts
    every private unlink and FD close, but failures or process death can leave
    staging files. Use a new attempt path rather than blindly reclaiming evidence.
    """

    def __init__(self, paths, inputs=()):
        self.paths = [Path(p).absolute() for p in paths]
        self.inputs = [Path(p).resolve() for p in inputs]
        self.staged = {}
        self.locks = {}
        self.published = []
        self.written = set()

    @staticmethod
    def _overlap(a, b):
        return a == b or a in b.parents or b in a.parents

    def __enter__(self):
        resolved = [p.resolve() for p in self.paths]
        locks = [p.with_name(f".{p.name}.niah.lock") for p in self.paths]
        all_outputs = resolved + [p.resolve() for p in locks]
        for i, path in enumerate(all_outputs):
            if any(self._overlap(path, other) for other in all_outputs[:i] + self.inputs):
                raise ValueError("input/output or artifact paths collide")
        for path in self.paths + locks:
            if os.path.lexists(path):
                raise FileExistsError(f"output or reservation already exists: {path}")
        try:
            for path, lock in zip(self.paths, locks):
                path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                self.locks[lock] = fd
                # Probe staging permissions and reserve storage before model/client IO.
                fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
                self.staged[path] = (fd, Path(temporary))
                if os.path.lexists(path):
                    raise FileExistsError(f"output already exists: {path}")
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def write(self, path, content):
        path = Path(path).absolute()
        if path in self.written:
            raise ValueError("artifact already written")
        fd, _ = self.staged[path]
        data = content.encode("utf-8") if isinstance(content, str) else content
        with os.fdopen(os.dup(fd), "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self.written.add(path)

    def write_json(self, path, record):
        self.write(path, json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def write_jsonl(self, path, records):
        self.write(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records))

    def write_csv(self, path, rows, fields):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        self.write(path, stream.getvalue())

    def __exit__(self, exc_type, exc_value, traceback):
        primary = exc_value
        try:
            if exc_type is None:
                if self.written != set(self.paths):
                    raise ValueError("not all reserved artifacts were written")
                for path in self.paths:
                    os.link(self.staged[path][1], path)  # Atomic no-clobber publication.
                    self.published.append(path)
                # Never roll back public paths: a check followed by unlink can
                # delete a non-cooperating writer's replacement.
        except BaseException as error:
            primary = error
            raise
        finally:
            failures = []
            for fd, temporary in self.staged.values():
                try:
                    os.close(fd)
                except BaseException as error:
                    failures.append(("staging-close", type(error).__name__))
                try:
                    temporary.unlink(missing_ok=True)
                except BaseException as error:
                    failures.append(("staging-unlink", type(error).__name__))
            # Reservation paths are permanent attempt tombstones, not removable
            # locks. Closing their FDs must not unlink a replacement sidecar.
            for fd in self.locks.values():
                try:
                    os.close(fd)
                except BaseException as error:
                    failures.append(("reservation-close", type(error).__name__))
            if failures:
                diagnostic = primary if primary is not None else RuntimeError("NIAH cleanup failed")
                for stage, error_type in failures:
                    diagnostic.add_note(f"NIAH cleanup {stage}: {error_type}")
                if primary is None:
                    raise diagnostic from None
        return False


def write_json(path, record):
    atomic_write(path, json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path, records):
    atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records))


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, stream.getvalue())
