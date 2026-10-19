"""Strict task-local wrapper attestation plus unchanged frozen SDK guard."""
import inspect
from pathlib import Path
import threading

_LOCK = threading.RLock()


def attach(parent, entry, wrapper, original, control):
    original_guard = parent.guard_native_sources
    wrapper_code = wrapper.__code__
    original_code = original.__code__
    def guarded():
        with _LOCK:
            control.require(entry.pretrain is wrapper and wrapper.__code__ is wrapper_code,
                            'unknown/changed local pretrain wrapper')
            control.require(wrapper.__wrapped__ is original and original.__code__ is original_code,
                            'underlying SDK pretrain identity changed')
            freeze_path = control.OUT/'source-freeze.json'
            frozen = control.read(freeze_path)
            local_sources = []
            for function, expected in ((wrapper, control.HERE/'native.py'), (attach, control.HERE/'source_guard.py')):
                path = Path(inspect.getsourcefile(function)).resolve()
                control.require(path == expected.resolve(), 'unexpected local wrapper source path')
                relative = str(path.relative_to(control.ROOT))
                digest = control.sha(path)
                control.require(frozen.get(relative) == digest, 'frozen local wrapper source changed: '+relative)
                local_sources.append(dict(path=str(path), sha256=digest))
            # Original runner traverses the SDK decorator chains and checks all
            # four pinned SDK hashes. Never skip it or substitute a fake receipt.
            entry.pretrain = original
            try:
                receipt = original_guard()
            finally:
                entry.pretrain = wrapper
            control.require(bool(receipt.get('decorated_callable_sources')), 'SDK source guard returned no sources')
            return dict(receipt, frozen_local_wrapper_sources=local_sources,
                        runtime_source_freeze_sha256=control.sha(freeze_path),
                        local_wrapper_identity_verified=True,
                        unchanged_sdk_guard_invoked=True)
    parent.guard_native_sources = guarded
    return guarded
