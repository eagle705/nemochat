"""Explicit callable boundaries share identities between Lens, NVTX and source."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import uuid

from .roles import ROLES
from .source import snapshot_function


def requested_nvtx():
    # CPU installation may expose NVTX stubs: test the actual requested path.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("NVTX requested but working CUDA/NVTX is unavailable")
    try:
        from torch.cuda import nvtx
        nvtx.range_push("lens-bridge availability probe")
        nvtx.range_pop()
    except Exception as exc:
        raise RuntimeError("NVTX requested but NVTX backend failed") from exc
    return nvtx


class Bridge:
    """One bridge per process, before other OTel setup; local-only Lens exporter.

    The output is diagnostic host scopes, NOT GPU duration or automatic attribution.
    User code must explicitly import and call this class. No monkeypatching.
    """
    def __init__(self, output, source_root, *, nvtx=False, lens=False):
        self.nvtx = requested_nvtx() if nvtx else None
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.source_root = Path(source_root).resolve()
        self.run_id = uuid.uuid4().hex
        self.handle = None
        self.lens_file = None
        self._closed = False
        self._cleanup_diagnostics = ()
        self.records = (self.output / "operations.jsonl").open("x", buffering=1)
        try:
            if lens:
                from nemo.lens import NemoLensConfig, setup_telemetry
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter
                from opentelemetry import trace
                # Refuse silently coexisting with another process-global exporter.
                if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
                    raise RuntimeError("Lens bridge requires fresh process OTel setup; existing provider refused")
                if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
                    raise RuntimeError("OTEL_SDK_DISABLED prevents requested Lens telemetry")
                self.lens_file = (self.output / "lens-spans.jsonl").open("x", buffering=1)
                exporter = ConsoleSpanExporter(out=self.lens_file,
                    formatter=lambda span: span.to_json(indent=None) + "\n")
                cfg = NemoLensConfig(enabled=True, exporter="console", metrics_enabled=False,
                                     span_groups="", logs_enabled=False, run_id=self.run_id,
                                     service_name="nemo-lens-nsight-educational-control")
                self.handle = setup_telemetry(cfg, span_exporter=exporter,
                    resource_attributes={"nv.dl.rank": 0, "nv.dl.world_size": 1})
        except BaseException:
            self.records.close()
            if self.lens_file:
                self.lens_file.close()
            raise

    def call(self, role, fn, *args, **kwargs):
        title, explanation = ROLES[role]
        ref = snapshot_function(fn, self.source_root, self.output)
        bridge_id = self.run_id + ":" + uuid.uuid4().hex[:16]
        label = f"MoE/{role}: {title} | bridge={bridge_id} src={ref['source_id'][:16]}"
        attrs = {"bridge.id": bridge_id, "bridge.role": role,
                 "bridge.explanation": explanation, "bridge.evidence": "explicit_python_scope",
                 "source.id": ref["source_id"], "source.path": ref["path"],
                 "source.start_line": ref["start_line"], "source.end_line": ref["end_line"],
                 "source.file_sha256": ref["file_sha256"],
                 "source.snippet_sha256": ref["snippet_sha256"], "bridge.nvtx_label": label}
        cm = self.handle.tracer.start_as_current_span(label, attributes=attrs) if self.handle else nullcontext()
        status = "error"
        pushed = False
        nvtx_error = None
        primary_failed = False
        failed = False
        start = time.monotonic_ns()
        try:
            with cm as span:
                if self.handle and not span.is_recording():
                    raise RuntimeError("Requested Lens span is not recording; check sampler/SDK configuration")
                if self.nvtx:
                    try:
                        self.nvtx.range_push(label)
                        pushed = True
                    except BaseException as exc:
                        nvtx_error = {"stage": "push", "type": type(exc).__name__}
                        raise
                try:
                    result = fn(*args, **kwargs)
                    status = "ok"
                    return result
                except BaseException:
                    primary_failed = True
                    raise
                finally:
                    if pushed:
                        try:
                            self.nvtx.range_pop()
                        except BaseException as exc:
                            status = "error"
                            nvtx_error = {"stage": "pop", "type": type(exc).__name__}
                            # Cleanup must not replace the wrapped callable's failure.
                            if not primary_failed:
                                raise
        except BaseException:
            failed = True
            status = "error"
            raise
        finally:
            # Attempt one record even on backend failure. Do not log exception
            # messages (which may contain customer data) or mask a primary error.
            try:
                self.records.write(json.dumps({"bridge_id": bridge_id, "role": role,
                    "title": title, "explanation": explanation, "nvtx_label": label,
                    "nvtx_emitted": pushed, "nvtx_error": nvtx_error,
                    "lens_enabled": self.handle is not None,
                    "evidence": "explicit_python_scope", "status": status,
                    "host_start_monotonic_ns": start, "host_end_monotonic_ns": time.monotonic_ns(),
                    "source": ref}) + "\n")
            except Exception:
                if not failed:
                    raise

    def close(self):
        """Attempt every cleanup once; repeated close is a no-op, even after failure.

        Shutdown precedes stream closure so queued spans can still be delivered.
        A failed cleanup is reported, not retried or treated as successful delivery.
        """
        if self._closed:
            return
        self._closed = True
        failures = []
        for stage, resource, method in (
            ("handle.shutdown", self.handle, "shutdown"),
            ("lens_file.close", self.lens_file, "close"),
            ("records.close", self.records, "close"),
        ):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except BaseException as exc:
                    failures.append((stage, exc))
        if failures:
            first = failures[0][1]
            # Keep trusted diagnostics separate from backend-owned exception notes.
            # Messages may contain customer data; report only stage and type.
            self._cleanup_diagnostics = tuple(
                f"Bridge cleanup failed during {stage}: {type(failure).__name__} (message omitted)"
                for stage, failure in failures
            )
            for note in self._cleanup_diagnostics:
                BaseException.add_note(first, note)
            raise first

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            for note in self._cleanup_diagnostics:
                BaseException.add_note(exc_value, note)
            if not self._cleanup_diagnostics:
                BaseException.add_note(exc_value,
                    f"Bridge cleanup failed: {type(cleanup_error).__name__} (message omitted)")
        return False
