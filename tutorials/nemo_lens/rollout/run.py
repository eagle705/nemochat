"""Executed CPU rollout observability lab: scripted planning, not an RL trainer."""
import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# Explicit source-tree reuse; do not modify the parent tutorial or its environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import clean_environment, local_otlp_session, loopback_endpoint

SERVICE = "nemochat-lens-rollout"
REVISION = "5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be"
STAGES = ("caller_queue", "generation", "tool.client", "reward")


class ToolFailure(RuntimeError):
    """Value-free failure: never record HTTP bodies or tool values in spans."""


def tool_session():
    import requests

    class LoopbackSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.trust_env = False

        def request(self, method, url, **kwargs):
            parsed = urlsplit(url)
            if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
                    or not parsed.port or parsed.path != "/calculate"
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise ValueError("tool endpoint must be credential-free numeric loopback")
            kwargs["allow_redirects"] = False
            response = super().request(method, url, **kwargs)
            if 300 <= response.status_code < 400:
                response.close()
                raise ToolFailure("tool redirect refused")
            return response
    return LoopbackSession()


def start_tool_server(tracer):
    from opentelemetry import context, propagate, trace

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path != "/calculate":
                self.send_error(404)
                return
            # Only the W3C traceparent/tracestate propagator is configured by the runner.
            token = context.attach(propagate.extract(dict(self.headers)))
            try:
                with tracer.start_as_current_span("demo.rl.tool.server", kind=trace.SpanKind.SERVER) as span:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ToolFailure("invalid tool request size")
                    try:
                        body = json.loads(self.rfile.read(length))
                        tid, delay = body["trajectory"], body["delay"]
                        a, b = body["a"], body["b"]
                        if (type(tid) is not int or type(a) is not int or type(b) is not int
                                or not isinstance(delay, (int, float)) or not math.isfinite(delay)
                                or not 0 <= delay <= 5):
                            raise ValueError()
                    except (ValueError, TypeError, KeyError):
                        raise ToolFailure("invalid tool request") from None
                    span.set_attribute("demo.trajectory.id", tid)
                    span.set_attribute("demo.injected_delay_ms", delay * 1000)
                    # The delay is an explicitly injected experimental intervention.
                    time.sleep(delay)
                    if body.get("fail", False):
                        span.set_status(trace.StatusCode.ERROR, "injected tool HTTP failure")
                        code, result = 503, {"error": "injected"}
                    else:
                        code, result = 200, {"result": a * a + b}
                    payload = json.dumps(result).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    if code != 200:
                        try:
                            self.wfile.write(payload)
                        except (BrokenPipeError, ConnectionResetError):
                            span.set_status(trace.StatusCode.ERROR, "client disconnected")
                        return
                # Server execution ends before the successful body becomes visible.
                # The client span owns delivery failures; do not alter ended intervals.
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            finally:
                context.detach(token)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    try:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    except BaseException as exc:
        try:
            server.server_close()
        except BaseException as cleanup:
            exc.add_note("server close failure: " + type(cleanup).__name__)
        raise
    return server, thread


def call_tool(tracer, endpoint, trajectory, delay, timeout, fail=False):
    from opentelemetry import propagate, trace
    with tracer.start_as_current_span("demo.rl.tool.client", kind=trace.SpanKind.CLIENT,
                                     attributes={"demo.trajectory.id": trajectory}) as span:
        headers = {}
        propagate.inject(headers)
        try:
            with tool_session() as session:
                with session.post(endpoint, json={"trajectory": trajectory, "a": trajectory + 2,
                                                  "b": 3, "delay": delay, "fail": fail},
                                  headers=headers, timeout=timeout, stream=True) as response:
                    span.set_attribute("http.response.status_code", response.status_code)
                    if response.status_code != 200:
                        raise ToolFailure("tool HTTP failure")
                    payload = response.raw.read(4097)
                    if len(payload) > 4096:
                        raise ToolFailure("tool response too large")
                    result = json.loads(payload)["result"]
                    if type(result) is not int:
                        raise ToolFailure("invalid tool response")
                    return result
        except Exception:
            # Do not expose request parameters, response bodies or raw exception text.
            raise ToolFailure("tool transport or response failure") from None


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2", revision=REVISION,
                                               local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", revision=REVISION,
                                              local_files_only=True)
    tokens = tokenizer("A short calculation", return_tensors="pt")

    def generate():
        with torch.inference_mode():
            output = model.generate(**tokens, max_new_tokens=8, min_new_tokens=8,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        return int(output.shape[1] - tokens["input_ids"].shape[1])
    generate()  # Untraced warmup; no engine prefill/decode or GPU timing claims.
    return generate


async def rollout_batch(tracer, scope, generate, endpoint, scenario, count, timeout, fail_id=-1):
    semaphore = asyncio.Semaphore(1)  # Caller-owned admission gate, NOT a vLLM queue.

    async def trajectory(index):
        attrs = {"demo.trajectory.id": index}
        with scope("trajectory", "demo.rl.trajectory", **attrs) as span:
            with scope("stage", "demo.rl.caller_queue", **attrs):
                await semaphore.acquire()
            try:
                with scope("stage", "demo.rl.generation", **attrs) as gen_span:
                    generated = await asyncio.to_thread(generate)
                    gen_span.set_attribute("demo.generated_tokens", generated)
            finally:
                semaphore.release()
            delay = 0.25 if scenario == "tool-slow" else (0.75 if scenario == "tool-straggler" and index == count - 1 else 0)
            value = await asyncio.to_thread(call_tool, tracer, endpoint, index, delay, timeout, index == fail_id)
            with scope("stage", "demo.rl.reward", **attrs) as reward_span:
                # Scripted deterministic correctness check, not policy optimization.
                correct = value == (index + 2) ** 2 + 3
                reward_span.set_attribute("demo.reward.correct", correct)
                if not correct:
                    raise ToolFailure("tool calculation incorrect")
            return {"trajectory_id": index, "span_id": format(span.get_span_context().span_id, "016x"),
                    "generated_tokens": generated, "correct": correct}

    # return_exceptions waits for ALL trajectories, preserving error traces before shutdown.
    with scope("batch", "demo.rl.batch", **{"demo.scenario": scenario, "demo.sample_size": count}) as root:
        results = await asyncio.gather(*(trajectory(i) for i in range(count)), return_exceptions=True)
        if any(isinstance(item, BaseException) for item in results):
            from opentelemetry.trace import StatusCode
            root.set_status(StatusCode.ERROR, "trajectory failed")
            raise ToolFailure("rollout batch failed")
        return format(root.get_span_context().trace_id, "032x"), results


def run(args):
    clean_environment()
    # Disable arbitrary ambient propagators (including baggage).
    from opentelemetry import propagate, trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    propagate.set_global_textmap(TraceContextTextMapPropagator())
    from nemo.lens import NemoLensConfig, SpanRegistry, managed_span, setup_telemetry
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    SpanRegistry.register("rollout_lab", {"batch", "trajectory", "stage"},
                          {"default": {"batch", "trajectory", "stage"}})
    stream = exporter = handle = server = thread = None
    primary = None
    cleanup_errors = []

    class Exporter(SpanExporter):
        def __init__(self):
            self.failed = False
            self.persisted = self.exported = 0
            self.remote = self.session = None
            self.closed = False

        def export(self, spans):
            # OTel catches exporter exceptions: failure must be sticky at this seam.
            try:
                for span in spans:
                    stream.write(json.dumps(json.loads(span.to_json())) + "\n")
                stream.flush()
                self.persisted += len(spans)
                result = self.remote.export(spans) if self.remote else SpanExportResult.SUCCESS
                if result == SpanExportResult.SUCCESS:
                    self.exported += len(spans)
                else:
                    self.failed = True
                return result
            except BaseException:
                self.failed = True
                return SpanExportResult.FAILURE

        def shutdown(self):
            if self.closed:
                return
            self.closed = True
            error = None
            for close in ([self.remote.shutdown] if self.remote is not None else []) + ([self.session.close] if self.session is not None else []):
                try:
                    close()
                except BaseException as exc:
                    self.failed = True
                    if error is None:
                        error = exc
                    else:
                        error.add_note("exporter cleanup failure: " + type(exc).__name__)
            if error is not None:
                raise error

    try:
        stream = (args.output / "spans.jsonl").open("x")
        exporter = Exporter()
        if args.mode == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter.session = local_otlp_session()
            exporter.remote = OTLPSpanExporter(endpoint=loopback_endpoint(args.endpoint), timeout=5,
                                               session=exporter.session)
        handle = setup_telemetry(NemoLensConfig(enabled=True, service_name=SERVICE, metrics_enabled=False,
                                              logs_enabled=False, run_id=args.output.name),
                                 resource_attributes={"tutorial.device": "cpu", "demo.workflow": "scripted"},
                                 span_exporter=exporter)
        generate = load_model()
        tracer = trace.get_tracer("nemochat.rollout")
        server, thread = start_tool_server(tracer)
        endpoint = f"http://127.0.0.1:{server.server_port}/calculate"
        trace_id, results = asyncio.run(rollout_batch(tracer, managed_span, generate, endpoint,
                                                    args.scenario, args.trajectories, args.timeout,
                                                    args.fail_id))
    except BaseException as exc:
        primary = exc
    finally:
        # Each close is independent; never replace the exact body exception.
        actions = []
        if server is not None:
            actions.append(server.shutdown)
        if thread is not None:
            actions.append(thread.join)
        if server is not None:
            actions.append(server.server_close)
        if handle is not None:
            actions.append(handle.shutdown)
        if exporter is not None:
            actions.append(exporter.shutdown)
        if stream is not None:
            actions.append(stream.close)
        for close in actions:
            try:
                close()
            except BaseException as exc:
                cleanup_errors.append(exc)
    if primary is None and cleanup_errors:
        primary = cleanup_errors.pop(0)
    expected = 1 + 6 * args.trajectories
    if primary is None and (exporter.failed or exporter.persisted != expected or exporter.exported != expected):
        primary = RuntimeError("incomplete or failed span export")
    if primary is not None:
        for exc in cleanup_errors:
            primary.add_note("cleanup failure: " + type(exc).__name__)
        try:
            with (args.output / "failure.json").open("x") as out:
                json.dump({"status": "failed", "error_type": type(primary).__name__}, out)
        except BaseException as exc:
            primary.add_note("failure record write failed: " + type(exc).__name__)
        raise primary
    result = {"status": "executed", "scenario": args.scenario, "mode": args.mode,
              "service": SERVICE, "run_id": args.output.name, "trace_id": trace_id,
              "expected_spans": 1 + 6 * args.trajectories, "trajectories": results,
              "model": "sshleifer/tiny-gpt2", "revision": REVISION,
              "workflow": "scripted; no RL optimizer, vLLM, NeMo RL or NeMo Gym",
              "ui_url": f"http://127.0.0.1:16687/trace/{trace_id}"}
    with (args.output / "result.json").open("x") as out:
        json.dump(result, out, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("file", "otlp"), default="otlp")
    parser.add_argument("--scenario", choices=("baseline", "tool-slow", "tool-straggler"), default="baseline")
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=2)
    parser.add_argument("--fail-id", type=int, default=-1, help="Inject HTTP 503 at this trajectory (test only)")
    parser.add_argument("--endpoint", type=loopback_endpoint, default="http://127.0.0.1:14318/v1/traces")
    args = parser.parse_args()
    if not 2 <= args.trajectories <= 32 or not math.isfinite(args.timeout) or not 0 < args.timeout <= 30:
        parser.error("require 2..32 trajectories and finite timeout in (0,30]")
    if not -1 <= args.fail_id < args.trajectories:
        parser.error("invalid failure trajectory ID")
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps(run(args)))


if __name__ == "__main__":
    main()
