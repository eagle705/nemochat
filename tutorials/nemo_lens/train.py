"""Offline CPU optimizer updates instrumented with the real pinned Lens SDK."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

MODES = ("baseline", "disabled", "file", "otlp")
SERVICE = "nemochat-lens-cpu"


def clean_environment():
    """Remove ambient telemetry destinations, identity and credentials before import."""
    for key in list(os.environ):
        if key.startswith(("OTEL_", "NEMO_LENS_", "WANDB_", "SLURM_", "KUBERNETES_")) or key in (
            "ENABLE_SINGLE_TRACEID", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "DEPLOYMENT_ENV", "ENVIRONMENT",
        ):
            os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] = "65536"
    os.environ["OTEL_BSP_SCHEDULE_DELAY"] = "1000"


def loopback_endpoint(value):
    url = urlsplit(value)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or url.username or url.password or url.query or url.fragment
            or url.path != "/v1/traces" or not url.port):
        raise ValueError("OTLP endpoint must be credential-free loopback HTTP /v1/traces")
    return value


def local_otlp_session():
    """Do not consult netrc/proxies or forward trace bodies through redirects."""
    # Lazy import keeps the baseline arm free of exporter dependencies.
    import requests

    class LocalOTLPSession(requests.Session):
        def __init__(self):
            super().__init__()
            self.trust_env = False

        def request(self, method, url, **kwargs):
            loopback_endpoint(url)
            kwargs["allow_redirects"] = False
            response = super().request(method, url, **kwargs)
            # Requests treats 3xx as .ok; OTel 1.43.0 would count that as success.
            # HTTPError is non-retryable in its exporter: no second POST is sent.
            if 300 <= response.status_code < 400:
                response.close()
                raise requests.HTTPError("OTLP redirects are not allowed", response=response)
            return response

    return LocalOTLPSession()


def run(mode, steps, warmup, output, endpoint):
    clean_environment()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 16))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x, y = torch.randn(32, 64), torch.randn(32, 16)
    handle, stream, exporter = None, None, None
    setup_start = time.perf_counter_ns()
    if mode != "baseline":
        from nemo.lens import NemoLensConfig, SpanRegistry, managed_span, setup_telemetry
        SpanRegistry.register("cpu_tutorial", {"run", "step", "phase"}, {"default": {"run", "step", "phase"}})
        if mode == "file":
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            stream = (output / "spans.jsonl").open("x")
            exporter = ConsoleSpanExporter(out=stream, formatter=lambda span: json.dumps(json.loads(span.to_json())) + "\n")
        elif mode == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=loopback_endpoint(endpoint), timeout=5,
                                        session=local_otlp_session())
        handle = setup_telemetry(
            NemoLensConfig(enabled=mode != "disabled", service_name=SERVICE, metrics_enabled=False,
                           logs_enabled=False, run_id=output.name),
            resource_attributes={"nv.dl.rank": 0, "nv.dl.world_size": 1, "tutorial.device": "cpu"},
            span_exporter=exporter,
        )
        def scope(group, name, **attrs):
            return managed_span(group, name, **attrs)
    setup_ns = time.perf_counter_ns() - setup_start

    def update(index, instrument):
        # Baseline does NOT call the Lens wrapper or even a nullcontext.
        if not instrument:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()
            return loss
        with scope("step", "train.step", **{"train.step": index}):
            optimizer.zero_grad(set_to_none=True)
            with scope("phase", "train.forward"):
                loss = torch.nn.functional.mse_loss(model(x), y)
            with scope("phase", "train.backward"):
                loss.backward()
            with scope("phase", "train.optimizer"):
                optimizer.step()
        return loss

    # Same warmup optimizer updates in every arm; outside timing and tracing.
    for index in range(warmup):
        update(index, False)
    losses = []
    trace_id = None
    start = time.perf_counter_ns()
    if mode == "baseline":
        for index in range(steps):
            losses.append(update(index, False).detach())
    else:
        with scope("run", "train.run", **{"tutorial.steps": steps}) as root:
            if root is not None and root.is_recording():
                trace_id = format(root.get_span_context().trace_id, "032x")
            for index in range(steps):
                losses.append(update(index, True).detach())
    training_ns = time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    if handle:
        handle.shutdown()
    if stream:
        stream.close()
    shutdown_ns = time.perf_counter_ns() - start
    digest = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    loss_values = [float(loss) for loss in losses]
    result = dict(mode=mode, steps=steps, warmup=warmup, training_ns=training_ns,
                  setup_ns=setup_ns, shutdown_ns=shutdown_ns, parameter_sha256=digest,
                  losses=loss_values, torch_version=torch.__version__, device="cpu",
                  expected_spans=1 + 4 * steps if mode in ("file", "otlp") else 0,
                  service=SERVICE, run_id=output.name, trace_id=trace_id)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="otlp")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", type=loopback_endpoint, default="http://127.0.0.1:14318/v1/traces")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps must be positive; warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    result = run(args.mode, args.steps, args.warmup, args.output, args.endpoint)
    print(json.dumps({key: val for key, val in result.items() if key != "losses"}))


if __name__ == "__main__":
    main()
