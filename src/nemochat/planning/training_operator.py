"""Explicit linear-layer forward/backward GEMM estimates, not GPU measurements."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from nemochat.planning.inference_roofline import calculate_roofline


LIMITATIONS = [
    "One bias-free dense linear layer, not attention, a Transformer update or MoE dispatch.",
    "Operand bytes count one read of each GEMM input and one output write; caches, tiling, workspace and accumulation are excluded.",
    "Logical operand traffic is not measured DRAM traffic; its FLOPs/byte is not a measured hardware roofline point.",
    "Backward assumes both input and weight gradients are required. Bias, activation, optimizer and communication are excluded.",
    "Individual operator bounds cannot be added as a training-step bound without critical-path and overlap analysis.",
    "Supplied ceilings and measurements are not authenticated. No GPU performance or achievable speedup is inferred.",
]


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 1_000_000:
        raise ValueError(f"{name} must be an integer in [1, 1000000]")
    return value


def _positive(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def estimate_linear(config: dict[str, Any]) -> dict[str, Any]:
    required = {"schema_version", "batch_tokens", "in_features", "out_features", "dtype", "provenance"}
    optional = {"effective_compute_flops_per_second", "effective_bandwidth_bytes_per_second", "measurements"}
    if not isinstance(config, dict) or not required <= config.keys() or config.keys() - required - optional:
        raise ValueError("missing or unknown linear operator fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    m = _integer(config["batch_tokens"], "batch_tokens")
    k = _integer(config["in_features"], "in_features")
    n = _integer(config["out_features"], "out_features")
    dtype = config["dtype"]
    if dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError("dtype must be float32, bfloat16 or float16")
    size = 4 if dtype == "float32" else 2
    provenance = config["provenance"]
    if (not isinstance(provenance, dict) or set(provenance) != {"kind", "description"}
            or provenance["kind"] not in ("synthetic", "source", "measurement")
            or not isinstance(provenance["description"], str) or not provenance["description"].strip()):
        raise ValueError("provenance needs kind and nonempty description")
    compute = config.get("effective_compute_flops_per_second")
    bandwidth = config.get("effective_bandwidth_bytes_per_second")
    if compute is not None:
        compute = _positive(compute, "effective_compute_flops_per_second")
    if bandwidth is not None:
        bandwidth = _positive(bandwidth, "effective_bandwidth_bytes_per_second")
    measurements = config.get("measurements", {})
    names = {"forward", "input_gradient", "weight_gradient"}
    if not isinstance(measurements, dict) or measurements.keys() - names:
        raise ValueError("measurements may contain forward, input_gradient and weight_gradient only")
    operators = []
    # Y = X @ W; dX = dY @ W.T; dW = X.T @ dY.
    for name, rows, inner, columns in (("forward", m, k, n),
                                       ("input_gradient", m, n, k),
                                       ("weight_gradient", k, m, n)):
        flops = 2 * rows * inner * columns
        logical_bytes = size * (rows * inner + inner * columns + rows * columns)
        record = {
            "name": name, "gemm_mnk": [rows, columns, inner],
            "flops": flops, "logical_operand_bytes": logical_bytes,
            "logical_intensity_flops_per_byte": flops / logical_bytes,
            "roofline": None, "status": "measurement_required",
        }
        if name in measurements:
            measured = measurements[name]
            if not isinstance(measured, dict) or set(measured) != {"dram_bytes", "observed_seconds", "description"}:
                raise ValueError(f"{name} measurement needs dram_bytes, observed_seconds and description")
            if not isinstance(measured["description"], str) or not measured["description"].strip():
                raise ValueError("measurement description must identify profiler/device/precision/scope")
            roofline = calculate_roofline({
                "scope": "single_operator", "operator": f"linear_{name}", "flops": flops,
                "dram_bytes": _positive(measured["dram_bytes"], "dram_bytes"),
                "observed_seconds": _positive(measured["observed_seconds"], "observed_seconds"),
                "effective_compute_flops_per_second": compute,
                "effective_bandwidth_bytes_per_second": bandwidth,
                "provenance": {"kind": provenance["kind"], "description": measured["description"]},
            })
            roofline["limitations"] = list(LIMITATIONS) + roofline["limitations"][1:]
            record["roofline"] = roofline
            record["status"] = roofline["observation"]["status"]
        operators.append(record)
    return {
        "schema_version": 1, "scope": "single_dense_linear_forward_backward",
        "provenance": dict(provenance), "dtype": dtype,
        "operators": operators, "total_gemm_flops": sum(op["flops"] for op in operators),
        "status": "analytical_estimate", "limitations": list(LIMITATIONS),
    }


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def report_from_json(raw: bytes, *, allow_synthetic: bool = False) -> dict[str, Any]:
    if len(raw) > 65536:
        raise ValueError("operator config exceeds 64 KiB")
    config = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_reject)
    report = estimate_linear(config)
    if report["provenance"]["kind"] == "synthetic" and not allow_synthetic:
        raise ValueError("synthetic inputs require --allow-synthetic")
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    return report
