"""Single-operator inference roofline estimates, using only user-supplied inputs.

The idealized lower bound is max(FLOPs/effective FLOP/s, DRAM bytes/effective
byte/s), assuming compute and traffic can overlap. Real overlap may be partial;
launch, synchronization and communication costs are excluded. Concurrent operators
share resources, so isolated ceilings need not apply under concurrency. Cache
reuse changes DRAM traffic: logical tensor sizes are not measured DRAM bytes.
Effective ceilings must match precision, operator shape, device and execution
conditions. This is not a whole-request latency model or a speedup prediction.
No hardware specifications or measurements are inferred by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

LIMITATIONS = [
    "Single operator only; not end-to-end request, prefill or decode latency.",
    "The max bound assumes ideal compute/DRAM overlap; partial overlap and launch, synchronization and communication costs are excluded.",
    "Concurrency shares resources; isolated effective ceilings may not apply under contention.",
    "Cache reuse changes DRAM traffic; logical tensor bytes are not measured DRAM bytes.",
    "User-supplied effective ceilings must match precision, shape, device and execution conditions; provenance is not independently verified.",
    "Utilization and headroom are model-relative estimates, not measured hardware utilization or achievable speedup.",
    "Balanced means compute and memory bounds differ by at most 5% of the larger bound.",
]


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _ratio(numerator: float, denominator: float, name: str) -> float:
    return _positive(numerator / denominator, name)


def calculate_roofline(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and estimate; omitted/null ceilings explicitly require measurement.

    Provenance kind is synthetic, source, or measurement, with a description
    identifying the source or measurement method/conditions. It labels the entire
    input set; it does not authenticate claims. Synthetic use requires a separate
    CLI opt-in. Inconsistent observed times yield no headroom estimate.
    """
    if not isinstance(config, dict):
        raise ValueError("input must be a JSON object")
    required = {"scope", "operator", "flops", "dram_bytes", "provenance"}
    allowed = required | {"effective_compute_flops_per_second",
                          "effective_bandwidth_bytes_per_second", "observed_seconds"}
    if required - config.keys():
        raise ValueError(f"missing fields: {sorted(required - config.keys())}")
    if config.keys() - allowed:
        raise ValueError(f"unknown fields: {sorted(config.keys() - allowed)}")
    if config["scope"] != "single_operator":
        raise ValueError("scope must be single_operator")
    if not isinstance(config["operator"], str) or not config["operator"].strip():
        raise ValueError("operator must be a nonempty string")
    provenance = config["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"kind", "description"}:
        raise ValueError("provenance must contain exactly kind and description")
    if provenance["kind"] not in ("synthetic", "source", "measurement"):
        raise ValueError("provenance.kind must be synthetic, source, or measurement")
    if not isinstance(provenance["description"], str) or not provenance["description"].strip():
        raise ValueError("provenance.description must identify source or measurement conditions")

    flops = _positive(config["flops"], "flops")
    dram = _positive(config["dram_bytes"], "dram_bytes")
    optional = {}
    for key in ("effective_compute_flops_per_second", "effective_bandwidth_bytes_per_second", "observed_seconds"):
        value = config.get(key)
        optional[key] = None if value is None else _positive(value, key)
    compute = optional["effective_compute_flops_per_second"]
    bandwidth = optional["effective_bandwidth_bytes_per_second"]
    observed = optional["observed_seconds"]
    compute_bound = None if compute is None else _ratio(flops, compute, "compute_lower_bound_seconds")
    memory_bound = None if bandwidth is None else _ratio(dram, bandwidth, "memory_lower_bound_seconds")
    complete = compute_bound is not None and memory_bound is not None
    bound = None
    classification = "measurement_required"
    if compute_bound is not None and memory_bound is not None:
        bound = max(compute_bound, memory_bound)
        if math.isclose(compute_bound, memory_bound, rel_tol=0.05):
            classification = "balanced"
        else:
            classification = "compute" if compute_bound > memory_bound else "bandwidth"
    warnings = []
    if provenance["kind"] == "synthetic":
        warnings.append("Synthetic illustrative inputs; not hardware specifications or measurements.")
    if not complete:
        warnings.append("Effective compute and bandwidth ceilings are both required for the combined bound and classification.")
    observation = None
    if observed is not None:
        inconsistent = any(b is not None and observed < b for b in (compute_bound, memory_bound))
        observation = {
            "observed_seconds": observed,
            "compute_utilization_fraction": None if compute_bound is None else _ratio(compute_bound, observed, "compute utilization"),
            "bandwidth_utilization_fraction": None if memory_bound is None else _ratio(memory_bound, observed, "bandwidth utilization"),
            "bound_utilization_fraction": None if bound is None else _ratio(bound, observed, "bound utilization"),
            "modeled_headroom_fraction": None,
            "observed_to_bound_ratio": None,
            "status": "assumptions_inconsistent" if inconsistent else ("model_relative_estimate" if complete else "measurement_required"),
        }
        if inconsistent:
            warnings.append("Observed time is below a modeled lower bound: ceiling/traffic assumptions are inconsistent. No real speedup potential can be inferred.")
        elif bound is not None:
            observation["modeled_headroom_fraction"] = 1.0 - bound / observed
            observation["observed_to_bound_ratio"] = _ratio(observed, bound, "observed to bound ratio")
    return {
        "schema_version": 1,
        "scope": "single_operator",
        "operator": config["operator"],
        "provenance": dict(provenance),
        "inputs": {"flops": flops, "dram_bytes": dram, **optional},
        "status": "model_estimate" if complete else "measurement_required",
        "arithmetic_intensity_flops_per_byte": _ratio(flops, dram, "arithmetic intensity"),
        "compute_lower_bound_seconds": compute_bound,
        "memory_lower_bound_seconds": memory_bound,
        "modeled_lower_bound_seconds": bound,
        "classification_model_estimate": classification,
        "observation": observation,
        "warnings": warnings,
        "limitations": list(LIMITATIONS),
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def report_from_json(raw: bytes, *, allow_synthetic: bool = False) -> dict[str, Any]:
    """Parse strict UTF-8 JSON and hash exact input bytes, not normalized JSON."""
    config = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                        parse_constant=_reject_constant)
    report = calculate_roofline(config)
    if report["provenance"]["kind"] == "synthetic" and not allow_synthetic:
        raise ValueError("synthetic inputs require --allow-synthetic")
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    return report
