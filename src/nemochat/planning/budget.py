"""Time/budget arithmetic from user measurements, never hardware-name forecasts."""
from __future__ import annotations

import math
import statistics

from .corpus import positive_int


def number(value, name, minimum=0, strict=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"invalid {name}: {value}")
    return value


def stage_measurement(stage, gbs):
    length = positive_int(stage.get("sequence_length"), "sequence_length")
    tokens = stage.get("tokens_per_update", gbs * length)
    number(tokens, "tokens_per_update", strict=True)
    if tokens > gbs * length:
        raise ValueError("tokens_per_update exceeds GBS * sequence_length")
    seconds = stage.get("step_seconds")
    throughput = stage.get("tokens_per_second")
    if seconds is not None and throughput is not None:
        raise ValueError("choose step_seconds OR tokens_per_second, not both")
    rate = None
    if seconds is not None:
        if "tokens_per_update" not in stage:
            raise ValueError("step timing requires measured tokens_per_update")
        if not isinstance(seconds, list) or not seconds:
            raise ValueError("step_seconds must be a nonempty measured list")
        for s in seconds:
            number(s, "step_seconds", strict=True)
        rate = tokens / statistics.mean(seconds)
    elif throughput is not None:
        rate = number(throughput, "tokens_per_second", strict=True)
    if rate is not None and not isinstance(stage.get("measurement_provenance"), str):
        raise ValueError("measured throughput requires measurement_provenance")
    scope = stage.get("timing_scope", "effective_wall")
    if scope not in {"effective_wall", "core_compute"}:
        raise ValueError("timing_scope must be effective_wall or core_compute")
    utilization = number(stage.get("utilization", 1), "utilization", strict=True)
    if utilization > 1:
        raise ValueError("utilization must be <= 1")
    checkpoint = number(stage.get("checkpoint_seconds", 0), "checkpoint_seconds")
    evaluation = number(stage.get("evaluation_seconds", 0), "evaluation_seconds")
    if scope == "effective_wall" and (utilization != 1 or checkpoint or evaluation):
        raise ValueError("effective_wall already includes utilization/checkpoint/eval; do not double count")
    if scope == "core_compute" and "utilization" not in stage:
        raise ValueError("core_compute requires explicit utilization for non-checkpoint/eval idle time")
    for duration, key in ((checkpoint, "checkpoint_every_updates"), (evaluation, "eval_every_updates")):
        if duration or key in stage:
            positive_int(stage.get(key), key)
    return tokens, rate * utilization if rate is not None else None


def stage_seconds(stage, updates, tokens_per_update, rate):
    compute = updates * tokens_per_update / rate
    # Intervals include an event at the last update if exactly divisible.
    checkpoint_count = updates // stage.get("checkpoint_every_updates", updates + 1)
    eval_count = updates // stage.get("eval_every_updates", updates + 1)
    overhead = checkpoint_count * stage.get("checkpoint_seconds", 0) + eval_count * stage.get("evaluation_seconds", 0)
    return compute + overhead, checkpoint_count, eval_count, overhead


def capacity_updates(stage, maximum, available, tokens_per_update, rate):
    lo, hi = 0, maximum
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if stage_seconds(stage, mid, tokens_per_update, rate)[0] <= available:
            lo = mid
        else:
            hi = mid - 1
    return lo


def plan_training(config):
    """Report three sensitivity scenarios around supplied stage-specific measurements.

    Elapsed time/cost are sunk. Stage completed_tokens are user-supplied progress.
    Scheduled intervals restart at the remaining segment, not a global resume
    cursor. Final partial updates are rounded up and reported as extra capacity.
    """
    dp = positive_int(config.get("data_parallel"), "data_parallel")
    mbs = positive_int(config.get("micro_batch_size"), "micro_batch_size")
    gas = positive_int(config.get("gradient_accumulation_steps"), "gradient_accumulation_steps")
    gbs = dp * mbs * gas
    if "global_batch_size" in config and config["global_batch_size"] != gbs:
        raise ValueError("global_batch_size must equal DP * MBS * GAS")
    nodes = positive_int(config.get("nodes"), "nodes")
    gpu_per_node = positive_int(config.get("gpus_per_node"), "gpus_per_node")
    price = number(config.get("price_per_hour"), "price_per_hour")
    unit = config.get("price_unit")
    if unit not in {"node", "gpu"}:
        raise ValueError("price_unit must be node or gpu")
    if dp > nodes * gpu_per_node:
        raise ValueError("DP replicas exceed total allocated GPUs")
    hourly = price * nodes * (gpu_per_node if unit == "gpu" else 1)
    elapsed = number(config.get("elapsed_seconds", 0), "elapsed_seconds")
    setup = number(config.get("remaining_setup_seconds", 0), "remaining_setup_seconds")
    time_limit = config.get("time_limit_seconds")
    budget_limit = config.get("budget_limit")
    if time_limit is not None:
        number(time_limit, "time_limit_seconds")
    if budget_limit is not None:
        number(budget_limit, "budget_limit")
    spent = number(config.get("spent_cost", elapsed / 3600 * hourly), "spent_cost")
    remaining_time = max(0, time_limit - elapsed) if time_limit is not None else None
    remaining_budget = max(0, budget_limit - spent) if budget_limit is not None else None
    available = math.inf
    if remaining_time is not None:
        available = min(available, remaining_time)
    if remaining_budget is not None and hourly > 0:
        available = min(available, remaining_budget / hourly * 3600)
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("nonempty stages required")
    parsed, names = [], set()
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("name"), str) or not stage["name"] or stage["name"] in names:
            raise ValueError("unique nonempty stage names required")
        names.add(stage["name"])
        target = positive_int(stage.get("target_tokens"), "target_tokens")
        completed = stage.get("completed_tokens", 0)
        if type(completed) is not int or not 0 <= completed <= target:
            raise ValueError("completed_tokens must be integer within target")
        tokens, rate = stage_measurement(stage, gbs)
        remaining = target - completed
        parsed.append((stage, remaining, tokens, rate))
    multipliers = config.get("throughput_multipliers", {"conservative": .8, "base": 1., "optimistic": 1.2})
    if not isinstance(multipliers, dict) or set(multipliers) != {"conservative", "base", "optimistic"}:
        raise ValueError("three scenario multipliers required")
    for value in multipliers.values():
        number(value, "throughput_multiplier", strict=True)
    if not (multipliers["conservative"] <= multipliers["base"] == 1 <= multipliers["optimistic"]):
        raise ValueError("multipliers must be conservative <= base=1 <= optimistic")
    scenarios = {}
    unknown = any(rate is None and remaining > 0 for _, remaining, _, rate in parsed)
    for scenario, multiplier in multipliers.items():
        if unknown:
            scenarios[scenario] = {"status": "measurement_required", "remaining_seconds": None,
                                   "remaining_cost": None, "capacity_tokens": None}
            continue
        total_seconds, available_seconds, capacity = setup, max(0, available - setup), 0
        stage_reports = []
        for stage, remaining, tokens, measured_rate in parsed:
            rate = (measured_rate or 1) * multiplier
            updates = ((remaining + tokens - 1) // tokens if type(tokens) is int
                       else math.ceil(remaining / tokens))
            wall, ckpts, evals, overhead = stage_seconds(stage, updates, tokens, rate)
            fit = capacity_updates(stage, updates, available_seconds, tokens, rate)
            consumed = stage_seconds(stage, fit, tokens, rate)[0]
            available_seconds = max(0, available_seconds - consumed)
            fit_tokens = min(remaining, fit * tokens)
            capacity += fit_tokens
            total_seconds += wall
            stage_reports.append({"name": stage["name"], "sequence_length": stage["sequence_length"],
                                  "measurement_provenance": stage.get("measurement_provenance"),
                                  "timing_scope": stage.get("timing_scope", "effective_wall"),
                                  "remaining_tokens": remaining, "tokens_per_update": tokens,
                                  "token_accounting": "user measured" if "tokens_per_update" in stage else "GBS * length (fully packed assumption)",
                                  "effective_tokens_per_second": rate, "updates": updates,
                                  "rounded_update_tokens": updates * tokens,
                                  "remaining_seconds": wall, "explicit_overhead_seconds": overhead,
                                  "checkpoint_events": ckpts, "eval_events": evals,
                                  "capacity_tokens": fit_tokens})
            # Capacity is sequential: later stages cannot run after an incomplete one.
            if fit < updates:
                available_seconds = 0
        storage = None
        if "storage" in config:
            spec = config["storage"]
            checkpoint_bytes = number(spec.get("checkpoint_bytes"), "checkpoint_bytes")
            retain = positive_int(spec.get("retain_checkpoints"), "retain_checkpoints")
            existing = positive_int(spec.get("existing_checkpoints", 1), "existing_checkpoints")
            eval_bytes = number(spec.get("eval_artifact_bytes", 0), "eval_artifact_bytes")
            eval_retain = positive_int(spec.get("retain_eval_artifacts", 1), "retain_eval_artifacts")
            events = sum(s["checkpoint_events"] for s in stage_reports)
            eval_events = sum(s["eval_events"] for s in stage_reports)
            storage = {"retained_bytes": min(retain, existing + events) * checkpoint_bytes + min(eval_retain, eval_events) * eval_bytes,
                       "peak_bytes_upper_bound": max(existing + (1 if events else 0), min(retain, existing + events) + (1 if events else 0)) * checkpoint_bytes + min(eval_retain, eval_events) * eval_bytes,
                       "scope": "initial checkpoint footprint included; prune after completed writes to retention; one transient checkpoint; excludes dataset, caches and replicas"}
        cost = total_seconds / 3600 * hourly
        scenarios[scenario] = {"status": "measured_input_projection", "throughput_multiplier": multiplier,
                               "remaining_seconds": total_seconds, "remaining_cost": cost,
                               "projected_total_seconds": elapsed + total_seconds, "projected_total_cost": spent + cost,
                               "capacity_tokens": capacity, "fits_limits": ((time_limit is None or elapsed + total_seconds <= time_limit)
                                      and (budget_limit is None or spent + cost <= budget_limit)),
                               "time_slack_seconds": time_limit - elapsed - total_seconds if time_limit is not None else None,
                               "budget_slack": budget_limit - spent - cost if budget_limit is not None else None,
                               "stages": stage_reports, "storage": storage}
    return {"schema": "nemochat-training-plan-v1", "hardware_label": config.get("hardware_label"),
            "global_batch_size": gbs, "formula": "DP * MBS * GAS", "hourly_cluster_price": hourly,
            "price_unit": unit, "elapsed_seconds": elapsed, "spent_cost": spent,
            "remaining_time_seconds": remaining_time, "remaining_budget": remaining_budget,
            "remaining_setup_seconds": setup, "scenarios": scenarios,
            "limitations": ["User measurements are not independently verified; hardware labels do not predict throughput.",
                            "Sensitivity scenarios are assumptions, not confidence intervals.",
                            "Setup counted once; effective-wall measurements forbid separate utilization/eval/checkpoint overhead.",
                            "Interval counters restart for remaining work; pass effective-wall timing for exact-resume overhead.",
                            "Billing assumes all specified nodes/GPUs are allocated during all remaining wall time."]}
