"""Diagnose actual exported span intervals without adding nested server time twice."""
import argparse
from datetime import datetime
import json
import re
import calendar
from collections import Counter
from pathlib import Path

STAGES = ("caller_queue", "generation", "tool.client", "reward")


def ns(value):
    # OTel ISO timestamps carry nanoseconds; preserve them without float rounding.
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,9}Z", value):
        raise ValueError("require UTC ISO timestamp with 1..9 fractional digits")
    stem, fraction = value[:-1].split(".")
    return calendar.timegm(datetime.fromisoformat(stem).timetuple()) * 10**9 + int(fraction.ljust(9, "0"))


def identity(value, width):
    if not isinstance(value, str):
        raise ValueError("invalid hexadecimal identity")
    value = value.removeprefix("0x")
    if not re.fullmatch(r"[0-9a-f]{%d}" % width, value) or int(value, 16) == 0:
        raise ValueError("invalid hexadecimal identity")
    return value


def span_id(span):
    return identity(span["context"]["span_id"], 16)


def analyze(spans):
    roots = [s for s in spans if s["name"] == "demo.rl.batch"]
    if len(roots) != 1:
        raise ValueError("require exactly one batch")
    root = roots[0]
    if len({s["context"]["trace_id"] for s in spans}) != 1:
        raise ValueError("mixed trace identities")
    ids = [span_id(s) for s in spans]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate span identity")
    for s in spans:
        identity(s["context"]["trace_id"], 32)
        if s["status"]["status_code"] not in ("UNSET", "OK"):
            raise ValueError("non-success span status")
        if ns(s["end_time"]) < ns(s["start_time"]):
            raise ValueError("negative span interval")
    count = root["attributes"]["demo.sample_size"]
    if type(count) is not int or not 2 <= count <= 32 or root.get("parent_id"):
        raise ValueError("invalid batch sample or parent")
    expected = {"demo.rl.batch": 1, **{"demo.rl." + k: count for k in
                ("trajectory", *STAGES, "tool.server")}}
    if Counter(s["name"] for s in spans) != expected:
        raise ValueError("incomplete operation coverage")
    trajectories = [s for s in spans if s["name"] == "demo.rl.trajectory"]
    tids = [s["attributes"].get("demo.trajectory.id") for s in trajectories]
    if any(type(t) is not int for t in tids) or set(tids) != set(range(count)):
        raise ValueError("invalid or duplicate trajectory identities")
    for t in trajectories:
        if (identity(t["parent_id"], 16) != span_id(root)
                or ns(t["start_time"]) < ns(root["start_time"])
                or ns(t["end_time"]) > ns(root["end_time"])):
            raise ValueError("trajectory outside batch or incorrect batch parent")
    rows = []
    for trajectory in [s for s in spans if s["name"] == "demo.rl.trajectory"]:
        children = [s for s in spans if (s.get("parent_id") or "").removeprefix("0x") == span_id(trajectory)]
        stages = {}
        for stage in STAGES:
            selected = [s for s in children if s["name"] == "demo.rl." + stage]
            if len(selected) != 1:
                raise ValueError("missing or duplicate sequential stage; cannot diagnose failed run")
            child_tid = selected[0]["attributes"].get("demo.trajectory.id")
            if type(child_tid) is not int or child_tid != trajectory["attributes"]["demo.trajectory.id"]:
                raise ValueError("stage trajectory identity differs")
            stages[stage] = selected[0]
        intervals = [(ns(stages[k]["start_time"]), ns(stages[k]["end_time"])) for k in STAGES]
        if any(end > intervals[i + 1][0] for i, (_, end) in enumerate(intervals[:-1])):
            raise ValueError("stage overlap: exclusive sequential assumption violated")
        start, end = ns(trajectory["start_time"]), ns(trajectory["end_time"])
        if any(a < start or b > end or b < a for a, b in intervals):
            raise ValueError("stage interval outside trajectory")
        durations = {k: (b - a) / 1e6 for k, (a, b) in zip(STAGES, intervals)}
        wall = (end - start) / 1e6
        servers = [s for s in spans if s["name"] == "demo.rl.tool.server"
                   and (s.get("parent_id") or "").removeprefix("0x") == span_id(stages["tool.client"])]
        if len(servers) != 1:
            raise ValueError("missing or duplicate nested tool server")
        server = servers[0]
        a, b = ns(server["start_time"]), ns(server["end_time"])
        client_a, client_b = intervals[2]
        server_tid = server["attributes"].get("demo.trajectory.id")
        if (type(server_tid) is not int or server_tid != trajectory["attributes"]["demo.trajectory.id"]
                or a < client_a or b > client_b or b < a
                or server["kind"] != "SpanKind.SERVER" or stages["tool.client"]["kind"] != "SpanKind.CLIENT"):
            raise ValueError("invalid server identity, kind or contained interval")
        server_ms = (b - a) / 1e6
        rows.append({"trajectory_id": trajectory["attributes"]["demo.trajectory.id"],
                     "nested_server_ms_excluded_from_totals": server_ms,
                     "injected_tool_delay_ms": server["attributes"]["demo.injected_delay_ms"],
                     "span_id": span_id(trajectory), "wall_ms": wall, "stages_ms": durations,
                     "unknown_gap_ms": wall - sum(durations.values()),
                     "end_ns": end, "intervals": intervals})
    if len(rows) < 2:
        raise ValueError("require at least two trajectories for tail diagnosis")
    ordered = sorted(rows, key=lambda row: row["end_ns"])
    last, previous = ordered[-1], ordered[-2]
    tail = {stage: max(0, b - max(a, previous["end_ns"])) / 1e6
            for stage, (a, b) in zip(STAGES, last["intervals"])}
    totals = {k: sum(row["stages_ms"][k] for row in rows) for k in STAGES}
    report = {"sample_size": len(rows), "batch_wall_ms": (ns(root["end_time"]) - ns(root["start_time"])) / 1e6,
              "sum_trajectory_wall_ms": sum(r["wall_ms"] for r in rows),
              "exclusive_stage_totals_ms": totals,
              "unknown_gap_total_ms": sum(r["unknown_gap_ms"] for r in rows),
              "last_finisher_id": last["trajectory_id"],
              "last_finisher_dominant_stage_candidate": max(last["stages_ms"], key=last["stages_ms"].get),
              "tail_after_penultimate_ms": (last["end_ns"] - previous["end_ns"]) / 1e6,
              "tail_stage_overlap_ms": tail,
              "tail_stage_candidate": max(tail, key=tail.get),
              "interpretation": "observed candidates, not guaranteed causal production bottlenecks; server nested in client excluded; concurrent sums are not wall time",
              "trajectories": [{k: v for k, v in row.items() if k not in ("end_ns", "intervals")} for row in rows]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spans = [json.loads(line) for line in args.spans.read_text().splitlines()]
    if any(s["status"]["status_code"] == "ERROR" for s in spans):
        raise ValueError("error spans present: not a successful measurement")
    report = analyze(spans)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
