"""Validate this lab's Jaeger identities, operation counts and remote hierarchy."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_jaeger import get_json


def attribute_matches(tag, value):
    """Compare Jaeger's semantic type, not JSON's representation of doubles."""
    semantic = {bool: "bool", int: "int64", float: "float64", str: "string"}.get(type(value))
    if semantic is None or tag.get("type") != semantic:
        return False
    remote = tag["value"]
    if type(value) is float:
        return (type(remote) in (int, float) and math.isfinite(remote)
                and math.isfinite(value) and remote == value)
    return type(remote) is type(value) and remote == value


def validate(trace, result, local):
    from diagnose import analyze, identity, span_id, ns
    from run import SERVICE, REVISION

    rows = result["trajectories"]
    count = len(rows)
    if (result["status"] != "executed" or not 2 <= count <= 32
            or result["mode"] not in ("file", "otlp")
            or result["scenario"] not in ("baseline", "tool-slow", "tool-straggler")
            or result["service"] != SERVICE or not isinstance(result["run_id"], str) or not result["run_id"]
            or result["model"] != "sshleifer/tiny-gpt2" or result["revision"] != REVISION
            or type(result["expected_spans"]) is not int or result["expected_spans"] != 1 + 6 * count):
        raise ValueError("invalid successful execution result")
    if identity(result["trace_id"], 32) != result["trace_id"]:
        raise ValueError("noncanonical result trace identity")
    tids = [r["trajectory_id"] for r in rows]
    if any(type(t) is not int for t in tids) or set(tids) != set(range(count)):
        raise ValueError("invalid result trajectory identities")
    result_ids = [identity(r["span_id"], 16) for r in rows]
    if len(set(result_ids)) != count or any(r["span_id"] != sid for r, sid in zip(rows, result_ids)):
        raise ValueError("invalid result span identities")
    if any(r["correct"] is not True or type(r["generated_tokens"]) is not int or r["generated_tokens"] != 8 for r in rows):
        raise ValueError("failed reward or incomplete generation")
    analyze(local)  # Unique identities, complete per-trajectory chains and contained intervals.
    declared = dict(zip(tids, result_ids))
    expected = {"demo.rl.batch": 1, **{f"demo.rl.{name}": len(result["trajectories"])
                for name in ("trajectory", "caller_queue", "generation", "tool.client", "tool.server", "reward")}}
    spans = trace["spans"]
    if trace["traceID"] != result["trace_id"] or Counter(s["operationName"] for s in spans) != expected:
        raise ValueError("trace identity or operation counts differ")
    by_id = {identity(s["spanID"], 16): s for s in spans}
    local_by_id = {span_id(s): s for s in local}
    if len(by_id) != len(spans) or len(local_by_id) != len(local) or set(by_id) != set(local_by_id):
        raise ValueError("local and Jaeger span identities differ")
    remote_intervals = {}
    for sid, span in by_id.items():
        start, duration = span["startTime"], span["duration"]
        if type(start) is not int or type(duration) is not int or start < 0 or duration < 0:
            raise ValueError("invalid remote timestamp interval")
        remote_start, remote_duration = start * 1000, duration * 1000
        remote_end = remote_start + remote_duration
        local_span = local_by_id[sid]
        local_start, local_end = ns(local_span["start_time"]), ns(local_span["end_time"])
        # Pinned OTel SDK ns_to_iso_str uses datetime.fromtimestamp(ns / 1e9)
        # and %f (six-digit rounded microseconds). Jaeger floors start and
        # duration separately to microseconds. Integer bounds cover that exact
        # quantization: <=1us start, <=2us duration/end; no relative epsilon.
        if (abs(remote_start - local_start) > 1000
                or abs(remote_duration - (local_end - local_start)) > 2000
                or abs(remote_end - local_end) > 2000):
            raise ValueError("local/remote timestamps differ")
        remote_intervals[sid] = (remote_start, remote_end)
    for sid, span in by_id.items():
        local_span = local_by_id[sid]
        if (span["traceID"] != result["trace_id"]
                or identity(local_span["context"]["trace_id"], 32) != result["trace_id"]
                or span["operationName"] != local_span["name"]):
            raise ValueError("local trace identity differs")
        process = trace["processes"][span["processID"]]
        resource = {tag["key"]: tag["value"] for tag in process["tags"]}
        local_resource = local_span["resource"]["attributes"]
        required = {"nemo.run.id": result["run_id"], "tutorial.device": "cpu", "demo.workflow": "scripted"}
        if (process["serviceName"] != result["service"] or local_resource.get("service.name") != result["service"]
                or any(resource.get(k) != v or local_resource.get(k) != v for k, v in required.items())):
            raise ValueError("incorrect resource identity")
        tags = {t["key"]: t["value"] for t in span["tags"]}
        status = local_span["status"]["status_code"]
        if (tags.get("error", False) is not False or status not in ("UNSET", "OK")
                or tags.get("otel.status_code", "UNSET") != status):
            raise ValueError("error or mismatched status cannot receive success receipt")
        attrs = local_span["attributes"]
        if len(tags) != len(span["tags"]) or len(resource) != len(process["tags"]):
            raise ValueError("duplicate remote tags")
        typed_tags = {t["key"]: t for t in span["tags"]}
        if any(k not in typed_tags or not attribute_matches(typed_tags[k], v) for k, v in attrs.items()):
            raise ValueError("local/remote attributes differ")
        parents = [ref for ref in span["references"] if ref["refType"] == "CHILD_OF"]
        parent_id = (local_span.get("parent_id") or "").removeprefix("0x")
        if span["operationName"] == "demo.rl.batch":
            if (parents or parent_id or attrs["demo.sample_size"] != count
                    or attrs["demo.scenario"] != result["scenario"]):
                raise ValueError("batch must be root")
            continue
        tid = attrs.get("demo.trajectory.id")
        if type(tid) is not int or tid not in declared or tags.get("demo.trajectory.id") != tid:
            raise ValueError("missing trajectory identity")
        if span["operationName"] == "demo.rl.trajectory" and declared[tid] != sid:
            raise ValueError("result trajectory span differs")
        if span["operationName"] == "demo.rl.generation" and (type(attrs.get("demo.generated_tokens")) is not int or attrs["demo.generated_tokens"] != 8):
            raise ValueError("local generated token count differs")
        if span["operationName"] == "demo.rl.reward" and attrs.get("demo.reward.correct") is not True:
            raise ValueError("local reward failed")
        if (len(parents) != 1 or parents[0]["traceID"] != result["trace_id"]
                or parents[0]["spanID"] != parent_id or parent_id not in by_id):
            raise ValueError("incorrect parent identity")
        parent = by_id[parent_id]
        start, end = remote_intervals[sid]
        parent_start, parent_end = remote_intervals[parent_id]
        # Flooring cannot invert start ordering. Separately floored durations
        # can put a contained child's computed end 1us beyond its parent's.
        if start < parent_start or end > parent_end + 1000:
            raise ValueError("span outside remote parent")
        desired = ("demo.rl.batch" if span["operationName"] == "demo.rl.trajectory" else
                   "demo.rl.tool.client" if span["operationName"] == "demo.rl.tool.server" else
                   "demo.rl.trajectory")
        if parent["operationName"] != desired:
            raise ValueError("incorrect hierarchy")
        if desired != "demo.rl.batch":
            parent_tags = {t["key"]: t["value"] for t in parent["tags"]}
            if tags.get("demo.trajectory.id") != parent_tags.get("demo.trajectory.id"):
                raise ValueError("trajectory context crossed concurrent tasks")
        if span["operationName"] in ("demo.rl.tool.server", "demo.rl.tool.client"):
            if tags.get("span.kind") != span["operationName"].rsplit(".", 1)[-1]:
                raise ValueError("incorrect client/server kind")
    return {"trace_id": trace["traceID"], "span_count": len(spans), "operations": expected,
            "all_local_span_ids_matched": True, "identity_counts_hierarchy_verified": True,
            "ui_url": result["ui_url"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--api", default="http://127.0.0.1:16687")
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    local = [json.loads(line) for line in (args.result.parent / "spans.jsonl").read_text().splitlines()]
    deadline = time.monotonic() + 30
    while True:
        try:
            payload = get_json(args.api, "/api/traces/" + result["trace_id"])
            receipt = validate(payload["data"][0], result, local)
            break
        except (OSError, ValueError, IndexError, KeyError, TypeError):
            if time.monotonic() >= deadline:
                raise RuntimeError("complete Jaeger rollout trace validation failed") from None
            time.sleep(0.1)
    for filename, value in (("jaeger_trace.json", payload), ("jaeger_receipt.json", receipt)):
        with (args.result.parent / filename).open("x") as stream:
            json.dump(value, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
