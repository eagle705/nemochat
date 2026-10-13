"""Readiness and complete-trace acceptance gates via the local Jaeger API."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit


MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def get_json(base, path):
    url = urlsplit(base)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
            or url.username or url.password or url.query or url.fragment or url.path not in ("", "/")):
        raise ValueError("Jaeger API must be credential-free loopback HTTP")
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise ValueError("Jaeger API redirects are not allowed")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(base.rstrip("/") + path, timeout=2) as response:
        payload = response.read(MAX_RESPONSE_BYTES + 1)
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ValueError("Jaeger API response too large")
        return json.loads(payload)


def wait_ready(base, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            return get_json(base, "/api/services")
        except (OSError, ValueError):
            time.sleep(0.1)  # bounded polling after an actual failed health request
    raise RuntimeError("Jaeger API did not become ready")


def validate_trace(trace, result):
    spans = trace["spans"]
    if len(spans) != result["expected_spans"]:
        raise ValueError("incomplete span count")
    names = Counter(span["operationName"] for span in spans)
    expected = {"train.run": 1, "train.step": result["steps"], "train.forward": result["steps"],
                "train.backward": result["steps"], "train.optimizer": result["steps"]}
    if names != expected:
        raise ValueError("unexpected operation counts")
    roots = [span for span in spans if span["operationName"] == "train.run"]
    by_id = {span["spanID"]: span for span in spans}
    for span in spans:
        process = trace["processes"][span["processID"]]
        tags = {tag["key"]: tag["value"] for tag in process["tags"]}
        if process["serviceName"] != result["service"] or tags.get("nemo.run.id") != result["run_id"]:
            raise ValueError("incorrect resource identity")
        if span["operationName"] != "train.run":
            parents = [ref["spanID"] for ref in span["references"] if ref["refType"] == "CHILD_OF"]
            parent_name = "train.run" if span["operationName"] == "train.step" else "train.step"
            if len(parents) != 1 or by_id[parents[0]]["operationName"] != parent_name:
                raise ValueError("incorrect parent hierarchy")
    return {"trace_id": trace["traceID"], "span_count": len(spans), "operations": dict(names),
            "root_span_id": roots[0]["spanID"], "identity_and_hierarchy_verified": True}


def fetch_trace(base, result, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = get_json(base, "/api/traces/" + result["trace_id"])
            trace = payload["data"][0]
            receipt = validate_trace(trace, result)
            return payload, receipt
        except (OSError, ValueError, IndexError, KeyError, TypeError):
            time.sleep(0.1)
    raise RuntimeError("Jaeger did not ingest a complete, correctly nested trace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:16687")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    print(json.dumps(wait_ready(args.api)))
    if args.result:
        result = json.loads(args.result.read_text())
        payload, receipt = fetch_trace(args.api, result)
        for name, value in (("jaeger_trace.json", payload), ("jaeger_receipt.json", receipt)):
            with (args.result.parent / name).open("x") as stream:
                json.dump(value, stream, indent=2)
        print(json.dumps(receipt))


if __name__ == "__main__":
    main()
