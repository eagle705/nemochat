"""Focused real HTTP/inference regressions; raw subprocess artifacts stay in results/."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from diagnose import analyze, ns, span_id
from run import ToolFailure, tool_session
from verify import get_json


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = HERE.parents[2] / "results" / "lens-rollout-tests" / uuid.uuid4().hex
        cls.root.mkdir(parents=True)
        cls.success = cls.execute("slow", "--scenario", "tool-slow")
        cls.spans = [json.loads(s) for s in (cls.root / "slow" / "spans.jsonl").read_text().splitlines()]

    @classmethod
    def execute(cls, name, *args):
        completed = subprocess.run([sys.executable, str(HERE / "run.py"), "--mode", "file",
                                    "--output", str(cls.root / name), *args],
                                   capture_output=True, text=True, timeout=90)
        (cls.root / (name + "-stdout.txt")).write_text(completed.stdout)
        (cls.root / (name + "-stderr.txt")).write_text(completed.stderr)
        return completed

    def test_real_generation_and_concurrency(self):
        self.assertEqual(self.success.returncode, 0, self.success.stderr)
        result = json.loads((self.root / "slow" / "result.json").read_text())
        self.assertEqual([r["generated_tokens"] for r in result["trajectories"]], [8] * 4)
        servers = [s for s in self.spans if s["name"] == "demo.rl.tool.server"]
        # All requests overlap: a sequential replay would violate this inequality.
        self.assertLess(max(ns(s["start_time"]) for s in servers), min(ns(s["end_time"]) for s in servers))
        queues = [s for s in self.spans if s["name"] == "demo.rl.caller_queue"]
        self.assertGreater(max(ns(s["end_time"]) - ns(s["start_time"]) for s in queues), 0)

    def test_w3c_identity_and_concurrent_parentage(self):
        by_id = {span_id(s): s for s in self.spans}
        self.assertEqual(len({s["context"]["trace_id"] for s in self.spans}), 1)
        for server in [s for s in self.spans if s["name"] == "demo.rl.tool.server"]:
            client = by_id[server["parent_id"].removeprefix("0x")]
            trajectory = by_id[client["parent_id"].removeprefix("0x")]
            self.assertEqual(client["name"], "demo.rl.tool.client")
            self.assertEqual(trajectory["name"], "demo.rl.trajectory")
            self.assertEqual(server["attributes"]["demo.trajectory.id"], client["attributes"]["demo.trajectory.id"])
            self.assertEqual(server["attributes"]["demo.trajectory.id"], trajectory["attributes"]["demo.trajectory.id"])
            self.assertEqual(server["kind"], "SpanKind.SERVER")
            self.assertEqual(client["kind"], "SpanKind.CLIENT")

    def test_exclusive_metrics_and_overlap_rejection(self):
        report = analyze(self.spans)
        self.assertEqual(report["sample_size"], 4)
        self.assertGreater(report["sum_trajectory_wall_ms"], report["batch_wall_ms"] * 2)
        self.assertNotIn("tool.server", report["exclusive_stage_totals_ms"])
        for row in report["trajectories"]:
            self.assertAlmostEqual(sum(row["stages_ms"].values()) + row["unknown_gap_ms"], row["wall_ms"])
        broken = copy.deepcopy(self.spans)
        generation = next(s for s in broken if s["name"] == "demo.rl.generation")
        queue = next(s for s in broken if s["name"] == "demo.rl.caller_queue"
                     and s["parent_id"] == generation["parent_id"])
        generation["start_time"] = queue["start_time"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            analyze(broken)

    def test_http_failure_stays_failure(self):
        result = self.execute("http-error", "--fail-id", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.root / "http-error" / "failure.json").exists())
        self.assertFalse((self.root / "http-error" / "result.json").exists())
        spans = [json.loads(s) for s in (self.root / "http-error" / "spans.jsonl").read_text().splitlines()]
        for name in ("demo.rl.tool.server", "demo.rl.tool.client", "demo.rl.trajectory", "demo.rl.batch"):
            self.assertTrue(any(s["name"] == name and s["status"]["status_code"] == "ERROR" for s in spans))

    def test_timeout_stays_failure(self):
        result = self.execute("timeout", "--scenario", "tool-straggler", "--timeout", "0.04")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.root / "timeout" / "failure.json").exists())
        self.assertFalse((self.root / "timeout" / "result.json").exists())
        spans = [json.loads(s) for s in (self.root / "timeout" / "spans.jsonl").read_text().splitlines()]
        self.assertTrue(any(s["name"] == "demo.rl.tool.client" and s["status"]["status_code"] == "ERROR" for s in spans))

    def test_fresh_output_refusal(self):
        before = (self.root / "slow" / "result.json").read_bytes()
        result = self.execute("slow")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, (self.root / "slow" / "result.json").read_bytes())

    def test_span_attribute_privacy(self):
        allowed = {"demo.trajectory.id", "demo.scenario", "demo.sample_size", "demo.generated_tokens",
                   "demo.reward.correct", "demo.injected_delay_ms", "http.response.status_code"}
        for span in self.spans:
            self.assertLessEqual(set(span["attributes"]), allowed)

    def test_http_hardening_actual_redirect_proxy_netrc(self):
        headers = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                headers.append(dict(self.headers))
                self.send_response(302)
                self.send_header("Location", "http://192.0.2.1/forbidden")
                self.send_header("Content-Length", "0")
                self.end_headers()
            def do_POST(self):
                headers.append(dict(self.headers))
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(302)
                self.send_header("Location", "http://192.0.2.1/forbidden")
                self.send_header("Content-Length", "0")
                self.end_headers()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        netrc = self.root / "dummy.netrc"
        netrc.write_text("machine 127.0.0.1 login dummy password dummy\n")
        try:
            with patch.dict(os.environ, {"NETRC": str(netrc), "HTTP_PROXY": "http://192.0.2.1:1",
                                         "ALL_PROXY": "http://192.0.2.1:1", "NO_PROXY": ""}):
                with tool_session() as session:
                    with self.assertRaises(ToolFailure):
                        session.post(f"http://127.0.0.1:{server.server_port}/calculate", json={}, timeout=1)
                    with self.assertRaises(ValueError):
                        session.post("http://192.0.2.1/calculate", json={})
                with self.assertRaises(ValueError):
                    get_json(f"http://127.0.0.1:{server.server_port}", "/api/services")
            self.assertEqual(len(headers), 2)
            self.assertFalse(any(k.lower() == "authorization" for entry in headers for k in entry))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
