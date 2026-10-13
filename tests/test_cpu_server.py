"""Exercise the actual CPU handler on loopback, without torch/HF downloads."""
import contextlib
import http.client
from http.server import HTTPServer
import importlib.util
import json
import socket
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CPUServerTests(unittest.TestCase):
    def test_nonobject_json_returns_structured_400_and_server_remains_usable(self):
        spec = importlib.util.spec_from_file_location(
            "serve_cpu_test", Path(__file__).resolve().parents[1] / "scripts" / "serve_cpu.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Output:
            def __getitem__(self, key):
                return [7]

        tokenizer = Mock(eos_token_id=0)
        tokenizer.return_value = {"input_ids": SimpleNamespace(shape=(1, 2))}
        tokenizer.decode.return_value = "completion"
        model = Mock(config=SimpleNamespace(max_position_embeddings=16))
        model.eval.return_value = model
        model.generate.return_value = Output()
        modules = {
            "torch": SimpleNamespace(set_num_threads=Mock(), inference_mode=contextlib.nullcontext),
            "transformers": SimpleNamespace(
                AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model))),
        }
        ready = threading.Event()
        servers, errors = [], []

        def make_server(address, handler):
            self.assertEqual(address[0], "127.0.0.1")
            server = HTTPServer(address, handler)
            server.handle_error = Mock()
            servers.append(server)
            ready.set()
            return server

        def run():
            try:
                module.main()
            except BaseException as exc:
                errors.append(exc)
                ready.set()

        def request(payload):
            connection = http.client.HTTPConnection("127.0.0.1", servers[0].server_port, timeout=5)
            try:
                connection.request("POST", "/v1/completions", json.dumps(payload),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.getheader("Content-Type"), "application/json")
                return response.status, json.loads(response.read())
            finally:
                connection.close()

        with patch.dict(sys.modules, modules), patch.object(module, "HTTPServer", make_server), \
                patch.object(sys, "argv", ["serve_cpu.py", "--model", "fake-tiny", "--port", "0"]), \
                patch.object(module, "REQUEST_TIMEOUT_SECONDS", 0.05):
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                self.assertTrue(ready.wait(5), "CPU server failed to start")
                self.assertEqual(errors, [])
                for payload in [[], [1], "text", 42, 1.5, True, None]:
                    with self.subTest(payload=payload):
                        self.assertEqual(request(payload),
                                         (400, {"error": "Invalid request or unsupported context"}))
                # Wait for EOF/408 rather than sleeping: both incomplete headers and
                # an incomplete declared body must release this single-threaded server.
                for partial in [b"POST /v1/completions HTTP/1.1\r\n",
                                b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                                b"Content-Length: 100\r\n\r\n{"]:
                    with self.subTest(partial=partial):
                        with socket.create_connection(servers[0].server_address, timeout=2) as stalled:
                            stalled.sendall(partial)
                            received = b""
                            while True:
                                chunk = stalled.recv(4096)
                                if not chunk:
                                    break
                                received += chunk
                            if b"Content-Length" in partial:
                                self.assertIn(b"408", received)
                        health = http.client.HTTPConnection(*servers[0].server_address, timeout=2)
                        try:
                            health.request("GET", "/health")
                            reply = health.getresponse()
                            self.assertEqual(reply.status, 200)
                            self.assertEqual(json.loads(reply.read())["status"], "ready")
                        finally:
                            health.close()
                # An abruptly disconnected body must also leave the server usable.
                with socket.create_connection(servers[0].server_address, timeout=2) as disconnected:
                    disconnected.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                                         b"Content-Length: 100\r\n\r\n{")
                tokenizer.assert_not_called()
                model.generate.assert_not_called()
                status, body = request({"model": "fake-tiny", "prompt": "hello", "max_tokens": 1})
                self.assertEqual(status, 200)
                self.assertEqual(body, {"model": "fake-tiny", "choices": [{"text": "completion"}],
                                        "usage": {"prompt_tokens": 2, "completion_tokens": 1,
                                                  "total_tokens": 3}})
                model.generate.assert_called_once()
                servers[0].handle_error.assert_not_called()
            finally:
                if servers:
                    servers[0].shutdown()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
