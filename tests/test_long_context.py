"""Correctness tests with tiny tokenizers and mocked HTTP, NOT performance tests."""
import io
import json
from pathlib import Path
import subprocess
import sys
import urllib.error
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from email.message import Message
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nemochat.long_context import (
    make_plan, step_at, load_cursor, advance_cursor, save_cursor,
    generate_fixtures, validate_fixture, run_predictions, exact_match, score_predictions,
)
from nemochat.long_context.io import read_jsonl
from nemochat.long_context.plot import plot_predictions


class TinyTokenizer:
    name_or_path = "tiny-character-fixture-test"

    def encode(self, text, add_special_tokens=True):
        return ([0] if add_special_tokens else []) + [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return "".join(chr(i) for i in ids if i)


@pytest.fixture
def tokenizer():
    return TinyTokenizer()


@pytest.fixture
def fixture(tokenizer):
    return generate_fixtures(tokenizer, [512], [.5], samples=1)[0]


def test_schedule_resume_and_exact_budget(tmp_path):
    plan = make_plan(long_only_steps=2, alternating_steps=4, effective_tokens=262144, world_size=2)
    path = tmp_path / "cursor.json"
    cursor = load_cursor(path, plan)
    first, cursor = advance_cursor(cursor, 3)
    save_cursor(path, cursor)
    remaining, cursor = advance_cursor(load_cursor(path, plan), 99)
    records = first + remaining
    assert [r["sequence_length"] for r in records] == [65536, 65536, 65536, 4096, 65536, 4096]
    assert records == [step_at(plan, i) for i in range(6)]
    for row in records:
        assert row["sequence_length"] * row["micro_batch_size"] * row["world_size"] * row["gradient_accumulation_steps"] == 262144
    assert advance_cursor(cursor)[0] == []
    with pytest.raises(ValueError):
        load_cursor(path, make_plan(effective_tokens=65536))


@pytest.mark.parametrize("options", [
    {"effective_tokens": 4096}, {"effective_tokens": 65537}, {"world_size": 3},
    {"micro_batch_size": 0}, {"long_only_steps": -1}, {"alternating_steps": 1.5},
    {"effective_tokens": True}, {"long_only_steps": 0, "alternating_steps": 0},
])
def test_impossible_plans(options):
    with pytest.raises(ValueError):
        make_plan(**options)


def test_invalid_cursor(tmp_path):
    path = tmp_path / "cursor.json"
    plan = make_plan()
    path.write_text(json.dumps({"version": 1, "plan": plan, "next_step": -1}))
    with pytest.raises(ValueError):
        load_cursor(path, plan)
    with pytest.raises(ValueError):
        advance_cursor({"version": 1, "plan": plan, "next_step": 999})


def test_frozen_fixtures_and_full_budget(tokenizer):
    options = dict(context_lengths=[512, 768], depths=[0., .5, 1.], samples=2, seed=123)
    rows = generate_fixtures(tokenizer, **options)
    assert rows == generate_fixtures(tokenizer, **options)
    assert len({r["expected"] for r in rows}) == 12
    for row in rows:
        assert validate_fixture(row, tokenizer) + row["generation_tokens"] <= row["context_length"]
        assert row["needle"] in row["prompt"]
        assert row["needle_offset_haystack_tokens"] == round(row["depth"] * row["haystack_tokens"])
        assert row["data_kind"] == "fixture"
    assert rows != generate_fixtures(tokenizer, **{**options, "seed": 124})


@pytest.mark.parametrize("options", [
    {"context_lengths": [64]}, {"depths": [1.1]}, {"depths": [float('nan')]},
    {"generation_tokens": 0}, {"samples": 0}, {"haystack": ""}, {"context_lengths": [512, 512]},
])
def test_invalid_fixtures(tokenizer, options):
    with pytest.raises(ValueError):
        generate_fixtures(tokenizer, **{ "context_lengths": [512], "depths": [.5], **options})


def test_tokenizer_roundtrip_rejected():
    class BrokenTokenizer(TinyTokenizer):
        def decode(self, ids, **kwargs):
            return super().decode(ids, **kwargs).lower()
    with pytest.raises(ValueError, match="round-trip"):
        generate_fixtures(BrokenTokenizer(), [512], [.5], 1)


def test_mismatch_before_network(monkeypatch, tokenizer, fixture):
    called = []
    monkeypatch.setattr("nemochat.long_context.niah.open_request", lambda *a, **k: called.append(1))
    changed = {**fixture, "prompt_tokens": fixture["prompt_tokens"] + 1}
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        run_predictions([changed], tokenizer, "https://example.test/v1/completions", "model")
    assert not called


def response(row, text=None, usage=True):
    reply = {"choices": [{"text": row["expected"] if text is None else text}]}
    if usage:
        reply["usage"] = {"prompt_tokens": row["prompt_tokens"]}
    return reply


def test_real_request_shape_mocked(monkeypatch, tokenizer, fixture):
    monkeypatch.setenv("NIAH_TEST_KEY", "test-environment-only")
    def request(req, timeout):
        payload = json.loads(req.data)
        assert payload["prompt"] == fixture["prompt"]
        assert payload["max_tokens"] == fixture["generation_tokens"]
        assert payload["stream"] is False
        assert req.get_header("Authorization") == "Bearer test-environment-only"
        assert timeout == 2
        return io.BytesIO(json.dumps(response(fixture)).encode())
    monkeypatch.setattr("nemochat.long_context.niah.open_request", request)
    rows = run_predictions([fixture], tokenizer, "https://example.test/v1/completions", "model", 2, "NIAH_TEST_KEY")
    assert rows[0]["status"] == "ok"
    assert rows[0]["data_kind"] == "measured"
    assert score_predictions(rows)[0]["accuracy"] == 1


@pytest.mark.parametrize("kind,status", [
    ("timeout", "timeout"), ("unsupported", "unsupported"), ("server", "error"),
    ("malformed", "error"), ("missing_usage", "tokenization_unverified"),
    ("wrong_usage", "tokenization_mismatch"), ("error_object", "error"),
])
def test_errors_are_excluded(monkeypatch, tokenizer, fixture, kind, status):
    def request(*args, **kwargs):
        if kind == "timeout":
            raise TimeoutError()
        if kind in ("unsupported", "server"):
            raise urllib.error.HTTPError("https://example.test", 413 if kind == "unsupported" else 500, "error", {}, None)
        reply = response(fixture)
        if kind == "malformed":
            reply = {"choices": []}
        elif kind == "error_object":
            reply = {"error": {"message": "context exceeded"}}
        elif kind == "missing_usage":
            reply.pop("usage")
        elif kind == "wrong_usage":
            reply["usage"]["prompt_tokens"] += 1
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr("nemochat.long_context.niah.open_request", request)
    rows = run_predictions([fixture], tokenizer, "https://example.test/v1/completions", "model")
    assert rows[0]["status"] == status
    cell = score_predictions(rows)[0]
    assert cell["accuracy"] is None and cell["count"] == 0 and cell["excluded"] == 1
    assert cell["wilson_low"] is None


def test_optional_hosted_client(tokenizer, fixture):
    class Client:
        completions = None
        def create(self, **kwargs):
            assert kwargs["timeout"] == 60
            return response(fixture)
    client = Client()
    client.completions = client
    assert run_predictions([fixture], tokenizer, "hosted-client", "model", client=client)[0]["status"] == "ok"


def test_strict_scoring_and_uncertainty(fixture):
    assert exact_match("  abc\n def ", "abc def")
    assert not exact_match("The code is abc", "abc")
    assert not exact_match("abc.", "abc")
    assert not exact_match("ABC", "abc")
    row = {**fixture, "data_kind": "measured", "model": "m", "status": "ok", "prediction": fixture["expected"]}
    wrong = {**row, "id": "wrong", "prediction": "code " + fixture["expected"]}
    timeout = {**row, "id": "timeout", "status": "timeout", "prediction": None}
    cell = score_predictions([row, wrong, timeout])[0]
    assert cell["count"] == 2 and cell["total"] == 3 and cell["accuracy"] == .5
    assert 0 < cell["wilson_low"] < .5 < cell["wilson_high"] < 1
    with pytest.raises(ValueError, match="duplicate"):
        score_predictions([row, row])
    with pytest.raises(ValueError, match="data_kind"):
        score_predictions([row, fixture])


def test_plots_fixture_and_measured(tmp_path, fixture):
    pytest.importorskip("matplotlib")
    prefix = tmp_path / "synthetic"
    paths = plot_predictions([fixture], prefix)
    assert all(Path(p).stat().st_size > 100 for p in paths)
    assert "SYNTHETIC FIXTURE" in Path(str(prefix) + ".svg").read_text()
    metadata = json.loads(Path(str(prefix) + ".metadata.json").read_text())
    assert metadata["measured"] is False and metadata["scale"] == [0, 1]
    assert metadata["cells"][0]["accuracy"] is None
    rows = [{**fixture, "data_kind": "measured", "status": "unsupported", "model": "test", "prediction": None},
            {**fixture, "id": "timeout", "depth": 1., "data_kind": "measured", "status": "timeout", "model": "test", "prediction": None}]
    plot_predictions(rows, tmp_path / "measured")
    svg = (tmp_path / "measured.svg").read_text()
    assert "unsupported" in svg and "timeout" in svg


def test_schedule_cli_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/lc_schedule.py"), "--cursor", str(tmp_path / "cursor.json"),
               "--output", str(tmp_path / "steps.jsonl"), "--count", "3"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert [r["step"] for r in read_jsonl(tmp_path / "steps.jsonl")] == [0, 1, 2]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert [r["step"] for r in read_jsonl(tmp_path / "steps.jsonl")] == [3, 4, 5]


def test_seeded_random_depths(tokenizer):
    rows = generate_fixtures(tokenizer, [512], [], samples=1, seed=99, random_depths=4)
    assert rows == generate_fixtures(tokenizer, [512], [], samples=1, seed=99, random_depths=4)
    assert len({row["depth"] for row in rows}) == 4
    assert all(0 <= row["depth"] <= 1 for row in rows)


@pytest.mark.parametrize("field,value", [("generation_tokens", -1), ("context_length", True), ("expected", "")])
def test_tampered_fixture_rejected(tokenizer, fixture, field, value):
    with pytest.raises(ValueError):
        validate_fixture({**fixture, field: value}, tokenizer)


def test_network_and_endpoint_validation(monkeypatch, tokenizer, fixture):
    def request(*args, **kwargs):
        raise urllib.error.URLError(TimeoutError())
    monkeypatch.setattr("nemochat.long_context.niah.open_request", request)
    assert run_predictions([fixture], tokenizer, "https://example.test/completions", "m")[0]["status"] == "timeout"
    with pytest.raises(ValueError):
        run_predictions([fixture], tokenizer, "https://user:password@example.test/completions", "m")
    with pytest.raises(ValueError):
        run_predictions([fixture], tokenizer, "file:///etc/passwd", "m")


def test_hf_local_tokenizer_and_cli(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    backend = tokenizers.Tokenizer(tokenizers.models.BPE())
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=300, initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet())
    backend.train_from_iterator(["The archive contains ordinary notes about weather, books, and gardens.\n"], trainer)
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend)
    directory = tmp_path / "tokenizer"
    tokenizer.save_pretrained(directory)
    rows = generate_fixtures(tokenizer, [512], [0., .5, 1.], 1)
    assert len(rows) == 3
    assert all(validate_fixture(row, tokenizer) <= 480 for row in rows)
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "fixtures.jsonl"
    subprocess.run([sys.executable, str(root / "scripts/niah.py"), "generate", "--tokenizer", str(directory),
                    "--local-files-only", "--context-lengths", "512", "--depths", "0", "0.5", "1",
                    "--samples", "1", "--output", str(output)], check=True, capture_output=True, text=True)
    generated = read_jsonl(output)
    assert len(generated) == 3 and generated[0]["data_kind"] == "fixture"
    subprocess.run([sys.executable, str(root / "scripts/niah.py"), "score", "--input", str(output),
                    "--output", str(tmp_path / "scores.jsonl"), "--csv", str(tmp_path / "scores.csv")],
                   check=True, capture_output=True, text=True)
    assert all(r["count"] == 0 for r in read_jsonl(tmp_path / "scores.jsonl"))


@pytest.mark.parametrize("endpoint", [
    "http://example.test/v1/completions",
    "http://localhost.evil/v1/completions",
    "http://127.0.0.1.evil/v1/completions",
    "http://192.0.2.1/v1/completions",
    "http://[2001:db8::1]/v1/completions",
    "http:///v1/completions",
    "http://:8000/v1/completions",
    "http://[::1/v1/completions",
    "http://localhost:invalid/v1/completions",
])
def test_authenticated_endpoint_rejected_before_network(monkeypatch, tokenizer, fixture, endpoint):
    from unittest.mock import Mock

    key = "dummy-niah-transport-key"
    monkeypatch.setenv("NIAH_TEST_KEY", key)
    request = Mock(return_value=io.BytesIO(json.dumps(response(fixture)).encode()))
    monkeypatch.setattr("nemochat.long_context.niah.open_request", request)
    with pytest.raises(ValueError) as error:
        run_predictions([fixture], tokenizer, endpoint, "model", api_key_env="NIAH_TEST_KEY")
    request.assert_not_called()
    assert key not in str(error.value)


@pytest.mark.parametrize("endpoint", [
    "https://example.test/v1/completions",
    "http://localhost:8000/v1/completions",
    "http://127.0.0.1:8000/v1/completions",
    "http://127.0.0.2:8000/v1/completions",
    "http://[::1]:8000/v1/completions",
])
def test_authenticated_https_and_loopback_http(monkeypatch, tokenizer, fixture, endpoint):
    from unittest.mock import Mock

    key = "dummy-niah-transport-key"
    monkeypatch.setenv("NIAH_TEST_KEY", key)
    def reply(request, timeout):
        assert request.get_header("Authorization") == "Bearer " + key
        return io.BytesIO(json.dumps(response(fixture)).encode())
    request = Mock(side_effect=reply)
    monkeypatch.setattr("nemochat.long_context.niah.open_request", request)
    rows = run_predictions([fixture], tokenizer, endpoint, "model", api_key_env="NIAH_TEST_KEY")
    request.assert_called_once()
    assert rows[0]["status"] == "ok"
    assert key not in json.dumps(rows)


@pytest.mark.parametrize("error_value", [{}, [], False, 0, "", {"message": "failed"}])
def test_nonnull_response_error_excluded(monkeypatch, tokenizer, fixture, error_value):
    reply = {**response(fixture), "error": error_value}
    monkeypatch.setattr("nemochat.long_context.niah.open_request",
                        lambda *args, **kwargs: io.BytesIO(json.dumps(reply).encode()))
    rows = run_predictions([fixture], tokenizer, "https://example.test/v1/completions", "model")
    assert rows[0]["status"] == "error"
    assert rows[0]["prediction"] is None
    cell = score_predictions(rows)[0]
    assert cell["count"] == 0 and cell["excluded"] == 1
    assert cell["accuracy"] is None


@pytest.mark.parametrize("include_null_error", [False, True])
def test_absent_or_null_response_error_accepted(monkeypatch, tokenizer, fixture, include_null_error):
    reply = response(fixture)
    if include_null_error:
        reply = {**reply, "error": None}
    monkeypatch.setattr("nemochat.long_context.niah.open_request",
                        lambda *args, **kwargs: io.BytesIO(json.dumps(reply).encode()))
    rows = run_predictions([fixture], tokenizer, "https://example.test/v1/completions", "model")
    assert rows[0]["status"] == "ok"
    assert score_predictions(rows)[0]["accuracy"] == 1



@pytest.mark.parametrize("redirect", [False, True])
def test_authenticated_loopback_bypasses_environment_proxy(monkeypatch, tokenizer, fixture, redirect):
    key = "dummy-proxy-regression-key"
    monkeypatch.setenv("NIAH_TEST_KEY", key)
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.headers.get("Authorization"))
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(302 if redirect else 200)
            self.send_header("Location", "http://192.0.2.1/credential-trap")
            content = json.dumps(response(fixture)).encode()
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server, socket.socket() as proxy:
        proxy.bind(("127.0.0.1", 0))
        proxy.listen()
        proxy.setblocking(False)
        proxy_url = f"http://127.0.0.1:{proxy.getsockname()[1]}"
        for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
            monkeypatch.setenv(name, proxy_url)
        for name in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(name, "")
        monkeypatch.delenv("REQUEST_METHOD", raising=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connections = []
        connect = socket.create_connection

        def guarded_connect(address, *args, **kwargs):
            connections.append(address)
            # Never contact the proxy or redirect target, even on the buggy path.
            if address != server.server_address:
                raise OSError("unexpected connection blocked by regression test")
            return connect(address, *args, **kwargs)

        try:
            with patch("socket.create_connection", side_effect=guarded_connect):
                rows = run_predictions([fixture], tokenizer,
                                       f"http://127.0.0.1:{server.server_port}/v1/completions",
                                       "model", timeout=2, api_key_env="NIAH_TEST_KEY")
            assert connections == [server.server_address]
            assert received == ["Bearer " + key]
            assert rows[0]["status"] == ("error" if redirect else "ok")
            if redirect:
                assert rows[0]["error"] == "HTTP 302"
            assert key not in json.dumps(rows)
            with pytest.raises(BlockingIOError):
                proxy.accept()
        finally:
            server.shutdown()
            thread.join(2)


@pytest.mark.parametrize("extra,status", [(0, "ok"), (1, "error")])
def test_http_response_size_bound(monkeypatch, tokenizer, fixture, extra, status):
    from nemochat.long_context import niah

    cap = 1024
    monkeypatch.setattr(niah, "MAX_RESPONSE_BYTES", cap, raising=False)
    content = json.dumps(response(fixture)).encode()
    content += b" " * (cap + extra - len(content))
    reads = []

    class BoundedStream(io.BytesIO):
        def read(self, size: int | None = -1):
            reads.append(size)
            return super().read(size)

    stream = BoundedStream(content)
    monkeypatch.setattr(niah, "open_request", lambda *a, **k: stream)
    rows = run_predictions([fixture], tokenizer, "https://example.test/v1/completions", "model")
    assert reads == [cap + 1]
    assert stream.closed
    assert rows[0]["status"] == status
    if extra:
        assert rows[0]["prediction"] is None
        assert rows[0]["error"] == "invalid response: ValueError"
        assert score_predictions(rows)[0]["count"] == 0



@pytest.mark.parametrize("code,status", [(413, "unsupported"), (500, "error")])
def test_oversize_http_error_body_not_read(monkeypatch, tokenizer, fixture, code, status):
    from nemochat.long_context.niah import MAX_RESPONSE_BYTES

    class ErrorBody(io.BytesIO):
        def read(self, size: int | None = -1):
            raise AssertionError("HTTP error bodies must not be read")

    stream = ErrorBody(b"x" * (MAX_RESPONSE_BYTES + 1))

    def request(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.test", code, "failure", Message(), stream)

    monkeypatch.setattr("nemochat.long_context.niah.open_request", request)
    rows = run_predictions([fixture], tokenizer, "https://example.test/v1/completions", "model")
    assert rows[0]["status"] == status
    assert rows[0]["error"] == f"HTTP {code}"
    assert rows[0]["prediction"] is None
    assert stream.closed
    assert score_predictions(rows)[0]["count"] == 0
