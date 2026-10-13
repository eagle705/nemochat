"""Offline exporter privacy regressions: synthetic auth and mocked HTTP transport only."""
from unittest.mock import Mock

import pytest
import requests
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

ENDPOINT = "http://127.0.0.1:14318/v1/traces"
BODY = b"synthetic-trace-body"


def exporter():
    from train import local_otlp_session
    return OTLPSpanExporter(endpoint=ENDPOINT, timeout=5, session=local_otlp_session())


def transport(monkeypatch, status):
    sent = []

    def send(adapter, request, **kwargs):
        sent.append((request, kwargs))
        response = requests.Response()
        response.status_code = status
        response.reason = "synthetic response"
        response.url = request.url
        response.request = request
        response._content = b""
        if status in (307, 308):
            response.headers["Location"] = "http://example.invalid/collect"
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return sent


def test_exporter_ignores_ambient_netrc_and_proxies(monkeypatch):
    # Mock the lookup itself: no real netrc file or credentials are ever read.
    netrc = Mock(return_value=("synthetic-user", "synthetic-password"))
    proxies = Mock(return_value={"http": "http://example.invalid:9999"})
    monkeypatch.setattr(requests.sessions, "get_netrc_auth", netrc)
    monkeypatch.setattr(requests.sessions, "get_environ_proxies", proxies)
    monkeypatch.setenv("HTTP_PROXY", "http://example.invalid:9999")
    monkeypatch.setenv("HTTPS_PROXY", "http://example.invalid:9999")
    monkeypatch.setenv("ALL_PROXY", "http://example.invalid:9999")
    sent = transport(monkeypatch, 200)
    instance = exporter()
    try:
        assert instance._session.trust_env is False
        assert instance._export(BODY).status_code == 200
        netrc.assert_not_called()
        proxies.assert_not_called()
        assert len(sent) == 1
        request, kwargs = sent[0]
        assert "Authorization" not in request.headers
        assert not kwargs["proxies"]
        assert request.url == ENDPOINT
        assert request.body == BODY
    finally:
        instance.shutdown()


@pytest.mark.parametrize("status", [307, 308])
def test_redirect_is_export_failure_without_second_post(monkeypatch, status):
    sent = transport(monkeypatch, status)
    # Keep the real pinned exporter export/retry path, with synthetic serialized spans.
    encoded = Mock()
    encoded.SerializePartialToString.return_value = BODY
    monkeypatch.setattr("opentelemetry.exporter.otlp.proto.http.trace_exporter.encode_spans",
                        Mock(return_value=encoded))
    instance = exporter()
    try:
        assert instance.export([]) is SpanExportResult.FAILURE
        assert len(sent) == 1
        assert sent[0][0].url == ENDPOINT
        assert sent[0][0].body == BODY
    finally:
        instance.shutdown()


@pytest.mark.parametrize("status", [307, 308])
def test_session_forces_redirects_off_even_if_requested(monkeypatch, status):
    from train import local_otlp_session
    sent = transport(monkeypatch, status)
    with local_otlp_session() as session:
        with pytest.raises(requests.HTTPError):
            session.post(ENDPOINT, data=BODY, allow_redirects=True)
    assert len(sent) == 1
    assert sent[0][0].url == ENDPOINT
    assert sent[0][0].body == BODY


def test_session_rejects_remote_before_transport(monkeypatch):
    from train import local_otlp_session
    sent = transport(monkeypatch, 200)
    with local_otlp_session() as session:
        with pytest.raises(ValueError):
            session.post("http://example.invalid/v1/traces", data=BODY)
    assert sent == []
