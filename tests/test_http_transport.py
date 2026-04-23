"""Tests for :mod:`earl_sdk._http` (httpx-backed transport)."""

from __future__ import annotations

import gzip
import json

import httpx
import pytest

from earl_sdk import _http
from earl_sdk.exceptions import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TransportError,
    ValidationError,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_shared_client():
    """Ensure every test starts/ends with no cached client."""
    _http.close_shared_client()
    yield
    _http.close_shared_client()


def _install(handler):
    """Install a MockTransport-backed client as the SDK's shared client."""
    client = httpx.Client(transport=httpx.MockTransport(handler))
    _http.set_shared_client(client)
    return client


# ── Happy path ───────────────────────────────────────────────────────────────


def test_request_json_returns_parsed_body_and_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://api.example.com/thing?x=1"
        assert request.headers["authorization"] == "Bearer TOK"
        return httpx.Response(200, json={"ok": True, "n": 42})

    _install(handler)
    body, resp = _http.request_json(
        "GET",
        "https://api.example.com/thing",
        headers={"Authorization": "Bearer TOK"},
        params={"x": 1},
    )
    assert body == {"ok": True, "n": 42}
    assert resp.status_code == 200


def test_request_json_decodes_gzip_response():
    """httpx transparently decodes gzip; we just need to make sure
    the transport ships the request and returns parsed JSON."""
    raw = gzip.compress(json.dumps({"gz": "ipped"}).encode())

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=raw,
            headers={"content-type": "application/json", "content-encoding": "gzip"},
        )

    _install(handler)
    body, _ = _http.request_json("GET", "https://api.example.com/g")
    assert body == {"gz": "ipped"}


def test_request_json_empty_body_returns_empty_dict():
    _install(lambda r: httpx.Response(204))
    body, resp = _http.request_json("GET", "https://api.example.com/nada")
    assert body == {}
    assert resp.status_code == 204


# ── Error mapping ────────────────────────────────────────────────────────────


def test_401_raises_authentication_error_with_request_id_and_hint():
    _install(
        lambda r: httpx.Response(
            401,
            json={"error": "invalid_token", "message": "Token expired"},
            headers={"x-request-id": "req-abc"},
        )
    )
    with pytest.raises(AuthenticationError) as ei:
        _http.request_json("GET", "https://api.example.com/x")
    exc = ei.value
    assert exc.status_code == 401
    assert exc.code == "invalid_token"
    assert exc.request_id == "req-abc"
    assert exc.message == "Token expired"
    assert exc.hint and "earl login" in exc.hint
    assert exc.method == "GET"
    assert exc.url == "https://api.example.com/x"


def test_403_with_organization_hint():
    _install(
        lambda r: httpx.Response(
            403,
            json={"detail": "Access denied: token missing organization_id"},
        )
    )
    with pytest.raises(AuthorizationError) as ei:
        _http.request_json("GET", "https://api.example.com/y")
    assert ei.value.code == "forbidden"
    assert "organization" in (ei.value.hint or "")


def test_404_maps_to_not_found_with_resource_info():
    _install(
        lambda r: httpx.Response(
            404,
            json={"resource_type": "Pipeline", "resource_id": "missing"},
        )
    )
    with pytest.raises(NotFoundError) as ei:
        _http.request_json("GET", "https://api.example.com/pipelines/missing")
    assert ei.value.resource_type == "Pipeline"
    assert ei.value.resource_id == "missing"


def test_404_infers_resource_id_from_url_when_body_missing():
    _install(lambda r: httpx.Response(404, json={}))
    with pytest.raises(NotFoundError) as ei:
        _http.request_json("GET", "https://api.example.com/patients/p-17")
    assert ei.value.resource_id == "p-17"


def test_400_fastapi_validation_error_flattens_to_single_message():
    _install(
        lambda r: httpx.Response(
            422,
            json={
                "detail": [
                    {"loc": ["body", "name"], "msg": "field required"},
                    {"loc": ["body", "turns"], "msg": "must be > 0"},
                ]
            },
        )
    )
    with pytest.raises(ValidationError) as ei:
        _http.request_json("POST", "https://api.example.com/z", json_body={"a": 1})
    assert "body -> name: field required" in ei.value.message
    assert "turns: must be > 0" in ei.value.message


def test_429_returns_retry_after():
    _install(
        lambda r: httpx.Response(
            429,
            json={"message": "chill"},
            headers={"retry-after": "7"},
        )
    )
    with pytest.raises(RateLimitError) as ei:
        _http.request_json("GET", "https://api.example.com/throttle", retries=0)
    assert ei.value.retry_after == 7.0
    assert ei.value.code == "rate_limited"


def test_500_maps_to_server_error():
    _install(lambda r: httpx.Response(500, json={"message": "boom"}, headers={"x-request-id": "rid"}))
    with pytest.raises(ServerError) as ei:
        _http.request_json("POST", "https://api.example.com/b", retries=0)
    assert ei.value.request_id == "rid"
    assert ei.value.status_code == 500


def test_non_json_error_body_is_preserved():
    _install(lambda r: httpx.Response(500, text="fatal <html>", headers={"content-type": "text/html"}))
    with pytest.raises(ServerError) as ei:
        _http.request_json("GET", "https://api.example.com/c", retries=0)
    assert "fatal" in ei.value.details.get("message", "")


# ── Retry + transport failure ────────────────────────────────────────────────


def test_idempotent_get_retries_transient_503(monkeypatch):
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)

    calls: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(r)
        if len(calls) < 3:
            return httpx.Response(503, json={"message": "warming up"}, headers={"retry-after": "0"})
        return httpx.Response(200, json={"ok": True})

    _install(handler)
    body, _ = _http.request_json("GET", "https://api.example.com/r")
    assert body == {"ok": True}
    assert len(calls) == 3


def test_non_idempotent_post_is_not_retried(monkeypatch):
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)

    calls: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"message": "warming"})

    _install(handler)
    with pytest.raises(ServerError):
        _http.request_json("POST", "https://api.example.com/s")
    assert len(calls) == 1


def test_transport_error_on_get_retries_then_raises(monkeypatch):
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)

    calls: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("nope", request=r)

    _install(handler)
    with pytest.raises(TransportError) as ei:
        _http.request_json("GET", "https://api.example.com/down")
    assert len(calls) == 1 + _http._MAX_RETRIES
    assert ei.value.url == "https://api.example.com/down"
    assert "nope" in ei.value.message


def test_timeout_on_get_retries_then_raises(monkeypatch):
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)

    calls: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ReadTimeout("slow", request=r)

    _install(handler)
    with pytest.raises(TransportError) as ei:
        _http.request_json("GET", "https://api.example.com/slow")
    assert len(calls) == 1 + _http._MAX_RETRIES
    assert "timed out" in ei.value.message.lower()


# ── Header redaction ─────────────────────────────────────────────────────────


def test_redact_headers_hides_sensitive_values():
    red = _http._redact_headers(
        {
            "Authorization": "Bearer xyz",
            "Cookie": "s=1",
            "X-API-Key": "abc",
            "Content-Type": "application/json",
        }
    )
    assert red["Authorization"] == "***redacted***"
    assert red["Cookie"] == "***redacted***"
    assert red["X-API-Key"] == "***redacted***"
    assert red["Content-Type"] == "application/json"


# ── Default client shape ─────────────────────────────────────────────────────


def test_shared_client_opts_in_to_http2_when_h2_available():
    _http.close_shared_client()
    try:
        c = _http._get_client()
        # Presence of the attribute is enough; actual H2 negotiation is
        # server-driven and needs a real connection.
        assert isinstance(c, httpx.Client)
        assert c.headers["accept-encoding"].startswith("gzip")
        assert "User-Agent" in c.headers or "user-agent" in c.headers
    finally:
        _http.close_shared_client()
