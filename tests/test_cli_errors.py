"""Tests for the rich CLI error renderer."""

from __future__ import annotations

import argparse
import io
import json

import pytest

from earl_sdk.cli import _errors
from earl_sdk.exceptions import (
    AuthenticationError,
    AuthorizationError,
    EarlError,
    NotFoundError,
    RateLimitError,
    ServerError,
    SimulationError,
    TransportError,
    ValidationError,
)


def _args(output: str = "text", debug: bool = False, verbose: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        output=output, json_output=(output == "json"), debug=debug, verbose=verbose
    )


# ── Exit codes ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "exc_factory, expected",
    [
        (lambda: ValidationError("bad"), 2),
        (lambda: AuthenticationError("nope"), 3),
        (lambda: AuthorizationError("nope"), 4),
        (lambda: NotFoundError("Pipeline", "x"), 5),
        (lambda: RateLimitError(retry_after=10), 6),
        (lambda: ServerError("kaboom"), 7),
        (lambda: TransportError("no net"), 8),
        (lambda: SimulationError("sim_1", "failed"), 9),
        (lambda: EarlError("misc"), 1),
        (lambda: RuntimeError("generic"), 1),
    ],
)
def test_exit_code_taxonomy(exc_factory, expected):
    assert _errors.exit_code_for(exc_factory()) == expected


# ── JSON mode ────────────────────────────────────────────────────────────────


def test_json_mode_emits_canonical_envelope():
    buf = io.StringIO()
    exc = AuthenticationError(
        "Token expired",
        request_id="req-1",
        url="https://api.example.com/x",
        method="GET",
        hint="Run `earl auth login`",
        details={"error": "invalid_token"},
    )
    code = _errors.render_error(exc, _args(output="json"), stream=buf)
    assert code == 3
    payload = json.loads(buf.getvalue())
    err = payload["error"]
    assert err["type"] == "AuthenticationError"
    assert err["status_code"] == 401
    assert err["code"] == "unauthorized"
    assert err["request_id"] == "req-1"
    assert err["url"] == "https://api.example.com/x"
    assert err["method"] == "GET"
    assert err["hint"].startswith("Run")
    assert err["details"] == {"error": "invalid_token"}
    assert "traceback" not in err  # no --debug


def test_json_mode_with_debug_includes_traceback():
    buf = io.StringIO()
    try:
        raise ValidationError("nope", details={"field": "x"})
    except ValidationError as exc:
        code = _errors.render_error(exc, _args(output="json", debug=True), stream=buf)
    assert code == 2
    payload = json.loads(buf.getvalue())
    assert "traceback" in payload["error"]
    assert isinstance(payload["error"]["traceback"], list)
    assert any("ValidationError" in line for line in payload["error"]["traceback"])


def test_json_mode_wraps_non_earl_exceptions():
    buf = io.StringIO()
    code = _errors.render_error(RuntimeError("unexpected"), _args(output="json"), stream=buf)
    assert code == 1
    payload = json.loads(buf.getvalue())
    assert payload["error"]["type"] == "RuntimeError"
    assert payload["error"]["message"] == "unexpected"


# ── Human mode ───────────────────────────────────────────────────────────────


def test_human_mode_prints_aligned_context():
    buf = io.StringIO()
    exc = AuthorizationError(
        "Access denied: token missing organization_id",
        request_id="req-42",
        url="https://api.example.com/pipelines",
        method="GET",
        hint="Run `earl auth login` with the right org.",
    )
    code = _errors.render_error(exc, _args(), stream=buf)
    out = buf.getvalue()
    assert code == 4
    assert "error: Access denied" in out
    assert "status: 403" in out
    assert "code: forbidden" in out
    assert "request: GET https://api.example.com/pipelines" in out
    assert "request id: req-42" in out
    assert "hint: Run" in out


def test_human_mode_hides_traceback_without_debug():
    buf = io.StringIO()
    try:
        raise ValidationError("bad argument")
    except ValidationError as exc:
        _errors.render_error(exc, _args(), stream=buf)
    assert "Traceback" not in buf.getvalue()


def test_human_mode_with_debug_shows_traceback_and_details():
    buf = io.StringIO()
    try:
        raise ServerError("boom", request_id="r", details={"trace": "abc"})
    except ServerError as exc:
        _errors.render_error(exc, _args(debug=True), stream=buf)
    out = buf.getvalue()
    assert "Traceback" in out
    assert "details:" in out
    assert "\"trace\": \"abc\"" in out


def test_rate_limit_error_displays_retry_after():
    buf = io.StringIO()
    exc = RateLimitError(retry_after=12.0)
    _errors.render_error(exc, _args(), stream=buf)
    out = buf.getvalue()
    assert "retry after: 12s" in out


def test_not_found_error_plays_nicely_with_extra_kwargs():
    """NotFoundError is constructed by _http with extra kwargs (request_id etc).
    Make sure that path doesn't regress the constructor."""
    exc = NotFoundError(
        "Patient",
        "p-99",
        request_id="r",
        url="https://api/patients/p-99",
        method="GET",
        hint="earl patients list",
    )
    assert exc.resource_type == "Patient"
    assert exc.resource_id == "p-99"
    assert exc.status_code == 404
    buf = io.StringIO()
    _errors.render_error(exc, _args(), stream=buf)
    assert "Patient not found: p-99" in buf.getvalue()
