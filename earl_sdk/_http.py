"""Shared HTTP transport for the EARL SDK.

This module replaces the earlier ``urllib.request``-based transport in
:mod:`earl_sdk.api`. Responsibilities:

- **Connection pooling**: one lazily-initialised :class:`httpx.Client` per
  process, so repeated API calls reuse TCP + TLS connections.
- **HTTP/2** (opportunistic): enabled when the ``h2`` package is importable,
  transparently downgraded to HTTP/1.1 otherwise.
- **gzip / br / zstd** compression: advertised via ``Accept-Encoding`` and
  decoded by httpx automatically.
- **Timeouts**: structured ``httpx.Timeout`` (connect / read / write / pool).
- **Retries**: idempotent verbs (``GET``/``HEAD``/``OPTIONS``) retry with
  exponential backoff on :class:`httpx.TransportError` and 502/503/504.
  POST/PUT/PATCH/DELETE are **not** retried automatically — the caller
  decides, since they may have side effects.
- **Debug logging**: structured request/response lines at ``DEBUG`` level
  with ``Authorization`` redacted. Full request + response bodies are
  available at ``TRACE``-equivalent (``DEBUG`` with ``EARL_TRACE_HTTP=1``).
- **Error mapping**: non-2xx responses are turned into the appropriate
  :class:`earl_sdk.exceptions.EarlError` subclass, enriched with
  ``x-request-id``, ``Retry-After``, and an actionable hint.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from typing import Any

import httpx

from .exceptions import (
    AuthenticationError,
    AuthorizationError,
    EarlError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TransportError,
    ValidationError,
)

logger = logging.getLogger("earl_sdk.http")

# ---------------------------------------------------------------------------
# Shared client
# ---------------------------------------------------------------------------

_USER_AGENT = "earl-sdk-python/httpx"
_DEFAULT_TIMEOUT_S = 60.0
_DEFAULT_CONNECT_S = 10.0
_MAX_KEEPALIVE = 10
_MAX_CONNECTIONS = 20
_IDEMPOTENT = frozenset({"GET", "HEAD", "OPTIONS"})
_RETRY_STATUS = frozenset({502, 503, 504})
_MAX_RETRIES = 3


def _http2_available() -> bool:
    """Return ``True`` iff the ``h2`` package is importable."""
    try:
        import h2  # noqa: F401

        return True
    except ImportError:
        return False


_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _get_client() -> httpx.Client:
    """Return the process-wide :class:`httpx.Client`, creating it on demand."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            limits = httpx.Limits(
                max_keepalive_connections=_MAX_KEEPALIVE,
                max_connections=_MAX_CONNECTIONS,
            )
            timeout = httpx.Timeout(
                _DEFAULT_TIMEOUT_S,
                connect=_DEFAULT_CONNECT_S,
            )
            _client = httpx.Client(
                http2=_http2_available(),
                limits=limits,
                timeout=timeout,
                headers={
                    "User-Agent": _USER_AGENT,
                    "Accept": "application/json",
                    # httpx auto-decodes gzip; advertising br/zstd is a no-op
                    # when those codecs aren't installed, so just ask for what
                    # httpx can actually handle.
                    "Accept-Encoding": "gzip, deflate",
                },
                follow_redirects=False,
            )
    return _client


def close_shared_client() -> None:
    """Close the shared HTTP client. Exposed mostly for tests + clean shutdown."""
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            finally:
                _client = None


def set_shared_client(client: httpx.Client | None) -> None:
    """Inject a custom :class:`httpx.Client` (tests use this with ``MockTransport``)."""
    global _client
    with _client_lock:
        _client = client


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _redact_headers(headers: dict[str, str] | httpx.Headers) -> dict[str, str]:
    """Return a copy of ``headers`` with sensitive values hidden."""
    redacted: dict[str, str] = {}
    for k, v in dict(headers).items():
        lk = k.lower()
        if lk in {"authorization", "cookie", "x-api-key", "x-auth-token"}:
            redacted[k] = "***redacted***"
        else:
            redacted[k] = v
    return redacted


def _trace_http_bodies() -> bool:
    return os.getenv("EARL_TRACE_HTTP", "").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    timeout: float | None = None,
    retries: int | None = None,
) -> tuple[Any, httpx.Response]:
    """Perform a JSON request and return ``(parsed_body, response)``.

    Raises :class:`EarlError` (or a subclass) on non-2xx responses and on
    transport failures. The returned response object is exposed so callers
    can inspect headers (``x-request-id``, pagination, etc.) if needed.
    """
    method = method.upper()
    client = _get_client()
    max_retries = _MAX_RETRIES if retries is None else max(0, retries)
    attempt = 0

    while True:
        start = time.monotonic()
        try:
            response = client.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            if method in _IDEMPOTENT and attempt < max_retries:
                _sleep_backoff(attempt)
                attempt += 1
                continue
            raise TransportError(
                f"Request timed out after {timeout or _DEFAULT_TIMEOUT_S}s: {exc}",
                url=url,
                method=method,
                hint=(
                    "Network or server is slow. Retry, or pass a larger "
                    "`request_timeout=` to `EarlClient(...)`."
                ),
            ) from exc
        except httpx.TransportError as exc:
            if method in _IDEMPOTENT and attempt < max_retries:
                _sleep_backoff(attempt)
                attempt += 1
                continue
            raise TransportError(
                f"Could not reach {url}: {exc}",
                url=url,
                method=method,
                hint=(
                    "Check your network connection and that the EARL API URL "
                    "for this environment is correct (`earl auth test`)."
                ),
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000.0
        _log_response(method, url, response, elapsed_ms, headers or {}, json_body)

        # Retry transient 5xx on idempotent verbs.
        if (
            response.status_code in _RETRY_STATUS
            and method in _IDEMPOTENT
            and attempt < max_retries
        ):
            _sleep_backoff(attempt, response)
            attempt += 1
            continue

        if 200 <= response.status_code < 300:
            return _parse_json(response), response

        _raise_for_response(response, method=method, url=url)
        # _raise_for_response always raises
        raise AssertionError("unreachable")  # pragma: no cover


def _sleep_backoff(attempt: int, response: httpx.Response | None = None) -> None:
    """Sleep before retrying, honouring ``Retry-After`` when present."""
    delay = 0.0
    if response is not None:
        ra = response.headers.get("retry-after")
        if ra:
            try:
                delay = float(ra)
            except ValueError:
                delay = 0.0
    if delay <= 0:
        # Exponential backoff with jitter: 0.25, 0.5, 1.0, … seconds.
        delay = (2**attempt) * 0.25
        delay += random.uniform(0, delay * 0.25)
    logger.debug("retrying in %.2fs (attempt %d)", delay, attempt + 1)
    time.sleep(delay)


def _log_response(
    method: str,
    url: str,
    response: httpx.Response,
    elapsed_ms: float,
    req_headers: dict[str, str],
    req_body: Any,
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    req_id = response.headers.get("x-request-id") or response.headers.get("x-trace-id") or "-"
    logger.debug(
        "HTTP %s %s -> %d (%.0fms, %s bytes, req_id=%s, http%s)",
        method,
        url,
        response.status_code,
        elapsed_ms,
        len(response.content),
        req_id,
        response.http_version.replace("HTTP/", ""),
    )
    if _trace_http_bodies():
        logger.debug(
            "  request headers: %s",
            json.dumps(_redact_headers(req_headers), default=str),
        )
        if req_body is not None:
            logger.debug("  request body: %s", json.dumps(req_body, default=str)[:4000])
        logger.debug(
            "  response headers: %s",
            json.dumps(_redact_headers(response.headers), default=str),
        )
        text = response.text
        if text:
            logger.debug("  response body: %s", text[:4000])


def _parse_json(response: httpx.Response) -> Any:
    if not response.content:
        return {}
    ctype = response.headers.get("content-type", "")
    if "json" not in ctype.lower():
        # Tolerate endpoints that return JSON without the content-type.
        try:
            return response.json()
        except ValueError:
            return response.text
    return response.json()


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _raise_for_response(response: httpx.Response, *, method: str, url: str) -> None:
    """Map a non-2xx httpx response to the right :class:`EarlError` subclass."""
    status = response.status_code
    body = _safe_parse_error_body(response)
    message = _extract_message(body, response)
    code = _extract_code(body, status)
    request_id = response.headers.get("x-request-id") or response.headers.get("x-trace-id")
    retry_after = _parse_retry_after(response)
    hint = _hint_for(status, code, body)

    common = {
        "status_code": status,
        "code": code,
        "request_id": request_id,
        "retry_after": retry_after,
        "url": url,
        "method": method,
        "hint": hint,
        "details": body if isinstance(body, dict) else {"raw": body},
    }

    if status == 401:
        raise AuthenticationError(message, **common)
    if status == 403:
        raise AuthorizationError(message, **common)
    if status == 404:
        rtype = (body.get("resource_type") if isinstance(body, dict) else None) or "Resource"
        rid = (body.get("resource_id") if isinstance(body, dict) else None) or _infer_resource_id(
            url
        )
        raise NotFoundError(rtype, rid, **common)
    if status in (400, 422):
        raise ValidationError(message, **common)
    if status == 429:
        raise RateLimitError(retry_after=retry_after, **{k: v for k, v in common.items() if k != "retry_after"})
    if status >= 500:
        raise ServerError(message, **common)
    raise EarlError(message, **common)


def _safe_parse_error_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (ValueError, UnicodeDecodeError):
        try:
            return {"message": response.text}
        except Exception:  # noqa: BLE001
            return {}


def _extract_message(body: Any, response: httpx.Response) -> str:
    """Pull the most useful human message out of a mixed-shape error body."""
    if isinstance(body, dict):
        raw = body.get("message") or body.get("error") or body.get("detail")
        if isinstance(raw, list):
            # FastAPI validation errors: [{loc, msg, type}, ...]
            parts: list[str] = []
            for item in raw:
                if isinstance(item, dict):
                    loc = " -> ".join(str(x) for x in (item.get("loc") or []))
                    msg = item.get("msg", "")
                    parts.append(f"{loc}: {msg}" if loc else str(msg))
                else:
                    parts.append(str(item))
            return "; ".join(p for p in parts if p)
        if raw:
            return str(raw)
    if isinstance(body, str) and body:
        return body
    return f"HTTP {response.status_code} {response.reason_phrase or ''}".strip()


def _extract_code(body: Any, status: int) -> str:
    if isinstance(body, dict):
        raw = body.get("code") or body.get("error_code") or body.get("type")
        # Some APIs (incl. Auth0) return ``error`` as a code string. Prefer
        # that over the free-form message when it's an identifier-shaped token.
        err = body.get("error")
        if isinstance(err, str) and err.replace("_", "").replace("-", "").isalnum():
            raw = raw or err
        if raw:
            return str(raw)
    return {
        400: "invalid_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "invalid_request",
        429: "rate_limited",
        500: "server_error",
        502: "bad_gateway",
        503: "service_unavailable",
        504: "gateway_timeout",
    }.get(status, f"http_{status}")


def _parse_retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _infer_resource_id(url: str) -> str:
    # Fallback when the server doesn't surface resource_id in the body.
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail.split("?", 1)[0] or "unknown"


def _hint_for(status: int, code: str, body: Any) -> str | None:
    """Return a short actionable hint for the given error, or ``None``."""
    msg = ""
    if isinstance(body, dict):
        msg = " ".join(str(v) for v in body.values() if isinstance(v, str)).lower()
    if status == 401:
        if "token" in msg and ("expired" in msg or "invalid" in msg):
            return (
                "Your access token is expired or invalid. "
                "Humans: run `earl login` (browser) or "
                "`earl login --headless` (device flow). "
                "CI / automation: verify `EARL_CLIENT_ID`, `EARL_CLIENT_SECRET`, "
                "and `EARL_ORG_ID` are set (see `earl service-account create`)."
            )
        return (
            "Authentication failed. Humans: `earl login`. CI: make sure "
            "`EARL_CLIENT_ID` / `EARL_CLIENT_SECRET` / `EARL_ORG_ID` match a "
            "valid service account (see `earl service-account list`)."
        )
    if status == 403:
        if "organization" in msg or "org_id" in msg:
            return (
                "The active profile's organization doesn't match the API. "
                "Run `earl auth profile list` and check the org column, "
                "or re-run `earl auth login` to pick a different org."
            )
        return (
            "Access denied. Check that the active profile's organization has "
            "permission for this operation."
        )
    if status == 404:
        return "Verify the resource ID. Use `earl <resource> list` to enumerate available items."
    if status == 429:
        return "Throttle your requests or wait the suggested retry-after seconds."
    if 500 <= status < 600:
        return (
            "Upstream server error — often transient. "
            "Retry in a few seconds; if it persists, file a bug with the "
            "`request_id` from the error output."
        )
    if code == "invalid_request":
        return "Check the command arguments against `earl schema` or `--help`."
    return None
