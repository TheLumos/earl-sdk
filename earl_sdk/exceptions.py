"""Exception classes for Earl SDK.

All SDK-layer errors inherit from :class:`EarlError` and carry a consistent,
machine-readable shape so the CLI can render them uniformly in either JSON
or human-readable form.

Fields (all optional unless noted):

- ``message``: Human-readable summary (required, positional).
- ``status_code``: HTTP status from the upstream API, when relevant.
- ``code``: Short machine-readable code. Prefers the server-provided
  ``error`` / ``code`` / ``type`` fields; falls back to a canonical tag
  (``"unauthorized"``, ``"forbidden"``, …).
- ``request_id``: Value of the ``x-request-id`` response header, if any.
  Lets users file bugs by pointing at a specific request in server logs.
- ``retry_after``: Seconds hint from ``Retry-After`` or the response body
  (set on :class:`RateLimitError`, but may be populated on 503s too).
- ``url`` / ``method``: The failing request, for debug/trace output.
- ``hint``: Actionable next step the user should take (``"run `earl auth
  login`"``, …). Populated by :mod:`earl_sdk._http` based on status + code.
- ``details``: The raw parsed response body (or best-effort dict), for
  ``--debug`` / JSON output.

The :meth:`to_dict` method produces the canonical error envelope consumed
by the CLI JSON renderer and by library callers that want structured errors.
"""

from __future__ import annotations

from typing import Any


class EarlError(Exception):
    """Base exception for all Earl SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
        url: str | None = None,
        method: str | None = None,
        hint: str | None = None,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retry_after = retry_after
        self.url = url
        self.method = method
        self.hint = hint
        self.details = details or {}

    def __str__(self) -> str:
        parts: list[str] = []
        if self.status_code is not None:
            parts.append(f"[{self.status_code}]")
        if self.code:
            parts.append(f"{self.code}:")
        parts.append(self.message)
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a stable JSON-friendly envelope."""
        out: dict[str, Any] = {
            "type": type(self).__name__,
            "message": self.message,
        }
        for name in ("status_code", "code", "request_id", "retry_after", "url", "method", "hint"):
            val = getattr(self, name)
            if val is not None:
                out[name] = val
        if self.details:
            out["details"] = self.details
        return out


class AuthenticationError(EarlError):
    """Raised when authentication fails (401)."""

    def __init__(self, message: str = "Authentication failed", **kwargs: Any):
        kwargs.setdefault("status_code", 401)
        kwargs.setdefault("code", "unauthorized")
        super().__init__(message, **kwargs)


class AuthorizationError(EarlError):
    """Raised when the user doesn't have permission (403)."""

    def __init__(self, message: str = "Access denied", **kwargs: Any):
        kwargs.setdefault("status_code", 403)
        kwargs.setdefault("code", "forbidden")
        super().__init__(message, **kwargs)


class NotFoundError(EarlError):
    """Raised when a resource is not found (404)."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        **kwargs: Any,
    ):
        message = f"{resource_type} not found: {resource_id}"
        details = kwargs.pop("details", None) or {}
        details.setdefault("resource_type", resource_type)
        details.setdefault("resource_id", resource_id)
        kwargs["details"] = details
        kwargs.setdefault("status_code", 404)
        kwargs.setdefault("code", "not_found")
        super().__init__(message, **kwargs)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ValidationError(EarlError):
    """Raised when request validation fails (400 / 422)."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        **kwargs: Any,
    ):
        kwargs.setdefault("status_code", 400)
        kwargs.setdefault("code", "invalid_request")
        super().__init__(message, **kwargs)
        self.field = field


class RateLimitError(EarlError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(self, retry_after: float | None = None, **kwargs: Any):
        message = "Rate limit exceeded"
        if retry_after:
            message += f". Retry after {int(retry_after)} seconds"
        details = kwargs.pop("details", None) or {}
        details.setdefault("retry_after", retry_after)
        kwargs["details"] = details
        kwargs.setdefault("status_code", 429)
        kwargs.setdefault("code", "rate_limited")
        kwargs["retry_after"] = retry_after
        super().__init__(message, **kwargs)


class ServerError(EarlError):
    """Raised when the server returns a 5xx."""

    def __init__(self, message: str = "Internal server error", **kwargs: Any):
        kwargs.setdefault("status_code", 500)
        kwargs.setdefault("code", "server_error")
        super().__init__(message, **kwargs)


class TransportError(EarlError):
    """Raised on connection / timeout / DNS failures (no HTTP status)."""

    def __init__(self, message: str, **kwargs: Any):
        kwargs.setdefault("code", "transport_error")
        super().__init__(message, **kwargs)


class SimulationError(EarlError):
    """Raised when a simulation fails."""

    def __init__(self, simulation_id: str, message: str, details: dict | None = None):
        details = dict(details or {})
        details.setdefault("simulation_id", simulation_id)
        super().__init__(
            f"Simulation {simulation_id} failed: {message}",
            code="simulation_failed",
            details=details,
        )
        self.simulation_id = simulation_id
