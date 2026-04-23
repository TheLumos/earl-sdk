"""OAuth 2.0 Authorization Code + PKCE (loopback redirect) helpers.

Implements RFC 7636-compliant Authorization Code flow with PKCE for native /
desktop clients, using a loopback HTTP server per RFC 8252 §7.3. This is the
primary interactive login path for humans on a developer workstation — the
browser opens Universal Login, which (thanks to Auth0 Organizations) handles
the org picker for multi-org users entirely server-side.

The two Auth0 endpoints we hit:

1. ``GET  /authorize`` — user-facing; rendered in their browser.
2. ``POST /oauth/token`` — back-channel exchange ``code`` → tokens.

Intentionally has zero dependency on :mod:`earl_sdk.auth` / :mod:`earl_sdk.api`,
so unit tests can drive it in isolation.
"""

from __future__ import annotations

import base64
import hashlib
import html
import http.server
import json
import logging
import os
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .exceptions import AuthenticationError, EarlError

logger = logging.getLogger("earl_sdk.pkce_flow")

DEFAULT_SCOPES: tuple[str, ...] = (
    "openid",
    "profile",
    "email",
    "offline_access",
)

CALLBACK_PATH = "/callback"

# Auth0 requires exact callback-URL matching on its per-app allowlist — it
# does NOT apply the RFC 8252 §7.3 "any port on loopback" grace. We therefore
# register a fixed set of loopback ports in the Auth0 Native-app settings and
# have the loopback helper pick the first one that is free. Keep this list in
# sync with the ``callbacks`` array on the EARL Cli Native apps in Auth0.
REGISTERED_LOOPBACK_PORTS: tuple[int, ...] = (
    8484, 8485, 8486, 8487, 8488, 8489, 8490, 8491, 8492, 8493,
    8494, 8495, 8496, 8497, 8498, 8499,
)

# How long we wait for the user to complete the browser login before giving up.
_DEFAULT_CALLBACK_TIMEOUT = 300.0  # 5 minutes


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PKCEChallenge:
    """S256 code verifier + challenge pair."""

    verifier: str
    challenge: str
    method: str = "S256"


@dataclass
class PKCETokenResponse:
    """Token response from ``POST /oauth/token``.

    The shape intentionally mirrors :class:`earl_sdk.device_flow.DeviceTokenResponse`
    so both flows can feed the same ``Auth0Client`` token cache.
    """

    access_token: str
    expires_in: int
    token_type: str = "Bearer"
    refresh_token: str | None = None
    scope: str = ""
    id_token: str | None = None
    expires_at: float = field(default_factory=lambda: 0.0)

    def with_expiry(self) -> "PKCETokenResponse":
        return PKCETokenResponse(
            access_token=self.access_token,
            expires_in=self.expires_in,
            token_type=self.token_type,
            refresh_token=self.refresh_token,
            scope=self.scope,
            id_token=self.id_token,
            expires_at=time.time() + max(0, self.expires_in),
        )


class PKCEFlowError(EarlError):
    """Raised when the browser flow fails for non-recoverable reasons."""


# ── PKCE primitives ──────────────────────────────────────────────────────────


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_pkce_pair() -> PKCEChallenge:
    """Generate an S256 code_verifier / code_challenge pair (RFC 7636)."""
    # RFC 7636 requires 43-128 chars of [A-Z,a-z,0-9,-._~]; 32 random bytes
    # base64url-encoded gives 43 URL-safe chars, which is the minimum length.
    verifier = _b64url(secrets.token_bytes(32))
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = _b64url(digest)
    return PKCEChallenge(verifier=verifier, challenge=challenge, method="S256")


# ── Loopback HTTP server ─────────────────────────────────────────────────────


class _CallbackResult:
    """Captures the callback from the browser for the main thread to consume."""

    def __init__(self) -> None:
        self.code: str | None = None
        self.state: str | None = None
        self.error: str | None = None
        self.error_description: str | None = None
        self.event = threading.Event()


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Single-shot request handler that parses ``?code=…&state=…``.

    We accept **any** path — the browser only hits us once and following
    strict path matching breaks when Auth0 (rarely) appends trailing
    artefacts. Instead we key off the presence of ``code`` or ``error``.
    """

    # Silence BaseHTTPRequestHandler's default stderr access log.
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        logger.debug("pkce loopback: " + format, *args)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

        result: _CallbackResult = self.server.result  # type: ignore[attr-defined]

        if "code" in params or "error" in params:
            result.code = params.get("code")
            result.state = params.get("state")
            result.error = params.get("error")
            result.error_description = params.get("error_description")

            if result.error:
                body = _error_page(
                    title="Login failed",
                    message=(
                        result.error_description
                        or result.error
                        or "Auth0 rejected the login."
                    ),
                )
                self.send_response(400)
            else:
                body = _success_page()
                self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # Private cache only; never index.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            result.event.set()
            return

        # Anything else — just 404; the browser hitting favicon.ico or the
        # user reloading the page shouldn't resolve the flow.
        self.send_response(404)
        self.end_headers()


def _success_page() -> bytes:
    return (
        b"<!doctype html><html><head><meta charset='utf-8'>"
        b"<title>Earl login complete</title>"
        b"<style>body{font:16px/1.4 system-ui,sans-serif;max-width:40em;"
        b"margin:4em auto;padding:0 1em;color:#111}h1{font-weight:600}"
        b"code{background:#f3f3f3;padding:.1em .3em;border-radius:3px}</style>"
        b"</head><body><h1>You're signed in.</h1>"
        b"<p>You can close this tab and return to the terminal.</p>"
        b"</body></html>"
    )


def _error_page(*, title: str, message: str) -> bytes:
    # HTML-escape both fields so a malicious Auth0 ``error_description``
    # (attacker-controlled via the query string) can't inject script tags
    # into the page the user sees. The loopback server only listens on
    # 127.0.0.1 so the blast radius is narrow, but raw interpolation of
    # untrusted strings into an HTML body is bad hygiene regardless.
    title_b = html.escape(title, quote=True).encode("utf-8", "replace")
    msg_b = html.escape(message, quote=True).encode("utf-8", "replace")
    return (
        b"<!doctype html><html><head><meta charset='utf-8'><title>"
        + title_b
        + b"</title><style>body{font:16px/1.4 system-ui,sans-serif;max-width:"
        b"40em;margin:4em auto;padding:0 1em;color:#111}h1{font-weight:600;"
        b"color:#b00020}</style></head><body><h1>"
        + title_b
        + b"</h1><p>"
        + msg_b
        + b"</p><p>You can close this tab.</p></body></html>"
    )


def _pick_free_port(
    candidates: tuple[int, ...] = REGISTERED_LOOPBACK_PORTS,
) -> int:
    """Return the first port in ``candidates`` that can bind on 127.0.0.1.

    Falls back to an OS-assigned ephemeral port if every candidate is busy —
    callers are expected to fail fast when Auth0 rejects the unregistered
    redirect_uri, with a clear message pointing back at the registered set.
    """
    for p in candidates:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", p))
                return p
        except OSError:
            continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── High-level API ───────────────────────────────────────────────────────────


def build_authorize_url(
    *,
    domain: str,
    client_id: str,
    audience: str,
    redirect_uri: str,
    state: str,
    challenge: PKCEChallenge,
    scopes: list[str] | tuple[str, ...] | None = None,
    organization: str | None = None,
    extra_params: dict[str, str] | None = None,
) -> str:
    """Build the ``/authorize`` URL the browser should open."""
    if not domain:
        raise ValueError("Auth0 domain is required")
    if not client_id:
        raise ValueError("Auth0 client_id is required")
    if not audience:
        raise ValueError("audience is required")

    q: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "audience": audience,
        "scope": " ".join(scopes or DEFAULT_SCOPES),
        "state": state,
        "code_challenge": challenge.challenge,
        "code_challenge_method": challenge.method,
    }
    if organization:
        q["organization"] = organization
    if extra_params:
        q.update(extra_params)
    return f"https://{domain}/authorize?" + urllib.parse.urlencode(q)


def _post_token(
    *,
    domain: str,
    form: dict[str, str],
    timeout: float = 15.0,
) -> tuple[int, dict]:
    url = f"https://{domain}/oauth/token"
    body = urllib.parse.urlencode(form).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "User-Agent": "EarlSDK-PKCE/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8") if e.fp is not None else ""
        try:
            data = json.loads(raw) if raw else {"error": "http_error"}
        except json.JSONDecodeError:
            data = {"error": "http_error", "error_description": raw or str(e)}
        return e.code, data
    except urllib.error.URLError as e:
        raise EarlError(f"Cannot reach Auth0 at {url}: {e}") from e
    except (TimeoutError, ConnectionError, OSError) as e:
        raise EarlError(f"Network error talking to Auth0: {e}") from e


def exchange_code_for_tokens(
    *,
    domain: str,
    client_id: str,
    code: str,
    redirect_uri: str,
    verifier: str,
    timeout: float = 15.0,
) -> PKCETokenResponse:
    """POST /oauth/token with ``grant_type=authorization_code``."""
    form = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }
    status, data = _post_token(domain=domain, form=form, timeout=timeout)
    if status != 200:
        err = data.get("error", "unknown_error")
        desc = data.get("error_description") or data.get("message") or ""
        raise AuthenticationError(
            f"Auth0 rejected code exchange ({err}): {desc}".rstrip(": "),
            status_code=status,
            code=str(err),
        )
    try:
        return PKCETokenResponse(
            access_token=data["access_token"],
            expires_in=int(data.get("expires_in", 3600)),
            token_type=data.get("token_type", "Bearer"),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope", ""),
            id_token=data.get("id_token"),
        ).with_expiry()
    except KeyError as e:
        raise AuthenticationError(
            f"Auth0 token response missing required field: {e}"
        ) from e


def run_loopback_login(
    *,
    domain: str,
    client_id: str,
    audience: str,
    scopes: list[str] | tuple[str, ...] | None = None,
    organization: str | None = None,
    open_browser: Callable[[str], bool] | None = None,
    port: int | None = None,
    callback_timeout: float = _DEFAULT_CALLBACK_TIMEOUT,
    print_instructions: Callable[[str], None] | None = None,
) -> PKCETokenResponse:
    """Run the full PKCE loopback login and return the token response.

    Steps:
    1. Pick an ephemeral localhost port (unless ``port`` provided) and bring
       up a one-shot HTTP server.
    2. Build a signed ``/authorize`` URL with PKCE challenge + random state.
    3. Open the URL in the default browser (or print it if headless).
    4. Wait for the callback, verify ``state``, exchange ``code`` for tokens.
    """
    chosen_port = port if port is not None else _pick_free_port()
    redirect_uri = f"http://127.0.0.1:{chosen_port}{CALLBACK_PATH}"
    challenge = generate_pkce_pair()
    state = _b64url(secrets.token_bytes(16))

    authorize_url = build_authorize_url(
        domain=domain,
        client_id=client_id,
        audience=audience,
        redirect_uri=redirect_uri,
        state=state,
        challenge=challenge,
        scopes=scopes,
        organization=organization,
    )

    # One-shot HTTPServer bound to the ephemeral port.
    server = http.server.HTTPServer(("127.0.0.1", chosen_port), _CallbackHandler)
    server.result = _CallbackResult()  # type: ignore[attr-defined]
    # Avoid holding the port open after the handler returns.
    server.timeout = callback_timeout

    serve_thread = threading.Thread(
        target=server.serve_forever, name="pkce-loopback", daemon=True
    )
    serve_thread.start()
    try:
        opener = open_browser or webbrowser.open
        # If the browser call fails, we still want the user to be able to
        # paste the URL manually.
        opened = False
        try:
            opened = bool(opener(authorize_url))
        except Exception as e:  # pragma: no cover - depends on env
            logger.warning("webbrowser.open failed: %s", e)

        if not opened and print_instructions is not None:
            print_instructions(authorize_url)
        elif not opened:
            logger.info("Open this URL to continue: %s", authorize_url)

        if not server.result.event.wait(timeout=callback_timeout):
            raise PKCEFlowError(
                "Timed out waiting for browser login. "
                "Close any stale tabs and run `earl login` again."
            )
    finally:
        server.shutdown()
        server.server_close()

    result: _CallbackResult = server.result  # type: ignore[attr-defined]
    if result.error:
        raise AuthenticationError(
            f"Auth0 returned error ({result.error}): "
            f"{result.error_description or ''}".strip(": "),
            code=str(result.error),
        )
    if not result.code:
        raise PKCEFlowError("Callback arrived without a code parameter.")
    if result.state != state:
        # Under no circumstances should we accept a callback whose state
        # doesn't match — this is the CSRF guard for the whole flow.
        raise PKCEFlowError(
            "State mismatch on loopback callback; aborting to prevent CSRF."
        )

    return exchange_code_for_tokens(
        domain=domain,
        client_id=client_id,
        code=result.code,
        redirect_uri=redirect_uri,
        verifier=challenge.verifier,
    )


def refresh_access_token(
    *,
    domain: str,
    client_id: str,
    refresh_token: str,
    scope: str | None = None,
    timeout: float = 15.0,
) -> PKCETokenResponse:
    """Refresh an access token issued by the PKCE flow.

    Mirrors :func:`earl_sdk.device_flow.refresh_access_token` so the
    ``Auth0Client`` refresh path can treat both flows uniformly.
    """
    form: dict[str, str] = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }
    if scope:
        form["scope"] = scope
    status, data = _post_token(domain=domain, form=form, timeout=timeout)
    if status != 200:
        err = data.get("error", "unknown_error")
        desc = data.get("error_description") or data.get("message") or ""
        raise AuthenticationError(
            f"Auth0 refresh failed ({err}): {desc}".rstrip(": "),
            status_code=status,
            code=str(err),
        )
    try:
        return PKCETokenResponse(
            access_token=data["access_token"],
            expires_in=int(data.get("expires_in", 3600)),
            token_type=data.get("token_type", "Bearer"),
            refresh_token=data.get("refresh_token") or refresh_token,
            scope=data.get("scope", ""),
            id_token=data.get("id_token"),
        ).with_expiry()
    except KeyError as e:
        raise AuthenticationError(
            f"Auth0 refresh response missing required field: {e}"
        ) from e


__all__ = [
    "PKCEChallenge",
    "PKCETokenResponse",
    "PKCEFlowError",
    "DEFAULT_SCOPES",
    "CALLBACK_PATH",
    "generate_pkce_pair",
    "build_authorize_url",
    "exchange_code_for_tokens",
    "refresh_access_token",
    "run_loopback_login",
]
