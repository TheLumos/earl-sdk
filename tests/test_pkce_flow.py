"""Unit tests for :mod:`earl_sdk.pkce_flow`.

The PKCE flow is a security-critical piece of the CLI: it's the human login
path. We want to lock down three properties:

1. The PKCE verifier/challenge pair is RFC 7636-compliant.
2. The authorize URL carries every required query parameter (and no extra
   secrets).
3. The loopback server's CSRF guard rejects mismatched ``state`` on the
   callback — under no circumstances may we trade a foreign code for tokens.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import urllib.parse
import urllib.request

import pytest

from earl_sdk import pkce_flow
from earl_sdk.exceptions import AuthenticationError
from earl_sdk.pkce_flow import (
    DEFAULT_SCOPES,
    PKCEChallenge,
    PKCEFlowError,
    PKCETokenResponse,
    build_authorize_url,
    exchange_code_for_tokens,
    generate_pkce_pair,
    refresh_access_token,
    run_loopback_login,
)


# ── PKCE primitives ──────────────────────────────────────────────────────────


def test_generate_pkce_pair_rfc7636_s256() -> None:
    """Verifier length is 43+ URL-safe chars; challenge is SHA256(verifier)."""
    pair = generate_pkce_pair()
    assert pair.method == "S256"
    assert 43 <= len(pair.verifier) <= 128
    # Every char must be URL-safe (no padding, no +/).
    assert all(c.isalnum() or c in "-._~" for c in pair.verifier)

    digest = hashlib.sha256(pair.verifier.encode("ascii")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    assert pair.challenge == expected


def test_generate_pkce_pair_is_random() -> None:
    """Distinct calls MUST produce distinct verifiers."""
    pairs = {generate_pkce_pair().verifier for _ in range(5)}
    assert len(pairs) == 5


# ── Authorize URL builder ────────────────────────────────────────────────────


def _parse_qs(url: str) -> tuple[str, dict[str, str]]:
    parsed = urllib.parse.urlparse(url)
    qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}", qs


def test_build_authorize_url_has_required_params() -> None:
    challenge = PKCEChallenge(verifier="v", challenge="c", method="S256")
    url = build_authorize_url(
        domain="earl.eu.auth0.com",
        client_id="cid_native",
        audience="https://earl.api",
        redirect_uri="http://127.0.0.1:54321/callback",
        state="state_abc",
        challenge=challenge,
    )
    base, qs = _parse_qs(url)
    assert base == "https://earl.eu.auth0.com/authorize"
    assert qs["response_type"] == "code"
    assert qs["client_id"] == "cid_native"
    assert qs["audience"] == "https://earl.api"
    assert qs["redirect_uri"] == "http://127.0.0.1:54321/callback"
    assert qs["state"] == "state_abc"
    assert qs["code_challenge"] == "c"
    assert qs["code_challenge_method"] == "S256"
    # Default scope set.
    assert set(qs["scope"].split()) == set(DEFAULT_SCOPES)


def test_build_authorize_url_includes_organization_when_set() -> None:
    challenge = PKCEChallenge(verifier="v", challenge="c", method="S256")
    url = build_authorize_url(
        domain="x",
        client_id="cid",
        audience="aud",
        redirect_uri="http://127.0.0.1:1/callback",
        state="s",
        challenge=challenge,
        organization="org_abc",
    )
    _, qs = _parse_qs(url)
    assert qs["organization"] == "org_abc"


def test_build_authorize_url_rejects_missing_inputs() -> None:
    challenge = PKCEChallenge(verifier="v", challenge="c", method="S256")
    with pytest.raises(ValueError):
        build_authorize_url(
            domain="",
            client_id="cid",
            audience="aud",
            redirect_uri="u",
            state="s",
            challenge=challenge,
        )
    with pytest.raises(ValueError):
        build_authorize_url(
            domain="d",
            client_id="",
            audience="aud",
            redirect_uri="u",
            state="s",
            challenge=challenge,
        )
    with pytest.raises(ValueError):
        build_authorize_url(
            domain="d",
            client_id="cid",
            audience="",
            redirect_uri="u",
            state="s",
            challenge=challenge,
        )


# ── Token exchange / refresh ────────────────────────────────────────────────


def _stub_post_token(monkeypatch: pytest.MonkeyPatch, status: int, body: dict) -> list[dict]:
    """Patch ``_post_token`` to return a fixed tuple and record every call."""
    calls: list[dict] = []

    def fake(*, domain: str, form: dict, timeout: float = 15.0) -> tuple[int, dict]:
        calls.append({"domain": domain, "form": dict(form), "timeout": timeout})
        return status, body

    monkeypatch.setattr(pkce_flow, "_post_token", fake)
    return calls


def test_exchange_code_for_tokens_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_post_token(
        monkeypatch,
        200,
        {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 120,
            "scope": "openid profile earl:read",
            "id_token": "idt",
        },
    )
    tok = exchange_code_for_tokens(
        domain="d", client_id="cid", code="thecode", redirect_uri="u", verifier="v"
    )
    assert isinstance(tok, PKCETokenResponse)
    assert tok.access_token == "at"
    assert tok.refresh_token == "rt"
    assert tok.expires_in == 120
    assert tok.expires_at > time.time()
    # Form fields are exactly what RFC 6749 §4.1.3 + PKCE require.
    assert calls[0]["form"] == {
        "grant_type": "authorization_code",
        "client_id": "cid",
        "code": "thecode",
        "redirect_uri": "u",
        "code_verifier": "v",
    }


def test_exchange_code_for_tokens_auth0_error_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_post_token(
        monkeypatch,
        403,
        {"error": "invalid_grant", "error_description": "code expired"},
    )
    with pytest.raises(AuthenticationError) as exc:
        exchange_code_for_tokens(
            domain="d", client_id="cid", code="c", redirect_uri="u", verifier="v"
        )
    assert "invalid_grant" in str(exc.value)
    assert "code expired" in str(exc.value)


def test_refresh_access_token_preserves_refresh_token_if_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Auth0 doesn't rotate the refresh token, we keep the old one."""
    _stub_post_token(
        monkeypatch,
        200,
        {"access_token": "new_at", "expires_in": 60},
    )
    tok = refresh_access_token(domain="d", client_id="cid", refresh_token="old_rt")
    assert tok.access_token == "new_at"
    assert tok.refresh_token == "old_rt"


def test_refresh_access_token_rejects_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_post_token(
        monkeypatch,
        401,
        {"error": "invalid_refresh_token", "error_description": "expired"},
    )
    with pytest.raises(AuthenticationError):
        refresh_access_token(domain="d", client_id="cid", refresh_token="rt")


# ── Loopback flow integration (state/CSRF guard) ─────────────────────────────


def _post_to_callback(url: str) -> None:
    """Hit the loopback server with a GET request — don't care about response."""
    try:
        urllib.request.urlopen(url, timeout=2)
    except Exception:
        # 400s for error pages are fine; we only need the request to reach
        # the handler so it can flip the callback event.
        pass


def test_run_loopback_login_rejects_state_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callback with wrong state MUST NOT lead to a token exchange.

    This is the core CSRF guard — we simulate a browser that returns the
    *wrong* state value. The server should still accept the request (so the
    user gets an error page), but the function must raise instead of
    exchanging the code.
    """
    captured_url: dict[str, str] = {}

    def fake_browser(url: str) -> bool:
        captured_url["url"] = url
        parsed = urllib.parse.urlparse(url)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        redirect = qs["redirect_uri"]
        # Race the browser: reply with a tampered state.
        t = threading.Thread(
            target=_post_to_callback,
            args=(f"{redirect}?code=attackercode&state=WRONG",),
            daemon=True,
        )
        t.start()
        return True

    # The token endpoint must never be called when state mismatches.
    def boom(**_kw) -> tuple[int, dict]:
        raise AssertionError("_post_token called despite state mismatch")

    monkeypatch.setattr(pkce_flow, "_post_token", boom)

    with pytest.raises(PKCEFlowError, match="State mismatch"):
        run_loopback_login(
            domain="earl.eu.auth0.com",
            client_id="cid_native",
            audience="https://earl.api",
            open_browser=fake_browser,
            callback_timeout=5.0,
        )
    assert "authorize" in captured_url["url"]


def test_run_loopback_login_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid state → code exchange → PKCETokenResponse returned."""
    calls: list[dict] = []

    def fake_post(*, domain: str, form: dict, timeout: float = 15.0) -> tuple[int, dict]:
        calls.append(form)
        return 200, {
            "access_token": "at",
            "refresh_token": "rt",
            "expires_in": 60,
            "scope": "openid",
        }

    monkeypatch.setattr(pkce_flow, "_post_token", fake_post)

    def fake_browser(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        redirect = qs["redirect_uri"]
        state = qs["state"]
        # Respect the state Auth0 would echo back.
        t = threading.Thread(
            target=_post_to_callback,
            args=(f"{redirect}?code=thecode&state={state}",),
            daemon=True,
        )
        t.start()
        return True

    tok = run_loopback_login(
        domain="earl.eu.auth0.com",
        client_id="cid_native",
        audience="https://earl.api",
        open_browser=fake_browser,
        callback_timeout=5.0,
    )
    assert tok.access_token == "at"
    assert tok.refresh_token == "rt"
    assert calls and calls[0]["grant_type"] == "authorization_code"
    assert calls[0]["code"] == "thecode"
    # The verifier field must be present; we don't care about the exact
    # value, only that it's a non-empty string.
    assert calls[0]["code_verifier"]


def test_run_loopback_login_propagates_error_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``error=access_denied`` in the callback → AuthenticationError."""
    def fake_browser(url: str) -> bool:
        parsed = urllib.parse.urlparse(url)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        redirect = qs["redirect_uri"]
        state = qs["state"]
        t = threading.Thread(
            target=_post_to_callback,
            args=(
                f"{redirect}?error=access_denied"
                f"&error_description=user+cancelled&state={state}",
            ),
            daemon=True,
        )
        t.start()
        return True

    with pytest.raises(AuthenticationError, match="access_denied"):
        run_loopback_login(
            domain="d.auth0.com",
            client_id="cid",
            audience="aud",
            open_browser=fake_browser,
            callback_timeout=5.0,
        )


def test_run_loopback_login_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """If no callback arrives, the function times out with a helpful error."""
    def never_opens(_url: str) -> bool:
        return True

    monkeypatch.setattr(pkce_flow, "_DEFAULT_CALLBACK_TIMEOUT", 0.5)
    with pytest.raises(PKCEFlowError, match="Timed out"):
        run_loopback_login(
            domain="d.auth0.com",
            client_id="cid",
            audience="aud",
            open_browser=never_opens,
            callback_timeout=0.25,
        )


# ── Smoke: PKCETokenResponse.with_expiry ─────────────────────────────────────


def test_pkce_token_response_expiry_is_relative_to_now() -> None:
    resp = PKCETokenResponse(access_token="at", expires_in=30)
    assert resp.expires_at == 0.0
    with_exp = resp.with_expiry()
    assert with_exp.expires_at >= time.time()
    assert with_exp.expires_at < time.time() + 31
