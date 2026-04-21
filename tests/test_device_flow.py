"""Unit tests for earl_sdk.device_flow and Auth0Client device-grant plumbing."""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import pytest

from earl_sdk import auth_storage, device_flow
from earl_sdk.auth import Auth0Client
from earl_sdk.auth_storage import CachedToken
from earl_sdk.device_flow import (
    DeviceAuthorization,
    DeviceFlowError,
    poll_for_token,
    refresh_access_token,
    start_device_authorization,
)
from earl_sdk.exceptions import AuthenticationError


@pytest.fixture
def tmp_token_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolate the on-disk token cache for every test."""
    d = tmp_path / "tokens"
    monkeypatch.setattr(auth_storage, "TOKEN_CACHE_DIR", d, raising=False)
    return d


# ── start_device_authorization ───────────────────────────────────────────────


def _post_form_returns(status: int, body: dict) -> mock.MagicMock:
    m = mock.MagicMock(return_value=(status, body))
    return m


def test_start_device_authorization_returns_device_code(monkeypatch):
    fake = _post_form_returns(
        200,
        {
            "device_code": "DEV-CODE",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://auth0.example/activate",
            "verification_uri_complete": "https://auth0.example/activate?user_code=ABCD-EFGH",
            "expires_in": 600,
            "interval": 5,
        },
    )
    monkeypatch.setattr(device_flow, "_post_form", fake)

    result = start_device_authorization(
        domain="tenant.auth0.com",
        client_id="public-client-id",
        audience="https://api.example",
    )
    assert isinstance(result, DeviceAuthorization)
    assert result.device_code == "DEV-CODE"
    assert result.user_code == "ABCD-EFGH"
    assert result.interval == 5

    url, form = fake.call_args.args[0], fake.call_args.args[1]
    assert url == "https://tenant.auth0.com/oauth/device/code"
    assert form["client_id"] == "public-client-id"
    assert form["audience"] == "https://api.example"
    # offline_access is part of the default scope set.
    assert "offline_access" in form["scope"]


def test_start_device_authorization_passes_organization(monkeypatch):
    fake = _post_form_returns(
        200,
        {
            "device_code": "d",
            "user_code": "u",
            "verification_uri": "x",
            "verification_uri_complete": "x",
            "expires_in": 60,
            "interval": 5,
        },
    )
    monkeypatch.setattr(device_flow, "_post_form", fake)

    start_device_authorization(
        domain="t.auth0.com", client_id="c", audience="a", organization="org_xyz"
    )
    form = fake.call_args.args[1]
    assert form["organization"] == "org_xyz"


def test_start_device_authorization_rejects_error_response(monkeypatch):
    monkeypatch.setattr(
        device_flow,
        "_post_form",
        _post_form_returns(400, {"error": "invalid_client", "error_description": "nope"}),
    )
    with pytest.raises(DeviceFlowError, match="invalid_client"):
        start_device_authorization(domain="t", client_id="c", audience="a")


# ── poll_for_token ───────────────────────────────────────────────────────────


class _FakeClock:
    """Deterministic sleep/time helpers for polling tests."""

    def __init__(self) -> None:
        self.now = 1_000_000.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def time(self) -> float:
        return self.now


def test_poll_for_token_success_after_pending(monkeypatch):
    responses = [
        (200, {"error": "authorization_pending"}),
        (200, {"error": "authorization_pending"}),
        (
            200,
            {
                "access_token": "at",
                "refresh_token": "rt",
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": "openid offline_access",
            },
        ),
    ]
    fake = mock.MagicMock(side_effect=responses)
    monkeypatch.setattr(device_flow, "_post_form", fake)

    clock = _FakeClock()
    tok = poll_for_token(
        domain="t",
        client_id="c",
        device_code="dc",
        interval=5,
        expires_in=60,
        sleep=clock.sleep,
        now=clock.time,
    )
    assert tok.access_token == "at"
    assert tok.refresh_token == "rt"
    assert tok.expires_at > clock.now - 1
    assert clock.sleeps == [5, 5]


def test_poll_for_token_respects_slow_down(monkeypatch):
    responses = [
        (200, {"error": "slow_down"}),
        (
            200,
            {
                "access_token": "at",
                "refresh_token": "rt",
                "expires_in": 3600,
                "token_type": "Bearer",
            },
        ),
    ]
    monkeypatch.setattr(device_flow, "_post_form", mock.MagicMock(side_effect=responses))

    clock = _FakeClock()
    poll_for_token(
        domain="t",
        client_id="c",
        device_code="dc",
        interval=5,
        expires_in=60,
        sleep=clock.sleep,
        now=clock.time,
    )
    # RFC 8628: slow_down must bump the poll interval by at least 5s.
    assert clock.sleeps == [10]


def test_poll_for_token_raises_on_access_denied(monkeypatch):
    monkeypatch.setattr(
        device_flow,
        "_post_form",
        _post_form_returns(200, {"error": "access_denied"}),
    )
    clock = _FakeClock()
    with pytest.raises(AuthenticationError, match="denied"):
        poll_for_token(
            domain="t",
            client_id="c",
            device_code="dc",
            interval=5,
            expires_in=60,
            sleep=clock.sleep,
            now=clock.time,
        )


def test_poll_for_token_raises_on_expired(monkeypatch):
    monkeypatch.setattr(
        device_flow,
        "_post_form",
        _post_form_returns(200, {"error": "expired_token"}),
    )
    clock = _FakeClock()
    with pytest.raises(DeviceFlowError, match="expired"):
        poll_for_token(
            domain="t",
            client_id="c",
            device_code="dc",
            interval=5,
            expires_in=60,
            sleep=clock.sleep,
            now=clock.time,
        )


def test_poll_for_token_times_out_when_deadline_reached(monkeypatch):
    # Always return "pending" so we rely on the deadline to bail out.
    monkeypatch.setattr(
        device_flow,
        "_post_form",
        _post_form_returns(200, {"error": "authorization_pending"}),
    )
    clock = _FakeClock()
    with pytest.raises(DeviceFlowError, match="expired"):
        poll_for_token(
            domain="t",
            client_id="c",
            device_code="dc",
            interval=5,
            expires_in=10,  # < interval*3, so deadline hits fast
            sleep=clock.sleep,
            now=clock.time,
        )


# ── refresh_access_token ─────────────────────────────────────────────────────


def test_refresh_access_token_happy_path(monkeypatch):
    monkeypatch.setattr(
        device_flow,
        "_post_form",
        _post_form_returns(
            200,
            {
                "access_token": "new-at",
                "expires_in": 1800,
                "token_type": "Bearer",
                "refresh_token": "rotated-rt",
            },
        ),
    )
    tok = refresh_access_token(domain="t", client_id="c", refresh_token="old-rt")
    assert tok.access_token == "new-at"
    assert tok.refresh_token == "rotated-rt"  # Auth0 rotated it


def test_refresh_access_token_keeps_old_rt_when_not_rotated(monkeypatch):
    monkeypatch.setattr(
        device_flow,
        "_post_form",
        _post_form_returns(
            200,
            {"access_token": "new-at", "expires_in": 1800, "token_type": "Bearer"},
        ),
    )
    tok = refresh_access_token(domain="t", client_id="c", refresh_token="old-rt")
    assert tok.refresh_token == "old-rt"


def test_refresh_access_token_invalid_grant_surfaces_auth_error(monkeypatch):
    monkeypatch.setattr(
        device_flow,
        "_post_form",
        _post_form_returns(
            403,
            {"error": "invalid_grant", "error_description": "revoked"},
        ),
    )
    with pytest.raises(AuthenticationError, match="invalid_grant"):
        refresh_access_token(domain="t", client_id="c", refresh_token="bad")


# ── Auth0Client with auth_kind="device" ──────────────────────────────────────


def test_auth0client_device_uses_refresh_token(tmp_token_dir: Path, monkeypatch):
    def fake_refresh(**kwargs):
        assert kwargs["client_id"] == "cid"
        assert kwargs["refresh_token"] == "rt-initial"
        return device_flow.DeviceTokenResponse(
            access_token="fresh-at",
            expires_in=3600,
            token_type="Bearer",
            refresh_token="rt-rotated",
            scope="openid offline_access",
        ).with_expiry()

    monkeypatch.setattr(device_flow, "refresh_access_token", fake_refresh)

    client = Auth0Client(
        client_id="cid",
        client_secret="",
        organization="",
        domain="t",
        audience="a",
        auth_kind="device",
        refresh_token="rt-initial",
    )

    assert client.get_token() == "fresh-at"
    # The rotated refresh token should now be in the in-memory state.
    assert client._token_info is not None
    assert client._token_info.refresh_token == "rt-rotated"

    # And persisted to the per-auth_kind cache so the next process sees it.
    key = auth_storage.token_cache_key("cid", "a", "", "t", "device")
    cached = auth_storage.load_token(key)
    assert cached is not None and cached.refresh_token == "rt-rotated"


def test_auth0client_device_reads_refresh_token_from_cache(tmp_token_dir: Path, monkeypatch):
    key = auth_storage.token_cache_key("cid", "a", "", "t", "device")
    auth_storage.save_token(
        key,
        CachedToken(
            access_token="still-valid",
            token_type="Bearer",
            expires_at=time.time() + 3600,
            refresh_token="rt-from-cache",
        ),
    )

    # Should NOT hit the network because the cached access token is fresh.
    refresh = mock.MagicMock()
    monkeypatch.setattr(device_flow, "refresh_access_token", refresh)

    client = Auth0Client(
        client_id="cid",
        client_secret="",
        organization="",
        domain="t",
        audience="a",
        auth_kind="device",
    )
    assert client.get_token() == "still-valid"
    refresh.assert_not_called()


def test_auth0client_device_without_refresh_token_raises(tmp_token_dir: Path):
    client = Auth0Client(
        client_id="cid",
        client_secret="",
        organization="",
        domain="t",
        audience="a",
        auth_kind="device",
    )
    # No cached token, no refresh token supplied.
    with pytest.raises(AuthenticationError, match="earl auth login"):
        client.get_token()


def test_auth0client_m2m_still_works(tmp_token_dir: Path, monkeypatch):
    from earl_sdk.auth import TokenInfo

    def fake_fetch(self):  # type: ignore[no-untyped-def]
        return TokenInfo(access_token="m2m-at", token_type="Bearer", expires_at=time.time() + 3600)

    monkeypatch.setattr(Auth0Client, "_fetch_m2m_token", fake_fetch)

    client = Auth0Client(
        client_id="cid",
        client_secret="s",
        organization="",
        domain="t",
        audience="a",
    )
    assert client.get_token() == "m2m-at"


def test_auth0client_rejects_empty_m2m_secret(tmp_token_dir: Path):
    with pytest.raises(ValueError, match="client_secret"):
        Auth0Client(
            client_id="cid",
            client_secret="",
            organization="",
            domain="t",
            audience="a",
        )


# ── JWT + org extraction helpers ─────────────────────────────────────────────


def _encode_jwt(payload: dict) -> str:
    """Build an unsigned JWT-shaped string for decoding tests."""
    import base64
    import json as _json

    def b64(obj):
        raw = _json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{b64({'alg': 'RS256'})}.{b64(payload)}.notasignature"


def test_decode_jwt_payload_returns_dict():
    tok = _encode_jwt({"sub": "auth0|abc", "org_id": "org_1", "email": "a@b.c"})
    claims = device_flow.decode_jwt_payload(tok)
    assert claims["sub"] == "auth0|abc"
    assert claims["org_id"] == "org_1"


@pytest.mark.parametrize(
    "bad_input",
    ["", "not-a-jwt", "one.two", "x.not-base64.y", "a." + "!" * 10 + ".b"],
)
def test_decode_jwt_payload_returns_empty_on_garbage(bad_input: str):
    assert device_flow.decode_jwt_payload(bad_input) == {}


def test_extract_org_info_reads_standard_claims():
    tok = _encode_jwt(
        {
            "sub": "auth0|u",
            "email": "alice@ex.com",
            "org_id": "org_abc",
            "org_name": "Acme Co.",
        }
    )
    info = device_flow.extract_org_info(tok)
    assert info.has_org
    assert info.org_id == "org_abc"
    assert info.org_name == "Acme Co."
    assert info.email == "alice@ex.com"


def test_extract_org_info_reads_namespaced_claims():
    tok = _encode_jwt(
        {
            "sub": "auth0|u",
            "https://earl/org_id": "org_xyz",
            "https://earl/org_name": "Lumos",
        }
    )
    info = device_flow.extract_org_info(tok)
    assert info.org_id == "org_xyz"
    assert info.org_name == "Lumos"


def test_extract_org_info_handles_missing_claims():
    info = device_flow.extract_org_info(_encode_jwt({"sub": "auth0|u"}))
    assert not info.has_org
    assert info.org_id == ""
    assert info.subject == "auth0|u"


@pytest.mark.parametrize(
    "name,org_id,expected",
    [
        ("Hippocratic AI", "org_abc", "hippocratic-ai"),
        ("  Acme   Co.  ", "org_abc", "acme-co"),
        ("", "org_jcq35GHp0Qxu9n2n", "Qxu9n2n1"[:-1] or "Qxu9n2n" * 0 + "Qxu9n2n"),  # last 8
        ("", "", ""),
        ("x" * 100, "org_abc", "x" * 32),
    ],
)
def test_slugify_org(name: str, org_id: str, expected: str):
    # Correct the parametrize for the org_id-only case.
    if not name and org_id:
        expected = org_id[-8:]
    assert device_flow.slugify_org(name, org_id) == expected
