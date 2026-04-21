"""CLI tests for `earl auth login`, `earl auth logout`, and device-clients."""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from earl_sdk import auth_storage, device_flow
from earl_sdk.cli.app import main as cli_main
from earl_sdk.interactive.storage import config_store as storage


@pytest.fixture
def isolated_earl_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point config + token dirs at a tmp path so tests never touch ~/.earl."""
    home = tmp_path / ".earl"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "EARL_DIR", home, raising=False)
    monkeypatch.setattr(storage, "CONFIG_PATH", home / "config.json", raising=False)
    monkeypatch.setattr(auth_storage, "EARL_DIR", home, raising=False)
    monkeypatch.setattr(auth_storage, "TOKEN_CACHE_DIR", home / "tokens", raising=False)
    # Keyring would write outside the tmp dir, so force the file backend.
    monkeypatch.setenv("EARL_SECRET_BACKEND", "file")
    monkeypatch.setattr(auth_storage, "_keyring_module", None, raising=False)
    monkeypatch.setattr(auth_storage, "_keyring_checked", True, raising=False)
    return home


def _run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(argv)) or 0
    except SystemExit as e:
        code = int(e.code or 0)
    return code, out.getvalue(), err.getvalue()


# ── earl auth device-clients ─────────────────────────────────────────────────


def test_device_clients_set_list_clear(isolated_earl_home: Path):
    code, _, _ = _run_cli("auth", "device-clients", "set", "--env", "test", "abc123")
    assert code == 0

    code, out, _ = _run_cli("--json", "auth", "device-clients", "list")
    assert code == 0
    assert json.loads(out)["test"] == "abc123"

    code, _, _ = _run_cli("auth", "device-clients", "clear", "--env", "test")
    assert code == 0

    code, out, _ = _run_cli("--json", "auth", "device-clients", "list")
    assert code == 0
    assert json.loads(out) == {}


def test_device_clients_set_dry_run_does_not_persist(isolated_earl_home: Path):
    code, out, _ = _run_cli(
        "--json", "--dry-run", "auth", "device-clients", "set", "--env", "prod", "ID"
    )
    assert code == 0
    assert json.loads(out)["action"] == "auth.device-clients.set"

    # Should still be empty after the dry-run.
    code, out, _ = _run_cli("--json", "auth", "device-clients", "list")
    assert json.loads(out) == {}


# ── earl auth login (dry-run) ────────────────────────────────────────────────


def test_auth_login_dry_run_reports_payload_without_network(isolated_earl_home: Path):
    code, out, _ = _run_cli(
        "--json",
        "--dry-run",
        "auth",
        "login",
        "--env",
        "test",
        "--device-client-id",
        "PUB-CID",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["action"] == "auth.login"
    assert payload["payload"]["env"] == "test"
    assert payload["payload"]["client_id"] == "PUB-CID"
    assert "offline_access" in payload["payload"]["scopes"]
    # No profile should have been created.
    assert not (isolated_earl_home / "config.json").exists()


def test_auth_login_fails_without_device_client_id(isolated_earl_home: Path):
    # No --device-client-id, no stored value, no env var.
    code, _, err = _run_cli("auth", "login", "--env", "dev")
    assert code != 0
    assert "device-flow client_id" in err.lower() or "device-clients set" in err.lower()


# ── earl auth login (full, mocked device flow) ───────────────────────────────


def test_auth_login_full_flow_persists_profile_and_token(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    # Register the public client id so `earl auth login` picks it up.
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("test", "PUB-CID")

    # Stub out the two HTTP calls.
    fake_auth = device_flow.DeviceAuthorization(
        device_code="DEV-CODE",
        user_code="ABCD-EFGH",
        verification_uri="https://t.auth0.com/activate",
        verification_uri_complete="https://t.auth0.com/activate?user_code=ABCD-EFGH",
        expires_in=600,
        interval=5,
    )
    monkeypatch.setattr(
        "earl_sdk.cli.app.start_device_authorization",
        lambda **kw: fake_auth,
        raising=False,
    )
    # app.py imports these lazily inside the handler, so patch via device_flow.
    monkeypatch.setattr(device_flow, "start_device_authorization", lambda **kw: fake_auth)

    def fake_poll(**kw):
        return device_flow.DeviceTokenResponse(
            access_token="AT-1",
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-1",
            scope="openid offline_access",
        ).with_expiry()

    monkeypatch.setattr(device_flow, "poll_for_token", fake_poll)

    # Stop the CLI from opening a real browser.
    monkeypatch.setattr("webbrowser.open", lambda *a, **kw: True)

    code, out, err = _run_cli("auth", "login", "--env", "test", "--no-browser")
    assert code == 0, err
    assert "Code:  ABCD-EFGH" in err
    # `_say` writes to stdout, the progress banner goes to stderr.
    assert "Signed in" in out

    # Profile was saved with auth_kind=device and marked active.
    cs2 = storage.ConfigStore(isolated_earl_home / "config.json")
    cfg = cs2.load()
    assert cfg.active_profile == "device-test"
    prof = cfg.profiles["device-test"]
    assert prof.auth_kind == "device"
    assert prof.client_id == "PUB-CID"
    assert prof.environment == "test"
    # Refresh token must be retrievable (file backend → base64 in client_secret).
    assert prof.refresh_token_clear() == "RT-1"

    # Access token cache is seeded so subsequent API calls skip Auth0.
    cache_key = auth_storage.token_cache_key(
        "PUB-CID",
        "https://earl-api.thelumos.xyz",
        "",
        "dev-f4675lf8h3k0i3me.us.auth0.com",
        "device",
    )
    cached = auth_storage.load_token(cache_key)
    assert cached is not None
    assert cached.access_token == "AT-1"
    assert cached.refresh_token == "RT-1"


def _make_fake_access_token(claims: dict) -> str:
    """Build a JWT with the given payload (header + signature are unverified)."""
    import base64
    import json as _json

    def b64(obj):
        raw = _json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{b64({'alg': 'RS256'})}.{b64(claims)}.sig-not-verified"


def test_auth_login_extracts_org_from_access_token(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """Happy path: user belongs to N orgs, Auth0 picks one in the browser,
    the returned token carries org_id/org_name, and the CLI names the profile
    after the resolved org without the user typing --organization."""
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("dev", "PUB-CID")

    fake_auth = device_flow.DeviceAuthorization(
        device_code="DEV-CODE",
        user_code="ABCD-EFGH",
        verification_uri="https://t.auth0.com/activate",
        verification_uri_complete="https://t.auth0.com/activate?user_code=ABCD-EFGH",
        expires_in=600,
        interval=5,
    )
    monkeypatch.setattr(device_flow, "start_device_authorization", lambda **kw: fake_auth)

    token_claims = {
        "sub": "auth0|abc",
        "email": "alice@hippocratic.com",
        "org_id": "org_jcq35GHp0Qxu9n2n",
        "org_name": "Hippocratic AI",
    }
    at = _make_fake_access_token(token_claims)
    monkeypatch.setattr(
        device_flow,
        "poll_for_token",
        lambda **kw: device_flow.DeviceTokenResponse(
            access_token=at,
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-1",
            scope="openid offline_access",
        ).with_expiry(),
    )
    monkeypatch.setattr("webbrowser.open", lambda *a, **kw: True)

    code, out, err = _run_cli("auth", "login", "--env", "dev", "--no-browser")
    assert code == 0, err
    # Profile slug comes from org_name, not the raw org_id.
    assert "Signed in to env=dev (alice@hippocratic.com)" in out
    assert "Hippocratic AI" in out

    cfg = storage.ConfigStore(isolated_earl_home / "config.json").load()
    assert "device-dev-hippocratic-ai" in cfg.profiles
    prof = cfg.profiles["device-dev-hippocratic-ai"]
    assert prof.organization == "org_jcq35GHp0Qxu9n2n"
    assert prof.environment == "dev"
    assert prof.auth_kind == "device"


def test_auth_login_warns_when_token_has_no_org_claim(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """If the Native app isn't configured for Business Users, the token won't
    carry an org_id. We still succeed, but emit a warning."""
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("dev", "PUB-CID")

    fake_auth = device_flow.DeviceAuthorization(
        device_code="D",
        user_code="U",
        verification_uri="x",
        verification_uri_complete="x",
        expires_in=60,
        interval=5,
    )
    monkeypatch.setattr(device_flow, "start_device_authorization", lambda **kw: fake_auth)

    # Token with a sub but no org_id / org_name.
    at = _make_fake_access_token({"sub": "auth0|x", "email": "u@ex.com"})
    monkeypatch.setattr(
        device_flow,
        "poll_for_token",
        lambda **kw: device_flow.DeviceTokenResponse(
            access_token=at,
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-1",
        ).with_expiry(),
    )
    monkeypatch.setattr("webbrowser.open", lambda *a, **kw: True)

    code, _, err = _run_cli("auth", "login", "--env", "dev", "--no-browser")
    assert code == 0
    assert "did not include an org_id claim" in err
    # Profile still created, just with empty organization.
    cfg = storage.ConfigStore(isolated_earl_home / "config.json").load()
    assert "device-dev" in cfg.profiles
    assert cfg.profiles["device-dev"].organization == ""


def test_auth_login_preselected_organization_still_works(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """Backward compat: an admin explicitly scripting --organization still
    short-circuits the Auth0 picker."""
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("dev", "PUB-CID")

    captured: dict = {}

    def _capture_auth(**kw):
        captured.update(kw)
        return device_flow.DeviceAuthorization(
            device_code="D",
            user_code="U",
            verification_uri="x",
            verification_uri_complete="x",
            expires_in=60,
            interval=5,
        )

    monkeypatch.setattr(device_flow, "start_device_authorization", _capture_auth)
    at = _make_fake_access_token({"sub": "auth0|x"})
    monkeypatch.setattr(
        device_flow,
        "poll_for_token",
        lambda **kw: device_flow.DeviceTokenResponse(
            access_token=at,
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-1",
        ).with_expiry(),
    )
    monkeypatch.setattr("webbrowser.open", lambda *a, **kw: True)

    code, _, err = _run_cli(
        "auth",
        "login",
        "--env",
        "dev",
        "--no-browser",
        "--organization",
        "org_explicit",
    )
    assert code == 0, err
    assert captured["organization"] == "org_explicit"
    cfg = storage.ConfigStore(isolated_earl_home / "config.json").load()
    # Slug falls back to last 8 chars of org_id when org_name is absent.
    assert "device-dev-explicit" in cfg.profiles
    assert cfg.profiles["device-dev-explicit"].organization == "org_explicit"


def test_auth_logout_removes_profile_and_clears_cache(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    # Seed a fake device profile + cached token.
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    prof = storage.AuthProfile.create_device(
        name="device-test",
        client_id="PUB-CID",
        clear_refresh_token="RT-1",
        environment="test",
    )
    cs.upsert_profile(prof)

    cache_key = auth_storage.token_cache_key(
        "PUB-CID",
        "https://earl-api.thelumos.xyz",
        "",
        "dev-f4675lf8h3k0i3me.us.auth0.com",
        "device",
    )
    auth_storage.save_token(
        cache_key,
        auth_storage.CachedToken(
            access_token="AT-1",
            token_type="Bearer",
            expires_at=time.time() + 3600,
            refresh_token="RT-1",
        ),
    )
    assert auth_storage.load_token(cache_key) is not None

    code, _, err = _run_cli("auth", "logout", "--name", "device-test")
    assert code == 0, err

    cs2 = storage.ConfigStore(isolated_earl_home / "config.json")
    assert "device-test" not in cs2.load().profiles
    assert auth_storage.load_token(cache_key) is None
