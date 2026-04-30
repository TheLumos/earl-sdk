"""CLI tests for `earl auth login`, `earl auth logout`, and device-clients."""

from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from earl_sdk import auth_storage, device_flow, pkce_flow
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
        # ``SystemExit("message")`` → code is a str; treat that as exit 1 and
        # write the message to stderr so the caller can assert on it.
        if isinstance(e.code, str):
            err.write(e.code + "\n")
            code = 1
        else:
            code = int(e.code or 0)
    return code, out.getvalue(), err.getvalue()


# ── earl auth device-clients ─────────────────────────────────────────────────


def test_device_clients_set_list_clear(isolated_earl_home: Path):
    code, _, _ = _run_cli("auth", "device-clients", "set", "--env", "staging", "abc123")
    assert code == 0

    code, out, _ = _run_cli("--json", "auth", "device-clients", "list")
    assert code == 0
    assert json.loads(out)["staging"] == "abc123"

    code, _, _ = _run_cli("auth", "device-clients", "clear", "--env", "staging")
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
        "staging",
        "--device-client-id",
        "PUB-CID",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["action"] == "auth.login"
    assert payload["payload"]["env"] == "staging"
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


def _stub_device_flow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    first_access_token: str,
    first_refresh_token: str = "RT-1",
    scope: str = "openid offline_access",
):
    """Install minimal stubs for ``start_device_authorization`` + ``poll_for_token``."""
    fake_auth = device_flow.DeviceAuthorization(
        device_code="DEV-CODE",
        user_code="ABCD-EFGH",
        verification_uri="https://t.auth0.com/activate",
        verification_uri_complete="https://t.auth0.com/activate?user_code=ABCD-EFGH",
        expires_in=600,
        interval=5,
    )
    monkeypatch.setattr(device_flow, "start_device_authorization", lambda **kw: fake_auth)

    def fake_poll(**kw):
        return device_flow.DeviceTokenResponse(
            access_token=first_access_token,
            expires_in=3600,
            token_type="Bearer",
            refresh_token=first_refresh_token,
            scope=scope,
        ).with_expiry()

    monkeypatch.setattr(device_flow, "poll_for_token", fake_poll)
    monkeypatch.setattr("webbrowser.open", lambda *a, **kw: True)


def test_auth_login_full_flow_single_org_autoselects(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """Common customer case: user belongs to exactly one org. CLI should
    autoselect it after a no-org device flow and mint an org-scoped token
    via refresh without prompting."""
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("staging", "PUB-CID")

    # First token: no-org, used only for org discovery.
    no_org_at = _make_fake_access_token({"sub": "auth0|alice"})
    _stub_device_flow(monkeypatch, first_access_token=no_org_at)

    # Orchestrator /auth/my-orgs → one org.
    monkeypatch.setattr(
        "earl_sdk.cli.app._discover_user_orgs",
        lambda **kw: [{"id": "org_only", "name": "solo", "display_name": "Solo Inc"}],
    )

    # Refresh-with-org call → org-scoped AT.
    refresh_calls: list = []

    def fake_refresh(**kw):
        refresh_calls.append(kw)
        org_at = _make_fake_access_token({
            "sub": "auth0|alice",
            "email": "alice@solo.example",
            "org_id": "org_only",
            "org_name": "Solo Inc",
        })
        return device_flow.DeviceTokenResponse(
            access_token=org_at,
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-scoped",
            scope="openid offline_access",
        ).with_expiry()

    monkeypatch.setattr(device_flow, "refresh_access_token_with_organization", fake_refresh)

    code, out, err = _run_cli("auth", "login", "--env", "staging", "--no-browser")
    assert code == 0, err
    assert "Signed in" in out
    assert "Solo Inc" in out

    # The refresh call must pass the chosen org.
    assert len(refresh_calls) == 1
    assert refresh_calls[0]["organization"] == "org_only"
    assert refresh_calls[0]["refresh_token"] == "RT-1"

    # Profile is saved with the org-scoped refresh token (not the initial one).
    cfg = storage.ConfigStore(isolated_earl_home / "config.json").load()
    assert cfg.active_profile == "device-staging-solo-inc"
    prof = cfg.profiles["device-staging-solo-inc"]
    assert prof.organization == "org_only"
    assert prof.refresh_token_clear() == "RT-scoped"
    assert prof.environment == "staging"

    # The seeded access-token cache points at the ORG-scoped token.
    cache_key = auth_storage.token_cache_key(
        "PUB-CID",
        "https://earl-api.thelumos.xyz",
        "org_only",
        "dev-f4675lf8h3k0i3me.us.auth0.com",
        "device",
    )
    cached = auth_storage.load_token(cache_key)
    assert cached is not None
    assert cached.refresh_token == "RT-scoped"


def test_auth_login_multi_org_uses_interactive_picker(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("staging", "PUB-CID")

    _stub_device_flow(monkeypatch, first_access_token=_make_fake_access_token({"sub": "auth0|b"}))

    orgs = [
        {"id": "org_a", "name": "alpha", "display_name": "Alpha"},
        {"id": "org_b", "name": "bravo", "display_name": "Bravo"},
        {"id": "org_c", "name": "charlie", "display_name": "Charlie"},
    ]
    monkeypatch.setattr("earl_sdk.cli.app._discover_user_orgs", lambda **kw: orgs)

    # Pretend we're on a TTY and answer "2" at the prompt.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "2")

    def fake_refresh(**kw):
        assert kw["organization"] == "org_b"
        return device_flow.DeviceTokenResponse(
            access_token=_make_fake_access_token({
                "sub": "auth0|b",
                "org_id": "org_b",
                "org_name": "Bravo",
            }),
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-b",
        ).with_expiry()

    monkeypatch.setattr(device_flow, "refresh_access_token_with_organization", fake_refresh)

    code, out, err = _run_cli("auth", "login", "--env", "staging", "--no-browser")
    assert code == 0, err
    assert "Bravo" in out
    cfg = storage.ConfigStore(isolated_earl_home / "config.json").load()
    assert "device-staging-bravo" in cfg.profiles
    assert cfg.profiles["device-staging-bravo"].organization == "org_b"


def test_auth_login_multi_org_non_tty_aborts_with_hint(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """Non-interactive runs (CI, pipes) with multiple orgs must NOT prompt —
    they must abort with the list of orgs so the user can re-run with
    --organization."""
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("staging", "PUB-CID")

    _stub_device_flow(monkeypatch, first_access_token=_make_fake_access_token({"sub": "auth0|c"}))
    orgs = [
        {"id": "org_a", "name": "alpha", "display_name": "Alpha"},
        {"id": "org_b", "name": "bravo", "display_name": "Bravo"},
    ]
    monkeypatch.setattr("earl_sdk.cli.app._discover_user_orgs", lambda **kw: orgs)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    # If the CLI wrongly tries to call refresh, blow up loudly.
    def _should_not_be_called(**kw):
        raise AssertionError("refresh_access_token_with_organization must not run in non-TTY multi-org")

    monkeypatch.setattr(
        device_flow, "refresh_access_token_with_organization", _should_not_be_called
    )

    code, _, err = _run_cli("auth", "login", "--env", "staging", "--no-browser")
    assert code != 0
    assert "org_a" in err
    assert "org_b" in err
    assert "--organization" in err


def test_auth_login_no_orgs_aborts(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("staging", "PUB-CID")

    _stub_device_flow(monkeypatch, first_access_token=_make_fake_access_token({"sub": "auth0|d"}))
    monkeypatch.setattr("earl_sdk.cli.app._discover_user_orgs", lambda **kw: [])

    code, _, err = _run_cli("auth", "login", "--env", "staging", "--no-browser")
    assert code != 0
    assert "not a member" in err.lower() or "no organizations" in err.lower()


def test_auth_login_my_orgs_unavailable_falls_back_gracefully(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """Old orchestrator that doesn't have /auth/my-orgs → the CLI should
    surface a readable error telling the user to re-run with --organization,
    rather than silently hang or crash with a traceback."""
    from earl_sdk.exceptions import NotFoundError

    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("staging", "PUB-CID")
    _stub_device_flow(monkeypatch, first_access_token=_make_fake_access_token({"sub": "auth0|e"}))

    def _boom(**kw):
        raise NotFoundError("endpoint", "/auth/my-orgs")

    monkeypatch.setattr("earl_sdk.cli.app._discover_user_orgs", _boom)

    code, _, err = _run_cli("auth", "login", "--env", "staging", "--no-browser")
    assert code != 0
    assert "--organization" in err


def _make_fake_access_token(claims: dict) -> str:
    """Build a JWT with the given payload (header + signature are unverified)."""
    import base64
    import json as _json

    def b64(obj):
        raw = _json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{b64({'alg': 'RS256'})}.{b64(claims)}.sig-not-verified"


def test_top_level_pkce_login_multi_org_uses_picker(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    cs.set_device_client_id("staging", "PUB-CID")

    monkeypatch.setattr(
        pkce_flow,
        "run_loopback_login",
        lambda **kw: pkce_flow.PKCETokenResponse(
            access_token=_make_fake_access_token({"sub": "auth0|pkce"}),
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-pkce",
            scope="openid offline_access",
        ).with_expiry(),
    )

    orgs = [
        {"id": "org_a", "name": "alpha", "display_name": "Alpha"},
        {"id": "org_b", "name": "bravo", "display_name": "Bravo"},
    ]
    monkeypatch.setattr("earl_sdk.cli.app._discover_user_orgs", lambda **kw: orgs)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "2")

    def fake_refresh(**kw):
        assert kw["organization"] == "org_b"
        return device_flow.DeviceTokenResponse(
            access_token=_make_fake_access_token(
                {
                    "sub": "auth0|pkce",
                    "email": "pkce@example.com",
                    "org_id": "org_b",
                    "org_name": "Bravo",
                }
            ),
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-pkce-org",
        ).with_expiry()

    monkeypatch.setattr(
        device_flow,
        "refresh_access_token_with_organization",
        fake_refresh,
    )

    code, out, err = _run_cli("login", "--env", "staging")
    assert code == 0, err
    assert "Bravo" in out

    cfg = storage.ConfigStore(isolated_earl_home / "config.json").load()
    assert cfg.active_profile == "pkce-staging-bravo"
    prof = cfg.profiles["pkce-staging-bravo"]
    assert prof.auth_kind == "pkce"
    assert prof.organization == "org_b"
    assert prof.refresh_token_clear() == "RT-pkce-org"


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


# NOTE: the pre-v0.7 test ``test_auth_login_warns_when_token_has_no_org_claim``
# was removed. In the new multi-org flow a no-org token is never persisted as
# a usable profile — the CLI instead calls ``/auth/my-orgs``, prompts, and
# mints an org-scoped token via refresh. The zero-org and non-TTY error paths
# are covered by ``test_auth_login_no_orgs_aborts`` and
# ``test_auth_login_multi_org_non_tty_aborts_with_hint`` above.


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
    # After poll returns a no-``org_id`` access token, the CLI re-mints
    # an org-scoped one off the refresh token. Mock that call.
    rebound_at = _make_fake_access_token(
        {"sub": "auth0|x", "org_id": "org_explicit"}
    )

    def fake_refresh(**kw):
        assert kw["organization"] == "org_explicit"
        return device_flow.DeviceTokenResponse(
            access_token=rebound_at,
            expires_in=3600,
            token_type="Bearer",
            refresh_token="RT-2",
        ).with_expiry()

    monkeypatch.setattr(
        device_flow, "refresh_access_token_with_organization", fake_refresh
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


# ── earl auth my-orgs ────────────────────────────────────────────────────────


def test_cli_my_orgs_lists_orgs_as_table(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    # Seed an active device profile.
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    prof = storage.AuthProfile.create_device(
        name="device-dev",
        client_id="PUB-CID",
        clear_refresh_token="RT-1",
        environment="dev",
    )
    cs.upsert_profile(prof)
    cs.set_active_profile("device-dev")

    # Short-circuit the Auth0 token fetch by stubbing _build_client → a fake
    # object exposing ``_auth.get_headers()``.
    from unittest import mock

    fake_client = mock.MagicMock()
    fake_client._auth.get_headers.return_value = {"Authorization": "Bearer AT-FAKE"}
    monkeypatch.setattr("earl_sdk.cli.app._build_client", lambda *a, **kw: fake_client)

    captured: dict = {}

    def fake_discover(**kw):
        captured.update(kw)
        return [
            {"id": "org_a", "name": "alpha", "display_name": "Alpha"},
            {"id": "org_b", "name": "bravo", "display_name": "Bravo"},
        ]

    monkeypatch.setattr("earl_sdk.cli.app._discover_user_orgs", fake_discover)

    code, out, err = _run_cli("auth", "my-orgs")
    assert code == 0, err
    assert "org_a" in out
    assert "org_b" in out
    assert captured["access_token"] == "AT-FAKE"
    assert captured["env"] == "dev"


def test_cli_my_orgs_json_mode(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    prof = storage.AuthProfile.create_device(
        name="device-staging",
        client_id="PUB-CID",
        clear_refresh_token="RT-1",
        environment="staging",
    )
    cs.upsert_profile(prof)
    cs.set_active_profile("device-staging")

    from unittest import mock

    fake_client = mock.MagicMock()
    fake_client._auth.get_headers.return_value = {"Authorization": "Bearer AT"}
    monkeypatch.setattr("earl_sdk.cli.app._build_client", lambda *a, **kw: fake_client)
    monkeypatch.setattr(
        "earl_sdk.cli.app._discover_user_orgs",
        lambda **kw: [{"id": "org_x", "name": "x", "display_name": "X"}],
    )

    code, out, _ = _run_cli("--json", "auth", "my-orgs")
    assert code == 0
    data = json.loads(out)
    assert data["environment"] == "staging"
    assert data["organizations"] == [{"id": "org_x", "name": "x", "display_name": "X"}]


def test_cli_my_orgs_requires_profile_or_env(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """No active profile and no --env → clear error, not a crash."""
    code, _, err = _run_cli("auth", "my-orgs")
    assert code != 0
    assert "profile" in err.lower() or "--env" in err


# ── organization picker helper ───────────────────────────────────────────────


def test_pick_organization_default_is_first_entry(monkeypatch: pytest.MonkeyPatch):
    from earl_sdk.cli.app import _pick_organization_interactive

    orgs = [
        {"id": "org_a", "name": "alpha", "display_name": "Alpha"},
        {"id": "org_b", "name": "bravo", "display_name": "Bravo"},
    ]
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "")  # pressed Enter
    stream = io.StringIO()
    picked = _pick_organization_interactive(orgs, stream=stream)
    assert picked["id"] == "org_a"


def test_pick_organization_default_honours_current_org_id(monkeypatch):
    from earl_sdk.cli.app import _pick_organization_interactive

    orgs = [
        {"id": "org_a", "name": "alpha", "display_name": "Alpha"},
        {"id": "org_b", "name": "bravo", "display_name": "Bravo"},
    ]
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "")
    picked = _pick_organization_interactive(orgs, current_org_id="org_b", stream=io.StringIO())
    assert picked["id"] == "org_b"


def test_pick_organization_matches_by_id_or_name(monkeypatch):
    from earl_sdk.cli.app import _pick_organization_interactive

    orgs = [
        {"id": "org_a", "name": "alpha", "display_name": "Alpha LLC"},
        {"id": "org_b", "name": "bravo", "display_name": "Bravo Inc"},
    ]
    monkeypatch.setattr("builtins.input", lambda *a, **kw: "bravo")
    assert _pick_organization_interactive(orgs, stream=io.StringIO())["id"] == "org_b"

    monkeypatch.setattr("builtins.input", lambda *a, **kw: "ORG_A")  # case-insensitive
    assert _pick_organization_interactive(orgs, stream=io.StringIO())["id"] == "org_a"


def test_pick_organization_retries_on_invalid_then_succeeds(monkeypatch):
    from earl_sdk.cli.app import _pick_organization_interactive

    orgs = [
        {"id": "org_a", "name": "alpha", "display_name": "Alpha"},
        {"id": "org_b", "name": "bravo", "display_name": "Bravo"},
    ]
    answers = iter(["bogus", "99", "2"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))
    picked = _pick_organization_interactive(orgs, stream=io.StringIO())
    assert picked["id"] == "org_b"


def test_pick_organization_gives_up_after_three_bad_answers(monkeypatch):
    from earl_sdk.cli.app import _pick_organization_interactive

    orgs = [{"id": "org_a", "name": "a", "display_name": "A"}]
    answers = iter(["nope", "also-nope", "still-no"])
    monkeypatch.setattr("builtins.input", lambda *a, **kw: next(answers))

    with pytest.raises(SystemExit):
        _pick_organization_interactive(orgs, stream=io.StringIO())


def test_pick_organization_ctrl_c_exits_cleanly(monkeypatch):
    from earl_sdk.cli.app import _pick_organization_interactive

    orgs = [{"id": "org_a", "name": "a", "display_name": "A"}]

    def raises_kbdinterrupt(*a, **kw):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", raises_kbdinterrupt)
    with pytest.raises(SystemExit) as exc:
        _pick_organization_interactive(orgs, stream=io.StringIO())
    msg = str(exc.value)
    assert "cancelled" in msg.lower()


def test_auth_logout_removes_profile_and_clears_cache(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    # Seed a fake device profile + cached token.
    cs = storage.ConfigStore(isolated_earl_home / "config.json")
    prof = storage.AuthProfile.create_device(
        name="device-staging",
        client_id="PUB-CID",
        clear_refresh_token="RT-1",
        environment="staging",
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

    code, _, err = _run_cli("auth", "logout", "--name", "device-staging")
    assert code == 0, err

    cs2 = storage.ConfigStore(isolated_earl_home / "config.json")
    assert "device-staging" not in cs2.load().profiles
    assert auth_storage.load_token(cache_key) is None
