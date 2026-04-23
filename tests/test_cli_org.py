"""CLI tests for ``earl org`` (create/list/show/update/delete/members/invite/
invitations/roles) and the ``EARL_ORGANIZATION`` → ``EARL_ORG_ID`` soft
deprecation.

The backend is mocked via ``_svc_request`` — we only care that the CLI:

1. Dispatches to the right ``/api/v1/organizations`` endpoint with the right
   body, query, and HTTP verb.
2. Honours ``--dry-run`` without making any network calls.
3. Emits clean ``--json`` output that tools can parse.
4. Refuses destructive operations (``delete``, ``members remove``,
   ``invitations revoke``) non-interactively without ``--yes``.
5. Prints a one-time deprecation warning when the caller only sets the old
   ``EARL_ORGANIZATION`` env var.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from earl_sdk import auth_storage
from earl_sdk.cli import app as cli_app
from earl_sdk.cli.app import main as cli_main
from earl_sdk.interactive.storage import config_store as storage


# ── Shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def isolated_earl_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / ".earl"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "EARL_DIR", home, raising=False)
    monkeypatch.setattr(storage, "CONFIG_PATH", home / "config.json", raising=False)
    monkeypatch.setattr(auth_storage, "EARL_DIR", home, raising=False)
    monkeypatch.setattr(
        auth_storage, "TOKEN_CACHE_DIR", home / "tokens", raising=False
    )
    monkeypatch.setenv("EARL_SECRET_BACKEND", "file")
    monkeypatch.setattr(auth_storage, "_keyring_module", None, raising=False)
    monkeypatch.setattr(auth_storage, "_keyring_checked", True, raising=False)
    # M2M creds short-circuit ``_build_client`` for every non-dry-run test.
    monkeypatch.setenv("EARL_CLIENT_ID", "cid-test")
    monkeypatch.setenv("EARL_CLIENT_SECRET", "sec-test")
    monkeypatch.setenv("EARL_ORG_ID", "org_caller")
    monkeypatch.setenv("EARL_ENVIRONMENT", "test")
    # Reset the one-time deprecation flag between tests so we can re-assert it.
    if hasattr(cli_app._build_client, "_org_deprecation_warned"):
        delattr(cli_app._build_client, "_org_deprecation_warned")
    return home


def _run_cli(*argv: str, stdin: str = "") -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with redirect_stdout(out), redirect_stderr(err):
            code = cli_main(list(argv)) or 0
    except SystemExit as e:
        if isinstance(e.code, str):
            err.write(e.code + "\n")
            code = 1
        else:
            code = int(e.code or 0)
    return code, out.getvalue(), err.getvalue()


def _stub_svc_request(
    monkeypatch: pytest.MonkeyPatch, responses: list
) -> list[dict]:
    calls: list[dict] = []
    queue = list(responses)

    def fake(client, method, path, *, json_body=None):  # noqa: ANN001
        calls.append({"method": method, "path": path, "json_body": json_body})
        if not queue:
            raise AssertionError(f"unexpected _svc_request call {method} {path}")
        nxt = queue.pop(0)
        if callable(nxt):
            return nxt(method, path, json_body)
        return nxt

    monkeypatch.setattr(cli_app, "_svc_request", fake)
    return calls


# ── list ──────────────────────────────────────────────────────────────────


def test_org_list_dry_run_emits_action(isolated_earl_home: Path):
    code, out, _ = _run_cli("--json", "--dry-run", "org", "list")
    assert code == 0
    payload = json.loads(out)
    assert payload["action"] == "org.list"


def test_org_list_calls_get_endpoint(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"organizations": [
            {"id": "org_a", "name": "alpha", "display_name": "Alpha"},
            {"id": "org_b", "name": "bravo", "display_name": "Bravo"},
        ]}],
    )
    code, out, _ = _run_cli("org", "list", "--format", "json")
    assert code == 0, out
    assert calls == [
        {"method": "GET", "path": "/api/v1/organizations", "json_body": None}
    ]
    body = json.loads(out)
    assert {o["id"] for o in body["organizations"]} == {"org_a", "org_b"}


# ── create ────────────────────────────────────────────────────────────────


def test_org_create_dry_run(isolated_earl_home: Path):
    code, out, _ = _run_cli(
        "--json", "--dry-run",
        "org", "create",
        "--name", "acme",
        "--display-name", "Acme Corp",
        "--metadata", "region=us-east",
        "--metadata", "tier=gold",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["action"] == "org.create"
    assert payload["payload"]["name"] == "acme"
    assert payload["payload"]["display_name"] == "Acme Corp"
    assert payload["payload"]["metadata"] == {"region": "us-east", "tier": "gold"}


def test_org_create_posts_expected_body(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"id": "org_acme", "name": "acme", "display_name": "Acme"}],
    )
    code, _, _ = _run_cli("org", "create", "--name", "acme", "--display-name", "Acme")
    assert code == 0
    assert calls[0]["method"] == "POST"
    assert calls[0]["path"] == "/api/v1/organizations"
    assert calls[0]["json_body"] == {"name": "acme", "display_name": "Acme"}


def test_org_create_rejects_bad_metadata(isolated_earl_home: Path):
    code, _, err = _run_cli(
        "org", "create", "--name", "acme",
        "--metadata", "badpair",
    )
    assert code == 1
    assert "Invalid --metadata" in err or "Expected KEY=VALUE" in err


# ── show / update / delete ────────────────────────────────────────────────


def test_org_show_falls_back_to_caller_org(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"id": "org_caller", "name": "caller", "display_name": "Caller"}],
    )
    code, _, _ = _run_cli("org", "show")
    assert code == 0
    assert calls[0]["path"] == "/api/v1/organizations/org_caller"


def test_org_show_explicit_overrides_caller(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"id": "org_x", "name": "x", "display_name": "X"}],
    )
    code, _, _ = _run_cli("org", "show", "org_x")
    assert code == 0
    assert calls[0]["path"] == "/api/v1/organizations/org_x"


def test_org_update_patches_display_name(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"id": "org_x", "name": "x", "display_name": "New"}],
    )
    code, _, _ = _run_cli(
        "org", "update", "org_x", "--display-name", "New"
    )
    assert code == 0
    assert calls[0]["method"] == "PATCH"
    assert calls[0]["path"] == "/api/v1/organizations/org_x"
    assert calls[0]["json_body"] == {"display_name": "New"}


def test_org_delete_requires_confirmation_non_interactive(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code, _, err = _run_cli("org", "delete", "org_x")
    assert code != 0
    assert "--yes" in err


def test_org_delete_with_yes_calls_delete(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(monkeypatch, [{}])
    code, _, _ = _run_cli("org", "delete", "org_x", "--yes")
    assert code == 0, calls
    assert calls[0] == {
        "method": "DELETE", "path": "/api/v1/organizations/org_x", "json_body": None
    }


# ── members ───────────────────────────────────────────────────────────────


def test_org_members_list_table_output(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"organization_id": "org_x", "members": [
            {"user_id": "auth0|alice", "email": "a@x.com", "name": "Alice",
             "roles": ["EARL_Org_Admin"]},
        ]}],
    )
    code, out, _ = _run_cli("org", "members", "list", "org_x")
    assert code == 0
    assert calls[0]["path"] == "/api/v1/organizations/org_x/members"
    assert "auth0|alice" in out
    assert "EARL_Org_Admin" in out


def test_org_members_remove_requires_yes(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code, _, err = _run_cli("org", "members", "remove", "org_x", "auth0|bob")
    assert code != 0
    assert "--yes" in err


def test_org_members_remove_happy(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(monkeypatch, [{}])
    code, _, _ = _run_cli(
        "org", "members", "remove", "org_x", "auth0|bob", "--yes"
    )
    assert code == 0
    assert calls[0]["method"] == "DELETE"
    assert calls[0]["path"] == "/api/v1/organizations/org_x/members/auth0|bob"


# ── invite ────────────────────────────────────────────────────────────────


def test_org_invite_dry_run(isolated_earl_home: Path):
    code, out, _ = _run_cli(
        "--json", "--dry-run",
        "org", "invite", "org_x",
        "--email", "new@x.com",
        "--role", "EARL_Org_Admin",
    )
    payload = json.loads(out)
    assert payload["action"] == "org.invite"
    assert payload["payload"]["org_id"] == "org_x"
    assert payload["payload"]["email"] == "new@x.com"
    assert payload["payload"]["roles"] == ["EARL_Org_Admin"]


def test_org_invite_prints_acceptance_url_on_stdout(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    _stub_svc_request(
        monkeypatch,
        [{
            "id": "inv_1", "email": "new@x.com", "organization_id": "org_x",
            "invitation_url": "https://auth0/accept/inv_1",
            "expires_at": "2026-05-01", "inviter": "earl", "roles": [],
        }],
    )
    code, out, err = _run_cli(
        "org", "invite", "org_x", "--email", "new@x.com"
    )
    assert code == 0
    # URL goes to stdout so scripts can capture & email it.
    assert "https://auth0/accept/inv_1" in out
    # Human-readable status line on stderr.
    assert "inv_1" in err


# ── invitations list/revoke ───────────────────────────────────────────────


def test_org_invitations_list(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"organization_id": "org_x", "invitations": [
            {"id": "inv_1", "email": "a@x.com", "expires_at": "2026-05-01"},
        ]}],
    )
    code, out, _ = _run_cli("org", "invitations", "list", "org_x")
    assert code == 0
    assert calls[0]["path"] == "/api/v1/organizations/org_x/invitations"
    assert "inv_1" in out


def test_org_invitations_revoke_needs_yes(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code, _, err = _run_cli(
        "org", "invitations", "revoke", "org_x", "inv_1"
    )
    assert code != 0
    assert "--yes" in err


def test_org_invitations_revoke_happy(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(monkeypatch, [{}])
    code, _, _ = _run_cli(
        "org", "invitations", "revoke", "org_x", "inv_1", "--yes"
    )
    assert code == 0
    assert calls[0]["method"] == "DELETE"
    assert calls[0]["path"] == "/api/v1/organizations/org_x/invitations/inv_1"


# ── roles ─────────────────────────────────────────────────────────────────


def test_org_roles_catalog_lists_names(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"roles": [
            {"id": "rol_1", "name": "EARL_Admin", "description": "global"},
            {"id": "rol_2", "name": "EARL_Org_Admin", "description": "per-org"},
        ]}],
    )
    code, out, _ = _run_cli("org", "roles", "catalog")
    assert code == 0
    assert calls[0]["path"] == "/api/v1/organizations/_roles/catalog"
    assert "EARL_Org_Admin" in out


def test_org_roles_grant_sends_roles_body(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"organization_id": "org_x", "user_id": "auth0|u",
          "roles": ["EARL_Org_Admin"]}],
    )
    code, _, _ = _run_cli(
        "org", "roles", "grant", "org_x", "auth0|u", "EARL_Org_Admin"
    )
    assert code == 0
    assert calls[0]["method"] == "POST"
    assert calls[0]["path"] == "/api/v1/organizations/org_x/members/auth0|u/roles"
    assert calls[0]["json_body"] == {"roles": ["EARL_Org_Admin"]}


def test_org_roles_revoke_uses_delete(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"organization_id": "org_x", "user_id": "auth0|u", "roles": []}],
    )
    code, _, _ = _run_cli(
        "org", "roles", "revoke", "org_x", "auth0|u", "EARL_Org_Admin"
    )
    assert code == 0
    assert calls[0]["method"] == "DELETE"
    assert calls[0]["path"] == "/api/v1/organizations/org_x/members/auth0|u/roles"
    assert calls[0]["json_body"] == {"roles": ["EARL_Org_Admin"]}


# ── EARL_ORGANIZATION deprecation ─────────────────────────────────────────


def test_earl_organization_alias_still_works(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("EARL_ORG_ID", raising=False)
    monkeypatch.setenv("EARL_ORGANIZATION", "org_legacy")
    calls = _stub_svc_request(monkeypatch, [{"organizations": []}])
    code, _, err = _run_cli("org", "list")
    assert code == 0, err
    # One-time deprecation warning went to stderr.
    assert "EARL_ORGANIZATION is deprecated" in err
    assert "EARL_ORG_ID" in err
    assert calls[0]["method"] == "GET"


def test_earl_org_id_suppresses_warning(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    # Both present but EARL_ORG_ID wins — no warning.
    monkeypatch.setenv("EARL_ORG_ID", "org_new")
    monkeypatch.setenv("EARL_ORGANIZATION", "org_old")
    _stub_svc_request(monkeypatch, [{"organizations": []}])
    code, _, err = _run_cli("org", "list")
    assert code == 0
    assert "deprecated" not in err


def test_earl_organization_warning_is_emitted_once(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("EARL_ORG_ID", raising=False)
    monkeypatch.setenv("EARL_ORGANIZATION", "org_legacy")
    _stub_svc_request(
        monkeypatch,
        [{"organizations": []}, {"organizations": []}],
    )
    # First call: warning.
    code1, _, err1 = _run_cli("org", "list")
    assert code1 == 0
    assert "deprecated" in err1
    # Second call in the same process: no repeated warning.
    code2, _, err2 = _run_cli("org", "list")
    assert code2 == 0
    assert "deprecated" not in err2
