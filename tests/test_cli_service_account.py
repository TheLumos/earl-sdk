"""CLI tests for ``earl service-account create|list|revoke``.

The backend is mocked — we only care that the CLI:

1. Builds a valid request body (name/scopes/description/org_id).
2. Prints credentials exactly once, with a "store this now" warning.
3. Emits valid ``--json`` payloads.
4. Refuses to revoke non-interactively without ``--yes``.
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


@pytest.fixture
def isolated_earl_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    home = tmp_path / ".earl"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(storage, "EARL_DIR", home, raising=False)
    monkeypatch.setattr(storage, "CONFIG_PATH", home / "config.json", raising=False)
    monkeypatch.setattr(auth_storage, "EARL_DIR", home, raising=False)
    monkeypatch.setattr(auth_storage, "TOKEN_CACHE_DIR", home / "tokens", raising=False)
    monkeypatch.setenv("EARL_SECRET_BACKEND", "file")
    monkeypatch.setattr(auth_storage, "_keyring_module", None, raising=False)
    monkeypatch.setattr(auth_storage, "_keyring_checked", True, raising=False)
    # Env-var M2M path is the shortest way past ``_build_client``; also
    # exercises the new credential resolution order.
    monkeypatch.setenv("EARL_CLIENT_ID", "cid-test")
    monkeypatch.setenv("EARL_CLIENT_SECRET", "sec-test")
    monkeypatch.setenv("EARL_ORG_ID", "org_caller")
    # Never touch the real Auth0.
    monkeypatch.setenv("EARL_ENVIRONMENT", "test")
    return home


def _run_cli(*argv: str) -> tuple[int, str, str]:
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
    """Record every ``_svc_request`` invocation and return queued responses.

    ``responses`` is consumed in order. Each entry is either the dict to
    return, or a callable ``(method, path, json_body) -> dict``.
    """
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


# ── create ───────────────────────────────────────────────────────────────────


def test_service_account_create_dry_run(isolated_earl_home: Path):
    code, out, _ = _run_cli(
        "--json",
        "--dry-run",
        "service-account",
        "create",
        "--name",
        "prod-ci",
        "--scopes",
        "earl:read earl:deploy",
        "--description",
        "CI pipeline",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["action"] == "service-account.create"
    assert payload["payload"]["name"] == "prod-ci"
    # Scope parsing: space/comma-separated both collapse to a clean list.
    assert payload["payload"]["scopes"] == ["earl:read", "earl:deploy"]
    assert payload["payload"]["description"] == "CI pipeline"
    assert "org_id" not in payload["payload"]


def test_service_account_create_scopes_comma_separated(isolated_earl_home: Path):
    code, out, _ = _run_cli(
        "--json",
        "--dry-run",
        "service-account",
        "create",
        "--name",
        "ci",
        "--scopes",
        "earl:read,earl:deploy, earl:admin",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["payload"]["scopes"] == [
        "earl:read",
        "earl:deploy",
        "earl:admin",
    ]


def test_service_account_create_prints_credentials_once_with_warning(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [
            {
                "id": "cid_new",
                "client_id": "cid_new",
                "client_secret": "super-secret-shh",
                "org_id": "org_caller",
                "name": "prod-ci",
                "scopes": ["earl:read"],
                "created_by": "auth0|u",
                "created_at": "2026-01-01T00:00:00Z",
                "warning": (
                    "Store the client_secret now — it will never be shown again."
                ),
            }
        ],
    )

    code, out, err = _run_cli(
        "service-account",
        "create",
        "--name",
        "prod-ci",
        "--scopes",
        "earl:read",
    )
    assert code == 0, err

    # Exactly one POST, to the right path, carrying a correct body.
    assert len(calls) == 1
    assert calls[0]["method"] == "POST"
    assert calls[0]["path"] == "/api/v1/service-accounts"
    assert calls[0]["json_body"]["name"] == "prod-ci"
    assert calls[0]["json_body"]["scopes"] == ["earl:read"]

    # Credentials are surfaced as env-var shaped lines on stdout so that
    # ``earl service-account create ... > .env`` works.
    assert "EARL_CLIENT_ID=cid_new" in out
    assert "EARL_CLIENT_SECRET=super-secret-shh" in out
    assert "EARL_ORG_ID=org_caller" in out

    # And the warning + human-readable context go to stderr.
    assert "never be shown again" in err.lower() or "cannot be shown" in err.lower()
    assert "prod-ci" in err


def test_service_account_create_json_output(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    body = {
        "id": "cid_j",
        "client_id": "cid_j",
        "client_secret": "s",
        "org_id": "org_caller",
        "name": "n",
        "scopes": ["earl:read"],
    }
    _stub_svc_request(monkeypatch, [body])
    code, out, _ = _run_cli(
        "--json",
        "service-account",
        "create",
        "--name",
        "n",
    )
    assert code == 0
    # --json surfaces the raw API response — a deterministic machine shape.
    assert json.loads(out) == body


def test_service_account_create_forwards_target_org(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    """``--org-id`` is the EARL_Admin knob for cross-org provisioning."""
    calls = _stub_svc_request(
        monkeypatch,
        [
            {
                "id": "cid_x",
                "client_id": "cid_x",
                "client_secret": "s",
                "org_id": "org_target",
                "name": "cross",
                "scopes": [],
                "warning": "store",
            }
        ],
    )
    code, _, err = _run_cli(
        "service-account",
        "create",
        "--name",
        "cross",
        "--org-id",
        "org_target",
    )
    assert code == 0, err
    assert calls[0]["json_body"]["org_id"] == "org_target"


# ── list ─────────────────────────────────────────────────────────────────────


def test_service_account_list_empty(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    _stub_svc_request(
        monkeypatch,
        [{"org_id": "org_caller", "service_accounts": []}],
    )
    code, out, err = _run_cli("service-account", "list")
    assert code == 0, err
    # Human-readable "empty" line; exact wording isn't contract but org is.
    assert "org_caller" in err + out


def test_service_account_list_json(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = {
        "org_id": "org_caller",
        "service_accounts": [
            {
                "id": "cid_1",
                "client_id": "cid_1",
                "name": "a",
                "scopes": ["earl:read"],
                "created_by": "auth0|x",
                "created_at": "2026-01-01T00:00:00Z",
            },
        ],
    }
    calls = _stub_svc_request(monkeypatch, [payload])
    code, out, _ = _run_cli("--json", "service-account", "list")
    assert code == 0
    assert json.loads(out) == payload
    assert calls[0]["method"] == "GET"
    assert calls[0]["path"] == "/api/v1/service-accounts"


def test_service_account_list_with_target_org(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(
        monkeypatch,
        [{"org_id": "org_target", "service_accounts": []}],
    )
    code, _, _ = _run_cli(
        "--json",
        "service-account",
        "list",
        "--org-id",
        "org_target",
    )
    assert code == 0
    # Target org is threaded through the query string.
    assert calls[0]["path"] == "/api/v1/service-accounts?org_id=org_target"


# ── revoke ───────────────────────────────────────────────────────────────────


def test_service_account_revoke_requires_yes_in_non_tty(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    # _svc_request must NOT be called when we bail on the safety check.
    def explode(*a, **kw):
        raise AssertionError("revoke must not call backend without --yes")

    monkeypatch.setattr(cli_app, "_svc_request", explode)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code, _, err = _run_cli("service-account", "revoke", "cid_doomed")
    assert code != 0
    assert "--yes" in err


def test_service_account_revoke_with_yes_deletes(
    isolated_earl_home: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = _stub_svc_request(monkeypatch, [{}])
    code, _, err = _run_cli(
        "service-account", "revoke", "cid_doomed", "--yes"
    )
    assert code == 0, err
    assert calls[0]["method"] == "DELETE"
    assert calls[0]["path"] == "/api/v1/service-accounts/cid_doomed"


def test_service_account_revoke_dry_run(isolated_earl_home: Path):
    code, out, _ = _run_cli(
        "--json",
        "--dry-run",
        "service-account",
        "revoke",
        "cid_abc",
        "--yes",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["action"] == "service-account.revoke"
    assert payload["payload"]["client_id"] == "cid_abc"
