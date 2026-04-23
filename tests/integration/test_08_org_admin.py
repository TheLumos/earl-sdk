"""Organization-administration integration tests.

Covers the CLI surface introduced with Earl's two-mode auth refactor:

* ``earl org list / show / create / update / delete``
* ``earl org members list / remove``
* ``earl org invite`` / ``earl org invitations list / revoke``
* ``earl org roles catalog / grant / revoke``

The tests deliberately *do not* send any real email (invitations are created
with a short TTL and immediately revoked) and never delete an org they did
not create.

Tiering
-------

The tests skip based on the kind of admin identity the active profile has:

* Profile with ``EARL_Admin``     — runs *all* happy-path + matrix tests.
* Profile with ``EARL_Org_Admin`` — runs the org-admin happy path and skips
  anything global-admin-only.
* Anything else                    — most tests skip; the ``requires_nonadmin``
  denial tests still assert 403s.
"""

from __future__ import annotations

import re
import time
from typing import Any

import pytest

from .cli_runner import CliRunner


# --------------------------------------------------------------------------- #
# Role detection (from `earl whoami`)
# --------------------------------------------------------------------------- #


def _roles(cli: CliRunner) -> list[str]:
    """Return the roles the backend sees for the active CLI profile."""
    data = cli.run("whoami").json()
    roles = data.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    return [r.strip() for r in roles if r]


@pytest.fixture(scope="module")
def admin_roles(cli: CliRunner) -> list[str]:
    return _roles(cli)


@pytest.fixture(scope="module")
def is_global_admin(admin_roles: list[str]) -> bool:
    return "EARL_Admin" in admin_roles


@pytest.fixture(scope="module")
def is_org_admin(admin_roles: list[str]) -> bool:
    return "EARL_Org_Admin" in admin_roles or "EARL_Admin" in admin_roles


def _skip_unless_global(is_global_admin: bool) -> None:
    if not is_global_admin:
        pytest.skip("active profile lacks the EARL_Admin role")


def _skip_unless_any_admin(is_org_admin: bool) -> None:
    if not is_org_admin:
        pytest.skip("active profile lacks EARL_Admin / EARL_Org_Admin")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _as_list(payload: Any, *keys: str) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in (*keys, "items", "data", "results"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []


def _find_org(orgs: list[dict], org_id: str) -> dict | None:
    for org in orgs:
        if isinstance(org, dict) and (
            org.get("id") == org_id or org.get("organization_id") == org_id
        ):
            return org
    return None


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,49}$")


# --------------------------------------------------------------------------- #
# Read-only flows — work for both global- and org-admins
# --------------------------------------------------------------------------- #


def test_org_show_self(cli: CliRunner, active_org_id: str, is_org_admin: bool) -> None:
    """``earl org show`` without an ORG argument returns the caller's own org.

    Documents::

        earl org show                       # implicit: caller's org
        earl org show org_xxxxxxxxxxxx      # any org (EARL_Admin only)
    """
    _skip_unless_any_admin(is_org_admin)
    data = cli.run("org", "show").json()
    assert data.get("id") == active_org_id or data.get("organization_id") == active_org_id


def test_org_members_list_self(cli: CliRunner, is_org_admin: bool) -> None:
    """``earl org members list`` defaults to the caller's org.

    Documents::

        earl --json org members list | jq 'length'
    """
    _skip_unless_any_admin(is_org_admin)
    payload = cli.run("org", "members", "list", "--format", "json").json()
    members = _as_list(payload, "members")
    # A live org always has at least the caller in it.
    assert members, payload


def test_org_invitations_list_self(cli: CliRunner, is_org_admin: bool) -> None:
    """Listing invitations must succeed (may be empty)."""
    _skip_unless_any_admin(is_org_admin)
    payload = cli.run(
        "org", "invitations", "list", "--format", "json"
    ).json()
    assert isinstance(_as_list(payload, "invitations"), list)


def test_org_roles_catalog_global_admin(
    cli: CliRunner, is_global_admin: bool
) -> None:
    """``earl org roles catalog`` is EARL_Admin-only."""
    _skip_unless_global(is_global_admin)
    payload = cli.run("org", "roles", "catalog", "--format", "json").json()
    roles = _as_list(payload, "roles")
    names = {r.get("name") for r in roles if isinstance(r, dict)}
    # Earl ships at least these two roles.
    assert {"EARL_Admin", "EARL_Org_Admin"} & names, roles


# --------------------------------------------------------------------------- #
# Global-admin mutations
# --------------------------------------------------------------------------- #


def test_org_list_global_admin(cli: CliRunner, is_global_admin: bool) -> None:
    """Tenant-wide org listing is EARL_Admin-only."""
    _skip_unless_global(is_global_admin)
    payload = cli.run("org", "list", "--format", "json").json()
    orgs = _as_list(payload, "organizations")
    assert orgs, payload


def test_org_create_update_delete_cycle(
    cli: CliRunner,
    run_id: str,
    is_global_admin: bool,
) -> None:
    """Global admin can create, update, and delete an organization.

    Documents::

        earl org create --name it-org-xxxx --display-name 'IT Org'
        earl org update org_xxxxxx --display-name 'IT Org renamed'
        earl org delete org_xxxxxx --yes
    """
    _skip_unless_global(is_global_admin)

    # Slugs must be <= 50 chars; keep the prefix short.
    slug = f"it-{run_id}".replace("_", "-")[:30].rstrip("-")
    if not _SLUG_RE.match(slug):
        pytest.skip(f"generated slug {slug!r} would fail validation")

    created = cli.run(
        "org", "create",
        "--name", slug,
        "--display-name", "Integration Test Org",
        "--metadata", "purpose=integration-test",
    ).json()
    org_id = created.get("id") or created.get("organization_id")
    assert org_id, created

    try:
        # Update
        updated = cli.run(
            "org", "update", org_id,
            "--display-name", "Integration Test Org (renamed)",
        ).json()
        assert "renamed" in (
            updated.get("display_name") or updated.get("displayName") or ""
        )

        # Should now appear in list
        listed = _as_list(
            cli.run("org", "list", "--format", "json").json(),
            "organizations",
        )
        assert _find_org(listed, org_id) is not None, (
            f"newly created org {org_id} did not appear in `earl org list`"
        )
    finally:
        # Cleanup — always, so the tenant doesn't accumulate junk orgs.
        cli.run("org", "delete", org_id, "--yes", check=False, timeout=30)


def test_org_invitation_create_then_revoke(
    cli: CliRunner, active_org_id: str, run_id: str, is_org_admin: bool
) -> None:
    """Org admins can mint and revoke invitations for their own org.

    Documents::

        earl org invite ORG --email user@example.com --ttl-seconds 300
        earl org invitations list ORG
        earl org invitations revoke ORG inv_xxx --yes
    """
    _skip_unless_any_admin(is_org_admin)

    email = f"earl-integration-{run_id}@example.invalid"
    created = cli.run(
        "org", "invite", active_org_id,
        "--email", email,
        "--ttl-seconds", "300",
    ).json()
    inv_id = created.get("id") or created.get("invitation_id")
    # Some backends return a wrapping envelope.
    if not inv_id and isinstance(created, dict):
        inv_id = (created.get("invitation") or {}).get("id")
    assert inv_id, created

    try:
        # Must appear in list
        deadline = time.time() + 15
        seen = False
        while time.time() < deadline:
            payload = cli.run(
                "org", "invitations", "list", active_org_id, "--format", "json"
            ).json()
            ids = {
                inv.get("id") or inv.get("invitation_id")
                for inv in _as_list(payload, "invitations")
                if isinstance(inv, dict)
            }
            if inv_id in ids:
                seen = True
                break
            time.sleep(1)
        assert seen, f"invitation {inv_id} not listed for org {active_org_id}"
    finally:
        cli.run(
            "org", "invitations", "revoke", active_org_id, inv_id, "--yes",
            check=False,
        )


# --------------------------------------------------------------------------- #
# Negative matrix — org-admin cannot touch other orgs / global admin surface
# --------------------------------------------------------------------------- #


@pytest.mark.requires_second_org
def test_org_admin_cannot_read_other_org_members(
    cli: CliRunner,
    second_org_id: str,
    is_global_admin: bool,
    is_org_admin: bool,
) -> None:
    """An ``EARL_Org_Admin`` must NOT be able to list members of a different org."""
    if is_global_admin:
        pytest.skip("test targets EARL_Org_Admin; active profile is global admin")
    _skip_unless_any_admin(is_org_admin)

    result = cli.run_expect_fail(
        "org", "members", "list", second_org_id, "--format", "json"
    )
    blob = (result.stdout + result.stderr).lower()
    assert any(
        kw in blob for kw in ("forbidden", "403", "not authorized", "outside")
    ), result.stdout + result.stderr


def test_org_admin_cannot_list_tenant_orgs(
    cli: CliRunner, is_global_admin: bool, is_org_admin: bool
) -> None:
    """Only EARL_Admin may list every org on the tenant."""
    if is_global_admin:
        pytest.skip("active profile is EARL_Admin — endpoint is allowed")
    _skip_unless_any_admin(is_org_admin)

    result = cli.run_expect_fail("org", "list", "--format", "json")
    blob = (result.stdout + result.stderr).lower()
    assert any(
        kw in blob for kw in ("forbidden", "403", "not authorized", "admin")
    ), result.stdout + result.stderr


def test_org_admin_cannot_delete_org(
    cli: CliRunner, active_org_id: str, is_global_admin: bool, is_org_admin: bool
) -> None:
    """Only EARL_Admin may delete an org."""
    if is_global_admin:
        pytest.skip("active profile is EARL_Admin — endpoint is allowed")
    _skip_unless_any_admin(is_org_admin)

    result = cli.run_expect_fail(
        "org", "delete", active_org_id, "--yes"
    )
    blob = (result.stdout + result.stderr).lower()
    assert any(
        kw in blob for kw in ("forbidden", "403", "not authorized", "admin")
    ), result.stdout + result.stderr


@pytest.mark.requires_nonadmin
def test_nonadmin_cannot_list_org_members(nonadmin_cli: CliRunner) -> None:
    """A plain member cannot list members of their own org."""
    result = nonadmin_cli.run_expect_fail(
        "org", "members", "list", "--format", "json"
    )
    blob = (result.stdout + result.stderr).lower()
    assert any(
        kw in blob for kw in ("forbidden", "403", "not authorized", "admin")
    ), result.stdout + result.stderr


@pytest.mark.requires_nonadmin
def test_nonadmin_cannot_invite(nonadmin_cli: CliRunner, active_org_id: str) -> None:
    """A plain member cannot invite users."""
    result = nonadmin_cli.run_expect_fail(
        "org", "invite", active_org_id,
        "--email", "nonadmin-should-fail@example.invalid",
    )
    blob = (result.stdout + result.stderr).lower()
    assert any(
        kw in blob for kw in ("forbidden", "403", "not authorized", "admin")
    ), result.stdout + result.stderr
