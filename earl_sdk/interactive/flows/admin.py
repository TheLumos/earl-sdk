"""Interactive admin flow — organizations, members, invitations, service
accounts, and role assignments.

Who sees what
-------------
The menu auto-detects the caller's role from the bearer token's ``roles``
claim (same claim the orchestrator uses for authorization) and hides
operations they can't perform:

* ``EARL_Admin`` (global internal operator) — full access: create/delete
  orgs, manage any org's members/invitations, assign ``EARL_Org_Admin``
  roles, provision service accounts.
* ``EARL_Org_Admin`` of a single org — everything inside that org except
  role grants: invite users, list/remove members, list/revoke
  invitations, create/revoke M2M service accounts.
* Anyone else — admin menu is disabled with a clear message explaining
  why.

Design notes
------------
* This flow *never* asks for Auth0 Management credentials; it talks only
  to the orchestrator's ``/api/v1/organizations`` and
  ``/api/v1/service-accounts`` endpoints, which are role-gated server
  side. Running the UI as a non-admin is safe — you'll get 403s, not
  escalation.
* Every destructive operation (delete org, remove member, revoke invite,
  revoke service account) goes through an explicit confirmation prompt.
* Secrets (service-account client_secret, invitation URLs) are shown
  once and the user is warned to copy them immediately — they are never
  cached on disk by the UI.

"""
from __future__ import annotations

from typing import Any, Optional

from ...client import EarlClient
from ...exceptions import EarlError
from ..storage.config_store import ConfigStore
from ..ui import (
    ask_confirm,
    ask_int,
    ask_text,
    console,
    datatable,
    error,
    info_panel,
    kvtable,
    muted,
    select_many,
    select_one,
    success,
    warn,
)


# ── Role detection ──────────────────────────────────────────────────────────


ROLE_GLOBAL_ADMIN = "EARL_Admin"
ROLE_ORG_ADMIN = "EARL_Org_Admin"


def _decode_roles_and_org(client: EarlClient) -> tuple[list[str], str]:
    """Inspect the bearer token to find the caller's roles and home org.

    Returns ``(roles, org_id)``. On any error — missing token, network
    failure decoding, malformed claims — returns ``([], "")`` so callers
    can fall through to the "no permissions" branch without crashing.
    """
    try:
        token = client._auth.get_token()  # noqa: SLF001 — in-tree UI
    except Exception:
        return [], ""
    try:
        from earl_sdk.device_flow import decode_jwt_payload

        claims = decode_jwt_payload(token) or {}
    except Exception:
        return [], ""

    roles: list[str] = []
    for key in (
        "https://earl/roles",
        "https://earl.thelumos.ai/roles",
        "https://earl.thelumos.xyz/roles",
        "https://earl-api.thelumos.xyz/roles",
        "https://api.earl.thelumos.ai/roles",
    ):
        val = claims.get(key)
        if isinstance(val, list):
            roles = [str(r) for r in val]
            break
    org_id = ""
    for k in ("org_id", "organization_id"):
        v = claims.get(k)
        if isinstance(v, str) and v:
            org_id = v
            break
    return roles, org_id


def _is_global_admin(roles: list[str]) -> bool:
    return any(r.lower() == ROLE_GLOBAL_ADMIN.lower() for r in roles)


def _is_org_admin(roles: list[str]) -> bool:
    return any(r.lower() == ROLE_ORG_ADMIN.lower() for r in roles)


# ── Backend wrappers routed through the typed ``client.admin`` surface ─────

# Historically the TUI reached into ``client.auth._request`` with
# ``_svc_get/_post/_patch/_delete`` helpers (tagged with SLF001 noqa).
# The typed :class:`earl_sdk.admin_api.OrgAdminAPI` now exposes the same
# routes as real methods; these wrappers exist only to bridge the legacy
# call sites below until each flow is migrated to the typed API one by one.


def _svc_get(client: EarlClient, path: str) -> dict:
    return client.admin.orgs._get(path)  # noqa: SLF001


def _svc_post(client: EarlClient, path: str, body: dict | None) -> dict:
    return client.admin.orgs._post(path, body)  # noqa: SLF001


def _svc_patch(client: EarlClient, path: str, body: dict | None) -> dict:
    return client.admin.orgs._patch(path, body)  # noqa: SLF001


def _svc_delete(client: EarlClient, path: str, body: dict | None = None) -> dict:
    return client.admin.orgs._delete(path, body)  # noqa: SLF001


# ── Entry point ────────────────────────────────────────────────────────────


def flow_admin(client: EarlClient, store: ConfigStore) -> None:
    """Top-level admin menu (role-aware)."""
    roles, caller_org_id = _decode_roles_and_org(client)
    if not _is_global_admin(roles) and not _is_org_admin(roles):
        info_panel(
            "Admin menu unavailable",
            [
                "The current session has no admin roles.",
                "",
                "If you are an EARL_Admin or EARL_Org_Admin, log in with "
                "[bold]earl login[/] and re-open this menu.",
                "",
                "Otherwise ask your operator to grant "
                f"[bold]{ROLE_ORG_ADMIN}[/] on your organization.",
                "",
                f"Detected roles: {roles or '(none)'}",
            ],
            border="yellow",
        )
        return

    while True:
        choices: list[tuple[str, str]] = []
        if _is_global_admin(roles):
            choices.append(
                ("orgs", "Organizations        — list, create, update, delete")
            )
        if _is_org_admin(roles) or _is_global_admin(roles):
            choices.extend([
                ("members", "Members              — list, remove users in an org"),
                ("invites", "Invitations          — invite users, revoke pending invites"),
                ("svc_accts", "Service Accounts     — create/revoke M2M credentials"),
            ])
        if _is_global_admin(roles):
            choices.append(
                ("roles", "Roles                — grant/revoke org roles, view catalog")
            )
        choices.append(("refresh", "Refresh role view    — re-decode the token"))

        action = select_one(
            f"Admin (env={client.environment}, you={caller_org_id or '(no org)'})",
            choices,
        )
        if action is None:
            return
        if action == "orgs":
            _flow_orgs(client)
        elif action == "members":
            _flow_members(client, roles, caller_org_id)
        elif action == "invites":
            _flow_invitations(client, roles, caller_org_id)
        elif action == "svc_accts":
            _flow_service_accounts(client, caller_org_id)
        elif action == "roles":
            _flow_roles(client, caller_org_id)
        elif action == "refresh":
            roles, caller_org_id = _decode_roles_and_org(client)
            success(f"Roles refreshed: {roles or '(none)'}")


# ── Organizations (global admin) ───────────────────────────────────────────


def _flow_orgs(client: EarlClient) -> None:
    while True:
        action = select_one("Organizations", [
            ("list", "List all organizations on this tenant"),
            ("show", "Show one organization"),
            ("create", "Create a new organization"),
            ("update", "Update display name / metadata"),
            ("delete", "Delete an organization (destructive)"),
        ])
        if action is None:
            return
        if action == "list":
            _orgs_list(client)
        elif action == "show":
            org_id = ask_text("Organization id to show")
            if not org_id:
                continue
            _orgs_show(client, org_id)
        elif action == "create":
            _orgs_create(client)
        elif action == "update":
            _orgs_update(client)
        elif action == "delete":
            _orgs_delete(client)


def _orgs_list(client: EarlClient) -> Optional[list[dict]]:
    try:
        resp = _svc_get(client, "/organizations")
    except EarlError as exc:
        error(f"List failed: {exc}")
        return None
    orgs = resp.get("organizations") or []
    if not orgs:
        muted("No organizations on this tenant.")
        return []
    datatable(
        [("id", "cyan"), ("name", "white"), ("display_name", "dim")],
        [[o.get("id", ""), o.get("name", ""), o.get("display_name", "")] for o in orgs],
        title="Organizations",
    )
    return orgs


def _orgs_show(client: EarlClient, org_id: str) -> None:
    try:
        org = _svc_get(client, f"/organizations/{org_id}")
    except EarlError as exc:
        error(f"Show failed: {exc}")
        return
    kvtable(
        [
            ("id", str(org.get("id", ""))),
            ("name", str(org.get("name", ""))),
            ("display_name", str(org.get("display_name", ""))),
        ],
        title=f"Organization {org_id}",
    )
    md = org.get("metadata") or {}
    if md:
        muted("metadata:")
        for k, v in md.items():
            muted(f"  {k}: {v}")


def _orgs_create(client: EarlClient) -> None:
    name = ask_text(
        "Slug (lowercase letters/digits/hyphens/underscores, 3-50 chars)"
    )
    if not name:
        return
    display = ask_text("Display name (optional, e.g. 'Acme Corp')", default="")
    body: dict[str, Any] = {"name": name}
    if display:
        body["display_name"] = display
    if not ask_confirm(f"Create organization {name!r}?", default=True):
        return
    try:
        created = _svc_post(client, "/organizations", body)
    except EarlError as exc:
        error(f"Create failed: {exc}")
        return
    success(f"Created {created.get('id')} ({created.get('name')})")


def _orgs_update(client: EarlClient) -> None:
    org_id = ask_text("Organization id to update")
    if not org_id:
        return
    display = ask_text("New display name (blank = leave unchanged)", default="")
    body: dict[str, Any] = {}
    if display:
        body["display_name"] = display
    if not body:
        muted("Nothing to update.")
        return
    if not ask_confirm(f"Apply update to {org_id}?", default=True):
        return
    try:
        _svc_patch(client, f"/organizations/{org_id}", body)
    except EarlError as exc:
        error(f"Update failed: {exc}")
        return
    success(f"Updated {org_id}.")


def _orgs_delete(client: EarlClient) -> None:
    org_id = ask_text("Organization id to DELETE")
    if not org_id:
        return
    warn(
        f"This is destructive. Users of {org_id} will lose access and the "
        "organization will be removed from Auth0."
    )
    confirm = ask_text(
        f"Type the org id ({org_id}) to confirm, or leave blank to abort"
    )
    if confirm != org_id:
        muted("Aborted.")
        return
    try:
        _svc_delete(client, f"/organizations/{org_id}")
    except EarlError as exc:
        error(f"Delete failed: {exc}")
        return
    success(f"Deleted {org_id}.")


# ── Members (org-admin or global admin) ───────────────────────────────────


def _pick_org(
    client: EarlClient,
    caller_org_id: str,
    *,
    allow_cross_org: bool,
) -> Optional[str]:
    """Ask the user for an org id, defaulting to their own for org admins."""
    if not allow_cross_org:
        if not caller_org_id:
            error(
                "Your token has no org_id claim. Re-run `earl login` and pick an "
                "organization."
            )
            return None
        return caller_org_id
    choice = ask_text(
        f"Target organization id (blank = {caller_org_id or 'list all first'})",
        default=caller_org_id or "",
    )
    if not choice:
        # Global admin without default — offer to list.
        if ask_confirm("List organizations first?", default=True):
            orgs = _orgs_list(client) or []
            ids = [o.get("id", "") for o in orgs if o.get("id")]
            if ids:
                pick = select_one(
                    "Pick an organization",
                    [(i, i) for i in ids],
                )
                return pick
        return None
    return choice


def _flow_members(
    client: EarlClient, roles: list[str], caller_org_id: str
) -> None:
    while True:
        action = select_one("Members", [
            ("list", "List members of an organization"),
            ("remove", "Remove a member from an organization"),
        ])
        if action is None:
            return
        org_id = _pick_org(
            client,
            caller_org_id,
            allow_cross_org=_is_global_admin(roles),
        )
        if not org_id:
            continue
        if action == "list":
            _members_list(client, org_id)
        elif action == "remove":
            _members_remove(client, org_id)


def _members_list(client: EarlClient, org_id: str) -> list[dict]:
    try:
        resp = _svc_get(client, f"/organizations/{org_id}/members")
    except EarlError as exc:
        error(f"List failed: {exc}")
        return []
    members = resp.get("members") or []
    if not members:
        muted(f"No members in {org_id}.")
        return []
    datatable(
        [
            ("user_id", "cyan"),
            ("email", "white"),
            ("name", "dim"),
            ("roles", "magenta"),
        ],
        [
            [
                m.get("user_id", ""),
                m.get("email", ""),
                m.get("name", ""),
                " ".join(m.get("roles") or []),
            ]
            for m in members
        ],
        title=f"Members of {org_id}",
    )
    return members


def _members_remove(client: EarlClient, org_id: str) -> None:
    members = _members_list(client, org_id)
    if not members:
        return
    target = select_one(
        "Remove which member?",
        [
            (m.get("user_id", ""), f"{m.get('email') or '(no email)'} {m.get('user_id', '')}")
            for m in members
            if m.get("user_id")
        ],
    )
    if not target:
        return
    if not ask_confirm(
        f"Remove {target} from {org_id}? (they keep their Auth0 account)",
        default=False,
    ):
        return
    try:
        _svc_delete(client, f"/organizations/{org_id}/members/{target}")
    except EarlError as exc:
        error(f"Remove failed: {exc}")
        return
    success(f"Removed {target} from {org_id}.")


# ── Invitations ────────────────────────────────────────────────────────────


def _flow_invitations(
    client: EarlClient, roles: list[str], caller_org_id: str
) -> None:
    while True:
        action = select_one("Invitations", [
            ("invite", "Invite a user by email"),
            ("list", "List pending invitations"),
            ("revoke", "Revoke a pending invitation"),
        ])
        if action is None:
            return
        org_id = _pick_org(
            client,
            caller_org_id,
            allow_cross_org=_is_global_admin(roles),
        )
        if not org_id:
            continue
        if action == "invite":
            _invitations_create(client, org_id, allow_earl_admin=_is_global_admin(roles))
        elif action == "list":
            _invitations_list(client, org_id)
        elif action == "revoke":
            _invitations_revoke(client, org_id)


def _invitations_create(
    client: EarlClient, org_id: str, *, allow_earl_admin: bool
) -> None:
    email = ask_text("Invitee email")
    if not email:
        return
    role_choices = [(ROLE_ORG_ADMIN, f"{ROLE_ORG_ADMIN} — full admin of {org_id}")]
    if allow_earl_admin:
        role_choices.append(
            (ROLE_GLOBAL_ADMIN, f"{ROLE_GLOBAL_ADMIN} — tenant-wide (careful!)")
        )
    granted = select_many(
        "Roles to grant on accept (space to toggle, Enter to confirm; "
        "skip for a plain member)",
        role_choices,
    )
    ttl_days: Optional[int] = None
    if ask_confirm("Override invitation TTL? (default = 7 days)", default=False):
        ttl_days = ask_int("TTL in days", default=7, min_val=1, max_val=30)
    body: dict[str, Any] = {"email": email}
    if granted:
        body["roles"] = list(granted)
    if ttl_days:
        body["ttl_seconds"] = ttl_days * 24 * 3600
    try:
        resp = _svc_post(
            client, f"/organizations/{org_id}/invitations", body
        )
    except EarlError as exc:
        error(f"Invite failed: {exc}")
        return
    success(
        f"Invited {resp.get('email')} to {resp.get('organization_id')} "
        f"(id={resp.get('id')}, expires {resp.get('expires_at') or 'default'})"
    )
    url = resp.get("invitation_url")
    if url:
        info_panel(
            "Acceptance link — share now, we cannot recover it",
            [url],
            border="yellow",
        )


def _invitations_list(client: EarlClient, org_id: str) -> list[dict]:
    try:
        resp = _svc_get(client, f"/organizations/{org_id}/invitations")
    except EarlError as exc:
        error(f"List failed: {exc}")
        return []
    invs = resp.get("invitations") or []
    if not invs:
        muted(f"No pending invitations in {org_id}.")
        return []
    datatable(
        [("id", "cyan"), ("email", "white"), ("expires_at", "dim"),
         ("roles", "magenta")],
        [
            [i.get("id", ""), i.get("email", ""), i.get("expires_at", ""),
             " ".join(i.get("roles") or [])]
            for i in invs
        ],
        title=f"Pending invitations in {org_id}",
    )
    return invs


def _invitations_revoke(client: EarlClient, org_id: str) -> None:
    invs = _invitations_list(client, org_id)
    if not invs:
        return
    target = select_one(
        "Revoke which invitation?",
        [
            (i.get("id", ""), f"{i.get('email', '?')} (expires {i.get('expires_at', '?')})")
            for i in invs
            if i.get("id")
        ],
    )
    if not target:
        return
    if not ask_confirm(f"Revoke invitation {target}?", default=False):
        return
    try:
        _svc_delete(client, f"/organizations/{org_id}/invitations/{target}")
    except EarlError as exc:
        error(f"Revoke failed: {exc}")
        return
    success(f"Revoked invitation {target}.")


# ── Service accounts (org-admin or global admin) ──────────────────────────


# Keep in lockstep with ``_ALLOWED_SCOPES`` in
# ``src/orchestrator/routes/service_accounts.py`` -- offering scopes the
# backend rejects would surface as opaque 422s in the TUI.
EARL_SCOPES = [
    "earl:read",
    "earl:deploy",
    "earl:admin",
]


def _flow_service_accounts(client: EarlClient, caller_org_id: str) -> None:
    while True:
        action = select_one("Service Accounts", [
            ("list", "List this org's service accounts"),
            ("create", "Create a new service account"),
            ("revoke", "Revoke a service account"),
        ])
        if action is None:
            return
        if action == "list":
            _svc_list(client)
        elif action == "create":
            _svc_create(client)
        elif action == "revoke":
            _svc_revoke(client)


def _svc_list(client: EarlClient) -> list[dict]:
    try:
        resp = _svc_get(client, "/service-accounts")
    except EarlError as exc:
        error(f"List failed: {exc}")
        return []
    rows = resp.get("service_accounts") or resp.get("items") or []
    if not rows:
        muted("No service accounts in this org.")
        return []
    datatable(
        [("name", "cyan"), ("client_id", "white"), ("scopes", "magenta"),
         ("created_by", "dim"), ("created_at", "dim")],
        [
            [
                r.get("name", ""),
                r.get("auth0_client_id") or r.get("client_id", ""),
                " ".join(r.get("scopes") or []),
                r.get("created_by", ""),
                r.get("created_at", ""),
            ]
            for r in rows
        ],
        title="Service accounts",
    )
    return rows


def _svc_create(client: EarlClient) -> None:
    name = ask_text("Name (e.g. prod-ci, nightly-deploy)")
    if not name:
        return
    picked = select_many(
        "Scopes to grant (space to toggle, Enter to confirm):",
        [(s, s) for s in EARL_SCOPES],
    )
    if not picked:
        warn("No scopes picked; service account will have zero API access.")
        if not ask_confirm("Continue anyway?", default=False):
            return
    desc = ask_text("Optional description", default="")
    body: dict[str, Any] = {"name": name, "scopes": list(picked)}
    if desc:
        body["description"] = desc
    if not ask_confirm(
        f"Create service account {name!r} with scopes={list(picked)}?",
        default=True,
    ):
        return
    try:
        resp = _svc_post(client, "/service-accounts", body)
    except EarlError as exc:
        error(f"Create failed: {exc}")
        return
    cid = resp.get("client_id", "")
    secret = resp.get("client_secret", "")
    info_panel(
        "Credentials — copy them NOW. The secret cannot be recovered.",
        [
            f"EARL_CLIENT_ID={cid}",
            f"EARL_CLIENT_SECRET={secret}",
            f"EARL_ORG_ID={resp.get('org_id', '')}",
            "",
            "Export these three env vars in CI to use this account.",
        ],
        border="yellow",
    )


def _svc_revoke(client: EarlClient) -> None:
    rows = _svc_list(client)
    if not rows:
        return
    target = select_one(
        "Revoke which service account?",
        [
            (
                r.get("auth0_client_id") or r.get("client_id", ""),
                f"{r.get('name', '?')} ({r.get('auth0_client_id') or r.get('client_id', '')})",
            )
            for r in rows
            if r.get("auth0_client_id") or r.get("client_id")
        ],
    )
    if not target:
        return
    if not ask_confirm(
        f"Revoke {target}? Any CI using this client_id will start getting 401s.",
        default=False,
    ):
        return
    try:
        _svc_delete(client, f"/service-accounts/{target}")
    except EarlError as exc:
        error(f"Revoke failed: {exc}")
        return
    success(f"Revoked {target}.")


# ── Roles (global admin only) ──────────────────────────────────────────────


def _flow_roles(client: EarlClient, caller_org_id: str) -> None:
    while True:
        action = select_one("Roles", [
            ("catalog", "Show the role catalog (assignable tenant-level roles)"),
            ("grant", "Grant a role to a member"),
            ("revoke", "Revoke a role from a member"),
        ])
        if action is None:
            return
        if action == "catalog":
            _roles_catalog(client)
        elif action == "grant":
            _roles_grant(client)
        elif action == "revoke":
            _roles_revoke(client)


def _roles_catalog(client: EarlClient) -> list[dict]:
    try:
        resp = _svc_get(client, "/organizations/_roles/catalog")
    except EarlError as exc:
        error(f"Catalog failed: {exc}")
        return []
    roles = resp.get("roles") or []
    if not roles:
        muted("No assignable tenant-level roles configured.")
        return []
    datatable(
        [("name", "cyan"), ("description", "dim")],
        [[r.get("name", ""), r.get("description", "")] for r in roles],
        title="Tenant role catalog",
    )
    return roles


def _roles_grant(client: EarlClient) -> None:
    org_id = ask_text("Organization id")
    if not org_id:
        return
    user_id = ask_text("User id (Auth0 sub, e.g. auth0|abc123)")
    if not user_id:
        return
    catalog = _roles_catalog(client)
    if not catalog:
        return
    picked = select_many(
        "Roles to grant (space to toggle, Enter to confirm):",
        [(r.get("name", ""), r.get("name", "")) for r in catalog if r.get("name")],
    )
    if not picked:
        muted("Nothing picked.")
        return
    if not ask_confirm(
        f"Grant {list(picked)} to {user_id} in {org_id}?", default=True
    ):
        return
    try:
        _svc_post(
            client,
            f"/organizations/{org_id}/members/{user_id}/roles",
            {"roles": list(picked)},
        )
    except EarlError as exc:
        error(f"Grant failed: {exc}")
        return
    success(f"Granted {list(picked)} to {user_id}.")


def _roles_revoke(client: EarlClient) -> None:
    org_id = ask_text("Organization id")
    if not org_id:
        return
    user_id = ask_text("User id (Auth0 sub)")
    if not user_id:
        return
    role_name = ask_text("Role to revoke (e.g. EARL_Org_Admin)")
    if not role_name:
        return
    if not ask_confirm(
        f"Revoke role {role_name!r} from {user_id} in {org_id}?", default=False
    ):
        return
    try:
        _svc_delete(
            client,
            f"/organizations/{org_id}/members/{user_id}/roles",
            {"roles": [role_name]},
        )
    except EarlError as exc:
        error(f"Revoke failed: {exc}")
        return
    success(f"Revoked {role_name} from {user_id}.")
