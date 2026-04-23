"""Typed client for the orchestrator's admin HTTP surface.

Before this module existed, both the TUI (:mod:`sdk.earl_sdk.interactive.flows.admin`)
and the CLI admin group reached into ``client.auth._request(...)`` with hand-rolled
``_svc_get/_post/_patch/_delete`` helpers and ``# noqa: SLF001`` markers.

The goal here is to expose exactly one stable, typed surface — ``client.admin.*`` —
that mirrors the routes under ``/api/v1/organizations``, ``/api/v1/invitations``,
and ``/api/v1/service-accounts`` so UIs just call methods and the underscore-private
request plumbing stays inside the SDK.

Return types are intentionally ``dict``/``list[dict]``: the orchestrator owns the
canonical response shape, and baking dataclasses in here would force lockstep
releases between the SDK and the backend for additive fields.
"""
from __future__ import annotations

from typing import Any

from .api import BaseAPI
from .auth import Auth0Client

_API_PREFIX = "/api/v1"


class _AdminSubAPI(BaseAPI):
    """Shared request plumbing for admin sub-clients.

    Subclassing ``BaseAPI`` lets us reuse ``_request`` through inheritance
    instead of reaching into another class's private API (which is what the
    old ``_svc_*`` helpers did under ``# noqa: SLF001``).
    """

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        return self._request("GET", f"{_API_PREFIX}{path}", params=params)

    def _post(self, path: str, body: dict | None = None) -> dict:
        return self._request("POST", f"{_API_PREFIX}{path}", data=body)

    def _patch(self, path: str, body: dict | None = None) -> dict:
        return self._request("PATCH", f"{_API_PREFIX}{path}", data=body)

    def _delete(self, path: str, body: dict | None = None) -> dict:
        return self._request("DELETE", f"{_API_PREFIX}{path}", data=body)


class OrganizationsAdminAPI(_AdminSubAPI):
    """CRUD + pagination for ``/organizations``."""

    def list(
        self, *, cursor: str | None = None, limit: int | None = None
    ) -> dict:
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return self._get("/organizations", params=params or None)

    def iter_all(self, *, limit: int | None = None) -> list[dict]:
        """Walk the paginated list to completion. Convenience for CLIs."""
        items: list[dict] = []
        cursor: str | None = None
        while True:
            page = self.list(cursor=cursor, limit=limit)
            items.extend(page.get("items") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                return items

    def get(self, org_id: str) -> dict:
        return self._get(f"/organizations/{org_id}")

    def create(
        self,
        *,
        name: str,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name}
        if display_name is not None:
            body["display_name"] = display_name
        if metadata is not None:
            body["metadata"] = metadata
        return self._post("/organizations", body)

    def update(
        self,
        org_id: str,
        *,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        body: dict[str, Any] = {}
        if display_name is not None:
            body["display_name"] = display_name
        if metadata is not None:
            body["metadata"] = metadata
        return self._patch(f"/organizations/{org_id}", body)

    def delete(self, org_id: str) -> dict:
        return self._delete(f"/organizations/{org_id}")


class MembersAdminAPI(_AdminSubAPI):
    """``/organizations/{id}/members`` and role grants."""

    def list(
        self, org_id: str, *, cursor: str | None = None, limit: int | None = None
    ) -> dict:
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return self._get(f"/organizations/{org_id}/members", params=params or None)

    def remove(self, org_id: str, user_id: str) -> dict:
        return self._delete(f"/organizations/{org_id}/members/{user_id}")

    def grant_roles(self, org_id: str, user_id: str, roles: list[str]) -> dict:
        return self._post(
            f"/organizations/{org_id}/members/{user_id}/roles", {"roles": roles}
        )

    def revoke_roles(self, org_id: str, user_id: str, roles: list[str]) -> dict:
        return self._delete(
            f"/organizations/{org_id}/members/{user_id}/roles", {"roles": roles}
        )

    def roles_catalog(self) -> dict:
        return self._get("/organizations/_roles/catalog")


class InvitationsAdminAPI(_AdminSubAPI):
    """``/organizations/{id}/invitations``."""

    def list(
        self, org_id: str, *, cursor: str | None = None, limit: int | None = None
    ) -> dict:
        params: dict[str, Any] = {}
        if cursor is not None:
            params["cursor"] = cursor
        if limit is not None:
            params["limit"] = limit
        return self._get(f"/organizations/{org_id}/invitations", params=params or None)

    def create(
        self,
        org_id: str,
        *,
        email: str,
        roles: list[str] | None = None,
        send_invitation_email: bool | None = None,
    ) -> dict:
        body: dict[str, Any] = {"email": email}
        if roles is not None:
            body["roles"] = roles
        if send_invitation_email is not None:
            body["send_invitation_email"] = send_invitation_email
        return self._post(f"/organizations/{org_id}/invitations", body)

    def revoke(self, org_id: str, invitation_id: str) -> dict:
        return self._delete(f"/organizations/{org_id}/invitations/{invitation_id}")


class ServiceAccountsAdminAPI(_AdminSubAPI):
    """``/service-accounts`` (M2M client provisioning)."""

    def list(self) -> dict:
        return self._get("/service-accounts")

    def create(
        self,
        *,
        name: str,
        scopes: list[str] | None = None,
        description: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"name": name}
        if scopes is not None:
            body["scopes"] = scopes
        if description is not None:
            body["description"] = description
        return self._post("/service-accounts", body)

    def revoke(self, client_id: str) -> dict:
        return self._delete(f"/service-accounts/{client_id}")

    def reconcile_orphans(self) -> dict:
        """Global-admin endpoint: reconcile dangling Auth0 clients."""
        return self._post("/service-accounts/admin/reconcile-orphans")


class OrgAdminAPI:
    """Top-level ``client.admin`` namespace.

    Pulls the four sub-clients together so callers can just do
    ``client.admin.orgs.list()`` / ``client.admin.members.remove(...)`` / etc.
    """

    def __init__(
        self,
        auth: Auth0Client,
        base_url: str,
        request_timeout: int | None = None,
    ) -> None:
        self.orgs = OrganizationsAdminAPI(auth, base_url, request_timeout)
        self.members = MembersAdminAPI(auth, base_url, request_timeout)
        self.invitations = InvitationsAdminAPI(auth, base_url, request_timeout)
        self.service_accounts = ServiceAccountsAdminAPI(auth, base_url, request_timeout)


__all__ = [
    "OrgAdminAPI",
    "OrganizationsAdminAPI",
    "MembersAdminAPI",
    "InvitationsAdminAPI",
    "ServiceAccountsAdminAPI",
]
