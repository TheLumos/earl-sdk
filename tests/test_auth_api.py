"""Tests for :class:`earl_sdk.api.AuthAPI`.

Focused on the ``my_orgs()`` endpoint shape, envelope tolerance, and error
propagation — we don't re-test the underlying httpx transport (that's covered
in ``test_http_transport.py``).
"""

from __future__ import annotations

from unittest import mock

import httpx
import pytest

from earl_sdk import _http
from earl_sdk.api import AuthAPI
from earl_sdk.exceptions import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
)


@pytest.fixture(autouse=True)
def _clean_shared_client():
    _http.close_shared_client()
    yield
    _http.close_shared_client()


def _install(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    _http.set_shared_client(client)
    return client


def _fake_auth():
    """Minimal stand-in for :class:`earl_sdk.auth.Auth0Client`.

    ``AuthAPI`` only touches ``auth.get_headers()`` and ``auth.invalidate_token()``.
    """
    a = mock.MagicMock()
    a.get_headers.return_value = {"Authorization": "Bearer TEST-TOK"}
    return a


# ── my_orgs() happy paths ───────────────────────────────────────────────────


def test_my_orgs_parses_envelope_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url).endswith("/auth/my-orgs")
        assert request.headers["authorization"] == "Bearer TEST-TOK"
        return httpx.Response(
            200,
            json={
                "organizations": [
                    {"id": "org_aLx", "name": "lumos", "display_name": "Lumos admins"},
                    {"id": "org_PZF", "name": "tempus", "display_name": "Tempus AI"},
                ]
            },
        )

    _install(handler)
    api = AuthAPI(_fake_auth(), "https://api.example.com/api/v1")
    orgs = api.my_orgs()

    assert len(orgs) == 2
    assert orgs[0]["id"] == "org_aLx"
    assert orgs[0]["display_name"] == "Lumos admins"
    assert orgs[1]["name"] == "tempus"


def test_my_orgs_tolerates_items_envelope():
    """Forward-compat: a future server release may rename the envelope key."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"items": [{"id": "org_x", "name": "x", "display_name": "X"}]},
        )

    _install(handler)
    api = AuthAPI(_fake_auth(), "https://api.example.com/api/v1")
    assert api.my_orgs() == [{"id": "org_x", "name": "x", "display_name": "X"}]


def test_my_orgs_tolerates_bare_list_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"id": "org_x", "name": "x", "display_name": "X"}],
        )

    _install(handler)
    api = AuthAPI(_fake_auth(), "https://api.example.com/api/v1")
    assert api.my_orgs() == [{"id": "org_x", "name": "x", "display_name": "X"}]


def test_my_orgs_returns_empty_when_user_has_no_orgs():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"organizations": []})

    _install(handler)
    api = AuthAPI(_fake_auth(), "https://api.example.com/api/v1")
    assert api.my_orgs() == []


def test_my_orgs_returns_empty_for_unexpected_shape():
    """A server giving us garbage shouldn't crash the CLI picker."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"some_other_key": "value"})

    _install(handler)
    api = AuthAPI(_fake_auth(), "https://api.example.com/api/v1")
    assert api.my_orgs() == []


# ── Error paths ─────────────────────────────────────────────────────────────


def test_my_orgs_401_invalidates_token_and_raises():
    """401 from the orchestrator should clear the cached bearer so the next
    call re-runs the device flow / refresh exchange."""
    auth = _fake_auth()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Token expired"})

    _install(handler)
    api = AuthAPI(auth, "https://api.example.com/api/v1")

    with pytest.raises(AuthenticationError):
        api.my_orgs()
    auth.invalidate_token.assert_called_once()


def test_my_orgs_403_surfaces_authorization_error():
    """M2M tokens hit /auth/my-orgs with a 403 (endpoint is user-only)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "User tokens only"})

    _install(handler)
    api = AuthAPI(_fake_auth(), "https://api.example.com/api/v1")
    with pytest.raises(AuthorizationError):
        api.my_orgs()


def test_my_orgs_404_when_feature_not_deployed():
    """Old orchestrator without the my-orgs route → NotFoundError so the CLI
    can fall back to --organization flag with a readable hint."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    _install(handler)
    api = AuthAPI(_fake_auth(), "https://api.example.com/api/v1")
    with pytest.raises(NotFoundError):
        api.my_orgs()
