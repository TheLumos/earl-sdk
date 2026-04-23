"""Service-account lifecycle integration test.

End-to-end happy + sad path for machine-to-machine credentials:

1. A PKCE-authenticated admin user creates a service account via the CLI.
2. The returned ``client_id`` + ``client_secret`` can mint an M2M token
   directly from Auth0 using the client_credentials grant.
3. That M2M token is accepted by the backend — the same CLI works end to
   end when ``EARL_CLIENT_ID`` / ``EARL_CLIENT_SECRET`` / ``EARL_ORG_ID``
   are set.
4. After revocation, the same credentials can no longer mint a token.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
import json as _json

import pytest

from .cli_runner import CliRunner
from .conftest import Cleanup


# --------------------------------------------------------------------------- #
# Helpers for direct Auth0 client_credentials grants
# --------------------------------------------------------------------------- #


def _auth0_for_env(env: str) -> tuple[str, str]:
    """Return (auth0_domain, audience) for a target env."""
    domain = "dev-f4675lf8h3k0i3me.us.auth0.com"
    audience = {
        "dev": "https://earl-api.thelumos.dev",
        "staging": "https://earl-api.thelumos.xyz",
        "prod": "https://earl-api.thelumos.ai",
    }[env]
    return domain, audience


def _mint_m2m_token(
    env: str, client_id: str, client_secret: str, organization: str
) -> dict:
    """Exchange client_credentials for an access token at Auth0 directly."""
    domain, audience = _auth0_for_env(env)
    body = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
            "audience": audience,
            "organization": organization,
        }
    ).encode()

    req = urllib.request.Request(
        f"https://{domain}/oauth/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
        return _json.loads(resp.read().decode())


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_service_account_full_lifecycle(
    cli: CliRunner,
    target_env: str,
    active_org_id: str,
    run_id: str,
    cleanup: Cleanup,
) -> None:
    """Create SA \u2192 mint M2M \u2192 use M2M against backend \u2192 revoke \u2192 denied.

    Documents::

        # as admin user
        earl service-account create --name <name> --scopes "earl:read earl:deploy"

        # as CI (env vars populated from the create response)
        EARL_CLIENT_ID=... EARL_CLIENT_SECRET=... EARL_ORG_ID=... earl cases list

        # back as admin user
        earl service-account revoke <client_id>

        # post-revoke \u2014 token exchange must fail
        curl https://<auth0>/oauth/token   # 401
    """
    sa_name = f"{run_id}-sa"

    # ---- 1. Create the service account --------------------------------
    created = cli.run(
        "service-account",
        "create",
        "--name",
        sa_name,
        "--scopes",
        "earl:read earl:deploy",
        "--description",
        "integration test SA — safe to revoke",
    ).json()

    client_id = created.get("client_id") or created.get("sa_client_id")
    client_secret = created.get("client_secret")
    assert client_id and client_secret, (
        f"service-account create response missing client_id/client_secret: {created}"
    )
    cleanup.service_account(client_id)

    # ---- 2. Mint a token directly via Auth0 ----------------------------
    token = _mint_m2m_token(target_env, client_id, client_secret, active_org_id)
    assert "access_token" in token, token
    assert token.get("token_type", "").lower() == "bearer", token

    # ---- 3. Use the M2M credentials via the CLI itself ------------------
    # ``EARL_ENVIRONMENT`` is required when running without a profile so the
    # CLI knows which Auth0 tenant / audience to use.
    m2m_cli = CliRunner(
        env=target_env,
        profile=None,
        extra_env={
            "EARL_CLIENT_ID": client_id,
            "EARL_CLIENT_SECRET": client_secret,
            "EARL_ORG_ID": active_org_id,
            "EARL_ENVIRONMENT": target_env,
        },
    )
    data = m2m_cli.run("cases", "list").json()
    assert isinstance(data, (list, dict)), data

    whoami = m2m_cli.run("whoami").json()
    # M2M tokens have no user; backend reports org + gty="client-credentials".
    assert whoami.get("organization") == active_org_id, whoami
    gty = whoami.get("gty") or whoami.get("grant_type") or ""
    if gty:
        assert "client" in gty.lower(), whoami

    # ---- 4. Verify the SA shows up in list ------------------------------
    listed = _as_list(cli.run("service-account", "list").json(), "service_accounts")
    listed_ids = [sa.get("client_id") for sa in listed if isinstance(sa, dict)]
    assert client_id in listed_ids, (client_id, listed_ids)

    # ---- 5. Revoke it ---------------------------------------------------
    cli.run("service-account", "revoke", client_id, "--yes")

    # ---- 6. Post-revoke: the same credentials cannot mint a token ------
    #   Auth0 may take a moment to propagate the delete; retry briefly.
    deadline = time.time() + 15
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            _mint_m2m_token(target_env, client_id, client_secret, active_org_id)
        except Exception as e:  # noqa: BLE001
            last_err = e
            break
        time.sleep(2)
    assert last_err is not None, (
        "expected Auth0 to reject the revoked client's token request"
    )

    # ---- 7. Post-revoke: SA no longer appears in list -------------------
    listed_after = _as_list(
        cli.run("service-account", "list").json(), "service_accounts"
    )
    listed_ids_after = [
        sa.get("client_id") for sa in listed_after if isinstance(sa, dict)
    ]
    assert client_id not in listed_ids_after, listed_ids_after


# --------------------------------------------------------------------------- #
# Scope validation
# --------------------------------------------------------------------------- #


def test_service_account_rejects_unknown_scopes(
    cli: CliRunner, run_id: str, cleanup: Cleanup
) -> None:
    """Requesting a scope outside the Earl API's allow-list must be rejected.

    Documents::

        earl service-account create --name <x> --scopes "earl:read earl:write"
        # \u2192 422 invalid_request: Unknown scope(s) ['earl:write']
    """
    bad_name = f"{run_id}-bad-scope"
    result = cli.run_expect_fail(
        "service-account",
        "create",
        "--name",
        bad_name,
        "--scopes",
        "earl:read earl:write",
    )
    cleanup.service_account(bad_name)  # no-op if nothing was created
    blob = (result.stdout + result.stderr).lower()
    assert any(
        kw in blob for kw in ("unknown scope", "invalid_request", "422", "scope")
    ), result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_list(payload: object, key: str) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in (key, "items", "data", "results"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []
