"""Auth layer integration tests.

These exercise the invariants every other test depends on:

* Protected endpoints reject unauthenticated requests with 401.
* A logged-in CLI profile can call ``whoami`` and see the expected org.
* ``auth my-orgs`` returns at least the active org.
* A token minted for env X is rejected by env Y (audience check).
* Refresh works — the CLI transparently mints fresh access tokens from a
  cached refresh token across subprocess invocations, without re-prompting.
"""

from __future__ import annotations

from .cli_runner import CliRunner


# --------------------------------------------------------------------------- #
# Unauthenticated access
# --------------------------------------------------------------------------- #


def test_whoami_requires_auth(unauth_cli: CliRunner) -> None:
    """Scratch ``$HOME`` + no M2M env vars + no ``--profile`` must fail with
    a clear auth-related error on any authenticated endpoint.

    Documents::

        earl whoami   # no profile, no EARL_CLIENT_ID/SECRET \u2192 must fail
    """
    result = unauth_cli.run_expect_fail("whoami")
    combined = (result.stdout + result.stderr).lower()
    assert any(
        kw in combined
        for kw in ("auth", "unauthor", "401", "client", "credentials", "login")
    ), f"expected auth-related error, got:\nstdout={result.stdout}\nstderr={result.stderr}"


def test_cases_list_requires_auth(unauth_cli: CliRunner) -> None:
    """Same check via a data-plane endpoint.

    Documents::

        earl cases list   # unauthenticated \u2192 401
    """
    result = unauth_cli.run_expect_fail("cases", "list")
    assert not result.ok


# --------------------------------------------------------------------------- #
# Authenticated happy path
# --------------------------------------------------------------------------- #


def test_whoami_happy_path(
    cli: CliRunner, active_org_id: str, target_env: str
) -> None:
    """``earl whoami`` returns the resolved connection metadata \u2014 env, org,
    and API URLs \u2014 pulled from the active profile.

    Note: this is a local-only command (no backend round trip). The backend
    identity check is covered by ``test_my_orgs_lists_active_org``.

    Documents::

        earl --profile <pkce-dev-...> whoami
    """
    result = cli.run("whoami")
    data = result.json()
    assert data.get("organization") == active_org_id, data
    assert data.get("environment") == target_env, data
    assert data.get("api_url", "").startswith("https://"), data


def test_auth_test_connection(cli: CliRunner) -> None:
    """``earl auth test`` exercises an actual backend round trip with the
    active profile's token. Complements ``whoami``, which is local-only.

    Documents::

        earl auth test
    """
    result = cli.run("auth", "test", json_output=False)
    assert "OK" in result.stdout or result.ok, (
        f"unexpected output: {result.stdout!r}, stderr={result.stderr!r}"
    )


def test_my_orgs_lists_active_org(cli: CliRunner, active_org_id: str) -> None:
    """``earl auth my-orgs`` includes the org the active token was issued for.

    Documents::

        earl auth my-orgs
    """
    result = cli.run("auth", "my-orgs")
    data = result.json()
    orgs = data.get("organizations", data if isinstance(data, list) else [])
    org_ids = {o.get("id") or o.get("org_id") for o in orgs}
    assert active_org_id in org_ids, (
        f"active org {active_org_id} not in my-orgs response: {data}"
    )


# --------------------------------------------------------------------------- #
# Cross-environment audience rejection
# --------------------------------------------------------------------------- #


def test_token_rejected_by_other_env(cli: CliRunner) -> None:
    """A token minted for env X must be rejected when sent to env Y.

    The CLI's ``--api-url`` flag lets us aim any command at an arbitrary base
    URL while keeping the profile's Auth0 audience (= the profile's env).
    The other env's API validates ``aud`` and must reject.

    Documents the regression check::

        earl --profile <dev-profile> --api-url https://earl-api.thelumos.xyz/api/v1 \\
             cases list   # must FAIL with audience / 401
    """
    other_env = "staging" if cli.env == "dev" else "dev"
    other_api = {
        "dev": "https://earl-api.thelumos.dev/api/v1",
        "staging": "https://earl-api.thelumos.xyz/api/v1",
    }[other_env]

    result = cli.run_expect_fail("--api-url", other_api, "cases", "list")
    blob = (result.stdout + result.stderr).lower()
    assert any(
        kw in blob for kw in ("audience", "unauth", "401", "invalid", "token")
    ), f"expected audience/401 error, got:\nstdout={result.stdout}\nstderr={result.stderr}"


# --------------------------------------------------------------------------- #
# Refresh flow (cached refresh token across subprocess invocations)
# --------------------------------------------------------------------------- #


def test_refresh_flow_still_works(cli: CliRunner) -> None:
    """Two authenticated calls in sequence must both succeed without prompting
    for re-login. Every ``earl`` call is a fresh Python process, so this
    implicitly exercises "pick up cached refresh token from
    ``~/.earl/config.json`` and mint a new access token if needed".

    Documents::

        earl auth my-orgs     # backend call #1 \u2014 may refresh silently
        earl cases list       # backend call #2 \u2014 uses cached / refreshed token
    """
    first = cli.run("auth", "my-orgs", timeout=30).json()
    orgs = first.get("organizations", first if isinstance(first, list) else [])
    assert isinstance(orgs, list) and orgs, first

    second = cli.run("cases", "list", timeout=30).json()
    assert isinstance(second, (list, dict)), second


# --------------------------------------------------------------------------- #
# Auth profile plumbing
# --------------------------------------------------------------------------- #


def test_auth_profile_list_includes_active(
    cli: CliRunner, active_profile: str
) -> None:
    """``earl auth profile list`` shows the active profile.

    Documents::

        earl auth profile list
    """
    result = cli.run("auth", "profile", "list")
    try:
        data = result.json()
        rows = data.get("profiles", data if isinstance(data, list) else [])
        names = [p.get("name") for p in rows if isinstance(p, dict)]
        assert active_profile in names, f"{active_profile} not in {names}"
    except ValueError:
        assert active_profile in result.stdout, (
            f"{active_profile!r} not in:\n{result.stdout}"
        )
