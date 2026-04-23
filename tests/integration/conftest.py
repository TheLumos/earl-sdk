"""Shared pytest fixtures for CLI-driven integration tests.

Environment gates
-----------------

The suite is **opt-in**. Nothing runs without:

    EARL_INTEGRATION_ENV=dev       # or "staging"

Optional for role/tenancy tests:

    EARL_INTEGRATION_NONADMIN_PROFILE=<profile-name>
        A profile in ~/.earl/config.json for a user that has a valid JWT but
        is NOT EARL_Admin / EARL_Org_Admin in the target env. Enables the
        role-enforcement tests; they skip without it.

    EARL_INTEGRATION_SECOND_ORG_PROFILE=<profile-name>
        A profile bound to a different Auth0 organization in the target env.
        Enables the cross-org tenancy tests; they skip without it.

Before running, make sure you have an active PKCE profile for the target env:

    earl login --env dev --organization <org_id>
    earl whoami   # sanity check
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import string
import time
from typing import Iterator

import pytest

from .cli_runner import CliRunner


ENV_VAR = "EARL_INTEGRATION_ENV"
NONADMIN_PROFILE_VAR = "EARL_INTEGRATION_NONADMIN_PROFILE"
SECOND_ORG_PROFILE_VAR = "EARL_INTEGRATION_SECOND_ORG_PROFILE"
_CONFIG_PATH = pathlib.Path.home() / ".earl" / "config.json"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_nonadmin: skip unless a non-admin profile is configured"
    )
    config.addinivalue_line(
        "markers", "requires_second_org: skip unless a second-org profile is configured"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if os.environ.get(ENV_VAR) not in ("dev", "staging", "local", "prod"):
        skip = pytest.mark.skip(
            reason=f"set {ENV_VAR}=dev|staging to run integration tests"
        )
        for item in items:
            # Cassette-backed tests replay offline, so they don't need
            # the env gate. Mark them with ``vcr`` (pytest-recording) and
            # they'll run in CI with no credentials.
            if item.get_closest_marker("vcr") is not None:
                continue
            item.add_marker(skip)
        return

    if not os.environ.get(NONADMIN_PROFILE_VAR):
        skip_nonadmin = pytest.mark.skip(
            reason=f"set {NONADMIN_PROFILE_VAR} to a non-admin CLI profile"
        )
        for item in items:
            if "requires_nonadmin" in item.keywords:
                item.add_marker(skip_nonadmin)

    if not os.environ.get(SECOND_ORG_PROFILE_VAR):
        skip_second = pytest.mark.skip(
            reason=f"set {SECOND_ORG_PROFILE_VAR} to a profile in another org"
        )
        for item in items:
            if "requires_second_org" in item.keywords:
                item.add_marker(skip_second)


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def target_env() -> str:
    env = os.environ.get(ENV_VAR)
    if not env:
        pytest.skip(f"{ENV_VAR} not set")
    return env


@pytest.fixture(scope="session")
def earl_config() -> dict:
    """Load ``~/.earl/config.json`` once per session."""
    if not _CONFIG_PATH.exists():
        pytest.skip(
            f"{_CONFIG_PATH} not found — run `earl login` first to create a profile."
        )
    with _CONFIG_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def active_profile(target_env: str, earl_config: dict) -> str:
    """Profile bound to the target env. Prefer the active profile if it
    matches the target env, otherwise pick the most-recent PKCE profile for
    that env.
    """
    profiles = earl_config.get("profiles", {})
    active = earl_config.get("active_profile", "")

    chosen = ""
    if active and profiles.get(active, {}).get("environment") == target_env:
        chosen = active
    else:
        # Pick any PKCE profile bound to the target env.
        candidates = [
            name
            for name, p in profiles.items()
            if p.get("environment") == target_env and p.get("auth_kind") == "pkce"
        ]
        if candidates:
            chosen = sorted(candidates)[-1]

    if not chosen:
        pytest.skip(
            f"no PKCE profile for env={target_env} in {_CONFIG_PATH}. "
            f"Run `earl login --env {target_env} --organization <org_id>` first."
        )
    return chosen


@pytest.fixture(scope="session")
def active_org_id(earl_config: dict, active_profile: str) -> str:
    org = earl_config["profiles"][active_profile].get("organization", "")
    if not org:
        pytest.skip(
            f"profile {active_profile} has no organization set; "
            f"re-run `earl login` with --organization"
        )
    return org


@pytest.fixture(scope="session")
def cli(target_env: str, active_profile: str) -> CliRunner:
    """A :class:`CliRunner` bound to the target env + active profile."""
    return CliRunner(env=target_env, profile=active_profile)


@pytest.fixture
def unauth_cli(target_env: str, tmp_path: pathlib.Path) -> CliRunner:
    """Runner with no profile, no M2M env, and a scratch ``$HOME`` so the
    CLI cannot fall back to the user's saved active profile.
    """
    (tmp_path / ".earl").mkdir(exist_ok=True)
    return CliRunner(
        env=target_env,
        profile=None,
        extra_env={"HOME": str(tmp_path)},
    )


@pytest.fixture(scope="session")
def nonadmin_cli(target_env: str, earl_config: dict) -> CliRunner:
    """Runner bound to a non-admin profile, if configured."""
    prof = os.environ.get(NONADMIN_PROFILE_VAR)
    if not prof:
        pytest.skip(f"{NONADMIN_PROFILE_VAR} not set")
    if prof not in earl_config.get("profiles", {}):
        pytest.skip(f"profile {prof!r} not found in {_CONFIG_PATH}")
    return CliRunner(env=target_env, profile=prof)


@pytest.fixture(scope="session")
def second_org_cli(target_env: str, earl_config: dict) -> CliRunner:
    """Runner bound to a profile in a different Auth0 organization."""
    prof = os.environ.get(SECOND_ORG_PROFILE_VAR)
    if not prof:
        pytest.skip(f"{SECOND_ORG_PROFILE_VAR} not set")
    profiles = earl_config.get("profiles", {})
    if prof not in profiles:
        pytest.skip(f"profile {prof!r} not found in {_CONFIG_PATH}")
    return CliRunner(env=target_env, profile=prof)


@pytest.fixture(scope="session")
def second_org_id(earl_config: dict) -> str:
    prof = os.environ.get(SECOND_ORG_PROFILE_VAR)
    if not prof:
        pytest.skip(f"{SECOND_ORG_PROFILE_VAR} not set")
    org = earl_config["profiles"].get(prof, {}).get("organization", "")
    if not org:
        pytest.skip(f"profile {prof!r} has no organization set")
    return org


# ---------------------------------------------------------------------------
# Utility fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def run_id() -> str:
    """Short random tag used to namespace per-run resources (pipelines, SAs)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"it-{stamp}-{suffix}"


@pytest.fixture
def tmp_pipeline_name(run_id: str, request: pytest.FixtureRequest) -> str:
    """A deterministic, test-unique pipeline name that is easy to recognise."""
    return f"{run_id}-{request.node.name}"[:60]


@pytest.fixture(scope="session")
def catalog_case_id(cli: CliRunner) -> str | None:
    """Return the id of one case assigned to the active org, or ``None``.

    Some environments don't have any catalog cases assigned to the test org;
    tests that optionally want to exercise the ``--case-id`` surface can use
    this fixture and fall back to "pure extra-verifier" pipelines when it is
    ``None``.
    """
    try:
        data = cli.run("cases", "list", timeout=20).json()
    except AssertionError:
        return None
    items = data if isinstance(data, list) else data.get("cases") if isinstance(data, dict) else []
    if not items:
        return None
    first = items[0]
    if isinstance(first, dict):
        return first.get("id") or first.get("case_id")
    return None


@pytest.fixture(scope="session")
def one_patient_id(cli: CliRunner) -> str:
    """The id of one patient visible to the active org (URL-encoded at call
    sites that need it)."""
    data = cli.run("patients", "list", "--limit", "1", timeout=20).json()
    items = data if isinstance(data, list) else data.get("patients", [])
    if not items:
        pytest.skip("no patients visible to this org")
    pid = items[0].get("id")
    if not pid:
        pytest.skip("first patient has no id")
    return pid


@pytest.fixture
def cleanup(cli: CliRunner) -> Iterator["Cleanup"]:
    """Register resources to be deleted after the test, best-effort."""
    c = Cleanup(cli)
    try:
        yield c
    finally:
        c.run()


class Cleanup:
    def __init__(self, cli: CliRunner):
        self._cli = cli
        self._pipelines: list[str] = []
        self._service_accounts: list[str] = []

    def pipeline(self, name: str) -> None:
        self._pipelines.append(name)

    def service_account(self, client_id: str) -> None:
        self._service_accounts.append(client_id)

    def run(self) -> None:
        for name in self._pipelines:
            try:
                self._cli.run("pipelines", "delete", name, check=False, timeout=20)
            except Exception as e:  # noqa: BLE001
                print(f"[cleanup] pipeline {name!r}: {e}")
        for cid in self._service_accounts:
            try:
                self._cli.run(
                    "service-account",
                    "revoke",
                    cid,
                    "--yes",
                    check=False,
                    timeout=20,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[cleanup] service-account {cid!r}: {e}")


# ---------------------------------------------------------------------------
# pytest-recording / VCR configuration
# ---------------------------------------------------------------------------
#
# Cassette-backed tests replay recorded HTTP traffic by default. Flip
# ``RECORD_CASSETTES=1`` in the environment to refresh the fixtures against
# a real Auth0 / orchestrator deployment. Sensitive headers are redacted so
# committed cassettes don't leak bearer tokens.


@pytest.fixture(scope="module")
def vcr_config() -> dict:
    """Per-module VCR config consumed by pytest-recording.

    Strip ``Authorization`` and cookie headers from both request and
    response so committed cassettes never carry live credentials. Match
    on ``method`` + ``path`` (not full URL) so the same cassette replays
    across dev / staging / local base URLs.
    """
    return {
        "filter_headers": [
            ("Authorization", "REDACTED"),
            ("Cookie", "REDACTED"),
            ("Set-Cookie", "REDACTED"),
        ],
        "filter_query_parameters": [
            ("access_token", "REDACTED"),
            ("refresh_token", "REDACTED"),
        ],
        "match_on": ["method", "path", "body"],
        "record_mode": (
            "rewrite" if os.environ.get("RECORD_CASSETTES") == "1" else "none"
        ),
    }
