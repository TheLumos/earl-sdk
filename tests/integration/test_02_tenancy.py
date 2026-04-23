"""Tenancy (org_id) enforcement tests.

Every JWT carries exactly one ``org_id`` claim. The backend derives tenancy
exclusively from that claim — never from request bodies, headers, or paths.
This module asserts:

1. The active user's token has exactly one ``org_id`` matching the profile.
2. Listing ``service-accounts`` returns only rows tagged with the active org.
3. When a second-org profile is configured (``EARL_INTEGRATION_SECOND_ORG_PROFILE``),
   resources created by org A are invisible to org B and vice-versa.
"""

from __future__ import annotations

import pytest

from .cli_runner import CliRunner
from .conftest import Cleanup


def test_whoami_reports_single_org(cli: CliRunner, active_org_id: str) -> None:
    """Token must carry exactly one ``org_id`` claim and it must match the
    active profile.

    Documents::

        earl whoami   # .organization is a single org_... value
    """
    data = cli.run("whoami").json()
    assert data.get("organization") == active_org_id, data
    assert isinstance(data.get("organization"), str), data


def test_service_accounts_list_scoped_to_active_org(
    cli: CliRunner, active_org_id: str
) -> None:
    """``service-account list`` returns only rows for the active org.

    Documents::

        earl service-account list
    """
    data = cli.run("service-account", "list").json()
    items = data.get("service_accounts", data if isinstance(data, list) else [])
    for sa in items:
        if "organization" in sa:
            assert sa["organization"] == active_org_id, sa
        elif "org_id" in sa:
            assert sa["org_id"] == active_org_id, sa


# --------------------------------------------------------------------------- #
# Cross-org isolation (requires a second profile)
# --------------------------------------------------------------------------- #


@pytest.mark.requires_second_org
def test_cross_org_pipeline_invisible(
    cli: CliRunner,
    second_org_cli: CliRunner,
    tmp_pipeline_name: str,
    cleanup: Cleanup,
) -> None:
    """A pipeline created in org A must be invisible to org B.

    Documents::

        # as org A
        earl pipelines create --name <X> --verifier-ids hard-gates/fabricated-ehr-data
        # as org B
        earl pipelines get <X>         # must FAIL: 404 / not_found
        earl pipelines list            # must NOT contain <X>
    """
    cli.run(
        "pipelines",
        "create",
        "--name",
        tmp_pipeline_name,
        "--verifier-ids",
        "hard-gates/fabricated-ehr-data",
    )
    cleanup.pipeline(tmp_pipeline_name)

    other_get = second_org_cli.run_expect_fail("pipelines", "get", tmp_pipeline_name)
    blob = (other_get.stdout + other_get.stderr).lower()
    assert any(
        kw in blob for kw in ("not_found", "404", "not found", "no such pipeline")
    ), other_get.stdout + other_get.stderr

    other_list = second_org_cli.run("pipelines", "list")
    names = _extract_pipeline_names(other_list.json())
    assert tmp_pipeline_name not in names, (
        f"leak: {tmp_pipeline_name} visible to second org: {names}"
    )


def _extract_pipeline_names(payload: object) -> set[str]:
    if isinstance(payload, list):
        return {p.get("name", "") for p in payload if isinstance(p, dict)}
    if isinstance(payload, dict):
        for key in ("pipelines", "items", "data", "results"):
            if key in payload and isinstance(payload[key], list):
                return {p.get("name", "") for p in payload[key] if isinstance(p, dict)}
    return set()
