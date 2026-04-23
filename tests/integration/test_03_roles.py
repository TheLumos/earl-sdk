"""Role-enforcement tests.

EARL has two admin roles:

* ``EARL_Admin``     — global: can manage any org
* ``EARL_Org_Admin`` — scoped: can manage their own org

These tests verify that a non-admin user in the same org is denied on
mutating endpoints. They require a second CLI profile set via
``EARL_INTEGRATION_NONADMIN_PROFILE`` and skip otherwise.
"""

from __future__ import annotations

import pytest

from .cli_runner import CliRunner


pytestmark = pytest.mark.requires_nonadmin


def test_nonadmin_cannot_create_service_account(nonadmin_cli: CliRunner) -> None:
    """Non-admin users must be denied when creating service accounts.

    Documents::

        earl --profile <non-admin-profile> service-account create \\
            --name nonadmin-should-fail --scopes earl:read    # \u2192 403
    """
    result = nonadmin_cli.run_expect_fail(
        "service-account",
        "create",
        "--name",
        "nonadmin-should-fail",
        "--scopes",
        "earl:read",
    )
    blob = (result.stdout + result.stderr).lower()
    assert any(
        kw in blob
        for kw in ("forbidden", "403", "not authorized", "admin", "permission")
    ), result.stdout + result.stderr


def test_nonadmin_cannot_delete_arbitrary_pipeline(
    nonadmin_cli: CliRunner, cli: CliRunner, tmp_pipeline_name: str
) -> None:
    """A non-admin user should not be able to delete a pipeline they did
    not create (and did not get explicit permission on). At minimum, the
    call must not succeed silently.

    Documents::

        earl --profile <non-admin-profile> pipelines delete <name>   # \u2192 403 / 404
    """
    cli.run(
        "pipelines",
        "create",
        "--name",
        tmp_pipeline_name,
        "--verifier-ids",
        "hard-gates/fabricated-ehr-data",
    )
    try:
        result = nonadmin_cli.run_expect_fail("pipelines", "delete", tmp_pipeline_name)
        blob = (result.stdout + result.stderr).lower()
        assert any(
            kw in blob for kw in ("forbidden", "403", "not authorized", "404")
        ), result.stdout + result.stderr
    finally:
        cli.run("pipelines", "delete", tmp_pipeline_name, check=False)


def test_nonadmin_can_read(nonadmin_cli: CliRunner) -> None:
    """Non-admin users can still read catalog data.

    Documents::

        earl --profile <non-admin-profile> cases list   # \u2192 200
        earl --profile <non-admin-profile> patients list   # \u2192 200
    """
    nonadmin_cli.run("cases", "list")
    nonadmin_cli.run("patients", "list")
