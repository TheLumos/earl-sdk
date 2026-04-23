"""Offline-replayable integration tests for ``client.admin.*``.

These tests mirror a slice of ``test_08_org_admin.py`` but reach the
orchestrator through the SDK's typed admin API
(:class:`earl_sdk.admin_api.OrgAdminAPI`) instead of shelling out to
``earl``. That matters because pytest-recording can only intercept HTTP
traffic that happens inside the test process; subprocess CLI calls
escape the harness.

Mode selection
--------------

* Default: cassettes in ``cassettes/`` are *replayed*. No network, no
  credentials — safe for every CI run.
* ``RECORD_CASSETTES=1``: the VCR marker switches to ``--record-mode=rewrite``
  (see :func:`_vcr_record_mode` in :mod:`conftest`). You must have a live
  PKCE profile for the target env.

See ``cassettes/README.md`` for the refresh workflow.
"""
from __future__ import annotations

import os

import pytest


pytest_recording = pytest.importorskip(
    "pytest_recording",
    reason="pytest-recording not installed; install sdk/tests/integration/requirements-dev.txt",
)


@pytest.fixture
def admin_client():
    """Build an ``EarlClient`` using the active PKCE profile (record mode)
    or synthetic creds that match the cassette (replay mode).
    """
    from earl_sdk import EarlClient

    if os.environ.get("RECORD_CASSETTES") == "1":
        env = os.environ.get("EARL_INTEGRATION_ENV", "dev")
        # The real PKCE flow will be exercised and cached locally; VCR
        # records only the HTTP side of the exchange.
        return EarlClient(environment=env, auth_kind="pkce")
    # In replay mode, concrete secrets don't matter — VCR matches by
    # method+URL+body and returns the stored response. We still set
    # something non-empty so SDK constructor validation is happy.
    return EarlClient(
        client_id="replay-client",
        client_secret="replay-secret",
        organization="org_replay1",
        environment="dev",
    )


@pytest.mark.vcr
def test_admin_orgs_list_via_typed_api(admin_client) -> None:
    """``client.admin.orgs.list()`` returns a paginated envelope."""
    page = admin_client.admin.orgs.list(limit=5)
    assert isinstance(page, dict)
    assert "items" in page
    assert isinstance(page["items"], list)


@pytest.mark.vcr
def test_admin_orgs_iter_all_via_typed_api(admin_client) -> None:
    """``iter_all`` walks cursors without the caller managing pagination."""
    orgs = admin_client.admin.orgs.iter_all(limit=2)
    assert isinstance(orgs, list)


@pytest.mark.vcr
def test_admin_members_list_via_typed_api(admin_client) -> None:
    """Members listing for the caller's own org should succeed."""
    org_id = admin_client.organization or "org_replay1"
    page = admin_client.admin.members.list(org_id, limit=5)
    assert isinstance(page, dict)
    assert "items" in page
