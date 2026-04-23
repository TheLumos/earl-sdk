"""Read-side catalog tests.

Cover every list/get endpoint a user needs before authoring a pipeline:

* patients: list, paginate, get
* cases: list, get (with embedded case_verifiers)
* dimensions: list, get
* verifiers: list (generic catalog of hard gates + scoring dimensions)

These all run on the happy path; failure modes are covered in test_01_auth
and test_02_tenancy.
"""

from __future__ import annotations

import pytest

from .cli_runner import CliRunner


# --------------------------------------------------------------------------- #
# Patients
# --------------------------------------------------------------------------- #


def test_patients_list(cli: CliRunner) -> None:
    """List every patient available to the active org.

    Documents::

        earl patients list
    """
    data = cli.run("patients", "list").json()
    items = _as_list(data, "patients")
    assert len(items) > 0, "expected at least one patient"
    assert "id" in items[0], items[0]


def test_patients_list_paginated(cli: CliRunner) -> None:
    """Pagination via ``--limit`` / ``--offset``.

    The CLI currently accepts both flags but the upstream Lumos patients API
    does not yet enforce them server-side \u2014 it returns the full catalog
    regardless. We still invoke the flags to ensure the CLI keeps accepting
    them (regression guard) and that the response is a well-formed list.

    Documents::

        earl patients list --limit 2 --offset 0
        earl patients list --limit 2 --offset 2
    """
    page1 = _as_list(
        cli.run("patients", "list", "--limit", "2", "--offset", "0").json(),
        "patients",
    )
    page2 = _as_list(
        cli.run("patients", "list", "--limit", "2", "--offset", "2").json(),
        "patients",
    )
    assert isinstance(page1, list) and page1, page1
    assert isinstance(page2, list) and page2, page2
    # When pagination is enforced upstream, --offset should shift the window.
    # We don't require it today; just warn if it's a no-op.
    if len(page1) > 2 and len(page2) > 2:
        import warnings

        warnings.warn(
            "patients list --limit/--offset is a no-op; upstream not paginating yet",
            stacklevel=1,
        )


def test_patients_get(cli: CliRunner) -> None:
    """Fetch one patient by id. Patient ids contain ``:`` and spaces, so the
    CLI handles URL encoding before sending.

    Documents::

        earl patients get "cancer_survivors:Brandon Kowalski"
    """
    patients = _as_list(
        cli.run("patients", "list", "--limit", "5").json(), "patients"
    )
    if not patients:
        pytest.skip("no patients visible to this org")
    pid = patients[0].get("id", "")
    if not pid:
        pytest.skip("first patient has no id")
    fetched = cli.run("patients", "get", pid).json()
    assert fetched.get("id") == pid, fetched


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #


def test_cases_list(cli: CliRunner) -> None:
    """List assigned cases.

    Documents::

        earl cases list
    """
    data = cli.run("cases", "list").json()
    cases = _as_list(data, "cases")
    assert isinstance(cases, list), data


def test_cases_get_has_verifier_counts(cli: CliRunner) -> None:
    """Each case detail should report its verifier, gate, and dimension counts.

    Documents::

        earl cases get <case_id>
    """
    cases = _as_list(cli.run("cases", "list").json(), "cases")
    if not cases:
        pytest.skip("no cases assigned to this org")
    case_id = cases[0].get("id") or cases[0].get("case_id")
    assert case_id, cases[0]
    detail = cli.run("cases", "get", case_id).json()
    # Accept either a flat structure or a nested ``totals``/``counts`` dict.
    totals = detail.get("totals") or detail.get("counts") or detail
    assert any(
        k in totals for k in ("case_verifiers", "hard_gates", "scoring_dimensions")
    ), detail


# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #


def test_dimensions_list(cli: CliRunner) -> None:
    """List dimensions (standard + custom).

    Documents::

        earl dimensions list
    """
    data = cli.run("dimensions", "list").json()
    dims = _as_list(data, "dimensions")
    assert len(dims) > 0, data


def test_dimensions_get(cli: CliRunner) -> None:
    """Fetch one dimension by id.

    Documents::

        earl dimensions get <id>
    """
    dims = _as_list(cli.run("dimensions", "list").json(), "dimensions")
    dim_id = dims[0].get("id")
    assert dim_id, dims[0]
    fetched = cli.run("dimensions", "get", dim_id).json()
    assert fetched.get("id") == dim_id, fetched


# --------------------------------------------------------------------------- #
# Verifiers (generic catalog \u2014 hard gates + scoring dimensions)
# --------------------------------------------------------------------------- #


def test_verifiers_list_has_gates_and_dims(cli: CliRunner) -> None:
    """The verifier catalog must expose at least one hard gate and one
    scoring dimension. Payload shape is
    ``{"hard_gates": [...], "scoring_dimensions": [...]}``.

    The pipeline-authoring test (test_05) and simulation test (test_06)
    depend on both bucket names being populated.

    Documents::

        earl verifiers list
    """
    data = cli.run("verifiers", "list").json()
    assert isinstance(data, dict), data
    gates = data.get("hard_gates") or []
    dims = data.get("scoring_dimensions") or []
    assert isinstance(gates, list) and gates, data
    assert isinstance(dims, list) and dims, data
    assert any(g.get("path", "").startswith("hard-gates/") for g in gates), gates[:3]
    assert any(
        d.get("path", "").startswith("scoring-dimensions/") for d in dims
    ), dims[:3]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_list(payload: object, key: str) -> list:
    """Accept either ``[...]`` or ``{"<key>": [...]}``-shaped responses."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in (key, "items", "data", "results"):
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []


