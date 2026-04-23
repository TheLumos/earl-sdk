"""Pipeline-authoring integration tests.

The orchestrator does not author *cases* — those live upstream in the Lumos
case-service. What Earl lets you do is:

1. Pick a catalog case (``--case-id``).
2. Attach **extra verifiers** at the pipeline level. The ``--verifier-ids``
   flag accepts a mix of ``hard-gates/*`` and ``scoring-dimensions/*`` IDs,
   which map to the "hard gates" and "scoring dimensions" the user asked
   about.
3. Pin a patient subset and doctor config.

This file verifies the full authoring surface: create, list, get (shows
every attached verifier), update (add another verifier), delete.

The downstream simulation test (test_06) picks up the pipeline produced
here and proves each attached verifier actually runs.
"""

from __future__ import annotations

import pytest

from .cli_runner import CliRunner
from .conftest import Cleanup


EXTRA_HARD_GATE = "hard-gates/fabricated-ehr-data"
EXTRA_SCORING_DIMS = [
    "scoring-dimensions/clinical-correctness",
    "scoring-dimensions/communication--empathy",
]


@pytest.fixture
def authored_pipeline(
    cli: CliRunner,
    tmp_pipeline_name: str,
    cleanup: Cleanup,
    catalog_case_id: str | None,
    one_patient_id: str,
) -> dict:
    """Create a pipeline that combines (optionally) a catalog case with
    extra hard gates and scoring dimensions, pinned to one patient,
    internal doctor, 2 turns.

    When the active org has no cases assigned (``catalog_case_id is None``)
    the pipeline is authored without ``--case-id`` and still runs through
    the attached verifiers. That exercises the same authoring path — the
    user's "create a case + add verifiers + add hard gates" scenario.

    Documents::

        earl pipelines create \\
            --name <name> \\
            [--case-id <catalog case>] \\
            --verifier-ids "hard-gates/fabricated-ehr-data,\\
                scoring-dimensions/clinical-correctness,\\
                scoring-dimensions/communication--empathy" \\
            --patient-id <pid> \\
            --doctor internal \\
            --conversation-initiator patient \\
            --max-turns 2
    """
    verifier_ids = [EXTRA_HARD_GATE, *EXTRA_SCORING_DIMS]

    argv = [
        "pipelines",
        "create",
        "--name",
        tmp_pipeline_name,
        "--verifier-ids",
        ",".join(verifier_ids),
        "--patient-id",
        one_patient_id,
        "--doctor",
        "internal",
        "--conversation-initiator",
        "patient",
        "--max-turns",
        "2",
    ]
    if catalog_case_id:
        argv += ["--case-id", catalog_case_id]

    created = cli.run(*argv).json()
    cleanup.pipeline(tmp_pipeline_name)

    return {
        "name": tmp_pipeline_name,
        "case_id": catalog_case_id,
        "patient_id": one_patient_id,
        "verifier_ids": verifier_ids,
        "response": created,
    }


# --------------------------------------------------------------------------- #
# Authoring surface
# --------------------------------------------------------------------------- #


def test_create_pipeline_with_case_and_extras(authored_pipeline: dict) -> None:
    """Created pipeline returns a payload that carries the requested name."""
    payload = authored_pipeline["response"]
    assert payload, "empty create response"
    # Pipeline APIs return either {"name": ...} directly or under .pipeline.
    name = payload.get("name") or payload.get("pipeline", {}).get("name")
    assert name == authored_pipeline["name"], payload


def test_list_pipelines_includes_new(cli: CliRunner, authored_pipeline: dict) -> None:
    """``earl pipelines list`` contains the newly created pipeline.

    Documents::

        earl pipelines list
    """
    data = cli.run("pipelines", "list").json()
    names = _extract_pipeline_names(data)
    assert authored_pipeline["name"] in names, (authored_pipeline["name"], names)


def test_get_pipeline_shows_all_attached_verifiers(
    cli: CliRunner, authored_pipeline: dict
) -> None:
    """``earl pipelines get`` shows every verifier we attached (hard-gate +
    both scoring dimensions) plus the case_id (when one was passed).

    Documents::

        earl pipelines get <name>
    """
    data = cli.run("pipelines", "get", authored_pipeline["name"]).json()
    verifiers = _extract_verifier_ids(data)
    for v in authored_pipeline["verifier_ids"]:
        assert v in verifiers, (
            f"{v} not in attached verifiers {verifiers}; payload={data}"
        )
    if authored_pipeline["case_id"]:
        case_id = _extract_case_id(data)
        assert case_id == authored_pipeline["case_id"], (case_id, data)


def test_update_pipeline_appends_verifier(
    cli: CliRunner, authored_pipeline: dict
) -> None:
    """``earl pipelines update`` replaces the verifier set atomically. We
    add one more hard gate and verify the resulting pipeline has it.

    Documents::

        earl pipelines update <name> --verifier-ids "<existing,new>"
    """
    extra = "hard-gates/unsafe-medication-recommendation"
    new_set = authored_pipeline["verifier_ids"] + [extra]
    cli.run(
        "pipelines",
        "update",
        authored_pipeline["name"],
        "--verifier-ids",
        ",".join(new_set),
    )
    data = cli.run("pipelines", "get", authored_pipeline["name"]).json()
    verifiers = _extract_verifier_ids(data)
    for v in new_set:
        assert v in verifiers, f"{v} missing after update; got {verifiers}"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_create_pipeline_rejects_bad_max_turns(
    cli: CliRunner, run_id: str, cleanup: Cleanup
) -> None:
    """``max-turns`` must be in [1, 50] — the CLI validates before calling
    the backend so this fails fast with a clear error.

    Documents::

        earl pipelines create --name <x> --max-turns 999  # \u2192 client-side error
    """
    bad_name = f"{run_id}-bad-turns"
    result = cli.run_expect_fail(
        "pipelines",
        "create",
        "--name",
        bad_name,
        "--verifier-ids",
        "hard-gates/fabricated-ehr-data",
        "--max-turns",
        "999",
    )
    cleanup.pipeline(bad_name)
    blob = (result.stdout + result.stderr).lower()
    assert "max_turns" in blob or "max-turns" in blob or "1" in blob, result.stderr


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


def _extract_pipeline_names(payload: object) -> set[str]:
    items = _as_list(payload, "pipelines")
    return {p.get("name", "") for p in items if isinstance(p, dict)}


def _extract_verifier_ids(payload: object) -> list[str]:
    """Walk the pipeline payload and pull every verifier id out of it.

    Acceptable shapes (we've seen all of these across backend versions):
      * ``{"dimension_ids": [...]}``                         (flat, current)
      * ``{"verifier_ids": [...]}``
      * ``{"config": {"judge": {"dimensions": [...]}}}``     (nested config)
      * ``{"pipeline": {...}}``                              (wrapped)
    """
    if not isinstance(payload, dict):
        return []
    candidates: list[str] = []
    payload = payload.get("pipeline", payload)
    if not isinstance(payload, dict):
        return []
    for key in ("dimension_ids", "verifier_ids", "verifiers"):
        val = payload.get(key)
        if isinstance(val, list):
            candidates.extend(str(v) for v in val)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    judge = config.get("judge", {}) if isinstance(config, dict) else {}
    for key in ("dimensions", "verifier_ids", "verifiers"):
        val = judge.get(key)
        if isinstance(val, list):
            candidates.extend(str(v) for v in val)
    return candidates


def _extract_case_id(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    payload = payload.get("pipeline", payload)
    if not isinstance(payload, dict):
        return None
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    judge = config.get("judge", {}) if isinstance(config, dict) else {}
    return judge.get("case_id") or payload.get("case_id")
