"""Tests for the live-progress per-episode "detail" cell.

The orchestrator persists two counters on each episode:

- ``dialogue_turns`` (legacy) — incremented for **every** message pushed
  to ``dialogue_history`` (doctor, patient, tool replays, system / EHR
  entries). Can exceed ``max_turns`` once tool use is in play.
- ``dialogue_exchanges`` (new) — incremented exactly once per completed
  doctor↔patient round. ``dialogue_exchanges <= max_turns`` is the
  invariant the SDK relies on to render "exch N/max" honestly.

These tests pin the label contract end-to-end so a future schema
rename can't silently regress the UX to the "turn 26/10" bug.
"""

from __future__ import annotations

from earl_sdk.interactive.flows.run import _episode_progress_detail


def _ep(**overrides):
    base = {"status": "conversation"}
    base.update(overrides)
    return base


# ── conversation / awaiting_doctor ──────────────────────────────────────────


def test_prefers_exchanges_with_known_max_turns() -> None:
    detail = _episode_progress_detail(
        _ep(dialogue_exchanges=4, dialogue_turns=11),
        max_turns=10,
    )
    assert detail == "exch 4/10"


def test_uses_exchanges_without_max_turns() -> None:
    detail = _episode_progress_detail(
        _ep(dialogue_exchanges=4, dialogue_turns=11),
        max_turns=None,
    )
    assert detail == "exch 4"


def test_exchanges_zero_is_still_rendered() -> None:
    """Zero is a valid value (doctor hasn't replied yet) and must render
    honestly rather than falling through to the message counter."""
    detail = _episode_progress_detail(
        _ep(dialogue_exchanges=0, dialogue_turns=1),
        max_turns=10,
    )
    assert detail == "exch 0/10"


def test_falls_back_to_msgs_when_server_does_not_publish_exchanges() -> None:
    """Older orchestrator deployments don't write ``dialogue_exchanges``;
    the SDK must avoid mis-labelling a message counter as 'turn' or
    'exch' — show "msgs N" so the user is not misled."""
    detail = _episode_progress_detail(
        _ep(dialogue_turns=26),  # no dialogue_exchanges key
        max_turns=10,
    )
    assert detail == "msgs 26"


def test_msgs_fallback_handles_missing_dialogue_turns() -> None:
    detail = _episode_progress_detail(_ep(), max_turns=10)
    assert detail == "msgs 0"


def test_awaiting_doctor_status_uses_same_logic() -> None:
    detail = _episode_progress_detail(
        _ep(status="awaiting_doctor", dialogue_exchanges=2),
        max_turns=10,
    )
    assert detail == "exch 2/10"


# ── other statuses (regression coverage for the existing branches) ──────────


def test_judging_shows_verifier_progress() -> None:
    detail = _episode_progress_detail(
        _ep(status="judging", categories_completed=3, categories_queued=9),
        max_turns=10,
    )
    assert detail == "verifiers 3/9"


def test_judging_without_total_falls_back() -> None:
    detail = _episode_progress_detail(_ep(status="judging"), max_turns=10)
    assert detail == "scoring..."


def test_completed_with_score() -> None:
    detail = _episode_progress_detail(
        _ep(status="completed", total_score=3.42),
        max_turns=10,
    )
    assert detail == "score 3.42"


def test_failed_truncates_long_error() -> None:
    err = "x" * 80
    detail = _episode_progress_detail(
        _ep(status="failed", error=err),
        max_turns=10,
    )
    assert detail == err[:40] + "..."


def test_pending_returns_empty_detail() -> None:
    assert _episode_progress_detail(_ep(status="pending"), max_turns=10) == ""
