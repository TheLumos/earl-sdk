"""
Unit tests for ``SimulationsAPI.wait_for_completion`` adaptive backoff.

The backoff exists to keep a forgotten ``earl simulations wait`` (or any
SDK-driven polling loop) from hammering the orchestrator if a simulation
gets orphaned and never transitions to a terminal state. We validate the
two important invariants:

1. Healthy / progressing simulations keep their tight, user-specified
   poll cadence (no surprise slow-down).
2. Stalled simulations back off exponentially up to ``max_poll_interval``
   and snap back to ``poll_interval`` as soon as anything moves again.

Run:
    pytest sdk/tests/test_simulations_wait_backoff.py -v
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import patch

import pytest

from earl_sdk.api import SimulationsAPI
from earl_sdk.models import Simulation, SimulationStatus


def _sim(
    *,
    status: SimulationStatus = SimulationStatus.RUNNING,
    completed_episodes: int = 0,
    current_episode: int = 0,
    updated_at: str = "2026-05-07T00:00:00Z",
) -> Simulation:
    return Simulation(
        id="sim-test",
        organization_id="org-test",
        pipeline_name="pipeline",
        status=status,
        total_episodes=10,
        completed_episodes=completed_episodes,
        current_episode=current_episode,
        started_at="2026-05-07T00:00:00Z",
        updated_at=updated_at,
    )


class _FakeAPI(SimulationsAPI):
    """SimulationsAPI subclass that yields a scripted sequence from .get()."""

    def __init__(self, sims_to_return: list[Simulation]) -> None:  # noqa: D401
        self._iter: Iterator[Simulation] = iter(sims_to_return)
        self.calls = 0

    def get(self, simulation_id: str) -> Simulation:  # type: ignore[override]
        self.calls += 1
        return next(self._iter)

    def get_episodes(self, simulation_id: str):  # type: ignore[override]
        return []


class TestWaitForCompletionBackoff:
    def test_progressing_simulation_keeps_tight_cadence(self) -> None:
        # Five distinct progress signatures → never stalls → always sleeps poll_interval.
        sims = [
            _sim(completed_episodes=1),
            _sim(completed_episodes=2),
            _sim(completed_episodes=3),
            _sim(completed_episodes=4),
            _sim(completed_episodes=10, status=SimulationStatus.COMPLETED),
        ]
        api = _FakeAPI(sims)

        sleep_calls: list[float] = []
        with patch("earl_sdk.api.time.sleep", side_effect=sleep_calls.append):
            done = api.wait_for_completion(
                "sim-test",
                poll_interval=5.0,
                max_poll_interval=60.0,
                stall_threshold=3,
                backoff_factor=2.0,
            )

        assert done.status == SimulationStatus.COMPLETED
        # 5 polls, terminal on the last → 4 sleeps, all at the base interval.
        assert sleep_calls == [5.0, 5.0, 5.0, 5.0]

    def test_stalled_simulation_backs_off_exponentially(self) -> None:
        # Same updated_at + completed_episodes for many polls → after 3
        # consecutive stalls, the interval doubles each subsequent poll
        # (capped at max_poll_interval).
        sims = [_sim() for _ in range(10)]
        sims[-1] = _sim(status=SimulationStatus.COMPLETED)
        api = _FakeAPI(sims)

        sleep_calls: list[float] = []
        with patch("earl_sdk.api.time.sleep", side_effect=sleep_calls.append):
            api.wait_for_completion(
                "sim-test",
                poll_interval=5.0,
                max_poll_interval=60.0,
                stall_threshold=3,
                backoff_factor=2.0,
            )

        # Iteration 1: first poll → no prior signature → no stall, sleep=5
        # Iter 2: stall_count=1, sleep=5
        # Iter 3: stall_count=2, sleep=5
        # Iter 4: stall_count=3 → trip backoff, sleep=10
        # Iter 5: stall_count=4 → sleep=20
        # Iter 6: stall_count=5 → sleep=40
        # Iter 7: stall_count=6 → sleep=60 (cap)
        # Iter 8: stall_count=7 → sleep=60 (cap)
        # Iter 9: stall_count=8 → sleep=60 (cap)
        # Iter 10: COMPLETED → return, no sleep
        assert sleep_calls == [5.0, 5.0, 5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0]

    def test_recovers_to_base_interval_when_progress_resumes(self) -> None:
        # Stalls for several polls, then progress resumes → interval snaps
        # back to the user-specified poll_interval.
        sims = [
            _sim(completed_episodes=0),                       # 1: first
            _sim(completed_episodes=0),                       # 2: stall #1
            _sim(completed_episodes=0),                       # 3: stall #2
            _sim(completed_episodes=0),                       # 4: stall #3 → trip
            _sim(completed_episodes=0),                       # 5: stall #4 → 20s
            _sim(completed_episodes=1),                       # 6: progress! reset
            _sim(completed_episodes=2),                       # 7: progress
            _sim(completed_episodes=10, status=SimulationStatus.COMPLETED),  # 8: done
        ]
        api = _FakeAPI(sims)

        sleep_calls: list[float] = []
        with patch("earl_sdk.api.time.sleep", side_effect=sleep_calls.append):
            api.wait_for_completion(
                "sim-test",
                poll_interval=5.0,
                max_poll_interval=60.0,
                stall_threshold=3,
                backoff_factor=2.0,
            )

        # Sleeps: 5, 5, 5, 10, 20, 5, 5  (then COMPLETED → no sleep)
        assert sleep_calls == [5.0, 5.0, 5.0, 10.0, 20.0, 5.0, 5.0]

    def test_max_poll_interval_equal_to_poll_interval_disables_backoff(self) -> None:
        sims = [_sim() for _ in range(8)]
        sims[-1] = _sim(status=SimulationStatus.COMPLETED)
        api = _FakeAPI(sims)

        sleep_calls: list[float] = []
        with patch("earl_sdk.api.time.sleep", side_effect=sleep_calls.append):
            api.wait_for_completion(
                "sim-test",
                poll_interval=5.0,
                max_poll_interval=5.0,
                stall_threshold=3,
                backoff_factor=2.0,
            )

        assert sleep_calls == [5.0] * 7

    def test_terminal_status_returns_immediately_without_extra_sleep(self) -> None:
        sims = [_sim(status=SimulationStatus.COMPLETED)]
        api = _FakeAPI(sims)

        sleep_calls: list[float] = []
        with patch("earl_sdk.api.time.sleep", side_effect=sleep_calls.append):
            done = api.wait_for_completion("sim-test", poll_interval=5.0)

        assert done.status == SimulationStatus.COMPLETED
        assert sleep_calls == []
        assert api.calls == 1


class TestWaitForCompletionFailureSemantics:
    def test_raises_on_failed_status(self) -> None:
        from earl_sdk.exceptions import SimulationError

        api = _FakeAPI([_sim(status=SimulationStatus.FAILED)])
        with patch("earl_sdk.api.time.sleep"):
            with pytest.raises(SimulationError):
                api.wait_for_completion("sim-test", poll_interval=5.0)

    def test_raises_on_stopped_status(self) -> None:
        from earl_sdk.exceptions import SimulationError

        api = _FakeAPI([_sim(status=SimulationStatus.STOPPED)])
        with patch("earl_sdk.api.time.sleep"):
            with pytest.raises(SimulationError):
                api.wait_for_completion("sim-test", poll_interval=5.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
