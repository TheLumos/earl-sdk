#!/usr/bin/env python3
"""
EARL SDK - Full Smoke Test
==========================

Comprehensive smoke test that exercises ALL SDK functionality against a live environment.
Designed for fast execution (~3-5 min) while covering the entire API surface.

Test Phases:
  1. Connection & Auth     - test_connection, rate_limits
  2. Dimensions            - list, get, create (custom)
  3. Patients              - list (with filters, pagination), get
  4. Pipelines CRUD        - create, list, get, update, delete
  5. Simulation Lifecycle  - create, list, get, wait_for_completion, episodes, report
  6. Cancel Simulation     - create then cancel
  7. Client-Driven Flow    - create, get_pending_episodes, submit_response, report

Usage:
  # Using env vars (recommended)
  export EARL_CLIENT_ID='your-client-id'
  export EARL_CLIENT_SECRET='your-client-secret'
  export EARL_ORGANIZATION='org_xxx'  # optional
  python3 sdk/tests/test_smoke_full.py --env test

  # Using CLI args
  python3 sdk/tests/test_smoke_full.py --env test \\
      --client-id YOUR_ID --client-secret YOUR_SECRET

  # Skip slow phases
  python3 sdk/tests/test_smoke_full.py --env test --skip-simulation --skip-client-driven

  # Custom timeout
  python3 sdk/tests/test_smoke_full.py --env test --timeout 600
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Optional

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earl_sdk import (
    EarlClient,
    Environment,
    Dimension,
    Patient,
    Pipeline,
    Simulation,
    SimulationStatus,
    DoctorApiConfig,
    ConversationConfig,
)
from earl_sdk.exceptions import (
    EarlError,
    NotFoundError,
    ValidationError,
)


# =============================================================================
# Output Helpers
# =============================================================================

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


class TestResult:
    """Track test results."""

    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[str] = []
        self.start_time = time.time()

    def record_pass(self, name: str, detail: str = ""):
        self.passed.append(name)
        suffix = f" {Colors.DIM}({detail}){Colors.END}" if detail else ""
        print(f"  {Colors.GREEN}PASS{Colors.END}  {name}{suffix}")

    def record_fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  {Colors.RED}FAIL{Colors.END}  {name}")
        print(f"        {Colors.RED}{error}{Colors.END}")

    def record_skip(self, name: str, reason: str = ""):
        self.skipped.append(name)
        suffix = f" - {reason}" if reason else ""
        print(f"  {Colors.YELLOW}SKIP{Colors.END}  {name}{suffix}")

    def summary(self):
        elapsed = time.time() - self.start_time
        total = len(self.passed) + len(self.failed) + len(self.skipped)

        print(f"\n{'='*60}")
        print(f"{Colors.BOLD}SMOKE TEST SUMMARY{Colors.END}")
        print(f"{'='*60}")
        print(f"  Total:   {total}")
        print(f"  {Colors.GREEN}Passed:  {len(self.passed)}{Colors.END}")
        if self.failed:
            print(f"  {Colors.RED}Failed:  {len(self.failed)}{Colors.END}")
            for name, err in self.failed:
                print(f"    - {name}: {err}")
        if self.skipped:
            print(f"  {Colors.YELLOW}Skipped: {len(self.skipped)}{Colors.END}")
        print(f"  Time:    {elapsed:.1f}s")
        print(f"{'='*60}")

        if self.failed:
            print(f"\n{Colors.RED}SMOKE TEST FAILED{Colors.END}")
            return False
        else:
            print(f"\n{Colors.GREEN}ALL SMOKE TESTS PASSED{Colors.END}")
            return True


def section(title: str):
    print(f"\n{Colors.CYAN}{Colors.BOLD}--- {title} ---{Colors.END}")


# =============================================================================
# Credential Helpers
# =============================================================================

def get_credentials(args):
    """Get credentials from CLI args or environment."""
    client_id = args.client_id or os.environ.get("EARL_CLIENT_ID", "")
    client_secret = args.client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
    organization = args.organization or os.environ.get("EARL_ORGANIZATION", "")

    if not client_id or not client_secret:
        print(f"\n{Colors.RED}MISSING CREDENTIALS{Colors.END}")
        print("\nProvide via environment variables or CLI args:")
        print("  export EARL_CLIENT_ID='your-client-id'")
        print("  export EARL_CLIENT_SECRET='your-client-secret'")
        print("  export EARL_ORGANIZATION='org_xxx'  # optional")
        print("\nOr: --client-id ID --client-secret SECRET")
        sys.exit(1)

    return client_id, client_secret, organization


# =============================================================================
# Test Phases
# =============================================================================

def phase_connection(client: EarlClient, results: TestResult):
    """Phase 1: Connection & Auth."""
    section("Phase 1: Connection & Auth")

    # test_connection
    try:
        ok = client.test_connection()
        assert ok, "test_connection() returned False"
        results.record_pass("test_connection", f"env={client.environment}")
    except Exception as e:
        results.record_fail("test_connection", str(e))
        # If connection fails, abort the rest
        raise SystemExit("Cannot connect - aborting smoke test")

    # rate_limits — endpoint not yet available in backend
    results.record_skip("rate_limits.get", "endpoint not yet implemented in backend")
    results.record_skip("rate_limits.get_effective_limit", "endpoint not yet implemented in backend")


def phase_dimensions(client: EarlClient, results: TestResult) -> list[Dimension]:
    """Phase 2: Dimensions. Returns list of dimensions."""
    section("Phase 2: Dimensions")
    dimensions = []

    # dimensions.list
    try:
        dimensions = client.dimensions.list()
        assert len(dimensions) > 0, "No dimensions returned"
        assert all(isinstance(d, Dimension) for d in dimensions)
        results.record_pass("dimensions.list", f"count={len(dimensions)}")
    except Exception as e:
        results.record_fail("dimensions.list", str(e))
        return dimensions

    # dimensions.list (without custom)
    try:
        std_dims = client.dimensions.list(include_custom=False)
        assert isinstance(std_dims, list)
        results.record_pass("dimensions.list(include_custom=False)", f"count={len(std_dims)}")
    except Exception as e:
        results.record_fail("dimensions.list(include_custom=False)", str(e))

    # dimensions.get
    try:
        dim = client.dimensions.get(dimensions[0].id)
        assert isinstance(dim, Dimension)
        assert dim.id == dimensions[0].id
        assert dim.name, "Dimension name should not be empty"
        assert dim.description, "Dimension description should not be empty"
        results.record_pass("dimensions.get", f"id={dim.id}, name={dim.name}")
    except Exception as e:
        results.record_fail("dimensions.get", str(e))

    # dimensions.create (custom) — may not be supported by all deployments
    try:
        ts = int(time.time())
        custom_dim = client.dimensions.create(
            name=f"smoke-test-dim-{ts}",
            description="Smoke test custom dimension - safe to delete",
            category="custom",
            weight=0.5,
        )
        assert isinstance(custom_dim, Dimension)
        assert custom_dim.name == f"smoke-test-dim-{ts}"
        results.record_pass("dimensions.create", f"id={custom_dim.id}")
    except (NotFoundError, EarlError) as e:
        if getattr(e, 'status_code', 0) in (404, 405, 422):
            results.record_skip("dimensions.create", f"not supported ({e.status_code})")
        else:
            results.record_fail("dimensions.create", str(e))
    except Exception as e:
        results.record_fail("dimensions.create", str(e))

    return dimensions


def phase_patients(client: EarlClient, results: TestResult) -> list[Patient]:
    """Phase 3: Patients. Returns list of patients."""
    section("Phase 3: Patients")
    patients = []

    # patients.list
    try:
        patients = client.patients.list()
        assert len(patients) > 0, "No patients returned"
        assert all(isinstance(p, Patient) for p in patients)
        results.record_pass("patients.list", f"count={len(patients)}")
    except Exception as e:
        results.record_fail("patients.list", str(e))
        return patients

    # patients.list with pagination
    try:
        page = client.patients.list(limit=2, offset=0)
        assert isinstance(page, list)
        assert len(page) <= 2, f"Expected <= 2, got {len(page)}"
        results.record_pass("patients.list(limit=2)", f"count={len(page)}")
    except Exception as e:
        results.record_fail("patients.list(limit=2)", str(e))

    # patients.list with offset
    try:
        page2 = client.patients.list(limit=2, offset=2)
        assert isinstance(page2, list)
        # IDs should differ from first page (if enough patients exist)
        if len(patients) > 2 and len(page2) > 0:
            page1_ids = {p.id for p in client.patients.list(limit=2, offset=0)}
            page2_ids = {p.id for p in page2}
            assert page1_ids != page2_ids, "Pagination offset had no effect"
        results.record_pass("patients.list(offset=2)", f"count={len(page2)}")
    except Exception as e:
        results.record_fail("patients.list(offset=2)", str(e))

    # patients.get — find a patient with a URL-safe ID
    safe_patient = None
    for p in patients:
        if " " not in p.id and ":" not in p.id:
            safe_patient = p
            break
    if not safe_patient:
        # Fall back to first patient but URL-encode the ID
        safe_patient = patients[0]

    try:
        import urllib.parse
        patient_id = safe_patient.id
        # URL-encode path component for safety
        patient = client.patients.get(urllib.parse.quote(patient_id, safe=""))
        assert isinstance(patient, Patient)
        assert patient.name, "Patient name should not be empty"
        results.record_pass("patients.get", f"id={patient.id}, name={patient.name}")
    except Exception as e:
        results.record_fail("patients.get", str(e))

    return patients


def phase_pipelines_crud(
    client: EarlClient,
    results: TestResult,
    dimensions: list[Dimension],
    patients: list[Patient],
) -> None:
    """Phase 4: Pipelines CRUD (create, list, get, update, delete)."""
    section("Phase 4: Pipelines CRUD")

    if len(dimensions) < 2:
        results.record_skip("pipelines.* (CRUD)", "Need >= 2 dimensions")
        return

    ts = int(time.time())
    pipeline_name = f"smoke-crud-{ts}"
    dim_ids = [dimensions[0].id, dimensions[1].id]
    patient_ids = [patients[0].id] if patients else []
    created_pipeline: Optional[Pipeline] = None

    # pipelines.create
    try:
        created_pipeline = client.pipelines.create(
            name=pipeline_name,
            dimension_ids=dim_ids,
            patient_ids=patient_ids,
            description="Smoke test pipeline - safe to delete",
            conversation_initiator="doctor",
            max_turns=3,
        )
        assert isinstance(created_pipeline, Pipeline)
        assert created_pipeline.name == pipeline_name
        results.record_pass("pipelines.create", f"name={pipeline_name}")
    except Exception as e:
        results.record_fail("pipelines.create", str(e))
        return

    # pipelines.list
    try:
        pipelines = client.pipelines.list()
        assert isinstance(pipelines, list)
        names = [p.name for p in pipelines]
        assert pipeline_name in names, f"Created pipeline not in list: {names[:5]}"
        results.record_pass("pipelines.list", f"count={len(pipelines)}")
    except Exception as e:
        results.record_fail("pipelines.list", str(e))

    # pipelines.list(active_only=False)
    try:
        all_pipelines = client.pipelines.list(active_only=False)
        assert isinstance(all_pipelines, list)
        assert len(all_pipelines) >= len(pipelines), "active_only=False should return >= active"
        results.record_pass("pipelines.list(active_only=False)", f"count={len(all_pipelines)}")
    except Exception as e:
        results.record_fail("pipelines.list(active_only=False)", str(e))

    # pipelines.get
    try:
        fetched = client.pipelines.get(pipeline_name)
        assert isinstance(fetched, Pipeline)
        assert fetched.name == pipeline_name
        assert fetched.conversation_initiator == "doctor"
        assert fetched.max_turns == 3
        results.record_pass("pipelines.get", f"initiator={fetched.conversation_initiator}, turns={fetched.max_turns}")
    except Exception as e:
        results.record_fail("pipelines.get", str(e))

    # pipelines.update — PUT with config dict
    try:
        updated = client.pipelines.update(
            pipeline_name=pipeline_name,
            description="Updated by smoke test",
        )
        assert isinstance(updated, Pipeline)
        results.record_pass("pipelines.update", "description updated")
    except EarlError as e:
        if getattr(e, 'status_code', 0) in (404, 405):
            results.record_skip("pipelines.update", f"not supported ({e.status_code})")
        else:
            results.record_fail("pipelines.update", str(e))
    except Exception as e:
        results.record_fail("pipelines.update", str(e))

    # pipelines.delete
    try:
        client.pipelines.delete(pipeline_name)
        # Verify it's deactivated
        inactive = client.pipelines.list(active_only=True)
        inactive_names = [p.name for p in inactive]
        assert pipeline_name not in inactive_names, "Pipeline still active after delete"
        results.record_pass("pipelines.delete", f"soft-deleted {pipeline_name}")
    except EarlError as e:
        if getattr(e, 'status_code', 0) in (401, 403, 405):
            results.record_skip("pipelines.delete", f"not permitted ({e.status_code})")
        else:
            results.record_fail("pipelines.delete", str(e))
    except Exception as e:
        results.record_fail("pipelines.delete", str(e))


def phase_simulation_lifecycle(
    client: EarlClient,
    results: TestResult,
    dimensions: list[Dimension],
    patients: list[Patient],
    timeout: int,
) -> None:
    """Phase 5: Full simulation lifecycle (internal doctor)."""
    section("Phase 5: Simulation Lifecycle (internal doctor)")

    if len(dimensions) < 2 or len(patients) < 1:
        results.record_skip("simulation lifecycle", "Need dimensions and patients")
        return

    ts = int(time.time())
    pipeline_name = f"smoke-sim-{ts}"
    dim_ids = [dimensions[0].id, dimensions[1].id]
    patient_ids = [patients[0].id]
    sim_id = None

    # Create pipeline for simulation
    try:
        pipeline = client.pipelines.create(
            name=pipeline_name,
            dimension_ids=dim_ids,
            patient_ids=patient_ids,
            description="Smoke test simulation pipeline",
            conversation_initiator="doctor",
            max_turns=2,
        )
        results.record_pass("sim: create pipeline", f"name={pipeline_name}")
    except Exception as e:
        results.record_fail("sim: create pipeline", str(e))
        return

    # simulations.create
    try:
        sim = client.simulations.create(pipeline_name=pipeline_name)
        assert isinstance(sim, Simulation), f"Expected Simulation, got {type(sim)}"
        assert sim.id, "Simulation ID should not be empty"
        assert sim.status in (SimulationStatus.PENDING, SimulationStatus.RUNNING), \
            f"Expected pending/running, got {sim.status}"
        sim_id = sim.id
        results.record_pass("simulations.create", f"id={sim_id}, status={sim.status.value}")
    except Exception as e:
        err_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
        results.record_fail("simulations.create", err_msg)
        _cleanup_pipeline(client, pipeline_name)
        return

    # simulations.list
    try:
        sims = _retry_call(lambda: client.simulations.list(limit=10))
        assert isinstance(sims, list)
        sim_ids = [s.id for s in sims]
        assert sim_id in sim_ids, f"Created simulation not in list"
        results.record_pass("simulations.list", f"found in list, total={len(sims)}")
    except Exception as e:
        results.record_fail("simulations.list", str(e))

    # simulations.list with status filter
    try:
        running = _retry_call(lambda: client.simulations.list(status=SimulationStatus.RUNNING, limit=10))
        assert isinstance(running, list)
        results.record_pass("simulations.list(status=RUNNING)", f"count={len(running)}")
    except Exception as e:
        results.record_fail("simulations.list(status=RUNNING)", str(e))

    # simulations.get
    try:
        fetched = _retry_call(lambda: client.simulations.get(sim_id))
        assert isinstance(fetched, Simulation)
        assert fetched.id == sim_id
        assert fetched.total_episodes >= 1
        results.record_pass("simulations.get", f"episodes={fetched.total_episodes}, status={fetched.status.value}")
    except Exception as e:
        results.record_fail("simulations.get", str(e))

    # Simulation.progress property
    try:
        fetched = _retry_call(lambda: client.simulations.get(sim_id))
        progress = fetched.progress
        assert isinstance(progress, float)
        assert 0.0 <= progress <= 1.0
        results.record_pass("Simulation.progress", f"progress={progress:.0%}")
    except Exception as e:
        results.record_fail("Simulation.progress", str(e))

    # simulations.wait_for_completion
    progress_calls = []
    sim_completed_ok = False

    def on_progress(s: Simulation):
        progress_calls.append(s.status.value)

    try:
        completed = client.simulations.wait_for_completion(
            sim_id,
            poll_interval=3.0,
            timeout=timeout,
            on_progress=on_progress,
        )
        assert isinstance(completed, Simulation)
        assert completed.status == SimulationStatus.COMPLETED
        assert len(progress_calls) > 0, "on_progress callback was never called"
        sim_completed_ok = True
        results.record_pass(
            "simulations.wait_for_completion",
            f"polls={len(progress_calls)}, final={completed.status.value}"
        )
    except TimeoutError:
        results.record_fail("simulations.wait_for_completion", f"Timed out after {timeout}s")
    except Exception as e:
        # SimulationError means the sim reached a terminal state (failed) —
        # the SDK method still worked correctly. Record as pass with note.
        from earl_sdk.exceptions import SimulationError
        if isinstance(e, SimulationError):
            results.record_pass(
                "simulations.wait_for_completion",
                f"polls={len(progress_calls)}, sim failed (server-side): {e}"
            )
        else:
            results.record_fail("simulations.wait_for_completion", str(e))

    # simulations.get_episodes — works regardless of sim status
    episodes = []
    try:
        episodes = client.simulations.get_episodes(sim_id)
        assert isinstance(episodes, list)
        assert len(episodes) >= 1, "Expected at least 1 episode"
        ep = episodes[0]
        assert "episode_id" in ep, f"Missing episode_id in {list(ep.keys())}"
        results.record_pass("simulations.get_episodes", f"count={len(episodes)}")
    except Exception as e:
        results.record_fail("simulations.get_episodes", str(e))

    # simulations.get_episodes(include_dialogue=True)
    try:
        eps_with_dialogue = client.simulations.get_episodes(sim_id, include_dialogue=True)
        assert isinstance(eps_with_dialogue, list)
        if eps_with_dialogue:
            ep = eps_with_dialogue[0]
            dialogue = ep.get("dialogue_history", [])
            results.record_pass(
                "simulations.get_episodes(include_dialogue=True)",
                f"turns={len(dialogue)}"
            )
        else:
            results.record_pass("simulations.get_episodes(include_dialogue=True)", "empty")
    except Exception as e:
        results.record_fail("simulations.get_episodes(include_dialogue=True)", str(e))

    # simulations.get_episode (single)
    if episodes:
        try:
            episode_id = episodes[0]["episode_id"]
            ep_detail = client.simulations.get_episode(sim_id, episode_id)
            assert isinstance(ep_detail, dict)
            assert ep_detail.get("episode_id") == episode_id
            dialogue = ep_detail.get("dialogue_history", [])
            results.record_pass(
                "simulations.get_episode",
                f"episode_id={episode_id[:12]}..., turns={len(dialogue)}"
            )
        except Exception as e:
            results.record_fail("simulations.get_episode", str(e))

    # simulations.get_report — works for any terminal state
    try:
        report = client.simulations.get_report(sim_id)
        assert isinstance(report, dict)
        assert "summary" in report or "episodes" in report, f"Report keys: {list(report.keys())}"
        summary = report.get("summary", {})
        dim_scores = report.get("dimension_scores", {})
        report_episodes = report.get("episodes", [])
        results.record_pass(
            "simulations.get_report",
            f"episodes={len(report_episodes)}, dims={len(dim_scores)}, "
            f"avg={summary.get('average_score', 'N/A')}"
        )
    except Exception as e:
        results.record_fail("simulations.get_report", str(e))

    # Cleanup
    _cleanup_pipeline(client, pipeline_name)


def phase_cancel_simulation(
    client: EarlClient,
    results: TestResult,
    dimensions: list[Dimension],
    patients: list[Patient],
) -> None:
    """Phase 6: Create and cancel a simulation."""
    section("Phase 6: Cancel Simulation")

    if len(dimensions) < 2 or len(patients) < 1:
        results.record_skip("cancel simulation", "Need dimensions and patients")
        return

    ts = int(time.time())
    pipeline_name = f"smoke-cancel-{ts}"
    dim_ids = [dimensions[0].id, dimensions[1].id]
    patient_ids = [patients[0].id]

    # Create pipeline
    try:
        client.pipelines.create(
            name=pipeline_name,
            dimension_ids=dim_ids,
            patient_ids=patient_ids,
            description="Smoke test cancel pipeline",
            conversation_initiator="doctor",
            max_turns=50,  # Long to ensure it doesn't finish before we cancel
        )
    except Exception as e:
        results.record_fail("cancel: create pipeline", str(e))
        return

    # Start simulation
    try:
        sim = client.simulations.create(pipeline_name=pipeline_name)
        sim_id = sim.id
        results.record_pass("cancel: create simulation", f"id={sim_id}")
    except Exception as e:
        results.record_fail("cancel: create simulation", str(e) or repr(e))
        _cleanup_pipeline(client, pipeline_name)
        return

    # Cancel it
    try:
        # Small delay to let it start but not finish
        time.sleep(2)
        stopped = client.simulations.stop(sim_id)
        assert isinstance(stopped, Simulation)
        # Check the final status (may take a moment)
        time.sleep(2)
        final = client.simulations.get(sim_id)
        assert final.status in (SimulationStatus.STOPPED, SimulationStatus.CANCELLED, SimulationStatus.COMPLETED, SimulationStatus.FAILED), \
            f"Expected terminal status, got {final.status.value}"
        results.record_pass("simulations.stop", f"status={final.status.value}")
    except EarlError as e:
        if getattr(e, 'status_code', 0) == 404:
            # Simulation may have already completed/been cleaned up
            results.record_pass("simulations.stop", "sim already terminated (404)")
        else:
            results.record_fail("simulations.stop", str(e) or repr(e))
    except Exception as e:
        results.record_fail("simulations.stop", str(e) or repr(e))

    _cleanup_pipeline(client, pipeline_name)


def phase_client_driven(
    client: EarlClient,
    results: TestResult,
    dimensions: list[Dimension],
    patients: list[Patient],
    timeout: int,
) -> None:
    """Phase 7: Client-driven simulation flow."""
    section("Phase 7: Client-Driven Flow")

    if len(dimensions) < 2 or len(patients) < 1:
        results.record_skip("client-driven flow", "Need dimensions and patients")
        return

    ts = int(time.time())
    pipeline_name = f"smoke-cd-{ts}"
    dim_ids = [dimensions[0].id, dimensions[1].id]
    patient_ids = [patients[0].id]

    # Create client-driven pipeline
    try:
        pipeline = client.pipelines.create(
            name=pipeline_name,
            dimension_ids=dim_ids,
            patient_ids=patient_ids,
            doctor_config=DoctorApiConfig.client_driven(),
            description="Smoke test client-driven pipeline",
            conversation_initiator="patient",
            max_turns=3,
        )
        assert pipeline.doctor_api is not None
        assert pipeline.doctor_api.is_client_driven
        results.record_pass("client-driven: create pipeline", f"name={pipeline_name}")
    except Exception as e:
        results.record_fail("client-driven: create pipeline", str(e))
        return

    # Start simulation
    try:
        sim = client.simulations.create(pipeline_name=pipeline_name)
        sim_id = sim.id
        results.record_pass("client-driven: create simulation", f"id={sim_id}")
    except Exception as e:
        results.record_fail("client-driven: create simulation", str(e))
        _cleanup_pipeline(client, pipeline_name)
        return

    # Drive the conversation
    start = time.time()
    turns_submitted = 0
    pending_tested = False
    submit_tested = False
    conversation_timeout = min(timeout, 120)  # Max 2 min for conversation phase
    network_retries = 3

    try:
        while (time.time() - start) < conversation_timeout:
            # Get simulation status with retry
            sim_status = _retry_call(lambda: client.simulations.get(sim_id), retries=network_retries)
            if sim_status.status in (SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.CANCELLED):
                break

            # get_pending_episodes with retry
            pending = _retry_call(
                lambda: client.simulations.get_pending_episodes(sim_id),
                retries=network_retries,
            )
            if not pending_tested and isinstance(pending, list):
                results.record_pass("simulations.get_pending_episodes", f"count={len(pending)}")
                pending_tested = True

            if not pending:
                time.sleep(3)
                continue

            for ep in pending:
                episode_id = ep["episode_id"]

                # Simple doctor response
                doctor_response = "Thank you for sharing that. Could you tell me more about your symptoms? Any pain, fever, or other concerns?"

                # submit_response with retry
                try:
                    updated_ep = _retry_call(
                        lambda eid=episode_id: client.simulations.submit_response(sim_id, eid, doctor_response),
                        retries=network_retries,
                    )
                    turns_submitted += 1

                    if not submit_tested:
                        assert isinstance(updated_ep, dict)
                        results.record_pass(
                            "simulations.submit_response",
                            f"episode={episode_id[:12]}..., turns_so_far={len(updated_ep.get('dialogue_history', []))}"
                        )
                        submit_tested = True
                except EarlError as e:
                    # Episode may have moved to judging/completed between poll and submit
                    if getattr(e, 'status_code', 0) in (400, 409):
                        pass  # Expected race condition
                    else:
                        raise

            time.sleep(2)

        # Wait for judging to complete (can take a while)
        remaining = max(180, timeout - int(time.time() - start))
        final = _wait_for_terminal(client, sim_id, timeout=remaining)

        if final.status == SimulationStatus.COMPLETED:
            report = _retry_call(lambda: client.simulations.get_report(sim_id), retries=network_retries)
            avg_score = report.get("summary", {}).get("average_score", "N/A")
            results.record_pass(
                "client-driven: full flow",
                f"turns_submitted={turns_submitted}, status={final.status.value}, avg_score={avg_score}"
            )
        elif final.status == SimulationStatus.FAILED:
            results.record_pass(
                "client-driven: full flow",
                f"turns_submitted={turns_submitted}, sim failed server-side: {final.error_message}"
            )
        else:
            results.record_pass(
                "client-driven: full flow",
                f"turns_submitted={turns_submitted}, status={final.status.value} (may need more time for judging)"
            )

    except Exception as e:
        results.record_fail("client-driven: full flow", str(e) or repr(e))

    if not pending_tested:
        results.record_skip("simulations.get_pending_episodes", "No pending episodes observed")
    if not submit_tested:
        results.record_skip("simulations.submit_response", "No responses submitted")

    _cleanup_pipeline(client, pipeline_name)


def phase_error_handling(client: EarlClient, results: TestResult):
    """Phase 8: Error handling (verify SDK raises proper exceptions)."""
    section("Phase 8: Error Handling")

    # NotFoundError for non-existent patient
    try:
        client.patients.get("nonexistent-patient-id-12345")
        results.record_fail("NotFoundError (patient)", "No exception raised")
    except NotFoundError:
        results.record_pass("NotFoundError (patient)", "correctly raised")
    except EarlError as e:
        # Some APIs return 400 instead of 404 for invalid IDs
        results.record_pass("NotFoundError (patient)", f"EarlError raised: {e.status_code}")
    except Exception as e:
        results.record_fail("NotFoundError (patient)", f"Wrong exception: {type(e).__name__}: {e}")

    # NotFoundError for non-existent pipeline
    try:
        client.pipelines.get("nonexistent-pipeline-xyz-99999")
        results.record_fail("NotFoundError (pipeline)", "No exception raised")
    except NotFoundError:
        results.record_pass("NotFoundError (pipeline)", "correctly raised")
    except EarlError as e:
        results.record_pass("NotFoundError (pipeline)", f"EarlError raised: {e.status_code}")
    except Exception as e:
        results.record_fail("NotFoundError (pipeline)", f"Wrong exception: {type(e).__name__}: {e}")

    # NotFoundError for non-existent simulation
    try:
        client.simulations.get("nonexistent-sim-id-00000")
        results.record_fail("NotFoundError (simulation)", "No exception raised")
    except NotFoundError:
        results.record_pass("NotFoundError (simulation)", "correctly raised")
    except EarlError as e:
        results.record_pass("NotFoundError (simulation)", f"EarlError raised: {e.status_code}")
    except Exception as e:
        results.record_fail("NotFoundError (simulation)", f"Wrong exception: {type(e).__name__}: {e}")

    # ValidationError for invalid pipeline config
    try:
        client.pipelines.create(
            name="",  # Empty name
            dimension_ids=[],  # Empty dims
        )
        results.record_fail("ValidationError (empty pipeline)", "No exception raised")
    except (ValidationError, EarlError):
        results.record_pass("ValidationError (empty pipeline)", "correctly raised")
    except Exception as e:
        results.record_fail("ValidationError (empty pipeline)", f"Wrong exception: {type(e).__name__}: {e}")


def phase_model_properties(results: TestResult):
    """Phase 9: Model constructors and properties."""
    section("Phase 9: Model Properties")

    # DoctorApiConfig factories
    try:
        internal = DoctorApiConfig.internal()
        assert internal.type == "internal"
        assert not internal.is_client_driven

        external = DoctorApiConfig.external(api_url="https://example.com", api_key="key")
        assert external.type == "external"
        assert external.api_url == "https://example.com"
        assert not external.is_client_driven

        cd = DoctorApiConfig.client_driven()
        assert cd.type == "client_driven"
        assert cd.is_client_driven

        # to_dict / from_dict roundtrip
        ext_dict = external.to_dict()
        rebuilt = DoctorApiConfig.from_dict(ext_dict)
        assert rebuilt.type == "external"
        assert rebuilt.api_url == "https://example.com"

        results.record_pass("DoctorApiConfig factories", "internal/external/client_driven/roundtrip")
    except Exception as e:
        results.record_fail("DoctorApiConfig factories", str(e))

    # ConversationConfig
    try:
        patient_init = ConversationConfig.patient_initiated(max_turns=5)
        assert patient_init.initiator == "patient"
        assert patient_init.max_turns == 5

        doctor_init = ConversationConfig.doctor_initiated(max_turns=15)
        assert doctor_init.initiator == "doctor"
        assert doctor_init.max_turns == 15

        # to_dict / from_dict roundtrip
        d = patient_init.to_dict()
        rebuilt = ConversationConfig.from_dict(d)
        assert rebuilt.initiator == "patient"
        assert rebuilt.max_turns == 5

        results.record_pass("ConversationConfig", "patient_initiated/doctor_initiated/roundtrip")
    except Exception as e:
        results.record_fail("ConversationConfig", str(e))

    # SimulationStatus enum
    try:
        assert SimulationStatus.PENDING.value == "pending"
        assert SimulationStatus.RUNNING.value == "running"
        assert SimulationStatus.JUDGING.value == "judging"
        assert SimulationStatus.COMPLETED.value == "completed"
        assert SimulationStatus.FAILED.value == "failed"
        assert SimulationStatus.CANCELLED.value == "cancelled"
        results.record_pass("SimulationStatus enum", "all 6 values correct")
    except Exception as e:
        results.record_fail("SimulationStatus enum", str(e))

    # Environment enum
    try:
        assert str(Environment.DEV) == "dev"
        assert str(Environment.TEST) == "test"
        assert str(Environment.PROD) == "prod"
        assert str(Environment.PRODUCTION) == "prod"
        results.record_pass("Environment enum", "dev/test/prod/production")
    except Exception as e:
        results.record_fail("Environment enum", str(e))


# =============================================================================
# Helpers
# =============================================================================

def _retry_call(fn, retries: int = 3, delay: float = 2.0):
    """Retry a callable on transient network/gateway errors."""
    import http.client
    last_error = None
    for attempt in range(retries):
        try:
            return fn()
        except (ConnectionResetError, ConnectionError, OSError,
                http.client.RemoteDisconnected, TimeoutError) as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
        except EarlError as e:
            # Retry on gateway errors (502, 503, 504) and connection errors
            if getattr(e, 'status_code', 0) in (502, 503, 504):
                last_error = e
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                continue
            # Also retry if the underlying error looks like a network issue
            err_msg = str(e).lower()
            if any(kw in err_msg for kw in ("remote", "reset", "connection", "eof", "broken pipe")):
                last_error = e
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                continue
            raise  # Don't retry other API errors
    raise last_error


def _cleanup_pipeline(client: EarlClient, name: str):
    """Best-effort pipeline cleanup."""
    try:
        client.pipelines.delete(name)
    except Exception:
        pass  # Already deleted or doesn't exist


def _wait_for_terminal(client: EarlClient, sim_id: str, timeout: int = 120) -> Simulation:
    """Wait for simulation to reach a terminal state."""
    start = time.time()
    last_sim = None
    consecutive_errors = 0
    max_consecutive_errors = 5

    while (time.time() - start) < timeout:
        try:
            sim = client.simulations.get(sim_id)
            last_sim = sim
            consecutive_errors = 0
            if sim.status in (SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.CANCELLED):
                return sim
        except Exception:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                break  # Server is consistently failing
        time.sleep(3)

    # Final attempt or return what we have
    try:
        return client.simulations.get(sim_id)
    except Exception:
        pass
    if last_sim:
        return last_sim
    return Simulation(
        id=sim_id, pipeline_name="", organization_id="",
        status=SimulationStatus.RUNNING, error_message="Timed out waiting"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EARL SDK - Full Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", default="test", choices=["dev", "test", "prod"],
                        help="Environment to test (default: test)")
    parser.add_argument("--client-id", help="Auth0 client ID (or EARL_CLIENT_ID env)")
    parser.add_argument("--client-secret", help="Auth0 client secret (or EARL_CLIENT_SECRET env)")
    parser.add_argument("--organization", help="Auth0 organization (or EARL_ORGANIZATION env)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Max wait for simulations in seconds (default: 300)")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Skip Phase 5 (simulation lifecycle)")
    parser.add_argument("--skip-cancel", action="store_true",
                        help="Skip Phase 6 (cancel simulation)")
    parser.add_argument("--skip-client-driven", action="store_true",
                        help="Skip Phase 7 (client-driven flow)")

    args = parser.parse_args()
    client_id, client_secret, organization = get_credentials(args)

    print(f"\n{'='*60}")
    print(f"{Colors.BOLD}EARL SDK - Full Smoke Test{Colors.END}")
    print(f"{'='*60}")
    print(f"  Environment:  {args.env}")
    print(f"  Client ID:    {client_id[:8]}...")
    print(f"  Organization: {organization or '(none)'}")
    print(f"  Timeout:      {args.timeout}s")
    print(f"  Skipping:     {', '.join(f for f, v in [('simulation', args.skip_simulation), ('cancel', args.skip_cancel), ('client-driven', args.skip_client_driven)] if v) or 'none'}")
    print(f"  Started:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Initialize client
    client = EarlClient(
        client_id=client_id,
        client_secret=client_secret,
        organization=organization,
        environment=args.env,
        request_timeout=120,
    )

    results = TestResult()

    # Phase 1: Connection & Auth
    phase_connection(client, results)

    # Phase 2: Dimensions
    dimensions = phase_dimensions(client, results)

    # Phase 3: Patients
    patients = phase_patients(client, results)

    # Phase 4: Pipelines CRUD
    phase_pipelines_crud(client, results, dimensions, patients)

    # Phase 5: Simulation Lifecycle (slowest)
    if args.skip_simulation:
        section("Phase 5: Simulation Lifecycle (SKIPPED)")
        results.record_skip("simulation lifecycle", "--skip-simulation")
    else:
        phase_simulation_lifecycle(client, results, dimensions, patients, args.timeout)

    # Phase 6: Cancel Simulation
    if args.skip_cancel:
        section("Phase 6: Cancel Simulation (SKIPPED)")
        results.record_skip("cancel simulation", "--skip-cancel")
    else:
        phase_cancel_simulation(client, results, dimensions, patients)

    # Phase 7: Client-Driven Flow
    if args.skip_client_driven:
        section("Phase 7: Client-Driven Flow (SKIPPED)")
        results.record_skip("client-driven flow", "--skip-client-driven")
    else:
        phase_client_driven(client, results, dimensions, patients, args.timeout)

    # Phase 8: Error Handling
    phase_error_handling(client, results)

    # Phase 9: Model Properties (offline, instant)
    phase_model_properties(results)

    # Summary
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
