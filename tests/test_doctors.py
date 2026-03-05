#!/usr/bin/env python3
"""
Earl SDK - Doctor Integration Tests (Lumos verifiers)

Tests the SDK with both internal (Earl's built-in) and external (customer-provided) doctors,
scored by the next-gen Lumos conversation verifiers.

Usage:
    # Quick test — internal doctor, Lumos verifiers, 2 turns:
    python3 test_doctors.py --env test --doctor internal --patients 1 --max-turns 2 --wait

    # Internal doctor with all Lumos dimensions:
    python3 test_doctors.py --env test --doctor internal --patients 1 --dimensions lumos-all --wait

    # External doctor with specific Lumos dimensions:
    python3 test_doctors.py --env test --doctor external --patients 1 --wait \
        --doctor-url "https://your-api.com/chat" --doctor-key "your-key" \
        --dimensions "scoring-dimensions/clinical-correctness,hard-gates/fabricated-ehr-data"

    # Legacy judge (old medical-conversation-judge):
    python3 test_doctors.py --env test --doctor internal --patients 1 --wait --judge legacy \
        --dimensions "turn_pacing,context_recall"

    # List patients only:
    python3 test_doctors.py --env test --list-only

    # Fire 20 simulations, poll reports:
    python3 test_doctors.py --env test --doctor external --patients 1 --repeat 20 \
        --poll-reports --poll-interval 300 --timeout 86400 \
        --doctor-url "..." --doctor-key "..."

Credentials:
- Pass via CLI: --client-id and --client-secret
- Or set env vars: EARL_CLIENT_ID and EARL_CLIENT_SECRET
- Or store in ~/.earl/config.json (auto-detected)
"""

import os
import sys
import argparse
import time
import json
from typing import Optional, List

# Add the SDK to path for development testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earl_sdk import EarlClient
from earl_sdk.models import DoctorApiConfig
from earl_sdk.exceptions import EarlError


# =============================================================================
# CONFIGURATION
# =============================================================================

# Lumos builtin dimensions — a sensible default subset for quick tests.
# Full list available via GET /judges on the conversation-verifiers service.
LUMOS_DEFAULT_DIMS = [
    "scoring-dimensions/clinical-correctness",
    "scoring-dimensions/communication--empathy",
    "scoring-dimensions/communication--clarity",
    "hard-gates/fabricated-ehr-data",
]

# Legacy dimensions (old medical-conversation-judge)
LEGACY_DEFAULT_DIMS = [
    "turn_pacing",
    "context_recall",
    "state_sensitivity",
    "patient_education",
    "empathetic_communication",
]

# All Lumos builtin scoring dimensions (fetched at runtime via --dimensions lumos-all)
# Hard gates are always included when using Lumos.
LUMOS_DEFAULT_HARD_GATES = [
    "hard-gates/contraindication-allergy-negligence",
    "hard-gates/fabricated-ehr-data",
    "hard-gates/patient-identity-confusion",
    "hard-gates/patient-specific-hallucination",
    "hard-gates/privacy-confidentiality-breach",
    "hard-gates/scope-violation",
    "hard-gates/unsafe-medication-guidance",
]


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def log_success(msg): print(f"{Colors.GREEN}✓ {msg}{Colors.END}")
def log_error(msg): print(f"{Colors.RED}✗ {msg}{Colors.END}")
def log_info(msg): print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")
def log_warning(msg): print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")
def log_section(title): print(f"\n{Colors.BOLD}{'='*60}\n{title}\n{'='*60}{Colors.END}")
def log_subsection(title): print(f"\n{Colors.CYAN}--- {title} ---{Colors.END}")


def get_credentials(cli_client_id=None, cli_client_secret=None, cli_organization=None):
    """Get credentials from CLI args, environment, or ~/.earl/config.json."""
    client_id = cli_client_id or os.environ.get("EARL_CLIENT_ID", "")
    client_secret = cli_client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
    organization = cli_organization or os.environ.get("EARL_ORGANIZATION", "")

    if not client_id or not client_secret:
        # Try ~/.earl/config.json (created by `earl` interactive CLI)
        config_path = os.path.expanduser("~/.earl/config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    cfg = json.load(f)
                profile_name = cfg.get("active_profile", "test")
                profile = cfg.get("profiles", {}).get(profile_name, {})
                if profile.get("client_id") and profile.get("client_secret"):
                    client_id = profile["client_id"]
                    raw_secret = profile["client_secret"]
                    if raw_secret.endswith("=="):
                        client_secret = base64.b64decode(raw_secret).decode()
                    else:
                        client_secret = raw_secret
                    organization = organization or profile.get("organization", "")
                    log_info(f"Loaded credentials from ~/.earl/config.json (profile: {profile_name})")
            except Exception as e:
                log_warning(f"Failed to read ~/.earl/config.json: {e}")

    if not client_id or not client_secret:
        print(f"\n{Colors.RED}{'='*60}")
        print("MISSING CREDENTIALS")
        print(f"{'='*60}{Colors.END}")
        print("\nProvide via CLI, env vars, or ~/.earl/config.json:\n")
        print("  --client-id '...' --client-secret '...'")
        print("  export EARL_CLIENT_ID='...' EARL_CLIENT_SECRET='...'")
        print("  earl  # interactive CLI stores creds in ~/.earl/config.json")
        print("")
        sys.exit(1)
    
    return client_id, client_secret, organization


def test_list_patients(client: EarlClient) -> List:
    """Test listing patients and return the list."""
    log_subsection("Listing Available Patients")
    
    try:
        patients = client.patients.list()
        log_success(f"Found {len(patients)} patients")
        
        for p in patients[:5]:  # Show first 5
            condition = p.condition or p.task or "N/A"
            print(f"   • {p.name or p.id} ({p.age}yo) - {condition}")
            if p.scenario:
                print(f"     {p.scenario[:60]}...")
        
        if len(patients) > 5:
            print(f"   ... and {len(patients) - 5} more")
        
        return patients
    except Exception as e:
        log_error(f"Failed to list patients: {e}")
        return []


def _is_report_terminal(report: dict) -> bool:
    """True when simulation is over: finished_at is set or any episode has conversation_ended_successfully != null."""
    if report.get("finished_at") is not None:
        return True
    for ep in report.get("episodes", []):
        if ep.get("conversation_ended_successfully") is not None:
            return True
    return False


def _poll_reports_and_write_jsonl(
    client: EarlClient,
    simulation_ids: List[str],
    pipeline_name: str,
    poll_interval_seconds: int = 300,
    timeout_seconds: int = 86400,
) -> None:
    """Poll GET /simulations/{id}/report every poll_interval_seconds; write each terminal report to JSONL."""
    results_file = f"earl_simulation_reports_{pipeline_name}.jsonl"
    log_info(f"Polling reports every {poll_interval_seconds}s, writing to {results_file} (max {timeout_seconds}s)")
    done: set = set()  # sim_ids we've written to JSONL
    start = time.time()

    def _get_report_with_retry(sim_id: str, max_retries: int = 2) -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                return client.simulations.get_report(sim_id)
            except EarlError as e:
                if attempt < max_retries - 1:
                    time.sleep(10)
                    continue
                log_warning(f"   Report failed for {sim_id[:12]}...: {e}")
                return None
        return None

    with open(results_file, "a") as f:
        while len(done) < len(simulation_ids) and (time.time() - start) < timeout_seconds:
            for sim_id in simulation_ids:
                if sim_id in done:
                    continue
                report = _get_report_with_retry(sim_id)
                if report is None:
                    continue
                if not _is_report_terminal(report):
                    continue
                line = json.dumps(report, default=str) + "\n"
                f.write(line)
                f.flush()
                done.add(sim_id)
                status = report.get("status", "?")
                err_info = ""
                for ep in report.get("episodes", []):
                    if ep.get("error"):
                        err_info = f" error={ep.get('error', '')[:60]}"
                        break
                    if ep.get("conversation_ended_successfully") is False:
                        err_info = " conversation_ended_successfully=false"
                        break
                log_info(f"   Wrote report for {sim_id[:12]}... status={status}{err_info} ({len(done)}/{len(simulation_ids)} done)")

            if len(done) >= len(simulation_ids):
                break
            elapsed = int(time.time() - start)
            print(f"\r   Polling... {len(done)}/{len(simulation_ids)} reports written ({elapsed}s)    ", end="", flush=True)
            time.sleep(poll_interval_seconds)

    print()
    log_success(f"Reports written to {results_file} ({len(done)} simulations)")
    if len(done) < len(simulation_ids):
        log_warning(f"   {len(simulation_ids) - len(done)} simulation(s) still in progress (timeout or skipped)")


def run_simulation(
    client: EarlClient,
    doctor_type: str,
    doctor_api_url: Optional[str] = None,
    doctor_api_key: Optional[str] = None,
    auth_type: str = "bearer",
    patient_count: int = 3,
    max_turns: int = 50,
    doctor_initiates: bool = False,
    parallel_count: int = 5,
    timeout: int = 1800,  # 30 minutes default
    wait: bool = True,
    save_results: bool = True,
    dimensions: List[str] = None,
    verifiers: str = "lumos",
    case_id: Optional[str] = None,
    repeat_simulations: int = 1,
    max_time_seconds: Optional[float] = None,
    poll_reports: bool = False,
    poll_interval: int = 300,
) -> bool:
    """Run a simulation with specified doctor configuration."""
    
    verifier_label = "Lumos" if verifiers == "lumos" else "Legacy"
    header = f"{doctor_type.upper()} Doctor + {verifier_label} Verifiers"
    if case_id:
        header += f" (case: {case_id})"
    log_section(header)

    if case_id:
        log_info(f"Using case: {case_id}")
        try:
            case_detail = client.cases.get(case_id)
            totals = case_detail.get("totals", {})
            log_success(
                f"Case loaded: {totals.get('case_verifiers', 0)} case verifiers, "
                f"{totals.get('hard_gates', 0)} hard gates, "
                f"{totals.get('scoring_dimensions', 0)} scoring dimensions"
            )
        except Exception as e:
            log_warning(f"Could not load case details: {e}")
    
    # Fetch patients
    try:
        all_patients = client.patients.list()
        if not all_patients:
            log_error("No patients available")
            return False
    except Exception as e:
        log_error(f"Failed to fetch patients: {e}")
        return False
    
    # Select patients
    selected_patients = all_patients[:patient_count]
    patient_ids = [p.id for p in selected_patients]
    num_patients = len(patient_ids)
    
    # Configure doctor
    if doctor_type == "internal":
        doctor_config = DoctorApiConfig.internal()
        log_info("Using Earl's internal doctor agent")
    else:
        if not doctor_api_url:
            log_error("External doctor requires --doctor-url")
            return False
        doctor_config = DoctorApiConfig.external(
            api_url=doctor_api_url,
            api_key=doctor_api_key,
            auth_type=auth_type,
        )
        log_info(f"External doctor: {doctor_api_url}")
        if doctor_api_key:
            log_info(f"API key: {'***' + doctor_api_key[-8:]}")
    
    # Settings
    parallel = min(num_patients, parallel_count)
    use_dimensions = dimensions if dimensions is not None else LUMOS_DEFAULT_DIMS
    initiator = "doctor" if doctor_initiates else "patient"
    
    log_info(f"Patients: {num_patients}, Parallel: {parallel}, Max turns: {max_turns}")
    log_info(f"Initiator: {initiator}, Timeout: {timeout}s")
    if use_dimensions:
        log_info(f"Dimensions: {', '.join(use_dimensions)}")
    elif case_id:
        log_info("Dimensions: from case defaults (server-side)")
    
    print(f"\n   Selected patients:")
    for p in selected_patients:
        condition = p.condition or p.task or "N/A"
        print(f"   • {p.id}: {p.name or 'N/A'} - {condition}")
    
    try:
        waited_via_poll = False
        # Create pipeline
        pipeline_name = f"sdk-test-{doctor_type}-{int(time.time())}"
        log_info(f"Creating pipeline: {pipeline_name}")
        
        create_kwargs = dict(
            name=pipeline_name,
            doctor_config=doctor_config,
            patient_ids=patient_ids,
            verifier_ids=use_dimensions,
            validate_doctor=False,
            conversation_initiator=initiator,
            max_turns=max_turns,
            verifiers=verifiers,
        )
        if case_id:
            create_kwargs["case_id"] = case_id
        client.pipelines.create(**create_kwargs)
        log_success(f"Pipeline created (verifiers: {verifiers}{', case: ' + case_id if case_id else ''})")

        # Start simulation(s): single run or batch (fire-and-forget)
        simulation_ids: List[str] = []
        start_wall = time.time()

        if repeat_simulations > 1:
            # Fire N simulations, print each ID, write to file as we go (so early exit still has IDs)
            ids_file = f"earl_simulation_ids_{pipeline_name}.txt"
            log_info(f"Starting {repeat_simulations} simulations (async, no wait)...")
            print(f"   IDs will be appended to: {ids_file}")
            for i in range(repeat_simulations):
                if max_time_seconds is not None and (time.time() - start_wall) >= max_time_seconds:
                    log_warning(f"Stopped after {max_time_seconds}s (started {len(simulation_ids)} simulations)")
                    break
                sim = client.simulations.create(
                    pipeline_name=pipeline_name,
                    num_episodes=num_patients,
                    parallel_count=parallel,
                )
                simulation_ids.append(sim.id)
                # Parseable line for extraction (e.g. grep EARL_SIMULATION_ID)
                print(f"EARL_SIMULATION_ID={sim.id}")
                # Persist immediately so we don't lose IDs if process exits early
                try:
                    with open(ids_file, "a") as f:
                        f.write(sim.id + "\n")
                except Exception:
                    pass
            print(f"\n   Started {len(simulation_ids)} simulations (running async on server)")
            if simulation_ids:
                print("   Simulation IDs (also in {}):".format(ids_file))
                for sid in simulation_ids:
                    print(f"   {sid}")
            simulation = sim  # last created (used only for cleanup path; no wait/save in repeat mode)
            wait = False
            results = None
            waited_via_poll = False

            # Poll /simulations/{id}/report every poll_interval until all finished or failed; write JSONL
            if poll_reports and simulation_ids:
                _poll_reports_and_write_jsonl(
                    client=client,
                    simulation_ids=simulation_ids,
                    pipeline_name=pipeline_name,
                    poll_interval_seconds=poll_interval,
                    timeout_seconds=timeout,
                )
                waited_via_poll = True
        else:
            # Single simulation
            log_info("Starting simulation...")
            simulation = client.simulations.create(
                pipeline_name=pipeline_name,
                num_episodes=num_patients,
                parallel_count=parallel,
            )
            log_success(f"Simulation started: {simulation.id}")
            print(f"   Status: {simulation.status.value}")
            print(f"   Episodes: {num_patients} ({parallel} parallel)")

        # Wait for completion (only for single run)
        results = None
        if wait:
            def _get_simulation_with_retry(sim_id: str, max_retries: int = 5):
                """Poll simulation status with retries on connection/timeout errors."""
                last_err = None
                for attempt in range(max_retries):
                    try:
                        return client.simulations.get(sim_id)
                    except (EarlError, TimeoutError, ConnectionError, OSError) as e:
                        last_err = e
                        err_str = str(e).lower()
                        if ("timed out" in err_str or "connect" in err_str or "reset" in err_str) and attempt < max_retries - 1:
                            wait_secs = 5 * (attempt + 1)  # backoff: 5, 10, 15, 20
                            log_warning(f"Retry {attempt + 1}/{max_retries - 1} after {wait_secs}s: {e}")
                            time.sleep(wait_secs)
                            continue
                        raise
                raise last_err  # unreachable if max_retries > 0

            log_info(f"Waiting for completion (max {timeout}s)...")
            start_time = time.time()

            last_progress_time = start_time
            last_completed = 0

            while time.time() - start_time < timeout:
                sim = _get_simulation_with_retry(simulation.id)
                if sim.status.value in ["completed", "failed", "stopped"]:
                    break
                elapsed = int(time.time() - start_time)
                completed = getattr(sim, 'completed_episodes', 0)
                total = getattr(sim, 'total_episodes', num_patients)

                # Track progress - reset timeout if making progress
                if completed > last_completed:
                    last_progress_time = time.time()
                    last_completed = completed

                print(f"\r   Status: {sim.status.value}, Progress: {completed}/{total} ({elapsed}s)", end="", flush=True)
                time.sleep(10)

            print()
            final_sim = _get_simulation_with_retry(simulation.id)
            
            if final_sim.status.value == "completed":
                log_success("Simulation completed!")
            elif final_sim.status.value == "failed":
                log_error("Simulation failed")
            else:
                log_warning(f"Simulation still running: {final_sim.status.value}")
            
            # Show summary
            if final_sim.summary:
                avg_score = final_sim.summary.get("average_score")
                completed = final_sim.summary.get("completed", 0)
                failed = final_sim.summary.get("failed", 0)
                print(f"\n   Summary:")
                print(f"   • Completed: {completed}/{num_patients}")
                print(f"   • Failed: {failed}")
                if avg_score is not None:
                    print(f"   • Average Score: {avg_score:.2f}/4")
            
            # Get detailed report
            try:
                report = client.simulations.get_report(simulation.id)
                results = report
                
                if "episodes" in report:
                    log_subsection("Episode Results")
                    
                    for ep in report["episodes"]:
                        score = ep.get("total_score")
                        status = ep.get("status", "?")
                        error = ep.get("error")
                        patient = ep.get("patient_name") or ep.get("patient_id", "?")
                        dialogue = ep.get("dialogue_history", [])
                        turns = len(dialogue)
                        
                        # Check for insights and termination
                        insights_count = 0
                        terminated_by_patient = False
                        for turn in dialogue:
                            metadata = turn.get("metadata", {}) or {}
                            if "v2_insights" in metadata:
                                insights_count += 1
                            if metadata.get("terminated"):
                                terminated_by_patient = True
                        
                        if status == "failed" and error:
                            log_error(f"Episode {ep.get('episode_number')}: {patient}")
                            print(f"      Error: {error[:80]}...")
                        elif score is not None:
                            log_success(f"Episode {ep.get('episode_number')}: {patient}")
                            print(f"      Score: {score:.2f}, Turns: {turns}")
                            
                            # Show structured results (new format)
                            hg = ep.get("hard_gates", [])
                            sd = ep.get("scoring_dimensions", [])
                            cv = ep.get("case_verifiers", [])
                            if hg:
                                passed = sum(1 for g in hg if g.get("passed"))
                                print(f"      Hard Gates: {passed}/{len(hg)} passed")
                            if sd:
                                activated = [d for d in sd if d.get("activated", True)]
                                if activated:
                                    avg = sum(d.get("score", 0) for d in activated) / len(activated)
                                    print(f"      Scoring Dims: {avg:.1f}/4 avg ({len(activated)} evaluated, {len(sd)-len(activated)} skipped)")
                            if cv:
                                triggered = [v for v in cv if v.get("triggered")]
                                total_pts = sum(v.get("points_awarded", 0) for v in cv)
                                print(f"      Case Verifiers: {len(triggered)}/{len(cv)} triggered ({total_pts:+d} pts)")
                            
                            # Fallback: flat judge_scores if no structured sections
                            if not hg and not sd and not cv:
                                judge_scores = ep.get("judge_scores", {})
                                if judge_scores:
                                    scores_str = [f"{d[:12]}: {s:.1f}" for d, s in judge_scores.items() if isinstance(s, (int, float))]
                                    if scores_str:
                                        print(f"      Dimensions: {', '.join(scores_str)}")
                            
                            if insights_count > 0:
                                print(f"      ✓ {insights_count} turns with patient insights")
                            if terminated_by_patient:
                                print(f"      🛑 Patient initiated termination")
                        else:
                            log_info(f"Episode {ep.get('episode_number')}: {patient}")
                            print(f"      Status: {status}, Turns: {turns}")
                    
                    # Dimension summary
                    if "dimension_scores" in report and report["dimension_scores"]:
                        log_subsection("Dimension Scores Summary")
                        for dim, scores in report["dimension_scores"].items():
                            avg = scores.get("average", 0)
                            print(f"   • {dim}: {avg:.2f}/4")
                
            except Exception as e:
                log_warning(f"Could not get report: {e}")
        else:
            if not waited_via_poll:
                log_info("Simulation started (not waiting for completion)")
        
        # Save results
        if save_results and results:
            results_file = f"{doctor_type}_results_{simulation.id[:8]}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            log_info(f"Results saved: {results_file}")
        
        # Keep pipeline for reproducibility / debugging
        log_info(f"Pipeline kept for debugging: {pipeline_name}")
        
        # Determine success: at least one episode must have completed
        if wait and final_sim:
            completed = (final_sim.summary or {}).get("completed", 0)
            failed = (final_sim.summary or {}).get("failed", 0)
            if completed == 0 and failed > 0:
                return False
            if final_sim.status.value == "failed":
                return False
        
        return True
        
    except Exception as e:
        log_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Earl SDK - Doctor Integration Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test internal doctor with 2 patients:
  python3 test_doctors.py --env test --doctor internal --patients 2 --wait
  
  # Test external doctor:
  python3 test_doctors.py --env test --doctor external \\
      --doctor-url "https://your-api.com/chat" \\
      --doctor-key "your-key" --patients 3 --wait
  
  # List patients only:
  python3 test_doctors.py --env test --list-only
        """
    )
    
    # Environment
    parser.add_argument("--env", choices=["dev", "test", "prod"], default="test",
                        help="Environment (default: test)")
    
    # Doctor selection
    parser.add_argument("--doctor", choices=["internal", "external"], default="internal",
                        help="Doctor type: internal (Earl's) or external (default: internal)")
    
    # External doctor config
    parser.add_argument("--doctor-url", type=str,
                        help="External doctor API URL (required for external)")
    parser.add_argument("--doctor-key", type=str,
                        help="External doctor API key")
    parser.add_argument("--auth-type", choices=["bearer", "api_key"], default="bearer",
                        help="API key auth type (default: bearer)")
    
    # Patient/simulation settings
    parser.add_argument("--patients", type=int, default=3,
                        help="Number of patients to use (default: 3)")
    parser.add_argument("--max-turns", type=int, default=50,
                        help="Max conversation turns (default: 50)")
    parser.add_argument("--parallel", type=int, default=5,
                        help="Parallel episodes (default: 5)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Timeout in seconds (default: 1800 = 30 min)")
    parser.add_argument("--doctor-initiates", action="store_true",
                        help="Doctor starts conversation (default: patient)")
    
    # Verifiers
    parser.add_argument("--verifiers", choices=["lumos", "legacy"], default="lumos",
                        help="Verifier backend: 'lumos' (next-gen, default) or 'legacy'")
    parser.add_argument("--case", type=str, default=None, metavar="CASE_ID",
                        help="Case ID to use (e.g. 'carla-hypertension-yasmin'). "
                             "Includes case verifiers + default hard gates + scoring dimensions automatically.")

    # Verifier IDs
    parser.add_argument("--dimensions", type=str, default=None,
                        help="Verifier IDs to evaluate. Defaults depend on --verifiers:\n"
                             "  lumos:  4 core dims + 1 hard gate (use 'lumos-all' for all 119+7)\n"
                             "  legacy: 5 core dims (use 'all' for all ~130)\n"
                             "Comma-separated for custom, e.g. 'scoring-dimensions/clinical-correctness,hard-gates/fabricated-ehr-data'")
    parser.add_argument("--list-dimensions", action="store_true",
                        help="List all available dimensions grouped by category and exit")
    
    # Execution options
    parser.add_argument("--wait", action="store_true",
                        help="Wait for simulation completion")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="Create N simulations (same pipeline, 1 episode each), print each ID (default: 1)")
    parser.add_argument("--max-time", type=float, default=None, metavar="SECS",
                        help="With --repeat: stop starting new sims after SECS seconds; may start fewer than N")
    parser.add_argument("--poll-reports", action="store_true", default=None,
                        help="With --repeat: wait until all sims finished/failed by polling reports (default: True when --repeat > 1)")
    parser.add_argument("--no-poll-reports", action="store_true",
                        help="With --repeat: do not wait; fire sims and exit (overrides default when --repeat > 1)")
    parser.add_argument("--poll-interval", type=int, default=300, metavar="SECS",
                        help="Seconds between report polls when waiting for reports (default: 300 = 5 min)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list patients, don't run simulation")
    
    # Credentials
    parser.add_argument("--client-id", type=str,
                        help="Auth0 client ID (or EARL_CLIENT_ID env)")
    parser.add_argument("--client-secret", type=str,
                        help="Auth0 client secret (or EARL_CLIENT_SECRET env)")
    parser.add_argument("--organization", type=str,
                        help="Organization ID (or EARL_ORGANIZATION env)")
    
    args = parser.parse_args()
    
    # Get credentials
    client_id, client_secret, organization = get_credentials(
        args.client_id, args.client_secret, args.organization
    )
    
    # Initialize client
    log_section("Initializing Earl SDK")
    # Use longer request timeout when waiting or when polling reports (get_report can be slow)
    request_timeout = 180 if (args.wait or args.poll_reports) else 120
    client = EarlClient(
        client_id=client_id,
        client_secret=client_secret,
        organization=organization,
        environment=args.env,
        request_timeout=request_timeout,
    )
    log_success(f"Client ready ({args.env} environment)")
    print(f"   API: {client.api_url}")
    
    # List dimensions if requested
    if args.list_dimensions:
        log_subsection("Listing Available Dimensions")
        try:
            dims = client.dimensions.list()
            # Group by category
            categories = {}
            for d in dims:
                cat = getattr(d, 'category', 'Unknown')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(d)
            
            print(f"\n   Total: {len(dims)} dimensions in {len(categories)} categories\n")
            for cat in sorted(categories.keys()):
                dim_list = categories[cat]
                print(f"   {Colors.BOLD}{cat}{Colors.END} ({len(dim_list)} dimensions):")
                for d in dim_list:
                    desc = getattr(d, 'description', '')[:60] if hasattr(d, 'description') else ''
                    print(f"      • {d.id}" + (f" - {desc}..." if desc else ""))
                print()
            
            log_success(f"Listed {len(dims)} dimensions")
        except Exception as e:
            log_error(f"Failed to list dimensions: {e}")
        sys.exit(0)
    
    # List patients
    patients = test_list_patients(client)
    if not patients:
        log_error("No patients available")
        sys.exit(1)
    
    if args.list_only:
        log_success("Done!")
        sys.exit(0)
    
    # Validate external doctor args
    if args.doctor == "external" and not args.doctor_url:
        log_error("External doctor requires --doctor-url")
        sys.exit(1)
    
    # When --repeat > 1: wait by default (poll reports) unless --no-poll-reports
    wait = args.wait and args.repeat <= 1
    poll_reports = (args.repeat > 1 and not args.no_poll_reports) if args.poll_reports is None else args.poll_reports
    if args.no_poll_reports:
        poll_reports = False

    # Resolve dimensions based on judge type
    dim_arg = (args.dimensions or "").strip().lower()

    if args.case and not dim_arg:
        dimensions = []
        log_info(f"Using case '{args.case}' — verifiers + default gates/dims loaded server-side")
    elif args.verifiers == "lumos":
        if not dim_arg or dim_arg == "default":
            dimensions = LUMOS_DEFAULT_DIMS
            log_info(f"Using Lumos defaults: {len(dimensions)} verifiers")
        elif dim_arg in ("lumos-all", "all"):
            dimensions = LUMOS_DEFAULT_HARD_GATES[:]
            log_info("Fetching all Lumos scoring dimensions from verifiers service...")
            try:
                import urllib.request, urllib.error
                api_base = client.api_url.replace("/api/v1", "")
                headers = client.auth.get_headers()
                headers["Accept"] = "application/json"
                req = urllib.request.Request(f"{api_base}/api/v1/verifiers/list", headers=headers)
                # The verifiers list endpoint might not exist yet; fall back to
                # the known comprehensive set from the Lumos service itself.
                raise NotImplementedError("Use hardcoded list")
            except Exception:
                pass
            # Comprehensive set: all 119 scoring dimensions from Lumos
            ALL_SCORING = [
                "scoring-dimensions/adaptive-dialogue--context-recall",
                "scoring-dimensions/adaptive-dialogue--question-management",
                "scoring-dimensions/adaptive-dialogue--state-sensitivity",
                "scoring-dimensions/alternative-treatment-options--completeness",
                "scoring-dimensions/alternative-treatment-options--personalization",
                "scoring-dimensions/clinical-correctness",
                "scoring-dimensions/clinical-reasoning--diagnostic-reasoning",
                "scoring-dimensions/clinical-reasoning--inferential-symptom-recognition",
                "scoring-dimensions/clinical-reasoning--procedures-reasoning",
                "scoring-dimensions/clinical-reasoning--symptom-severity-assessment",
                "scoring-dimensions/clinical-reasoning--treatment-reasoning",
                "scoring-dimensions/clinical-reasoning-and-differential",
                "scoring-dimensions/cognitive-load",
                "scoring-dimensions/communication--adaptability",
                "scoring-dimensions/communication--clarity",
                "scoring-dimensions/communication--empathy",
                "scoring-dimensions/communication--professionalism-and-tone",
                "scoring-dimensions/communication--responsiveness",
                "scoring-dimensions/communication-quality",
                "scoring-dimensions/contextual-awareness--continuity",
                "scoring-dimensions/contextual-awareness--diagnostic-context-management",
                "scoring-dimensions/contextual-awareness--patient-context",
                "scoring-dimensions/contextual-awareness--prevention",
                "scoring-dimensions/contextual-awareness--resources-awareness",
                "scoring-dimensions/contextual-awareness--system-coordination",
                "scoring-dimensions/counseling-and-education-quality",
                "scoring-dimensions/cultural-competence-and-sensitivity",
                "scoring-dimensions/differential-diagnosis--bias-awareness",
                "scoring-dimensions/differential-diagnosis--completeness",
                "scoring-dimensions/differential-diagnosis--prioritization",
                "scoring-dimensions/differential-diagnosis--rare-disease-inclusion",
                "scoring-dimensions/ehr-grounding",
                "scoring-dimensions/elicitation-depth",
                "scoring-dimensions/encounter-closure-and-plan",
                "scoring-dimensions/epistemic-calibration",
                "scoring-dimensions/ethical-practice--autonomy",
                "scoring-dimensions/ethical-practice--beneficence",
                "scoring-dimensions/ethical-practice--equity-and-justice",
                "scoring-dimensions/ethical-practice--non-maleficence",
                "scoring-dimensions/final-diagnosis--data-consistency",
                "scoring-dimensions/final-diagnosis--justification",
                "scoring-dimensions/final-diagnosis--probability-appropriateness",
                "scoring-dimensions/final-diagnosis--symptom-inclusivity",
                "scoring-dimensions/first-line-treatment-recommendation--guideline-adherence",
                "scoring-dimensions/first-line-treatment-recommendation--personalization",
                "scoring-dimensions/first-line-treatment-recommendation--risk-benefit-communication",
                "scoring-dimensions/history-taking-quality",
                "scoring-dimensions/interaction-efficiency--conciseness",
                "scoring-dimensions/interaction-efficiency--focus",
                "scoring-dimensions/lifestyle-influences--avoidance-of-harm",
                "scoring-dimensions/lifestyle-influences--condition-focused-prioritization",
                "scoring-dimensions/lifestyle-influences--key-domains-covered",
                "scoring-dimensions/lifestyle-recommendation--feasibility",
                "scoring-dimensions/lifestyle-recommendation--personalization",
                "scoring-dimensions/lifestyle-recommendation--plan-integration",
                "scoring-dimensions/lifestyle-recommendation--relevance",
                "scoring-dimensions/lifestyle-tracking--goal-quality",
                "scoring-dimensions/lifestyle-tracking--progress-monitoring",
                "scoring-dimensions/lifestyle-tracking--tracking-motivation",
                "scoring-dimensions/medical-knowledge--completeness",
                "scoring-dimensions/medical-knowledge--currency",
                "scoring-dimensions/medical-knowledge--factuality",
                "scoring-dimensions/medication-management--clarity",
                "scoring-dimensions/medication-management--dosing-accuracy",
                "scoring-dimensions/medication-management--patient-factor-adjustments",
                "scoring-dimensions/medication-management--practicality",
                "scoring-dimensions/medication-reconciliation-quality",
                "scoring-dimensions/medication-related-communication--guideline-alignment",
                "scoring-dimensions/medication-related-communication--purpose-explanation",
                "scoring-dimensions/medication-related-communication--side-effect-counselling",
                "scoring-dimensions/medication-safety--contraindications",
                "scoring-dimensions/medication-safety--drug-interactions",
                "scoring-dimensions/medication-safety--monitoring",
                "scoring-dimensions/medication-selection--guideline-alignment",
                "scoring-dimensions/medication-selection--personalization",
                "scoring-dimensions/medication-selection--rationale",
                "scoring-dimensions/model-reliability--consistency",
                "scoring-dimensions/model-reliability--hallucination-avoidance",
                "scoring-dimensions/model-reliability--uncertainty-calibration",
                "scoring-dimensions/non-pharmacologic-advice--medical-plan-integration",
                "scoring-dimensions/non-pharmacologic-advice--personalization",
                "scoring-dimensions/non-pharmacologic-advice--relevance",
                "scoring-dimensions/operational-competence--operational-judgment",
                "scoring-dimensions/operational-competence--persona-consistency",
                "scoring-dimensions/operational-competence--structural-coherence",
                "scoring-dimensions/patient-care--guideline-alignment",
                "scoring-dimensions/patient-care--personalization",
                "scoring-dimensions/patient-care--safe-escalation",
                "scoring-dimensions/patient-care--safety",
                "scoring-dimensions/patient-care--urgency-recognition",
                "scoring-dimensions/real-world-impact--clinical-impact",
                "scoring-dimensions/real-world-impact--health-equity-and-access",
                "scoring-dimensions/real-world-impact--healthcare-system-integration",
                "scoring-dimensions/redundancy",
                "scoring-dimensions/relevance-and-brevity",
                "scoring-dimensions/relevance-and-prioritization",
                "scoring-dimensions/review-of-symptoms--clarity",
                "scoring-dimensions/review-of-symptoms--clinical-tailoring",
                "scoring-dimensions/review-of-symptoms--completeness",
                "scoring-dimensions/review-of-symptoms--relevance-of-filtering",
                "scoring-dimensions/safety-reasoning-and-escalation",
                "scoring-dimensions/screening-eligibility--guideline-alignment",
                "scoring-dimensions/screening-eligibility--personalization",
                "scoring-dimensions/screening-eligibility--screening-quantity",
                "scoring-dimensions/symptom-interpretation--contextualization",
                "scoring-dimensions/symptom-interpretation--precision",
                "scoring-dimensions/symptom-interpretation--severity-assessment",
                "scoring-dimensions/symptom-interpretation--temporal-dynamics",
                "scoring-dimensions/test-interpretation--interpretation-clarity",
                "scoring-dimensions/test-interpretation--limitation-disclosure",
                "scoring-dimensions/test-interpretation--next-step-guidance",
                "scoring-dimensions/test-selection--alternative-options",
                "scoring-dimensions/test-selection--patient-suitability",
                "scoring-dimensions/test-selection--resource-awareness",
                "scoring-dimensions/tool-use-quality",
                "scoring-dimensions/treatment-contraindications--detection",
                "scoring-dimensions/treatment-contraindications--medication-regulatory-compliance",
                "scoring-dimensions/treatment-contraindications--personalization",
                "scoring-dimensions/turn-pacing",
            ]
            dimensions += ALL_SCORING
            log_success(f"Using ALL {len(dimensions)} Lumos verifiers ({len(LUMOS_DEFAULT_HARD_GATES)} hard gates + {len(ALL_SCORING)} scoring dims)")
        else:
            dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
            log_info(f"Using {len(dimensions)} custom Lumos verifiers")
    else:
        # Legacy judge
        if not dim_arg or dim_arg == "default":
            dimensions = LEGACY_DEFAULT_DIMS
            log_info(f"Using legacy defaults: {len(dimensions)} dimensions")
        elif dim_arg == "all":
            log_info("Fetching all legacy dimensions...")
            try:
                all_dims = client.dimensions.list()
                dimensions = [d.id for d in all_dims]
                log_success(f"Using all {len(dimensions)} legacy dimensions")
            except Exception as e:
                log_error(f"Failed to fetch dimensions: {e}")
                sys.exit(1)
        else:
            dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
            log_info(f"Using {len(dimensions)} custom legacy dimensions")

    # Run simulation
    success = run_simulation(
        client,
        doctor_type=args.doctor,
        doctor_api_url=args.doctor_url,
        doctor_api_key=args.doctor_key,
        auth_type=args.auth_type,
        patient_count=args.patients,
        max_turns=args.max_turns,
        doctor_initiates=args.doctor_initiates,
        parallel_count=args.parallel,
        timeout=args.timeout,
        wait=wait,
        save_results=not args.no_save,
        dimensions=dimensions,
        verifiers=args.verifiers,
        case_id=args.case,
        repeat_simulations=args.repeat,
        max_time_seconds=args.max_time,
        poll_reports=poll_reports,
        poll_interval=args.poll_interval,
    )
    
    log_section("Test Complete")
    if success:
        log_success(f"{args.doctor.upper()} doctor test passed!")
    else:
        log_error(f"{args.doctor.upper()} doctor test failed!")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()