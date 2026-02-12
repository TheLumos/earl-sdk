#!/usr/bin/env python3
"""
Earl SDK - Doctor Integration Tests

Tests the SDK with both internal (Earl's built-in) and external (customer-provided) doctors.

Usage:
    # Test internal doctor (Earl's built-in):
    python3 test_doctors.py --env test --doctor internal --patients 2 --wait \
        --client-id "your-client-id" \
        --client-secret "your-client-secret"
    
    # Test external doctor:
    python3 test_doctors.py --env test --doctor external --patients 3 --wait \
        --client-id "your-client-id" \
        --client-secret "your-client-secret" \
        --doctor-url "https://your-doctor.example.com/v1/chat/completions" \
        --doctor-key "your-api-key"
    
    # Quick test with 1 patient:
    python3 test_doctors.py --env test --doctor internal --patients 1 --wait \
        --client-id "your-client-id" --client-secret "your-client-secret"
    
    # List available patients only:
    python3 test_doctors.py --env test --list-only \
        --client-id "your-client-id" --client-secret "your-client-secret"

    # Fire 20 simulations (same config), print IDs, exit; simulations run async:
    python3 test_doctors.py --env test --doctor external --patients 1 --repeat 20 --max-time 60 \
        --doctor-url "..." --doctor-key "..." --client-id "..." --client-secret "..."
    # Extract IDs: grep EARL_SIMULATION_ID
    # IDs are also appended to earl_simulation_ids_<pipeline_name>.txt as they're created (safe if process exits early).
    # To recover IDs from API if you lost them: python3 list_recent_simulations.py --env test --limit 100 --ids-only ...

    # 20 conversations: start 20 sims, poll /simulations/{id}/report every 5 min, write terminal reports to JSONL:
    python3 test_doctors.py --env test --doctor external --patients 1 --repeat 20 --poll-reports --poll-interval 300 \\
        --timeout 86400 --doctor-url "..." --doctor-key "..." --client-id "..." --client-secret "..."
    # Output: earl_simulation_reports_<pipeline_name>.jsonl (one JSON object per line per simulation).
    # Polling stops for a sim when report has status completed/failed or any episode has error or conversation_ended_successfully=false.

Features tested:
- Patient listing from unified Patient API
- Pipeline creation with internal/external doctor
- Simulation execution with parallel episodes
- Session-based patient conversations
- Evaluation with customizable dimensions
- Patient-initiated termination handling

Credentials:
- Pass via CLI: --client-id and --client-secret
- Or set env vars: EARL_CLIENT_ID and EARL_CLIENT_SECRET
- For external doctor: also provide --doctor-url and optionally --doctor-key
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

# Default evaluation dimensions (used when --dimensions not specified)
DEFAULT_DIMENSIONS = [
    "turn_pacing",
    "context_recall", 
    "state_sensitivity",
    "patient_education",
    "empathetic_communication",
]

# Special dimension values
DIMENSION_ALL = "all"  # Use all available dimensions
DIMENSION_DEFAULT = "default"  # Use DEFAULT_DIMENSIONS


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
    """Get credentials from CLI args or environment."""
    client_id = cli_client_id or os.environ.get("EARL_CLIENT_ID", "")
    client_secret = cli_client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
    organization = cli_organization or os.environ.get("EARL_ORGANIZATION", "")
    
    if not client_id or not client_secret:
        print(f"\n{Colors.RED}{'='*60}")
        print("MISSING CREDENTIALS")
        print(f"{'='*60}{Colors.END}")
        print("\nPlease provide credentials via environment variables or CLI args:\n")
        print("  Environment Variables:")
        print("    export EARL_CLIENT_ID='your-client-id'")
        print("    export EARL_CLIENT_SECRET='your-client-secret'")
        print("    export EARL_ORGANIZATION='org_xxx'  # optional")
        print("\n  Or CLI Arguments:")
        print("    --client-id 'your-client-id'")
        print("    --client-secret 'your-client-secret'")
        print("    --organization 'org_xxx'  # optional")
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
    repeat_simulations: int = 1,
    max_time_seconds: Optional[float] = None,
    poll_reports: bool = False,
    poll_interval: int = 300,
) -> bool:
    """Run a simulation with specified doctor configuration."""
    
    log_section(f"{doctor_type.upper()} Doctor Test")
    
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
    use_dimensions = dimensions if dimensions else DEFAULT_DIMENSIONS
    initiator = "doctor" if doctor_initiates else "patient"
    
    log_info(f"Patients: {num_patients}, Parallel: {parallel}, Max turns: {max_turns}")
    log_info(f"Initiator: {initiator}, Timeout: {timeout}s")
    log_info(f"Dimensions: {', '.join(use_dimensions)}")
    
    print(f"\n   Selected patients:")
    for p in selected_patients:
        condition = p.condition or p.task or "N/A"
        print(f"   • {p.id}: {p.name or 'N/A'} - {condition}")
    
    try:
        waited_via_poll = False
        # Create pipeline
        pipeline_name = f"sdk-test-{doctor_type}-{int(time.time())}"
        log_info(f"Creating pipeline: {pipeline_name}")
        
        client.pipelines.create(
            name=pipeline_name,
            doctor_config=doctor_config,
            patient_ids=patient_ids,
            dimension_ids=use_dimensions,
            validate_doctor=False,  # Skip validation for cold-start APIs
            conversation_initiator=initiator,
            max_turns=max_turns,
        )
        log_success("Pipeline created")

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
            def _get_simulation_with_retry(sim_id: str, max_retries: int = 3):
                """Poll simulation status with retries on connection/timeout errors."""
                last_err = None
                for attempt in range(max_retries):
                    try:
                        return client.simulations.get(sim_id)
                    except EarlError as e:
                        last_err = e
                        err_str = str(e).lower()
                        if ("timed out" in err_str or "connect" in err_str) and attempt < max_retries - 1:
                            time.sleep(5)
                            continue
                        raise
                raise last_err  # unreachable if max_retries > 0

            log_info(f"Waiting for completion (max {timeout}s)...")
            start_time = time.time()

            last_progress_time = start_time
            last_completed = 0

            while time.time() - start_time < timeout:
                sim = _get_simulation_with_retry(simulation.id)
                if sim.status.value in ["completed", "failed"]:
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
                            print(f"      Score: {score:.2f}/4, Turns: {turns}")
                            
                            # Show per-dimension scores
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
    
    # Dimension options
    parser.add_argument("--dimensions", type=str, required=True,
                        help="Dimensions to evaluate (REQUIRED): 'all' for all ~130 dimensions, 'default' for 5 core dimensions, "
                             "or comma-separated list like 'turn_pacing,context_recall,empathy'")
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

    # Resolve dimensions
    if args.dimensions.lower() == DIMENSION_ALL:
        log_info("Fetching all available dimensions...")
        try:
            all_dims = client.dimensions.list()
            dimensions = [d.id for d in all_dims]
            log_success(f"Using all {len(dimensions)} dimensions")
        except Exception as e:
            log_error(f"Failed to fetch dimensions: {e}")
            sys.exit(1)
    elif args.dimensions.lower() == DIMENSION_DEFAULT:
        dimensions = DEFAULT_DIMENSIONS
        log_info(f"Using default {len(dimensions)} dimensions")
    else:
        # Comma-separated list
        dimensions = [d.strip() for d in args.dimensions.split(",") if d.strip()]
        log_info(f"Using {len(dimensions)} specified dimensions")

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
