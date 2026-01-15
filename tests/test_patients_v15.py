#!/usr/bin/env python3
"""
Earl SDK - V1.5 Generative Patients Integration Test

Tests the SDK with v1.5 Generative Patients API which provides:
- Rich emotional and cognitive state
- Session-based conversation management  
- Termination signals when patient wants to end conversation
- Internal thoughts and behavioral patterns
- Consistent session_id across all turns

Usage:
    # Set credentials first
    export EARL_CLIENT_ID="your-client-id"
    export EARL_CLIENT_SECRET="your-client-secret"
    
    # Test with all 9 v1.5 patients (9 parallel threads):
    python3 test_patients_v15.py --env test \\
        --doctor-url "https://your-doctor-api.com/chat" \\
        --doctor-key "your-api-key"
    
    # Quick test with 2 patients:
    python3 test_patients_v15.py --env test --patients 2 \\
        --doctor-url "https://your-doctor-api.com/chat" \\
        --doctor-key "your-api-key"

Features tested:
- V1.5 patient API with session management
- 9 parallel episodes (one per patient)
- Session ID consistency across turns
- Engagement level tracking
- Patient-initiated termination
- 5 evaluation dimensions

Required:
- EARL_CLIENT_ID and EARL_CLIENT_SECRET (env vars or CLI)
- --doctor-url and --doctor-key (CLI arguments)
"""

import os
import sys
import argparse
import time
import json

# Add the SDK to path for development testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earl_sdk import EarlClient
from earl_sdk.models import DoctorApiConfig


# =============================================================================
# V1.5 PATIENTS (Generative Patients API)
# =============================================================================
# These are the 9 available v1.5 patients with rich emotional/cognitive state
V15_PATIENTS = {
    # Anxiety Patients - Olen Hills (66 years old, male)
    "anxiety-focused-clinical-encounter": {
        "name": "Olen Hills", 
        "age": 66, 
        "condition": "anxiety",
        "task": "Focused Clinical Encounter (Remote)",
        "behaviors": "looping, validation-seeking, cautious"
    },
    "anxiety-medication-reconciliation": {
        "name": "Olen Hills", 
        "age": 66, 
        "condition": "anxiety",
        "task": "Medication History / Med Reconciliation",
        "behaviors": "looping, validation-seeking, cautious"
    },
    "anxiety-pre-visit-intake-history": {
        "name": "Olen Hills", 
        "age": 66, 
        "condition": "anxiety",
        "task": "Pre-visit Intake History",
        "behaviors": "looping, validation-seeking, cautious"
    },
    # Anxiety Impaired Patients - Shemeka Gutmann (25 years old, female)
    "anxiety-impaired-focused-clinical-encounter": {
        "name": "Shemeka Gutmann", 
        "age": 25, 
        "condition": "anxiety (impaired)",
        "task": "Focused Clinical Encounter (Remote)",
        "behaviors": "fragmented, relief-seeking, dependent"
    },
    "anxiety-impaired-pre-visit-intake-history": {
        "name": "Shemeka Gutmann", 
        "age": 25, 
        "condition": "anxiety (impaired)",
        "task": "Pre-visit Intake History",
        "behaviors": "fragmented, relief-seeking, dependent"
    },
    "anxiety-impaired-schedule-appointment": {
        "name": "Shemeka Gutmann", 
        "age": 25, 
        "condition": "anxiety (impaired)",
        "task": "Schedule Appointment (Chat)",
        "behaviors": "fragmented, relief-seeking, dependent"
    },
    # Asthma Patients - Darleen Zulauf (54 years old, female)
    "asthma-chronic-symptom-monitoring": {
        "name": "Darleen Zulauf", 
        "age": 54, 
        "condition": "asthma",
        "task": "Chronic Symptom Monitoring Check-in",
        "behaviors": "negotiator, avoidance, theme-driven"
    },
    "asthma-focused-clinical-encounter": {
        "name": "Darleen Zulauf", 
        "age": 54, 
        "condition": "asthma",
        "task": "Focused Clinical Encounter (Remote)",
        "behaviors": "negotiator, avoidance, theme-driven"
    },
    "asthma-medication-adherence": {
        "name": "Darleen Zulauf", 
        "age": 54, 
        "condition": "asthma",
        "task": "Medication Adherence Support",
        "behaviors": "negotiator, avoidance, theme-driven"
    },
}

V15_PATIENT_IDS = list(V15_PATIENTS.keys())

# Dimensions to use for evaluation
EVALUATION_DIMENSIONS = [
    "turn_pacing",
    "context_recall", 
    "state_sensitivity",
    "patient_education",
    "empathetic_communication",
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


def test_list_v15_patients(client: EarlClient) -> bool:
    """Test listing v1.5 patients."""
    log_subsection("Testing V1.5 Patient Listing")
    
    try:
        patients = client.patients.list_v15()
        log_success(f"Listed {len(patients)} v1.5 patients")
        
        for p in patients:
            print(f"   • {p.id}: {p.name} ({p.age}yo, {p.condition})")
            if p.scenario:
                print(f"     Scenario: {p.scenario[:80]}...")
            if p.behaviors:
                print(f"     Behaviors: {p.behaviors}")
        
        return len(patients) > 0
    except Exception as e:
        log_error(f"Failed to list v1.5 patients: {e}")
        return False


def test_v15_external_doctor_workflow(
    client: EarlClient,
    doctor_api_url: str,
    doctor_api_key: str,
    auth_type: str = "bearer",
    wait_for_completion: bool = True,
    patient_count: int = 9,
    judge_timeout: int = 1200,  # 20 minutes default for 9 patients
    skip_validation: bool = False,
    parallel_count: int = None,  # None = match patient count
    doctor_initiates: bool = False,
    max_turns: int = 50,
    save_results: bool = True,
) -> bool:
    """Test workflow with v1.5 patients and external doctor API."""
    log_section("V1.5 Generative Patients Workflow Test")
    
    # Select patients first to determine parallel count
    patient_ids = V15_PATIENT_IDS[:patient_count]
    num_patients = len(patient_ids)
    
    # Default parallel_count to number of patients (max 10)
    if parallel_count is None:
        parallel_count = min(num_patients, 10)
    
    log_info(f"Doctor API URL: {doctor_api_url}")
    log_info(f"Doctor API Key: {'***' + doctor_api_key[-8:] if doctor_api_key else 'None'}")
    log_info(f"Auth Type: {auth_type}")
    log_info(f"Conversation initiator: {'doctor' if doctor_initiates else 'patient'}")
    log_info(f"Max turns: {max_turns}")
    log_info(f"Patient version: v1.5 (Generative Patients)")
    log_info(f"Parallel episodes: {parallel_count} (running {num_patients} patients concurrently)")
    log_info(f"Timeout: {judge_timeout}s ({judge_timeout // 60} minutes)")
    if skip_validation:
        log_warning("Skipping doctor API validation (cold-start mode)")
    
    try:
        log_info(f"Using {num_patients} v1.5 patients:")
        for pid in patient_ids:
            info = V15_PATIENTS[pid]
            print(f"   • {pid}")
            print(f"     {info['name']} ({info['age']}yo) - {info['condition']}")
            print(f"     Task: {info['task']}")
            print(f"     Behaviors: {info['behaviors']}")

        # Create pipeline with v1.5 patients
        pipeline_name = f"sdk-test-v15-{int(time.time())}"
        log_info(f"Creating pipeline: {pipeline_name}")
        
        initiator = "doctor" if doctor_initiates else "patient"
        
        client.pipelines.create(
            name=pipeline_name,
            doctor_config=DoctorApiConfig.external(
                api_url=doctor_api_url,
                api_key=doctor_api_key,
                auth_type=auth_type,
            ),
            patient_ids=patient_ids,
            patient_version="v1.5",  # Use v1.5 Generative Patients API
            dimension_ids=EVALUATION_DIMENSIONS,
            validate_doctor=not skip_validation,
            conversation_initiator=initiator,
            max_turns=max_turns,
        )
        log_success(f"Pipeline created with v1.5 patients")
        log_info(f"   Patient version: v1.5")
        log_info(f"   Dimensions: {', '.join(EVALUATION_DIMENSIONS)}")

        # Start simulation
        log_info("Starting simulation...")
        num_episodes = len(patient_ids)
        parallel = min(num_episodes, parallel_count)
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=num_episodes,
            parallel_count=parallel,
        )
        log_success(f"Simulation started: {simulation.id}")
        print(f"   Status: {simulation.status.value}")
        print(f"   Episodes: {num_episodes} ({parallel} parallel)")

        # Wait for completion
        results = None
        if wait_for_completion:
            log_info(f"Waiting for completion (max {judge_timeout}s)...")
            start_time = time.time()
            
            while time.time() - start_time < judge_timeout:
                sim = client.simulations.get(simulation.id)
                if sim.status.value in ["completed", "failed"]:
                    break
                elapsed = int(time.time() - start_time)
                print(f"\r   Status: {sim.status.value}, Episodes: {sim.completed_episodes}/{sim.total_episodes} ({elapsed}s)", end="", flush=True)
                time.sleep(15)
            
            print()
            final_sim = client.simulations.get(simulation.id)
            
            if final_sim.status.value == "completed":
                log_success(f"Simulation completed!")
            elif final_sim.status.value == "failed":
                log_error(f"Simulation failed")
            else:
                log_warning(f"Simulation still running: {final_sim.status.value}")
            
            # Show results
            if final_sim.summary:
                avg_score = final_sim.summary.get("average_score")
                completed = final_sim.summary.get("completed", 0)
                failed = final_sim.summary.get("failed", 0)
                print(f"\n   Summary:")
                print(f"   • Completed: {completed}/{num_episodes}")
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
                        
                        # Check for v1.5 insights in dialogue
                        v15_insights_count = 0
                        terminated_by_patient = False
                        session_ids = set()
                        engagement_levels = []
                        
                        for turn in dialogue:
                            metadata = turn.get("metadata", {}) or {}
                            if "v2_insights" in metadata:
                                v15_insights_count += 1
                                insights = metadata.get("v2_insights", {})
                                # Track session_id consistency
                                if "session_id" in insights:
                                    session_ids.add(insights["session_id"])
                                # Track engagement levels
                                if "engagement_level" in insights:
                                    engagement_levels.append(insights["engagement_level"])
                                elif "internal_state" in insights:
                                    internal = insights.get("internal_state", {})
                                    if "engagement_level" in internal:
                                        engagement_levels.append(internal["engagement_level"])
                            if metadata.get("terminated"):
                                terminated_by_patient = True
                        
                        if status == "failed" and error:
                            log_error(f"Episode {ep.get('episode_number')}: {patient}")
                            print(f"      Status: FAILED - {error[:80]}...")
                        else:
                            if score is not None:
                                log_success(f"Episode {ep.get('episode_number')}: {patient}")
                                print(f"      Score: {score:.2f}/4, Turns: {turns}")
                            else:
                                log_info(f"Episode {ep.get('episode_number')}: {patient}")
                                print(f"      Status: {status}, Turns: {turns}")
                            
                            # V1.5 specific insights
                            if v15_insights_count > 0:
                                print(f"      ✓ V1.5 Insights: {v15_insights_count} turns with internal state")
                                # Check session_id consistency (should be 1 unique session per episode)
                                if len(session_ids) == 1:
                                    print(f"      ✓ Session ID: consistent across all turns")
                                elif len(session_ids) > 1:
                                    print(f"      ⚠ Session IDs: {len(session_ids)} different (expected 1)")
                                # Show engagement level trend
                                if engagement_levels:
                                    avg_engagement = sum(engagement_levels) / len(engagement_levels)
                                    print(f"      ✓ Avg Engagement: {avg_engagement:.1f}/10")
                            if terminated_by_patient:
                                print(f"      🛑 Patient initiated termination")
                    
                    # Show dimension scores
                    if "dimension_scores" in report and report["dimension_scores"]:
                        log_subsection("Dimension Scores")
                        for dim, scores in report["dimension_scores"].items():
                            avg = scores.get("average", 0)
                            print(f"   • {dim}: {avg:.2f}/4")
                
            except Exception as e:
                log_warning(f"Could not get report: {e}")
        else:
            log_info("Simulation started (not waiting for completion)")

        # Save results to file
        if save_results and results:
            results_file = f"v15_test_results_{simulation.id[:8]}.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            log_info(f"Results saved to: {results_file}")

        # Cleanup
        try:
            client.pipelines.delete(pipeline_name)
            log_success("Test pipeline deleted")
        except Exception as e:
            log_info(f"Cleanup note: {e}")

        return True

    except Exception as e:
        log_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Earl SDK - V1.5 Generative Patients Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set credentials via environment
  export EARL_CLIENT_ID="your-client-id"
  export EARL_CLIENT_SECRET="your-client-secret"

  # Test all 9 v1.5 patients (9 parallel threads):
  python3 test_patients_v15.py --env test \\
      --doctor-url "https://your-doctor.modal.run/v1/chat/completions" \\
      --doctor-key "your-api-key"

  # Quick test with 2 patients:
  python3 test_patients_v15.py --env test --patients 2 \\
      --doctor-url "https://your-doctor.modal.run/v1/chat/completions" \\
      --doctor-key "your-api-key"

Required:
  EARL_CLIENT_ID      Your Auth0 client ID (env var or --client-id)
  EARL_CLIENT_SECRET  Your Auth0 client secret (env var or --client-secret)
  --doctor-url        External doctor API URL
  --doctor-key        External doctor API key
        """
    )
    
    # Environment
    parser.add_argument("--env", choices=["dev", "test", "prod"], default="test",
                        help="Environment to use (default: test)")
    parser.add_argument("--wait", action="store_true", default=True,
                        help="Wait for simulation completion (default: True)")
    parser.add_argument("--no-wait", action="store_true",
                        help="Don't wait for simulation completion")
    
    # Patient selection
    parser.add_argument("--patients", type=int, default=9,
                        help="Number of v1.5 patients to use (1-9, default: 9)")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list v1.5 patients, don't run simulation")
    
    # Simulation settings
    parser.add_argument("--judge-timeout", type=int, default=1200,
                        help="Max seconds to wait for completion (default: 1200 = 20 minutes)")
    parser.add_argument("--parallel", type=int, default=None,
                        help="Number of parallel episodes (default: match patient count, max 10)")
    parser.add_argument("--max-turns", type=int, default=50,
                        help="Maximum conversation turns (default: 50)")
    parser.add_argument("--doctor-initiates", action="store_true",
                        help="Doctor starts the conversation (default: patient)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")
    
    # Credentials
    parser.add_argument("--client-id", type=str, default=None,
                        help="Auth0 client ID (or set EARL_CLIENT_ID)")
    parser.add_argument("--client-secret", type=str, default=None,
                        help="Auth0 client secret (or set EARL_CLIENT_SECRET)")
    parser.add_argument("--organization", type=str, default=None,
                        help="Organization ID (or set EARL_ORGANIZATION)")
    
    # External doctor (required)
    parser.add_argument("--doctor-url", type=str, required=True,
                        help="External doctor API URL (required)")
    parser.add_argument("--doctor-key", type=str, required=True,
                        help="External doctor API key (required)")
    parser.add_argument("--auth-type", choices=["bearer", "api_key"], default="bearer",
                        help="API key auth method (default: bearer)")
    parser.add_argument("--skip-validation", action="store_true", default=True,
                        help="Skip doctor API validation (default: True for cold-start APIs)")
    parser.add_argument("--validate-doctor", action="store_true",
                        help="Validate doctor API before starting (overrides --skip-validation)")
    
    args = parser.parse_args()

    # Get credentials
    client_id, client_secret, organization = get_credentials(
        args.client_id, args.client_secret, args.organization
    )

    # Initialize client
    log_section("Initializing Earl SDK Client")
    client = EarlClient(
        client_id=client_id,
        client_secret=client_secret,
        organization=organization,
        environment=args.env,
    )
    log_success(f"Client created for {args.env} environment")
    print(f"   API URL: {client.api_url}")

    # Test listing v1.5 patients
    if not test_list_v15_patients(client):
        log_error("Failed to list v1.5 patients - check API connection")
        sys.exit(1)
    
    if args.list_only:
        log_success("List-only mode - done!")
        sys.exit(0)

    # Validate patient count
    patient_count = max(1, min(args.patients, 9))
    if patient_count != args.patients:
        log_warning(f"Adjusted patient count from {args.patients} to {patient_count}")

    # Determine if we should wait (--no-wait overrides default)
    should_wait = not args.no_wait
    
    # Determine if we should skip validation (--validate-doctor overrides default skip)
    should_skip_validation = args.skip_validation and not args.validate_doctor
    
    # Run the main test
    result = test_v15_external_doctor_workflow(
        client,
        doctor_api_url=args.doctor_url,
        doctor_api_key=args.doctor_key,
        auth_type=args.auth_type,
        wait_for_completion=should_wait,
        patient_count=patient_count,
        judge_timeout=args.judge_timeout,
        skip_validation=should_skip_validation,
        parallel_count=args.parallel,
        doctor_initiates=args.doctor_initiates,
        max_turns=args.max_turns,
        save_results=not args.no_save,
    )

    log_section("Test Complete")
    if result:
        log_success("V1.5 Generative Patients test passed!")
    else:
        log_error("V1.5 Generative Patients test failed!")

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
