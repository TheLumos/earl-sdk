#!/usr/bin/env python3
"""
Earl SDK - External Doctor Integration Test

Tests the SDK with a customer-provided external doctor API.

Usage:
    # With custom external doctor:
    python3 test_external_doctor.py --env test --wait \
        --doctor-url "https://your-doctor-api.com/chat" \
        --doctor-key "your-api-key"
"""

import os
import sys
import argparse
import time

# Add the SDK to path for development testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earl_sdk import EarlClient
from earl_sdk.models import DoctorApiConfig


# =============================================================================
# PATIENT SELECTION OPTIONS
# =============================================================================
SINGLE_PATIENT_ID = "Adrian_Cruickshank"

KNOWN_PATIENTS = {
    "Adrian_Cruickshank": {"name": "Adrian Cruickshank", "age": 26},
    "Delorse_Evelynn_Wiza": {"name": "Delorse Wiza", "age": 32},
    "Gianna_Corrine_Hahn": {"name": "Gianna Hahn", "age": 55},
}
THREE_KNOWN_PATIENT_IDS = list(KNOWN_PATIENTS.keys())


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def log_success(msg): print(f"{Colors.GREEN}✓ {msg}{Colors.END}")
def log_error(msg): print(f"{Colors.RED}✗ {msg}{Colors.END}")
def log_info(msg): print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")
def log_warning(msg): print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")
def log_section(title): print(f"\n{Colors.BOLD}{'='*60}\n{title}\n{'='*60}{Colors.END}")


def get_credentials(cli_client_id=None, cli_client_secret=None, cli_organization=None):
    """Get credentials from CLI args or environment."""
    client_id = cli_client_id or os.environ.get("EARL_CLIENT_ID", "")
    client_secret = cli_client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
    organization = cli_organization or os.environ.get("EARL_ORGANIZATION", "")
    
    if not client_id or not client_secret:
        log_error("Missing credentials. Set EARL_CLIENT_ID and EARL_CLIENT_SECRET or use --client-id/--client-secret")
        sys.exit(1)
    
    return client_id, client_secret, organization


def test_external_doctor_workflow(
    client: EarlClient,
    doctor_api_url: str,
    doctor_api_key: str,
    wait_for_completion: bool = True,
    use_default_pipeline: bool = False,
    patient_mode: str = "single",
    judge_timeout: int = 600,
) -> bool:
    """Test workflow with an external doctor API."""
    log_section("External Doctor Workflow Test")
    
    log_info(f"Doctor API URL: {doctor_api_url}")
    log_info(f"Doctor API Key: {doctor_api_key[:20]}..." if doctor_api_key else "No API key")
    
    try:
        # Select patients
        if patient_mode == "single":
            patient_ids = [SINGLE_PATIENT_ID]
            log_info(f"Using single patient: {SINGLE_PATIENT_ID}")
        elif patient_mode == "three":
            patient_ids = THREE_KNOWN_PATIENT_IDS
            log_info(f"Using three known patients: {patient_ids}")
        elif patient_mode == "random":
            patients = client.patients.list(limit=3)
            patient_ids = [p.id for p in patients[:3]]
            log_info(f"Using random patients: {patient_ids}")
        else:
            patient_ids = None
            log_info("Using default_pipeline patients")

        # Create or use pipeline
        if use_default_pipeline:
            pipeline_name = "default_pipeline"
            log_info(f"Using existing pipeline: {pipeline_name}")
        else:
            pipeline_name = f"sdk-test-external-{int(time.time())}"
            log_info(f"Creating pipeline: {pipeline_name}")
            
            client.pipelines.create(
                name=pipeline_name,
                doctor_config=DoctorApiConfig.external(
                    api_url=doctor_api_url,
                    api_key=doctor_api_key,
                ),
                patient_ids=patient_ids,
                dimension_ids=["turn_pacing", "context_recall", "state_sensitivity"],
            )
            log_success("Pipeline created with external doctor")

        # Start simulation
        log_info("Starting simulation...")
        num_episodes = len(patient_ids) if patient_ids else 1
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=num_episodes,
        )
        log_success(f"Simulation started: {simulation.id}")
        print(f"   Status: {simulation.status.value}")
        print(f"   Episodes: {num_episodes}")

        # Wait for completion
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
                log_error(f"Simulation failed: {getattr(final_sim, 'error', 'unknown')}")
            else:
                log_warning(f"Simulation still running: {final_sim.status.value}")
            
            # Show results
            if final_sim.summary:
                avg_score = final_sim.summary.get("average_score")
                if avg_score is not None:
                    print(f"   Average Score: {avg_score:.2f}/4")

            # Show episode details
            try:
                report = client.simulations.get_report(simulation.id)
                if "episodes" in report:
                    log_info("Episode Results:")
                    for ep in report["episodes"]:
                        score = ep.get("total_score")
                        status = ep.get("status", "?")
                        error = ep.get("error")
                        patient = ep.get("patient_name") or ep.get("patient_id", "?")
                        turns = len(ep.get("dialogue_history", []))
                        
                        if status == "failed" and error:
                            log_error(f"Episode {ep.get('episode_number')}: {patient} - FAILED: {error[:80]}...")
                        elif score is not None:
                            log_success(f"Episode {ep.get('episode_number')}: {patient} - {turns} msgs - Score: {score:.2f}/4")
                        else:
                            log_info(f"   Episode {ep.get('episode_number')}: {patient} - Status: {status}")
            except Exception as e:
                log_warning(f"Could not get report: {e}")
        else:
            log_info("Simulation started (not waiting for completion)")

        # Cleanup
        if not use_default_pipeline:
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
    parser = argparse.ArgumentParser(description="Earl SDK - External Doctor Test")
    parser.add_argument("--env", choices=["dev", "test", "prod"], default="test")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    parser.add_argument("--patients", choices=["single", "three", "random", "default"], default="single")
    parser.add_argument("--use-default-pipeline", action="store_true")
    parser.add_argument("--judge-timeout", type=int, default=600)
    parser.add_argument("--client-id", type=str, default=None)
    parser.add_argument("--client-secret", type=str, default=None)
    parser.add_argument("--organization", type=str, default=None)
    
    # External doctor specific
    parser.add_argument(
        "--doctor-url",
        type=str,
        required=True,
        help="External doctor API URL (e.g., https://your-api.com/v1/chat/completions)",
    )
    parser.add_argument(
        "--doctor-key",
        type=str,
        default=None,
        help="External doctor API key",
    )
    
    args = parser.parse_args()

    client_id, client_secret, organization = get_credentials(
        args.client_id, args.client_secret, args.organization
    )

    log_section("Initializing Earl SDK Client")
    client = EarlClient(
        client_id=client_id,
        client_secret=client_secret,
        organization=organization,
        environment=args.env,
    )
    log_success(f"Client created for {args.env} environment")
    print(f"   API URL: {client.api_url}")

    result = test_external_doctor_workflow(
        client,
        doctor_api_url=args.doctor_url,
        doctor_api_key=args.doctor_key,
        wait_for_completion=args.wait,
        use_default_pipeline=args.use_default_pipeline,
        patient_mode=args.patients,
        judge_timeout=args.judge_timeout,
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()

