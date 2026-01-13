#!/usr/bin/env python3
"""
Earl SDK - V2 Patients Integration Test

Tests the SDK with the new V2 patient scenarios (Anxiety & Asthma patients).

Usage:
    python3 test_patients_v2.py --env test --wait
    python3 test_patients_v2.py --env test --wait --subset anxiety
    python3 test_patients_v2.py --env test --wait --subset asthma
"""

import os
import sys
import argparse
import time
from typing import Optional

# Add the SDK to path for development testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earl_sdk import EarlClient
from earl_sdk.models import DoctorApiConfig


# =============================================================================
# V2 PATIENT DEFINITIONS
# =============================================================================
V2_PATIENTS = {
    # Anxiety Patients - Olen Hills (66 years old)
    "Anxiety_Focused_Clinical_Encounter": {
        "name": "Olen Hills",
        "age": 66,
        "condition": "Anxiety",
        "encounter_type": "telehealth",
        "difficulty": "medium",
    },
    "Anxiety_Medication_Reconciliation": {
        "name": "Olen Hills",
        "age": 66,
        "condition": "Anxiety",
        "encounter_type": "telehealth",
        "difficulty": "medium",
    },
    "Anxiety_Pre_Visit_Intake_History": {
        "name": "Olen Hills",
        "age": 66,
        "condition": "Anxiety",
        "encounter_type": "telehealth",
        "difficulty": "medium",
    },
    
    # Anxiety Patients - Shemeka Gutmann (25 years old, impaired)
    "Anxiety_Impaired_Focused_Clinical_Encounter": {
        "name": "Shemeka Gutmann",
        "age": 25,
        "condition": "Anxiety (Impaired)",
        "encounter_type": "telehealth",
        "difficulty": "medium",
    },
    "Anxiety_Impaired_Pre_Visit_Intake_History": {
        "name": "Shemeka Gutmann",
        "age": 25,
        "condition": "Wellness Check (Anxiety Impaired)",
        "encounter_type": "medical screening",
        "difficulty": "medium",
    },
    "Anxiety_Impaired_Schedule_Appointment": {
        "name": "Shemeka Gutmann",
        "age": 25,
        "condition": "Anxiety (Impaired)",
        "encounter_type": "telehealth",
        "difficulty": "medium",
    },
    
    # Asthma Patients - Darleen Zulauf (54 years old)
    "Asthma_Chronic_Symptom_Monitoring": {
        "name": "Darleen Zulauf",
        "age": 54,
        "condition": "Asthma",
        "encounter_type": "telehealth",
        "difficulty": "medium",
    },
    "Asthma_Focused_Clinical_Encounter": {
        "name": "Darleen Zulauf",
        "age": 54,
        "condition": "Asthma",
        "encounter_type": "telehealth",
        "difficulty": "medium",
    },
    "Asthma_Medication_Adherence": {
        "name": "Darleen Zulauf",
        "age": 54,
        "condition": "Asthma",
        "encounter_type": "telehealth",
        "difficulty": "medium",
    },
}

# Subsets
ANXIETY_PATIENTS = [
    "Anxiety_Focused_Clinical_Encounter",
    "Anxiety_Impaired_Focused_Clinical_Encounter",
    "Anxiety_Impaired_Pre_Visit_Intake_History",
    "Anxiety_Impaired_Schedule_Appointment",
    "Anxiety_Medication_Reconciliation",
    "Anxiety_Pre_Visit_Intake_History",
]

ASTHMA_PATIENTS = [
    "Asthma_Chronic_Symptom_Monitoring",
    "Asthma_Focused_Clinical_Encounter",
    "Asthma_Medication_Adherence",
]

ALL_V2_PATIENTS = list(V2_PATIENTS.keys())


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
def log_highlight(msg): print(f"{Colors.CYAN}{Colors.BOLD}{msg}{Colors.END}")
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


def test_v2_patients_workflow(
    client: EarlClient,
    patient_ids: list,
    wait_for_completion: bool = True,
    judge_timeout: int = 900,
    max_turns: int = 20,
) -> dict:
    """
    Test V2 patients with the internal doctor.
    
    Returns:
        dict with test results including scores
    """
    log_section(f"V2 Patients Test ({len(patient_ids)} patients)")
    
    pipeline_name = f"v2-patients-test-{int(time.time())}"
    pipeline_created = False
    
    results = {
        "success": False,
        "patients_tested": len(patient_ids),
        "episodes_completed": 0,
        "episodes_failed": 0,
        "average_score": None,
        "episode_results": [],
        "error": None,
    }
    
    try:
        # Show patient info
        log_info("Patients to test:")
        for pid in patient_ids:
            info = V2_PATIENTS.get(pid, {})
            name = info.get("name", "Unknown")
            condition = info.get("condition", "Unknown")
            print(f"   • {pid}")
            print(f"     {name} - {condition}")
        
        # Get dimensions
        log_info("Getting available dimensions...")
        dimensions = client.dimensions.list()
        dimension_ids = [d.id for d in dimensions]
        log_success(f"Found {len(dimension_ids)} dimensions: {dimension_ids}")
        
        # Create pipeline
        log_info(f"Creating pipeline: {pipeline_name}")
        log_info(f"   max_turns={max_turns}")
        
        pipeline = client.pipelines.create(
            name=pipeline_name,
            doctor_config=DoctorApiConfig.internal(),
            patient_ids=patient_ids,
            dimension_ids=dimension_ids,
            description=f"V2 Patients Test - {len(patient_ids)} patients",
            conversation_initiator="doctor",
            max_turns=max_turns,
        )
        pipeline_created = True
        log_success(f"Pipeline created: {pipeline.name}")
        
        # Start simulation (1 episode per patient)
        num_episodes = len(patient_ids)
        log_info(f"Starting simulation with {num_episodes} episodes...")
        
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=num_episodes,
            parallel_count=min(3, num_episodes),  # Run up to 3 in parallel
        )
        log_success(f"Simulation started: {simulation.id}")
        print(f"   Status: {simulation.status.value}")
        print(f"   Episodes: {num_episodes}")
        
        # Wait for completion
        if wait_for_completion:
            log_info(f"Waiting for completion (max {judge_timeout}s)...")
            start_time = time.time()
            last_status = ""
            
            while time.time() - start_time < judge_timeout:
                sim = client.simulations.get(simulation.id)
                
                # Get episode progress
                try:
                    episodes = client.simulations.get_episodes(simulation.id)
                    completed = sum(1 for e in episodes if e.get("status") in ["completed", "judged"])
                    in_progress = sum(1 for e in episodes if e.get("status") in ["running", "awaiting_doctor_response"])
                    failed = sum(1 for e in episodes if e.get("status") == "failed")
                except:
                    completed, in_progress, failed = 0, 0, 0
                
                status_msg = f"{sim.status.value} | Episodes: {completed}/{num_episodes} done, {in_progress} running, {failed} failed"
                
                if status_msg != last_status:
                    elapsed = int(time.time() - start_time)
                    print(f"\r   [{elapsed}s] {status_msg}      ", end="", flush=True)
                    last_status = status_msg
                
                if sim.status.value in ["completed", "failed"]:
                    break
                    
                time.sleep(5)
            
            print()
            
            # Get final results
            final_sim = client.simulations.get(simulation.id)
            
            if final_sim.status.value == "completed":
                log_success("Simulation completed!")
            elif final_sim.status.value == "failed":
                log_error(f"Simulation failed: {getattr(final_sim, 'error', 'unknown')}")
            else:
                log_warning(f"Simulation still running after timeout: {final_sim.status.value}")
            
            # Collect episode results
            log_section("Episode Results")
            
            try:
                episodes = client.simulations.get_episodes(simulation.id)
                
                for ep in episodes:
                    episode_id = ep.get("episode_id")
                    patient_id = ep.get("patient_id", "Unknown")
                    status = ep.get("status", "unknown")
                    
                    # Get full episode details
                    try:
                        full_ep = client.simulations.get_episode(simulation.id, episode_id)
                        dialogue = full_ep.get("dialogue_history", [])
                        turn_count = sum(1 for m in dialogue if m.get("role") == "doctor")
                        scores = full_ep.get("scores", {})
                        avg_score = full_ep.get("average_score")
                    except:
                        dialogue = []
                        turn_count = 0
                        scores = {}
                        avg_score = None
                    
                    info = V2_PATIENTS.get(patient_id, {})
                    name = info.get("name", "Unknown")
                    
                    # Track results
                    ep_result = {
                        "patient_id": patient_id,
                        "patient_name": name,
                        "status": status,
                        "turns": turn_count,
                        "scores": scores,
                        "average_score": avg_score,
                    }
                    results["episode_results"].append(ep_result)
                    
                    if status in ["completed", "judged"]:
                        results["episodes_completed"] += 1
                    elif status == "failed":
                        results["episodes_failed"] += 1
                    
                    # Display
                    status_icon = "✓" if status in ["completed", "judged"] else "✗" if status == "failed" else "⋯"
                    status_color = Colors.GREEN if status in ["completed", "judged"] else Colors.RED if status == "failed" else Colors.YELLOW
                    
                    print(f"\n{status_color}{status_icon} {patient_id}{Colors.END}")
                    print(f"   Patient: {name}")
                    print(f"   Status: {status}")
                    print(f"   Turns: {turn_count}")
                    
                    if scores:
                        print(f"   Scores:")
                        for dim, score in scores.items():
                            print(f"      {dim}: {score}/4")
                    
                    if avg_score is not None:
                        print(f"   Average: {avg_score:.2f}/4")
            
            except Exception as e:
                log_error(f"Could not get episode details: {e}")
            
            # Summary
            if final_sim.summary:
                avg = final_sim.summary.get("average_score")
                if avg is not None:
                    results["average_score"] = avg
            
            # Calculate average from episodes if not in summary
            if results["average_score"] is None:
                scores_list = [r["average_score"] for r in results["episode_results"] if r["average_score"] is not None]
                if scores_list:
                    results["average_score"] = sum(scores_list) / len(scores_list)
            
            log_section("Summary")
            log_info(f"Patients tested: {len(patient_ids)}")
            log_info(f"Episodes completed: {results['episodes_completed']}/{num_episodes}")
            log_info(f"Episodes failed: {results['episodes_failed']}")
            
            if results["average_score"] is not None:
                log_highlight(f"Overall Average Score: {results['average_score']:.2f}/4")
            
            results["success"] = results["episodes_completed"] > 0
            
        else:
            log_info("Simulation started (not waiting for completion)")
            log_info(f"Simulation ID: {simulation.id}")
            results["success"] = True
        
        return results
        
    except Exception as e:
        log_error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
        return results
        
    finally:
        # Cleanup
        if pipeline_created:
            try:
                client.pipelines.delete(pipeline_name)
                log_info(f"Cleaned up pipeline: {pipeline_name}")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Earl SDK - V2 Patients Test")
    parser.add_argument("--env", choices=["dev", "test", "prod"], default="test",
                        help="Environment to test against")
    parser.add_argument("--wait", action="store_true",
                        help="Wait for completion and show results")
    parser.add_argument("--subset", choices=["all", "anxiety", "asthma"], default="all",
                        help="Which patient subset to test")
    parser.add_argument("--max-turns", type=int, default=20,
                        help="Maximum conversation turns")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Timeout in seconds for judge completion")
    parser.add_argument("--client-id", help="Override EARL_CLIENT_ID")
    parser.add_argument("--client-secret", help="Override EARL_CLIENT_SECRET")
    parser.add_argument("--organization", help="Override EARL_ORGANIZATION")
    
    args = parser.parse_args()
    
    # Get credentials
    client_id, client_secret, organization = get_credentials(
        args.client_id, args.client_secret, args.organization
    )
    
    # Select patients based on subset
    if args.subset == "anxiety":
        patient_ids = ANXIETY_PATIENTS
    elif args.subset == "asthma":
        patient_ids = ASTHMA_PATIENTS
    else:
        patient_ids = ALL_V2_PATIENTS
    
    log_section("Earl SDK - V2 Patients Test")
    log_info(f"Environment: {args.env}")
    log_info(f"Patient subset: {args.subset} ({len(patient_ids)} patients)")
    log_info(f"Max turns: {args.max_turns}")
    log_info(f"Wait for completion: {args.wait}")
    
    # Create client
    try:
        client = EarlClient(
            client_id=client_id,
            client_secret=client_secret,
            organization=organization,
            environment=args.env,
        )
        log_success(f"Client created for {args.env} environment")
        print(f"   API URL: {client.api_url}")
    except Exception as e:
        log_error(f"Failed to create client: {e}")
        sys.exit(1)
    
    # Run test
    results = test_v2_patients_workflow(
        client,
        patient_ids=patient_ids,
        wait_for_completion=args.wait,
        judge_timeout=args.timeout,
        max_turns=args.max_turns,
    )
    
    # Final status
    log_section("FINAL RESULT")
    if results["success"]:
        log_success("TEST COMPLETED")
        if results["average_score"] is not None:
            log_highlight(f"Average Score: {results['average_score']:.2f}/4")
        sys.exit(0)
    else:
        log_error("TEST FAILED")
        if results["error"]:
            log_error(f"Error: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
