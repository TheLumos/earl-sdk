#!/usr/bin/env python3
"""
Earl SDK - V2 Patients Client-Driven Test

Tests the V2 patients with client-driven doctor (external doctor API).

Usage:
    python3 test_patients_v2_client_driven.py --env test
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
# V2 PATIENT IDs
# =============================================================================
V2_PATIENTS = [
    "Anxiety_Focused_Clinical_Encounter",
    "Anxiety_Impaired_Focused_Clinical_Encounter",
    "Anxiety_Impaired_Pre_Visit_Intake_History",
    "Anxiety_Impaired_Schedule_Appointment",
    "Anxiety_Medication_Reconciliation",
    "Anxiety_Pre_Visit_Intake_History",
    "Asthma_Chronic_Symptom_Monitoring",
    "Asthma_Focused_Clinical_Encounter",
    "Asthma_Medication_Adherence",
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
def log_highlight(msg): print(f"{Colors.CYAN}{Colors.BOLD}{msg}{Colors.END}")
def log_section(title): print(f"\n{Colors.BOLD}{'='*60}\n{title}\n{'='*60}{Colors.END}")


def mock_doctor_response(dialogue_history: list, turn_number: int) -> str:
    """Generate a mock doctor response that keeps the conversation going."""
    if turn_number == 0:
        return "Hello! I'm Dr. Smith. How can I help you today? Please tell me about your symptoms or concerns."
    
    questions = [
        "I understand. Can you tell me more about when these symptoms started?",
        "How severe would you rate this on a scale of 1-10?",
        "Have you noticed any patterns or triggers?",
        "Are you experiencing any other symptoms I should know about?",
        "Have you tried any treatments or medications for this?",
        "How is this affecting your daily activities?",
        "Do you have any family history of similar conditions?",
        "Have you seen any other healthcare providers for this issue?",
        "Are you currently taking any medications or supplements?",
        "How has your sleep been lately?",
        "Have you noticed any changes in your appetite or weight?",
        "Are you experiencing any stress or anxiety related to this?",
        "How long have you been dealing with this issue?",
        "Have you had any recent changes in your lifestyle?",
        "Are there any activities that make this better or worse?",
        "Thank you for sharing all of this. Based on what you've told me, I'd like to discuss some next steps. Goodbye!",
    ]
    
    idx = min(turn_number - 1, len(questions) - 1)
    return questions[idx]


def test_single_patient(
    client: EarlClient,
    patient_id: str,
    max_turns: int = 10,
    poll_interval: float = 2.0,
) -> dict:
    """
    Test a single V2 patient with client-driven doctor.
    
    Returns dict with success, turns, error
    """
    pipeline_name = f"v2-test-{patient_id[:20]}-{int(time.time())}"
    pipeline_created = False
    
    result = {
        "patient_id": patient_id,
        "success": False,
        "turns": 0,
        "error": None,
    }
    
    try:
        # Get dimensions
        dimensions = client.dimensions.list()
        dimension_ids = [d.id for d in dimensions[:3]]  # Just 3 for speed
        
        # Create pipeline
        pipeline = client.pipelines.create(
            name=pipeline_name,
            dimension_ids=dimension_ids,
            patient_ids=[patient_id],
            doctor_config=DoctorApiConfig.client_driven(),
            description=f"V2 patient test: {patient_id}",
            conversation_initiator="doctor",
            max_turns=max_turns,
        )
        pipeline_created = True
        log_info(f"Pipeline created: {pipeline_name}")
        
        # Start simulation
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=1,
            parallel_count=1,
        )
        log_info(f"Simulation: {simulation.id}")
        
        time.sleep(2)  # Wait for episode to initialize
        
        # Run conversation loop
        episode_id = None
        turn_count = 0
        max_iterations = 100
        iteration = 0
        last_dialogue_len = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check simulation status
            sim = client.simulations.get(simulation.id)
            if sim.status.value in ["completed", "failed"]:
                log_info(f"Simulation {sim.status.value}")
                break
            
            # Get episodes
            try:
                episodes = client.simulations.get_episodes(simulation.id)
            except Exception as e:
                log_warning(f"Could not get episodes: {e}")
                time.sleep(poll_interval)
                continue
            
            if not episodes:
                time.sleep(poll_interval)
                continue
            
            # Get the episode
            ep = episodes[0]
            episode_id = ep.get("episode_id")
            status = ep.get("status", "unknown")
            
            if status == "failed":
                error_msg = ep.get("error", "Unknown error")
                log_error(f"Episode failed: {error_msg}")
                result["error"] = error_msg
                break
            
            if status in ["completed", "judged"]:
                log_info(f"Episode {status}")
                break
            
            # Get full episode with dialogue
            try:
                full_ep = client.simulations.get_episode(simulation.id, episode_id)
                dialogue = full_ep.get("dialogue_history", [])
            except Exception as e:
                log_warning(f"Could not get episode: {e}")
                time.sleep(poll_interval)
                continue
            
            current_len = len(dialogue)
            
            # Count turns
            turn_count = sum(1 for msg in dialogue if msg.get("role") == "doctor")
            
            # Check if we need to send a doctor response
            needs_response = False
            
            if current_len == 0:
                needs_response = True
            elif current_len > last_dialogue_len:
                last_msg = dialogue[-1] if dialogue else None
                if last_msg and last_msg.get("role") == "patient":
                    needs_response = True
            
            if needs_response and status == "awaiting_doctor_response":
                doctor_msg = mock_doctor_response(dialogue, turn_count)
                
                try:
                    client.simulations.submit_response(
                        simulation_id=simulation.id,
                        episode_id=episode_id,
                        response=doctor_msg,
                    )
                    turn_count += 1
                    last_dialogue_len = current_len + 1
                    print(f"   Turn {turn_count}: Doctor responded")
                    
                except Exception as e:
                    if "Goodbye" in doctor_msg or "goodbye" in doctor_msg:
                        log_info(f"Conversation ended at turn {turn_count}")
                        break
                    log_warning(f"Submit failed: {e}")
            
            time.sleep(poll_interval)
        
        result["turns"] = turn_count
        result["success"] = turn_count >= 3  # At least 3 turns = success
        
        return result
        
    except Exception as e:
        log_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
        return result
        
    finally:
        if pipeline_created:
            try:
                client.pipelines.delete(pipeline_name)
                log_info(f"Cleaned up: {pipeline_name}")
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description="Earl SDK - V2 Patients Client-Driven Test")
    parser.add_argument("--env", choices=["dev", "test", "prod"], default="test")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--patient", type=str, default=None, help="Test single patient ID")
    parser.add_argument("--client-id", help="Override EARL_CLIENT_ID")
    parser.add_argument("--client-secret", help="Override EARL_CLIENT_SECRET")
    parser.add_argument("--organization", help="Override EARL_ORGANIZATION")
    
    args = parser.parse_args()
    
    # Get credentials
    client_id = args.client_id or os.environ.get("EARL_CLIENT_ID", "")
    client_secret = args.client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
    organization = args.organization or os.environ.get("EARL_ORGANIZATION", "")
    
    if not client_id or not client_secret:
        log_error("Missing credentials")
        sys.exit(1)
    
    # Select patients
    if args.patient:
        patient_ids = [args.patient]
    else:
        patient_ids = V2_PATIENTS
    
    log_section("Earl SDK - V2 Patients Client-Driven Test")
    log_info(f"Environment: {args.env}")
    log_info(f"Patients to test: {len(patient_ids)}")
    log_info(f"Max turns per patient: {args.max_turns}")
    
    # Create client
    try:
        client = EarlClient(
            client_id=client_id,
            client_secret=client_secret,
            organization=organization,
            environment=args.env,
        )
        log_success(f"Client created for {args.env} environment")
    except Exception as e:
        log_error(f"Failed to create client: {e}")
        sys.exit(1)
    
    # Test each patient
    results = []
    for i, patient_id in enumerate(patient_ids, 1):
        log_section(f"Patient {i}/{len(patient_ids)}: {patient_id}")
        result = test_single_patient(
            client, 
            patient_id, 
            max_turns=args.max_turns
        )
        results.append(result)
        
        if result["success"]:
            log_success(f"Patient {patient_id}: {result['turns']} turns")
        else:
            log_error(f"Patient {patient_id}: FAILED - {result.get('error', 'Unknown')}")
    
    # Summary
    log_section("Summary")
    passed = sum(1 for r in results if r["success"])
    failed = sum(1 for r in results if not r["success"])
    
    log_info(f"Total: {len(results)} patients")
    log_success(f"Passed: {passed}")
    if failed > 0:
        log_error(f"Failed: {failed}")
        for r in results:
            if not r["success"]:
                print(f"   - {r['patient_id']}: {r.get('error', 'Unknown error')}")
    
    sys.exit(0 if passed > 0 else 1)


if __name__ == "__main__":
    main()
