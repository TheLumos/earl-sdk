#!/usr/bin/env python3
"""
Earl SDK - Max Turns Verification Test

This test verifies that the max_turns feature works correctly by:
1. Installing earl-sdk==0.3.1 from PyPI
2. Creating a pipeline with max_turns=30
3. Running a simulation and verifying conversations can exceed 20 turns

Usage:
    python3 test_max_turns.py --env test
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
# Configuration
# =============================================================================
PIPELINE_MAX_TURNS = 30  # Pipeline max turns setting
TEST_MAX_TURNS = 35      # Test loop max (slightly higher than pipeline)
MIN_SUCCESS_TURNS = 21   # Minimum turns to consider test successful (>20)


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
    
    # Keep asking follow-up questions to extend the conversation
    questions = [
        "I see. Can you tell me more about when these symptoms started?",
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
        "Have you had any similar episodes in the past?",
        "Are you experiencing any pain associated with this?",
        "How is your energy level throughout the day?",
        "Have you noticed any changes in your mood?",
        "Is there anything else you'd like to share about your symptoms?",
        "Let me ask a few more clarifying questions. How about your hydration?",
        "Are you getting enough rest and exercise?",
        "Have you traveled recently?",
        "Any recent illnesses or infections?",
        "How about your diet - any changes recently?",
        "Do you have any allergies I should know about?",
        "Are you up to date on your vaccinations?",
        "How is your overall mental health?",
        "Any recent life changes or stressors?",
        "Thank you for all this information. I have a clearer picture now. Goodbye!",
    ]
    
    idx = min(turn_number - 1, len(questions) - 1)
    return questions[idx]


def test_max_turns_workflow(
    client: EarlClient,
    poll_interval: float = 3.0,
) -> dict:
    """
    Test that conversations can exceed 20 turns with max_turns=30.
    
    Returns:
        dict with test results including max_turns_reached
    """
    log_section(f"Max Turns Verification Test (target: >{MIN_SUCCESS_TURNS-1} turns)")
    
    pipeline_name = f"sdk-test-max-turns-{int(time.time())}"
    pipeline_created = False
    results = {
        "success": False,
        "max_turns_reached": 0,
        "pipeline_max_turns": PIPELINE_MAX_TURNS,
        "target_turns": MIN_SUCCESS_TURNS,
        "error": None,
    }
    
    try:
        # Step 1: Get patient and dimensions
        log_info("Step 1: Getting patient and dimensions...")
        
        patient_id = "Adrian_Cruickshank"  # Known good patient
        dimensions = client.dimensions.list()
        dimension_ids = [d.id for d in dimensions[:2]]  # Just 2 dimensions for speed
        
        log_success(f"Using patient: {patient_id}")
        log_success(f"Using {len(dimension_ids)} dimensions")
        
        # Step 2: Create pipeline with max_turns=30
        log_info(f"Step 2: Creating pipeline with max_turns={PIPELINE_MAX_TURNS}...")
        
        pipeline = client.pipelines.create(
            name=pipeline_name,
            dimension_ids=dimension_ids,
            patient_ids=[patient_id],
            doctor_config=DoctorApiConfig.client_driven(),
            description=f"SDK test - max_turns verification (max_turns={PIPELINE_MAX_TURNS})",
            conversation_initiator="doctor",
            max_turns=PIPELINE_MAX_TURNS,  # THE KEY PARAMETER!
        )
        pipeline_created = True
        log_success(f"Created pipeline: {pipeline.name}")
        log_highlight(f"   Pipeline max_turns: {PIPELINE_MAX_TURNS}")
        
        # Verify the pipeline has max_turns set
        fetched = client.pipelines.get(pipeline_name)
        if hasattr(fetched, 'max_turns'):
            log_success(f"   Verified pipeline.max_turns: {fetched.max_turns}")
        elif hasattr(fetched, 'conversation') and fetched.conversation:
            log_success(f"   Verified conversation.max_turns: {fetched.conversation.max_turns}")
        
        # Step 3: Start simulation
        log_info("Step 3: Starting simulation...")
        
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=1,
            parallel_count=1,
        )
        log_success(f"Simulation started: {simulation.id}")
        
        time.sleep(3)  # Wait for episode to initialize
        
        # Step 4: Run conversation loop
        log_info(f"Step 4: Running conversation (aiming for >{MIN_SUCCESS_TURNS-1} turns)...")
        
        episode_id = None
        turn_count = 0
        max_iterations = 150  # Safety limit
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
            
            if status in ["completed", "judged", "failed"]:
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
            
            # Count turns (each doctor message = 1 turn)
            turn_count = sum(1 for msg in dialogue if msg.get("role") == "doctor")
            
            # Check if we need to send a doctor response
            needs_response = False
            
            if current_len == 0:
                # No messages yet, doctor initiates
                needs_response = True
            elif current_len > last_dialogue_len:
                # New message received
                last_msg = dialogue[-1] if dialogue else None
                if last_msg and last_msg.get("role") == "patient":
                    needs_response = True
            
            if needs_response and status == "awaiting_doctor_response":
                # Generate and submit doctor response
                doctor_msg = mock_doctor_response(dialogue, turn_count)
                
                try:
                    client.simulations.submit_response(
                        simulation_id=simulation.id,
                        episode_id=episode_id,
                        response=doctor_msg,
                    )
                    turn_count += 1
                    last_dialogue_len = current_len + 1
                    
                    # Log progress
                    if turn_count <= 5 or turn_count % 5 == 0 or turn_count >= MIN_SUCCESS_TURNS:
                        marker = ""
                        if turn_count == MIN_SUCCESS_TURNS:
                            marker = f" {Colors.GREEN}← TARGET REACHED!{Colors.END}"
                        elif turn_count > 20:
                            marker = f" {Colors.CYAN}(exceeded 20!){Colors.END}"
                        print(f"   Turn {turn_count}: Doctor responded{marker}")
                    
                except Exception as e:
                    if "Goodbye" in doctor_msg or "goodbye" in doctor_msg:
                        log_info(f"Conversation ended naturally at turn {turn_count}")
                        break
                    log_warning(f"Submit failed: {e}")
            
            time.sleep(poll_interval)
        
        # Record results
        results["max_turns_reached"] = turn_count
        
        # Step 5: Evaluate success
        log_section("Test Results")
        
        log_info(f"Pipeline max_turns setting: {PIPELINE_MAX_TURNS}")
        log_info(f"Target minimum turns: {MIN_SUCCESS_TURNS}")
        log_highlight(f"Actual turns reached: {turn_count}")
        
        if turn_count >= MIN_SUCCESS_TURNS:
            log_success(f"SUCCESS! Conversation reached {turn_count} turns (> 20)")
            log_success("This proves the max_turns feature is working!")
            results["success"] = True
        else:
            log_warning(f"Conversation ended at {turn_count} turns (target: {MIN_SUCCESS_TURNS})")
            log_info("This may be due to natural conversation ending - try running again")
            results["success"] = False
        
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
    parser = argparse.ArgumentParser(description="Earl SDK - Max Turns Verification Test")
    parser.add_argument("--env", choices=["dev", "test", "prod"], default="test",
                        help="Environment to test against")
    parser.add_argument("--client-id", help="Override EARL_CLIENT_ID")
    parser.add_argument("--client-secret", help="Override EARL_CLIENT_SECRET")
    parser.add_argument("--organization", help="Override EARL_ORGANIZATION")
    parser.add_argument("--retries", type=int, default=3,
                        help="Number of retries if conversation ends early")
    
    args = parser.parse_args()
    
    # Get credentials
    client_id = args.client_id or os.environ.get("EARL_CLIENT_ID", "")
    client_secret = args.client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
    organization = args.organization or os.environ.get("EARL_ORGANIZATION", "")
    
    if not client_id or not client_secret:
        log_error("Missing credentials. Set EARL_CLIENT_ID and EARL_CLIENT_SECRET")
        sys.exit(1)
    
    log_section("Earl SDK - Max Turns Verification Test")
    log_info(f"Environment: {args.env}")
    log_info(f"SDK will create pipeline with max_turns={PIPELINE_MAX_TURNS}")
    log_info(f"Success criteria: conversation exceeds 20 turns")
    
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
    
    # Run test with retries
    for attempt in range(1, args.retries + 1):
        log_section(f"Attempt {attempt}/{args.retries}")
        
        results = test_max_turns_workflow(client)
        
        if results["success"]:
            log_section("FINAL RESULT")
            log_success(f"TEST PASSED on attempt {attempt}!")
            log_success(f"Conversation reached {results['max_turns_reached']} turns")
            log_success("The max_turns feature is working correctly!")
            sys.exit(0)
        else:
            if attempt < args.retries:
                log_warning(f"Attempt {attempt} did not reach target. Retrying...")
                time.sleep(5)
    
    # All retries exhausted
    log_section("FINAL RESULT")
    log_error(f"TEST FAILED after {args.retries} attempts")
    log_error(f"Could not reach >{MIN_SUCCESS_TURNS-1} turns")
    log_info("Check if the backend has the max_turns handling deployed")
    sys.exit(1)


if __name__ == "__main__":
    main()
