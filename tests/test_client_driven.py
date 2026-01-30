#!/usr/bin/env python3
"""
Earl SDK - Client-Driven Integration Test

Tests the SDK client-driven workflow for scenarios where the doctor API
is behind a VPN or firewall and cannot be reached directly by the orchestrator.

This test acts as MIDDLEWARE between:
- EARL Orchestrator (patient messages)
- Local/VPN Doctor API (doctor responses)

Usage:
    # With mock doctor (for testing):
    python3 test_client_driven.py --env test \
        --client-id "your-client-id" \
        --client-secret "your-client-secret"

    # With a local doctor API:
    python3 test_client_driven.py --env test \
        --client-id "your-client-id" \
        --client-secret "your-client-secret" \
        --local-doctor-url "http://localhost:8080/chat" \
        --local-doctor-key "your-key"

    # With an OpenAI-compatible external doctor:
    python3 test_client_driven.py --env test \
        --client-id "your-client-id" \
        --client-secret "your-client-secret" \
        --local-doctor-url "https://your-doctor-api.com/v1/chat/completions" \
        --local-doctor-key "your-api-key"

Credentials:
- Pass via CLI: --client-id and --client-secret
- Or set env vars: EARL_CLIENT_ID and EARL_CLIENT_SECRET
"""

import os
import sys
import argparse
import time
import requests
from typing import Optional

# Add the SDK to path for development testing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earl_sdk import EarlClient
from earl_sdk.models import DoctorApiConfig


# =============================================================================
# CONFIGURATION
# =============================================================================
# Patients are fetched dynamically from the API - no hardcoded IDs


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


def mock_doctor_response(dialogue_history: list, turn_number: int) -> str:
    """Generate a mock doctor response for testing."""
    if turn_number == 0:
        return "Hello! I'm Dr. Smith. How can I help you today? Please tell me about your symptoms or concerns."

    responses = [
        "I see. Can you tell me more about when these symptoms started and how severe they are?",
        "Thank you for that information. Have you noticed any patterns or triggers?",
        "I understand. Are you experiencing any other symptoms I should know about?",
        "Based on what you've described, I have a few recommendations. Have you tried any medications?",
        "Thank you for sharing all of this. I recommend scheduling a follow-up appointment to monitor your condition. Take care and goodbye!",
    ]
    return responses[min(turn_number - 1, len(responses) - 1)]


def call_local_doctor(dialogue_history: list, url: str, key: str, turn_number: int) -> Optional[str]:
    """
    Call a local/VPN-accessible doctor API (OpenAI-compatible format).
    """
    # Build messages in OpenAI format
    messages = [{
        "role": "system", 
        "content": "You are a compassionate medical AI assistant. Have a helpful conversation with the patient. After 4-5 exchanges, politely conclude the conversation."
    }]

    for msg in dialogue_history:
        role = "assistant" if msg.get("role") == "doctor" else "user"
        messages.append({"role": role, "content": msg.get("content", "")})

    # If no dialogue yet (doctor initiates), add a prompt
    if not dialogue_history:
        messages.append({"role": "user", "content": "Please greet me and ask how you can help."})

    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
        headers["Authorization"] = f"Bearer {key}"

    try:
        log_info(f"      Calling local doctor API: {url}")
        resp = requests.post(
            url, 
            json={
                "model": "default",  # Some APIs need a model field
                "messages": messages,
                "max_tokens": 256,
            }, 
            headers=headers, 
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle OpenAI format
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")
        else:
            content = data.get("response") or data.get("message") or data.get("content", "")

        if content:
            log_success(f"      Doctor API responded: {content[:60]}...")
            return content
        else:
            log_warning("      Doctor API returned empty response")
            return None

    except requests.exceptions.Timeout:
        log_warning(f"      Doctor API timeout after 120s")
        return None
    except requests.exceptions.RequestException as e:
        log_warning(f"      Doctor API error: {e}")
        return None
    except Exception as e:
        log_warning(f"      Doctor API unexpected error: {e}")
        return None


def get_doctor_response(dialogue_history: list, turn_number: int, 
                        local_doctor_url: Optional[str], local_doctor_key: Optional[str]) -> str:
    """Get doctor response from local API or mock."""
    if local_doctor_url:
        response = call_local_doctor(dialogue_history, local_doctor_url, local_doctor_key, turn_number)
        if response:
            return response
        log_warning("      Falling back to mock doctor")

    return mock_doctor_response(dialogue_history, turn_number)


def test_client_driven_workflow(
    client: EarlClient,
    local_doctor_url: Optional[str] = None,
    local_doctor_key: Optional[str] = None,
    patient_mode: str = "single",
    doctor_initiates: bool = True,
    judge_timeout: int = 600,
    poll_interval: float = 5.0,
    max_turns: int = 6,
) -> bool:
    """
    Test client-driven simulation workflow.
    
    This test acts as MIDDLEWARE between the EARL orchestrator and a local doctor API.
    """
    log_section("Integration Test: Client-Driven Workflow (VPN-Safe)")

    pipeline_name = f"sdk-test-client-driven-{int(time.time())}"
    pipeline_created = False

    log_info("This test acts as MIDDLEWARE between orchestrator and your doctor API")
    log_info(f"Doctor initiates: {doctor_initiates}")
    log_info(f"Local doctor URL: {local_doctor_url or 'MOCK (no URL provided)'}")

    try:
        # Step 1: Fetch patients from API
        log_info(f"Step 1: Fetching patients from API (mode: {patient_mode})...")
        try:
            all_patients = client.patients.list()
            if not all_patients:
                log_error("No patients available")
                return False
            log_info(f"Found {len(all_patients)} patients")
        except Exception as e:
            log_error(f"Failed to fetch patients: {e}")
            return False
        
        # Select patients based on mode
        if patient_mode == "single":
            patient_ids = [all_patients[0].id]
        elif patient_mode == "three":
            patient_ids = [p.id for p in all_patients[:3]]
        else:  # random
            patient_ids = [p.id for p in all_patients[:2]]
        
        log_success(f"Using {len(patient_ids)} patient(s)")
        for pid in patient_ids:
            print(f"   • {pid}")

        # Step 2: Get dimensions
        log_info("Step 2: Querying available dimensions...")
        dimensions = client.dimensions.list()
        dimension_ids = [d.id for d in dimensions[:3]]
        log_success(f"Selected {len(dimension_ids)} dimensions")

        # Step 3: Create CLIENT-DRIVEN pipeline
        log_info("Step 3: Creating CLIENT-DRIVEN pipeline...")
        initiator = "doctor" if doctor_initiates else "patient"

        pipeline = client.pipelines.create(
            name=pipeline_name,
            dimension_ids=dimension_ids,
            patient_ids=patient_ids,
            doctor_config=DoctorApiConfig.client_driven(),
            description="SDK test - client-driven mode for VPN scenarios",
            conversation_initiator=initiator,
        )
        pipeline_created = True
        log_success(f"Created pipeline: {pipeline.name}")

        # Step 4: Start simulation
        log_info("Step 4: Starting simulation...")
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=len(patient_ids),
            parallel_count=1,
        )
        log_success(f"Simulation started: {simulation.id}")
        print(f"   Episodes: {simulation.total_episodes}")

        time.sleep(3)  # Wait for episodes to initialize

        # Step 5: ORCHESTRATE the conversation
        log_info("Step 5: Orchestrating conversation (middleware mode)...")
        
        episode_states = {}
        max_total_iterations = 100
        iteration = 0

        while iteration < max_total_iterations:
            iteration += 1
            
            sim = client.simulations.get(simulation.id)
            if sim.status.value in ["completed", "failed"]:
                print()
                log_success(f"Simulation {sim.status.value}!")
                break

            try:
                episodes = client.simulations.get_episodes(simulation.id)
            except Exception as e:
                log_warning(f"Could not get episodes: {e}")
                time.sleep(poll_interval)
                continue

            if not episodes:
                time.sleep(poll_interval)
                continue

            active_episodes = 0
            for ep in episodes:
                ep_id = ep.get("episode_id")
                ep_num = ep.get("episode_number", "?")
                status = ep.get("status", "unknown")

                # Fetch full episode for dialogue
                try:
                    full_episode = client.simulations.get_episode(simulation.id, ep_id)
                    dialogue = full_episode.get("dialogue_history", [])
                except Exception:
                    dialogue = []

                if ep_id not in episode_states:
                    episode_states[ep_id] = {"turns": 0, "done": False, "last_dialogue_len": 0}

                state = episode_states[ep_id]

                if state["done"] or status in ["completed", "failed", "judging"]:
                    state["done"] = True
                    continue

                active_episodes += 1
                turn_number = state["turns"]
                current_dialogue_len = len(dialogue)

                needs_doctor_response = False

                if status == "awaiting_doctor":
                    if current_dialogue_len > state["last_dialogue_len"]:
                        if dialogue and dialogue[-1].get("role") == "patient":
                            needs_doctor_response = True
                    elif state["last_dialogue_len"] == 0 and state["turns"] == 0:
                        if doctor_initiates and current_dialogue_len == 0:
                            needs_doctor_response = True
                        elif current_dialogue_len > 0:
                            needs_doctor_response = True

                if needs_doctor_response:
                    print(f"\n   === Episode {ep_num} - Turn {turn_number + 1} ===")

                    if dialogue:
                        patient_msgs = [m for m in dialogue if m.get("role") == "patient"]
                        if patient_msgs:
                            last_patient = patient_msgs[-1].get("content", "")
                            print(f"   Patient: {last_patient[:80]}{'...' if len(last_patient) > 80 else ''}")

                    doctor_response = get_doctor_response(
                        dialogue, turn_number, local_doctor_url, local_doctor_key
                    )
                    print(f"   Doctor: {doctor_response[:80]}{'...' if len(doctor_response) > 80 else ''}")

                    try:
                        updated_ep = client.simulations.submit_response(
                            simulation.id, ep_id, doctor_response
                        )
                        new_status = updated_ep.get("status", "unknown")
                        new_dialogue = updated_ep.get("dialogue_history", [])
                        state["turns"] += 1
                        state["last_dialogue_len"] = len(new_dialogue)
                        print(f"   -> Status: {new_status}, Msgs: {len(new_dialogue)}")

                        if new_status in ["completed", "judging", "failed"]:
                            state["done"] = True

                        if state["turns"] >= max_turns:
                            state["done"] = True

                    except Exception as e:
                        log_error(f"   Failed to submit: {e}")

            done_count = sum(1 for s in episode_states.values() if s["done"])
            print(f"\r   Progress: {done_count}/{len(episodes)} episodes done   ", end="", flush=True)

            if all(s["done"] for s in episode_states.values()) and len(episode_states) == len(episodes):
                print()
                log_success("All episodes completed!")
                break

            time.sleep(poll_interval)

        print()

        # Step 6: Wait for judging
        log_info(f"Step 6: Waiting for judging (max {judge_timeout}s)...")
        judge_start = time.time()

        while time.time() - judge_start < judge_timeout:
            final_sim = client.simulations.get(simulation.id)
            if final_sim.status.value in ["completed", "failed"]:
                break
            elapsed = int(time.time() - judge_start)
            print(f"\r   Waiting for judging... ({elapsed}s)", end="", flush=True)
            time.sleep(20)

        print()
        final_sim = client.simulations.get(simulation.id)

        if final_sim.status.value == "failed":
            log_error(f"Final status: {final_sim.status.value}")
        elif final_sim.status.value == "completed":
            log_success(f"Final status: {final_sim.status.value}")
        else:
            log_warning(f"Final status: {final_sim.status.value}")

        print(f"   Episodes: {final_sim.completed_episodes}/{final_sim.total_episodes}")

        if final_sim.summary:
            avg_score = final_sim.summary.get("average_score")
            if avg_score is not None:
                print(f"   Average Score: {avg_score:.2f}/4")

        # Show episode results
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
                        log_error(f"Episode {ep.get('episode_number')}: {patient} - FAILED: {error[:60]}...")
                    elif score is not None and score > 0:
                        log_success(f"Episode {ep.get('episode_number')}: {patient} - {turns} msgs - Score: {score:.2f}/4")
                    else:
                        log_warning(f"Episode {ep.get('episode_number')}: {patient} - {turns} msgs - Status: {status}")
        except Exception as e:
            log_warning(f"Could not get report: {e}")

        # Cleanup
        log_info("Step 7: Cleanup...")
        try:
            client.pipelines.delete(pipeline_name)
            log_success("Test pipeline deleted")
        except Exception as e:
            log_info(f"Cleanup note: {e}")

        if final_sim.status.value == "completed":
            log_success("Client-driven workflow completed successfully!")
        else:
            log_warning(f"Workflow ended with status: {final_sim.status.value}")

        return True

    except Exception as e:
        log_error(f"Error: {e}")
        import traceback
        traceback.print_exc()

        if pipeline_created:
            try:
                client.pipelines.delete(pipeline_name)
            except:
                pass

        return False


def main():
    parser = argparse.ArgumentParser(description="Earl SDK - Client-Driven Test")
    parser.add_argument("--env", choices=["dev", "test", "prod"], default="test")
    parser.add_argument("--patients", choices=["single", "three", "random"], default="single")
    parser.add_argument("--judge-timeout", type=int, default=600)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--client-id", type=str, default=None)
    parser.add_argument("--client-secret", type=str, default=None)
    parser.add_argument("--organization", type=str, default=None)
    
    # Local doctor API (for VPN scenarios)
    parser.add_argument(
        "--local-doctor-url",
        type=str,
        default=None,
        help="Local doctor API URL (OpenAI-compatible). If not provided, uses mock doctor.",
    )
    parser.add_argument(
        "--local-doctor-key",
        type=str,
        default=None,
        help="API key for the local doctor API",
    )
    parser.add_argument(
        "--patient-initiates",
        action="store_true",
        help="Let patient speak first (default: doctor initiates)",
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

    result = test_client_driven_workflow(
        client,
        local_doctor_url=args.local_doctor_url,
        local_doctor_key=args.local_doctor_key,
        patient_mode=args.patients,
        doctor_initiates=not args.patient_initiates,
        judge_timeout=args.judge_timeout,
        poll_interval=args.poll_interval,
        max_turns=args.max_turns,
    )

    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()

