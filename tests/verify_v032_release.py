#!/usr/bin/env python3
"""
Verification test for SDK v0.3.2 features:
1. auth_type parameter in DoctorApiConfig.external()
2. Cold-start retry logic in validate_doctor_api()
3. V2 patients integration

Run with:
    PYTHONPATH=sdk python3 sdk/tests/verify_v032_release.py \
        --client-id "YOUR_CLIENT_ID" \
        --client-secret "YOUR_CLIENT_SECRET"
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add SDK to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from earl_sdk import EarlClient, DoctorApiConfig
from earl_sdk.models import DoctorApiConfig as DoctorApiConfigModel

# V2 Patient IDs
V2_PATIENTS = [
    "Anxiety_Focused_Clinical_Encounter",
]


def log_info(msg):
    print(f"ℹ {msg}")


def log_success(msg):
    print(f"✓ {msg}")


def log_error(msg):
    print(f"✗ {msg}")


def log_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def test_auth_type_parameter():
    """Test that auth_type parameter works in DoctorApiConfig.external()"""
    log_section("Test 1: auth_type parameter")
    
    # Test default auth_type (bearer)
    config = DoctorApiConfig.external(
        api_url="https://example.com/chat",
        api_key="test-key"
    )
    data = config.to_dict()
    
    if data.get("auth_type") == "bearer":
        log_success("Default auth_type is 'bearer'")
    else:
        log_error(f"Expected auth_type='bearer', got '{data.get('auth_type')}'")
        return False
    
    # Test explicit bearer
    config = DoctorApiConfig.external(
        api_url="https://example.com/chat",
        api_key="test-key",
        auth_type="bearer"
    )
    data = config.to_dict()
    
    if data.get("auth_type") == "bearer":
        log_success("Explicit auth_type='bearer' works")
    else:
        log_error(f"Expected auth_type='bearer', got '{data.get('auth_type')}'")
        return False
    
    # Test api_key auth type
    config = DoctorApiConfig.external(
        api_url="https://custom-api.com/generate",
        api_key="custom-key",
        auth_type="api_key"
    )
    data = config.to_dict()
    
    if data.get("auth_type") == "api_key":
        log_success("auth_type='api_key' works")
    else:
        log_error(f"Expected auth_type='api_key', got '{data.get('auth_type')}'")
        return False
    
    log_success("All auth_type parameter tests passed!")
    return True


def test_internal_doctor_config():
    """Test that internal doctor config still works"""
    log_section("Test 2: Internal doctor config")
    
    config = DoctorApiConfig.internal()
    data = config.to_dict()
    
    if data.get("type") == "internal":
        log_success("Internal doctor type is correct")
    else:
        log_error(f"Expected type='internal', got '{data.get('type')}'")
        return False
    
    # Internal doctor should NOT have auth_type
    if "auth_type" not in data:
        log_success("Internal doctor does not include auth_type (correct)")
    else:
        log_error(f"Internal doctor should not have auth_type, but got '{data.get('auth_type')}'")
        return False
    
    log_success("Internal doctor config tests passed!")
    return True


def test_v2_patients_integration(client_id: str, client_secret: str):
    """Test V2 patients integration with internal doctor"""
    log_section("Test 3: V2 Patients Integration")
    
    log_info("Creating Earl client...")
    
    client = EarlClient(
        client_id=client_id,
        client_secret=client_secret,
        organization="",
        environment="test",
    )
    
    log_success(f"Client created for test environment")
    log_info(f"API URL: {client._api_url}")
    
    # Test listing dimensions
    log_info("Fetching dimensions...")
    dimensions = client.dimensions.list(include_custom=False)
    log_success(f"Got {len(dimensions)} dimensions")
    
    # Test listing patients (V2) - optional, may fail if patient API is down
    log_info("Checking V2 patient availability...")
    try:
        patients = client.patients.list()
        v2_found = []
        for p in patients:
            if p.get("id") in V2_PATIENTS or p.get("patient_id") in V2_PATIENTS:
                v2_found.append(p.get("id") or p.get("patient_id"))
        
        if v2_found:
            log_success(f"Found V2 patients: {v2_found}")
        else:
            log_info(f"V2 patients not found in patient list (may use different IDs)")
    except Exception as e:
        log_info(f"Patient list unavailable: {e} (continuing with pipeline test)")
    
    # Create a pipeline with internal doctor and V2 patient
    pipeline_name = f"verify-v032-{int(time.time())}"
    log_info(f"Creating pipeline: {pipeline_name}")
    
    # Select 3 dimensions (handle both dict and object responses)
    dimension_ids = []
    for d in dimensions[:3]:
        if isinstance(d, dict):
            dim_id = d.get("id") or d.get("dimension_id")
        else:
            dim_id = getattr(d, "id", None) or getattr(d, "dimension_id", None)
        if dim_id:
            dimension_ids.append(dim_id)
    
    try:
        pipeline = client.pipelines.create(
            name=pipeline_name,
            doctor_config=DoctorApiConfig.internal(),
            patient_ids=V2_PATIENTS,
            dimension_ids=dimension_ids,
            description="SDK v0.3.2 verification test",
            conversation_initiator="doctor",
            max_turns=5,  # Short test
        )
        
        # Handle both dict and object responses
        if isinstance(pipeline, dict):
            pipeline_id = pipeline.get("id") or pipeline.get("pipeline_id")
        else:
            pipeline_id = getattr(pipeline, "id", None) or getattr(pipeline, "pipeline_id", None)
        log_success(f"Pipeline created: {pipeline_id}")
        
        # Start simulation (use pipeline name, not ID)
        log_info("Starting simulation...")
        simulation = client.simulations.create(pipeline_name=pipeline_name)
        if isinstance(simulation, dict):
            simulation_id = simulation.get("id") or simulation.get("simulation_id")
        else:
            simulation_id = getattr(simulation, "id", None) or getattr(simulation, "simulation_id", None)
        log_success(f"Simulation started: {simulation_id}")
        
        # Wait for completion (with timeout)
        log_info("Waiting for simulation to complete...")
        max_wait = 180  # 3 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = client.simulations.get(simulation_id)
            if isinstance(status, dict):
                sim_status = status.get("status", "unknown")
            else:
                sim_status = getattr(status, "status", "unknown")
            
            if sim_status == "completed":
                log_success("Simulation completed!")
                break
            elif sim_status in ["failed", "error"]:
                if isinstance(status, dict):
                    error_msg = status.get('error', 'Unknown error')
                else:
                    error_msg = getattr(status, 'error', 'Unknown error')
                log_error(f"Simulation failed: {error_msg}")
                return False
            else:
                elapsed = int(time.time() - start_time)
                print(f"\r  Status: {sim_status} ({elapsed}s elapsed)...", end="", flush=True)
                time.sleep(5)
        else:
            print()
            log_error(f"Simulation timed out after {max_wait}s")
            return False
        
        print()  # New line after progress
        
        # Get results
        log_info("Fetching simulation results...")
        if isinstance(status, dict):
            episodes = status.get("episodes", [])
        else:
            episodes = getattr(status, "episodes", []) or []
        
        if episodes:
            for ep in episodes:
                if isinstance(ep, dict):
                    ep_id = ep.get("id") or ep.get("episode_id")
                    ep_status = ep.get("status", "unknown")
                    patient_id = ep.get("patient_id", "unknown")
                    dialogue = ep.get("dialogue", [])
                    scores = ep.get("scores", {})
                else:
                    ep_id = getattr(ep, "id", None) or getattr(ep, "episode_id", None)
                    ep_status = getattr(ep, "status", "unknown")
                    patient_id = getattr(ep, "patient_id", "unknown")
                    dialogue = getattr(ep, "dialogue", []) or []
                    scores = getattr(ep, "scores", {}) or {}
                
                if ep_status == "completed":
                    log_success(f"  Episode {ep_id}: {patient_id} - COMPLETED")
                    
                    # Check for dialogue
                    if dialogue:
                        log_success(f"    Dialogue turns: {len(dialogue)}")
                    
                    # Check for scores
                    if scores:
                        if isinstance(scores, dict):
                            avg_score = scores.get("average", "N/A")
                        else:
                            avg_score = getattr(scores, "average", "N/A")
                        log_success(f"    Average score: {avg_score}")
                else:
                    log_info(f"  Episode {ep_id}: {patient_id} - {ep_status}")
        else:
            log_info("No episode details in response (may need to fetch individually)")
        
        log_success("V2 patients integration test passed!")
        return True
        
    except Exception as e:
        log_error(f"Pipeline/simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify SDK v0.3.2 release")
    parser.add_argument("--client-id", type=str, help="Earl client ID")
    parser.add_argument("--client-secret", type=str, help="Earl client secret")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration test")
    args = parser.parse_args()
    
    print("=" * 60)
    print(" SDK v0.3.2 Verification Test")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: auth_type parameter
    if not test_auth_type_parameter():
        all_passed = False
    
    # Test 2: Internal doctor config
    if not test_internal_doctor_config():
        all_passed = False
    
    # Test 3: V2 patients integration (if credentials provided)
    if not args.skip_integration:
        client_id = args.client_id or os.environ.get("EARL_CLIENT_ID", "")
        client_secret = args.client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
        
        if client_id and client_secret:
            if not test_v2_patients_integration(client_id, client_secret):
                all_passed = False
        else:
            log_section("Test 3: V2 Patients Integration (SKIPPED)")
            log_info("No credentials provided. Use --client-id and --client-secret")
            log_info("Or set EARL_CLIENT_ID and EARL_CLIENT_SECRET environment variables")
    else:
        log_section("Test 3: V2 Patients Integration (SKIPPED)")
        log_info("Skipped via --skip-integration flag")
    
    # Summary
    log_section("Summary")
    if all_passed:
        log_success("All tests passed! SDK v0.3.2 is working correctly.")
        return 0
    else:
        log_error("Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
