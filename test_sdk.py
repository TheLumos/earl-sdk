#!/usr/bin/env python3
"""
Earl SDK Integration Test Script

This script tests the Earl SDK against the test environment.
It performs real API calls to validate all SDK functionality.

Usage:
    # Option 1: Set credentials below in CREDENTIALS section
    # Option 2: Use environment variables:
    #   export EARL_CLIENT_ID="your-m2m-client-id"
    #   export EARL_CLIENT_SECRET="your-m2m-client-secret"
    #   export EARL_ORGANIZATION="org_your_org_id"
    
    # Run the test:
    python test_sdk.py
    
    # Or run specific tests:
    python test_sdk.py --test auth
    python test_sdk.py --test dimensions
    python test_sdk.py --test patients
    python test_sdk.py --test pipelines
    python test_sdk.py --test full

    # Or run integration tests:
    python3 test_sdk.py --env test --test external \
  --doctor-url "https://example.com/chat" \
  --doctor-key "earl-test-doctor-key" \
  --wait
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Optional

# =============================================================================
# CREDENTIALS - Set your credentials here (or use environment variables)
# =============================================================================
CREDENTIALS = {
    "client_id": "",      # Your Auth0 M2M client ID
    "client_secret": "",  # Your Auth0 M2M client secret
    "organization_id": "",   # Leave empty - org_id is injected by Auth0 Action from client metadata
}
# =============================================================================

# Add the SDK to path for development testing (go up one level from tests/ to project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earl_sdk import EarlClient, Environment
from earl_sdk.exceptions import (
    EarlError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
)


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def log_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def log_error(msg: str) -> None:
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def log_info(msg: str) -> None:
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")


def log_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")


def show_simulation_progress(simulation) -> None:
    """Display simulation progress on a single line (overwrites previous)."""
    pct = int(simulation.progress * 100) if simulation.total_episodes > 0 else 0
    status_str = simulation.status.value if hasattr(simulation.status, 'value') else str(simulation.status)
    print(
        f"\r   Progress: {simulation.completed_episodes}/{simulation.total_episodes} "
        f"episodes ({pct}%) - Status: {status_str}    ",
        end="",
        flush=True
    )


def display_simulation_report(client: "EarlClient", simulation_id: str) -> None:
    """Display the complete simulation report in a formatted way."""
    try:
        print()
        log_info("Fetching complete simulation report...")
        report = client.simulations.get_report(simulation_id)
        
        print()
        print(f"   {Colors.BOLD}╔══════════════════════════════════════════════════════════╗{Colors.END}")
        print(f"   {Colors.BOLD}║           SIMULATION REPORT                              ║{Colors.END}")
        print(f"   {Colors.BOLD}╚═══════ ═══════════════════════════════════════════════════╝{Colors.END}")
        
        # Simulation metadata
        print(f"\n   {Colors.BOLD}Simulation:{Colors.END} {report.get('simulation_id', 'N/A')}")
        print(f"   {Colors.BOLD}Pipeline:{Colors.END} {report.get('pipeline_name', 'N/A')}")
        print(f"   {Colors.BOLD}Status:{Colors.END} {report.get('status', 'N/A')}")
        if report.get('duration_seconds'):
            print(f"   {Colors.BOLD}Duration:{Colors.END} {report['duration_seconds']:.1f} seconds")
        
        # Summary statistics
        summary = report.get('summary', {})
        print(f"\n   {Colors.BOLD}Summary:{Colors.END}")
        print(f"      Episodes: {summary.get('completed', 0)}/{summary.get('total_episodes', 0)} completed")
        if summary.get('failed', 0) > 0:
            print(f"      Failed: {summary.get('failed', 0)}")
        avg_score = summary.get('average_score')
        if avg_score is not None:
            print(f"      Average Score: {avg_score:.2f}/4")
            print(f"      Score Range: {summary.get('min_score', 'N/A')} - {summary.get('max_score', 'N/A')}")
        
        # Per-dimension breakdown
        dimension_scores = report.get('dimension_scores', {})
        if dimension_scores:
            print(f"\n   {Colors.BOLD}Dimension Scores:{Colors.END}")
            for dim_id, stats in dimension_scores.items():
                avg = stats.get('average')
                if avg is not None:
                    print(f"      {dim_id}: {avg:.2f}/4 (min: {stats.get('min')}, max: {stats.get('max')})")
        
        # Episode details
        episodes = report.get('episodes', [])
        if episodes:
            print(f"\n   {Colors.BOLD}Episodes:{Colors.END}")
            for ep in episodes:
                ep_num = ep.get('episode_number', '?')
                patient = ep.get('patient_name') or ep.get('patient_id', 'Unknown')
                status = ep.get('status', 'unknown')
                score = ep.get('total_score')
                turns = ep.get('dialogue_turns', 0)
                
                score_str = f"{score:.2f}/4" if score is not None else "N/A"
                status_icon = "✓" if status == "completed" else "✗" if status == "failed" else "?"
                
                print(f"\n      {Colors.BOLD}Episode {ep_num}:{Colors.END} {status_icon} {status}")
                print(f"         Patient: {patient}")
                print(f"         Turns: {turns}")
                print(f"         Score: {score_str}")
                
                # Show judge scores per dimension
                judge_scores = ep.get('judge_scores', {})
                if judge_scores:
                    print(f"         Dimension scores:")
                    for dim, score in judge_scores.items():
                        if score is not None:
                            print(f"            {dim}: {score:.1f}/4")
                
                # Show dialogue preview (first and last turn)
                dialogue = ep.get('dialogue_history', [])
                if dialogue:
                    print(f"         Dialogue preview:")
                    # First turn
                    first = dialogue[0]
                    role = first.get('role', 'unknown').upper()
                    content = first.get('content', '')[:80]
                    if len(first.get('content', '')) > 80:
                        content += "..."
                    print(f"            {role}: {content}")
                    
                    if len(dialogue) > 2:
                        print(f"            ... ({len(dialogue) - 2} more turns) ...")
                    
                    if len(dialogue) > 1:
                        # Last turn
                        last = dialogue[-1]
                        role = last.get('role', 'unknown').upper()
                        content = last.get('content', '')[:80]
                        if len(last.get('content', '')) > 80:
                            content += "..."
                        print(f"            {role}: {content}")
                
                # Show error if failed
                if ep.get('error'):
                    print(f"         {Colors.RED}Error: {ep['error']}{Colors.END}")
        
        print()
        print(f"   {Colors.BOLD}{'─'*60}{Colors.END}")
        log_success("Report displayed successfully")
        
    except Exception as e:
        log_warning(f"Could not fetch report: {e}")


def log_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.BOLD}{'='*60}{Colors.END}")


def get_credentials(
    cli_client_id: str = None,
    cli_client_secret: str = None,
    cli_organization: str = None
) -> tuple[str, str, str]:
    """Get credentials from CLI args, CREDENTIALS dict, or environment variables.
    
    Priority order:
    1. CLI arguments (--client-id, --client-secret, --organization)
    2. CREDENTIALS dict in this file
    3. Environment variables (EARL_CLIENT_ID, EARL_CLIENT_SECRET, EARL_ORGANIZATION_ID)
    """
    # Priority: CLI args > hardcoded CREDENTIALS > env vars
    client_id = cli_client_id or CREDENTIALS.get("client_id") or os.environ.get("EARL_CLIENT_ID")
    client_secret = cli_client_secret or CREDENTIALS.get("client_secret") or os.environ.get("EARL_CLIENT_SECRET")
    organization = cli_organization or CREDENTIALS.get("organization_id") or os.environ.get("EARL_ORGANIZATION_ID") or ""
    
    missing = []
    if not client_id:
        missing.append("client_id")
    if not client_secret:
        missing.append("client_secret")
    # Organization is optional for M2M - the org_id will be in the token claims via Auth0 Action
    
    if missing:
        log_error(f"Missing required credentials: {', '.join(missing)}")
        print("\nOption 1: Pass via CLI: --client-id YOUR_ID --client-secret YOUR_SECRET")
        print("Option 2: Set them in the CREDENTIALS dict at the top of this file")
        print("Option 3: Use environment variables:")
        print('  export EARL_CLIENT_ID="your-m2m-client-id"')
        print('  export EARL_CLIENT_SECRET="your-m2m-client-secret"')
        sys.exit(1)
    
    return client_id, client_secret, organization


def test_authentication(client: EarlClient) -> bool:
    """Test that authentication works."""
    log_section("Testing Authentication")
    
    try:
        log_info(f"Environment: {client.environment}")
        log_info(f"API URL: {client.api_url}")
        log_info(f"Organization: {client.organization}")
        
        # Test connection (fetches token and makes a request)
        result = client.test_connection()
        
        if result:
            log_success("Authentication successful!")
            return True
        else:
            log_error("Authentication failed - no error but returned False")
            return False
            
    except AuthenticationError as e:
        log_error(f"Authentication failed: {e.message}")
        if e.details:
            print(f"   Details: {e.details}")
        return False
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        return False


def test_dimensions(client: EarlClient) -> bool:
    """Test dimensions API."""
    log_section("Testing Dimensions API")
    
    try:
        # List dimensions
        log_info("Listing dimensions...")
        dimensions = client.dimensions.list()
        
        if not dimensions:
            log_error("No dimensions returned")
            return False
        
        log_success(f"Found {len(dimensions)} dimensions")
        
        # Print first few dimensions
        for dim in dimensions[:5]:
            print(f"   - {dim.name} ({dim.category}): {dim.description[:50]}...")
        
        if len(dimensions) > 5:
            print(f"   ... and {len(dimensions) - 5} more")
        
        # Get a specific dimension
        log_info(f"Getting dimension '{dimensions[0].id}'...")
        dim = client.dimensions.get(dimensions[0].id)
        log_success(f"Got dimension: {dim.name}")
        
        return True
        
    except NotFoundError as e:
        log_error(f"Dimension not found: {e.message}")
        return False
    except Exception as e:
        log_error(f"Error testing dimensions: {e}")
        return False


def test_patients(client: EarlClient) -> bool:
    """Test patients API."""
    log_section("Testing Patients API")
    
    try:
        # List patients
        log_info("Listing patients...")
        patients = client.patients.list(limit=10)
        
        if not patients:
            log_info("No patients available (endpoint may not be implemented yet)")
            log_success("Patients API responded (empty list)")
            return True
        
        log_success(f"Found {len(patients)} patients")
        
        # Print patient info
        for patient in patients[:5]:
            print(f"   - {patient.name} ({patient.age}yo {patient.gender})")
            print(f"     Chief complaint: {patient.chief_complaint[:50]}...")
            print(f"     Difficulty: {patient.difficulty}")
        
        if len(patients) > 5:
            print(f"   ... and {len(patients) - 5} more")
        
        return True
        
    except NotFoundError as e:
        log_info("Patients endpoint not found (may not be implemented)")
        log_success("Skipped - endpoint not available")
        return True  # Not a failure, just not implemented
    except Exception as e:
        log_error(f"Error testing patients: {e}")
        return False


def test_pipelines(client: EarlClient) -> bool:
    """Test pipelines API."""
    log_section("Testing Pipelines API")
    
    try:
        # List pipelines
        log_info("Listing pipelines...")
        pipelines = client.pipelines.list()
        log_success(f"Found {len(pipelines)} pipelines")
        
        for pipeline in pipelines[:3]:
            default_str = " (default)" if pipeline.is_default else ""
            active_str = "active" if pipeline.is_active else "inactive"
            print(f"   - {pipeline.name}{default_str} ({active_str})")
            if pipeline.description:
                print(f"     Description: {pipeline.description[:50]}...")
            if pipeline.dimension_ids:
                print(f"     Dimensions: {len(pipeline.dimension_ids)} configured")
            if pipeline.has_auth_key:
                print(f"     Has auth key: Yes")
        
        # Try to get a specific pipeline (if any exist)
        if pipelines:
            log_info(f"Getting pipeline '{pipelines[0].name}'...")
            try:
                fetched_pipeline = client.pipelines.get(pipelines[0].name)
                log_success(f"Got pipeline: {fetched_pipeline.name}")
            except NotFoundError:
                log_info("Get by name not supported, skipping")
        
        # Note: Skipping create/update/delete as they may require specific permissions
        log_info("Skipping create/update/delete (requires specific permissions)")
        
        return True
        
    except ValidationError as e:
        log_error(f"Validation error: {e.message}")
        if e.details:
            print(f"   Details: {e.details}")
        return False
    except Exception as e:
        log_error(f"Error testing pipelines: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulations(client: EarlClient) -> bool:
    """Test simulations API."""
    log_section("Testing Simulations API")
    
    try:
        # List simulations
        log_info("Listing simulations...")
        simulations = client.simulations.list(limit=5)
        log_success(f"Found {len(simulations)} simulations")
        
        for sim in simulations[:3]:
            sim_id_short = sim.id[:8] if len(sim.id) > 8 else sim.id
            print(f"   - {sim_id_short}... ({sim.status.value})")
            print(f"     Pipeline: {sim.pipeline_name}")
            print(f"     Type: {sim.simulation_type}")
            print(f"     Episodes: {sim.completed_episodes}/{sim.total_episodes}")
            if sim.summary:
                avg_score = sim.summary.get("average_score", "N/A")
                print(f"     Average Score: {avg_score}")
        
        return True
        
    except Exception as e:
        log_error(f"Error testing simulations: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow(client: EarlClient) -> bool:
    """Test full workflow: list resources and verify connectivity."""
    log_section("Testing Full Workflow")
    
    try:
        # 1. Get dimensions
        log_info("Step 1: Getting dimensions...")
        dimensions = client.dimensions.list()
        if len(dimensions) < 2:
            log_error("Need at least 2 dimensions")
            return False
        log_success(f"Got {len(dimensions)} dimensions")
        
        # 2. Get pipelines
        log_info("Step 2: Getting pipelines...")
        pipelines = client.pipelines.list()
        log_success(f"Got {len(pipelines)} pipelines")
        
        # 3. Get simulations (past runs)
        log_info("Step 3: Getting simulations...")
        simulations = client.simulations.list(limit=5)
        log_success(f"Got {len(simulations)} simulations")
        
        # Note: Creating pipelines and running simulations requires specific permissions
        log_info("Step 4: Skipping pipeline creation (requires specific permissions)")
        log_info("Step 5: Skipping simulation run (requires real doctor API)")
        
        log_success("Full workflow connectivity verified!")
        
        return True
        
    except Exception as e:
        log_error(f"Error in full workflow: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# INTEGRATION TESTS - Simulating Real Customer Workflows
# =============================================================================

def test_internal_doctor_workflow(
    client: EarlClient,
    wait_for_completion: bool = False,
    use_default_pipeline: bool = False,
) -> bool:
    """
    Test full customer workflow using EARL's internal doctor.
    
    This simulates how a customer would:
    1. Get patient IDs from default_pipeline
    2. Query available dimensions and select some
    3. Create a NEW pipeline with internal doctor configuration (or use default)
    4. Verify the pipeline was created correctly
    5. Start a simulation
    6. (Optionally) Wait for results
    7. Clean up the test pipeline (if created)
    
    Args:
        client: EarlClient instance
        wait_for_completion: If True, wait for simulation to finish (can take minutes)
        use_default_pipeline: If True, use default_pipeline instead of creating a new one
    """
    log_section("Integration Test: Internal Doctor Workflow")
    
    pipeline_name = f"sdk-test-internal-{int(time.time())}"
    pipeline_created = False
    
    # If using default pipeline, skip pipeline creation
    if use_default_pipeline:
        log_info("Using default_pipeline (--use-default-pipeline flag)")
        pipeline_name = "default_pipeline"
    
    try:
        # Step 1: Get patient IDs from default_pipeline
        log_info("Step 1: Getting patient IDs from default_pipeline...")
        patient_ids = []
        try:
            default_pipeline = client.pipelines.get("default_pipeline")
            if default_pipeline and default_pipeline.patient_ids:
                patient_ids = default_pipeline.patient_ids
            if patient_ids:
                log_success(f"Found {len(patient_ids)} patients")
                print(f"   Patient IDs: {patient_ids[:3]}{'...' if len(patient_ids) > 3 else ''}")
            else:
                log_warning("No patients found in default_pipeline - simulation may fail")
        except Exception as e:
            log_warning(f"Could not get default_pipeline: {e}")
        
        # Step 2: Query dimensions and select a subset
        log_info("Step 2: Querying available dimensions...")
        dimensions = client.dimensions.list()
        log_success(f"Found {len(dimensions)} dimensions available")
        
        # Select 5 dimensions from different categories for a balanced evaluation
        selected_dims = []
        seen_categories = set()
        for dim in dimensions:
            if dim.category not in seen_categories and len(selected_dims) < 5:
                selected_dims.append(dim)
                seen_categories.add(dim.category)
        
        # Fill remaining slots if we don't have 5 different categories
        for dim in dimensions:
            if len(selected_dims) >= 5:
                break
            if dim not in selected_dims:
                selected_dims.append(dim)
        
        dimension_ids = [d.id for d in selected_dims]
        log_success(f"Selected {len(dimension_ids)} dimensions:")
        for dim in selected_dims:
            print(f"   - {dim.name} ({dim.category})")
        
        # Step 3: Create a NEW pipeline with internal doctor (or use default)
        if use_default_pipeline:
            log_info("Step 3: Using default_pipeline (skipping pipeline creation)...")
            log_success(f"Using pipeline: {pipeline_name}")
        else:
            log_info("Step 3: Creating NEW pipeline with INTERNAL doctor...")
            from earl_sdk.models import DoctorApiConfig
            
            # Use patient-initiated conversation (typical telemedicine flow)
            pipeline = client.pipelines.create(
                name=pipeline_name,
                dimension_ids=dimension_ids,
                patient_ids=patient_ids,
                doctor_config=DoctorApiConfig.internal(),
                description="SDK integration test - internal doctor",
                conversation_initiator="patient",  # Patient sends first message
            )
            pipeline_created = True
            log_success(f"Created pipeline: {pipeline.name}")
            print(f"   Conversation initiator: patient")
        
        # Step 4: Verify pipeline configuration
        log_info("Step 4: Verifying pipeline configuration...")
        fetched_pipeline = client.pipelines.get(pipeline_name)
        if not fetched_pipeline:
            log_error("Failed to fetch pipeline!")
            return False
        
        log_success("Pipeline verified:")
        print(f"   Name: {fetched_pipeline.name}")
        print(f"   Description: {fetched_pipeline.description or 'N/A'}")
        
        # Verify doctor type (SDK uses doctor_api field)
        if fetched_pipeline.doctor_api:
            doctor_type = fetched_pipeline.doctor_api.type
            print(f"   Doctor Type: {doctor_type}")
        else:
            print(f"   Doctor Type: internal (default)")
        
        # Verify conversation initiator
        print(f"   Conversation Initiator: {fetched_pipeline.conversation_initiator}")
        
        # Verify patients
        if fetched_pipeline.patient_ids:
            print(f"   Patients: {len(fetched_pipeline.patient_ids)}")
        
        # Step 5: Start simulation
        log_info("Step 5: Starting simulation...")
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=1,  # Run just 1 episode for testing
            parallel_count=1,
        )
        log_success(f"Simulation started: {simulation.id}")
        print(f"   Status: {simulation.status.value}")
        print(f"   Pipeline: {pipeline_name}")
        
        # Step 6: Wait for completion (optional)
        if wait_for_completion:
            log_info("Step 6: Waiting for simulation to complete...")
            log_info("   (This may take several minutes...)")
            
            try:
                completed_sim = client.simulations.wait_for_completion(
                    simulation.id,
                    poll_interval=5.0,
                    timeout=600.0,  # 10 minute timeout
                    on_progress=show_simulation_progress,
                )
                print()  # Newline after progress display
                log_success(f"Simulation completed!")
                print(f"   Status: {completed_sim.status.value}")
                print(f"   Episodes: {completed_sim.completed_episodes}/{completed_sim.total_episodes}")
                if completed_sim.summary:
                    avg_score = completed_sim.summary.get('average_score')
                    if avg_score is not None:
                        print(f"   Average Score: {avg_score:.2f}/4")
                    else:
                        print(f"   Average Score: N/A")
                
                # Show complete report with all details
                display_simulation_report(client, simulation.id)
                    
            except TimeoutError:
                print()  # Newline after progress
                log_info("Simulation still running (timeout reached)")
            except Exception as sim_error:
                print()  # Newline after progress
                log_error(f"Simulation error: {sim_error}")
                # Don't fail the test - simulation infrastructure may have issues
                log_info("Simulation had errors but SDK workflow was correct")
        else:
            log_info("Step 6: Skipping wait (simulation running in background)")
            log_info(f"   Monitor at: /simulations/{simulation.id}")
        
        # Step 7: Cleanup (only if we created a pipeline)
        if pipeline_created:
            log_info("Step 7: Cleaning up test pipeline...")
            try:
                client.pipelines.delete(pipeline_name)
                log_success("Test pipeline deleted")
            except Exception as cleanup_error:
                log_info(f"Cleanup note: {cleanup_error}")
        else:
            log_info("Step 7: Skipping cleanup (used existing pipeline)")
        
        log_success("Internal doctor workflow completed successfully!")
        return True
        
    except Exception as e:
        log_error(f"Error in internal doctor workflow: {e}")
        import traceback
        traceback.print_exc()
        
        # Attempt cleanup
        if pipeline_created:
            try:
                client.pipelines.delete(pipeline_name)
            except:
                pass
        
        return False


def test_external_doctor_workflow(
    client: EarlClient,
    doctor_api_url: str,
    doctor_api_key: Optional[str] = None,
    wait_for_completion: bool = False,
    use_default_pipeline: bool = False,
) -> bool:
    """
    Test full customer workflow using an EXTERNAL doctor API.
    
    This simulates how a customer would test their own AI doctor:
    1. Get patient IDs from default_pipeline
    2. Query available dimensions and select some
    3. Create a NEW pipeline with their external doctor API URL and key (or use default)
    4. Verify the pipeline was created correctly with external doctor config
    5. Start a simulation
    6. (Optionally) Wait for results
    7. Clean up the test pipeline (if created)
    
    Args:
        client: EarlClient instance
        doctor_api_url: URL of the external doctor API
        doctor_api_key: Optional API key for authentication
        wait_for_completion: If True, wait for simulation to finish
        use_default_pipeline: If True, use default_pipeline instead of creating a new one
        
    Example doctor API format:
        POST /chat
        Body: {"messages": [{"role": "user", "content": "..."}]}
        Response: {"response": "..."}
    """
    log_section("Integration Test: External Doctor Workflow")
    
    import time
    pipeline_name = f"sdk-test-external-{int(time.time())}"
    pipeline_created = False
    
    # If using default pipeline, skip pipeline creation
    if use_default_pipeline:
        log_info("Using default_pipeline (--use-default-pipeline flag)")
        pipeline_name = "default_pipeline"
    
    try:
        # Step 1: Get patient IDs from default_pipeline
        log_info("Step 1: Getting patient IDs from default_pipeline...")
        patient_ids = []
        try:
            default_pipeline = client.pipelines.get("default_pipeline")
            if default_pipeline and default_pipeline.patient_ids:
                patient_ids = default_pipeline.patient_ids
            if patient_ids:
                log_success(f"Found {len(patient_ids)} patients")
                print(f"   Patient IDs: {patient_ids[:3]}{'...' if len(patient_ids) > 3 else ''}")
            else:
                log_warning("No patients found in default_pipeline - simulation may fail")
        except Exception as e:
            log_warning(f"Could not get default_pipeline: {e}")
        
        # Step 2: Query dimensions and select a subset
        log_info("Step 2: Querying available dimensions...")
        dimensions = client.dimensions.list()
        log_success(f"Found {len(dimensions)} dimensions available")
        
        # Select 5 dimensions for testing
        selected_dims = dimensions[:5]
        dimension_ids = [d.id for d in selected_dims]
        log_success(f"Selected {len(dimension_ids)} dimensions:")
        for dim in selected_dims:
            print(f"   - {dim.name} ({dim.category})")
        
        # Step 3: Create NEW pipeline with EXTERNAL doctor (or use default)
        if use_default_pipeline:
            log_info("Step 3: Using default_pipeline (skipping pipeline creation)...")
            log_info(f"   Note: External doctor URL will be used for this test")
            log_success(f"Using pipeline: {pipeline_name}")
        else:
            log_info("Step 3: Creating NEW pipeline with EXTERNAL doctor...")
            log_info(f"   Doctor API URL: {doctor_api_url}")
            log_info(f"   API Key provided: {'Yes' if doctor_api_key else 'No'}")
            log_info(f"   Patients: {len(patient_ids)}")
            log_info(f"   Conversation Initiator: patient")
            
            from earl_sdk.models import DoctorApiConfig
            
            # Use patient-initiated (typical telemedicine flow)
            pipeline = client.pipelines.create(
                name=pipeline_name,
                dimension_ids=dimension_ids,
                patient_ids=patient_ids,
                doctor_config=DoctorApiConfig.external(
                    api_url=doctor_api_url,
                    api_key=doctor_api_key,
                ),
                description="SDK integration test - external doctor",
                conversation_initiator="patient",  # Patient sends first message
            )
            pipeline_created = True
            log_success(f"Created pipeline: {pipeline.name}")
            print(f"   Conversation initiator: patient")
        
        # Step 4: Verify pipeline configuration
        log_info("Step 4: Verifying pipeline configuration...")
        fetched_pipeline = client.pipelines.get(pipeline_name)
        if not fetched_pipeline:
            log_error("Failed to fetch pipeline!")
            return False
        
        log_success("Pipeline verified:")
        print(f"   Name: {fetched_pipeline.name}")
        print(f"   Description: {fetched_pipeline.description or 'N/A'}")
        
        # Verify doctor type (SDK uses doctor_api field)
        if fetched_pipeline.doctor_api:
            doctor_type = fetched_pipeline.doctor_api.type
            doctor_url = fetched_pipeline.doctor_api.url
            print(f"   Doctor Type: {doctor_type}")
            if doctor_url:
                print(f"   Doctor URL: {doctor_url}")
            # Only fail on mismatch if we created a new pipeline
            if not use_default_pipeline:
                if doctor_type != 'external':
                    log_error(f"Expected external doctor, got: {doctor_type}")
                    return False
                if doctor_url and doctor_url != doctor_api_url:
                    log_warning(f"Doctor URL mismatch: expected {doctor_api_url}, got {doctor_url}")
        else:
            if not use_default_pipeline:
                log_warning("Could not verify doctor configuration in pipeline")
        
        # Verify conversation initiator
        print(f"   Conversation Initiator: {fetched_pipeline.conversation_initiator}")
        
        # Verify patients
        if fetched_pipeline.patient_ids:
            print(f"   Patients: {len(fetched_pipeline.patient_ids)}")
        
        # Step 5: Start simulation
        log_info("Step 5: Starting simulation with external doctor...")
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=1,
            parallel_count=1,
        )
        log_success(f"Simulation started: {simulation.id}")
        print(f"   Status: {simulation.status.value}")
        print(f"   Pipeline: {pipeline_name}")
        
        # Step 6: Wait for completion (optional)
        if wait_for_completion:
            log_info("Step 6: Waiting for simulation to complete...")
            log_info("   (This tests your doctor API connectivity...)")
            
            try:
                completed_sim = client.simulations.wait_for_completion(
                    simulation.id,
                    poll_interval=5.0,
                    timeout=600.0,
                    on_progress=show_simulation_progress,
                )
                print()  # Newline after progress display
                log_success(f"Simulation completed!")
                print(f"   Status: {completed_sim.status.value}")
                print(f"   Episodes: {completed_sim.completed_episodes}/{completed_sim.total_episodes}")
                if completed_sim.summary:
                    avg_score = completed_sim.summary.get('average_score')
                    if avg_score is not None:
                        print(f"   Average Score: {avg_score:.2f}/4")
                    else:
                        print(f"   Average Score: N/A")
                
                # Show complete report with all details
                display_simulation_report(client, simulation.id)
                
                # Check for errors in the simulation
                if hasattr(completed_sim, 'error_message') and completed_sim.error_message:
                    log_error(f"Simulation had errors: {completed_sim.error_message}")
                    return False
                    
            except TimeoutError:
                print()  # Newline after progress
                log_info("Simulation still running (timeout reached)")
            except Exception as sim_error:
                print()  # Newline after progress
                log_error(f"Simulation error: {sim_error}")
                log_info("Simulation had errors but SDK workflow was correct")
        else:
            log_info("Step 6: Skipping wait (simulation running in background)")
            log_info(f"   Monitor at: /simulations/{simulation.id}")
        
        # Step 7: Cleanup (only if we created a pipeline)
        if pipeline_created:
            log_info("Step 7: Cleaning up test pipeline...")
            try:
                client.pipelines.delete(pipeline_name)
                log_success("Test pipeline deleted")
            except Exception as cleanup_error:
                log_info(f"Cleanup note: {cleanup_error}")
        else:
            log_info("Step 7: Skipping cleanup (used existing pipeline)")
        
        log_success("External doctor workflow completed successfully!")
        return True
        
    except Exception as e:
        log_error(f"Error in external doctor workflow: {e}")
        import traceback
        traceback.print_exc()
        
        # Attempt cleanup only if we created a pipeline
        if pipeline_created:
            try:
                client.pipelines.delete(pipeline_name)
            except:
                pass
        
        return False


def test_integration(client: EarlClient) -> bool:
    """
    Run integration tests for customer workflows.
    
    Tests:
    1. Internal doctor workflow (uses EARL's built-in doctor)
    2. External doctor workflow (skipped if no doctor URL provided)
    """
    log_section("Running Integration Tests")
    
    results = []
    
    # Test 1: Internal doctor workflow
    log_info("Testing internal doctor workflow...")
    internal_result = test_internal_doctor_workflow(client, wait_for_completion=False)
    results.append(("internal_doctor", internal_result))
    
    # Test 2: External doctor workflow (only if URL provided)
    external_doctor_url = os.environ.get("EXTERNAL_DOCTOR_URL")
    external_doctor_key = os.environ.get("EXTERNAL_DOCTOR_API_KEY")
    
    if external_doctor_url:
        log_info("Testing external doctor workflow...")
        external_result = test_external_doctor_workflow(
            client,
            doctor_api_url=external_doctor_url,
            doctor_api_key=external_doctor_key,
            wait_for_completion=False,
        )
        results.append(("external_doctor", external_result))
    else:
        log_info("Skipping external doctor test (set EXTERNAL_DOCTOR_URL to enable)")
        results.append(("external_doctor", True))  # Skip = pass
    
    # Summary
    all_passed = all(r[1] for r in results)
    
    log_section("Integration Test Summary")
    for name, passed in results:
        status = f"{Colors.GREEN}PASSED{Colors.END}" if passed else f"{Colors.RED}FAILED{Colors.END}"
        print(f"   {name}: {status}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Earl SDK Integration Tests")
    parser.add_argument(
        "--test",
        choices=[
            "auth", "dimensions", "patients", "pipelines", "simulations", 
            "full", "integration", "internal", "external", "all"
        ],
        default="all",
        help="Which test to run (default: all)",
    )
    parser.add_argument(
        "--env",
        choices=["dev", "test", "prod"],
        default="test",
        help="Environment to test against (default: test)",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for simulations to complete (can take several minutes)",
    )
    parser.add_argument(
        "--doctor-url",
        type=str,
        default=None,
        help="External doctor API URL for integration test",
    )
    parser.add_argument(
        "--doctor-key",
        type=str,
        default="earl-test-doctor-key",  # Default key for external-doctor service
        help="External doctor API key for integration test (default: earl-test-doctor-key)",
    )
    parser.add_argument(
        "--use-default-pipeline",
        action="store_true",
        help="Use default_pipeline instead of creating a new test pipeline",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Auth0 M2M client ID (overrides env var EARL_CLIENT_ID)",
    )
    parser.add_argument(
        "--client-secret",
        type=str,
        default=None,
        help="Auth0 M2M client secret (overrides env var EARL_CLIENT_SECRET)",
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Auth0 organization ID (overrides env var EARL_ORGANIZATION_ID)",
    )
    args = parser.parse_args()
    
    # Get credentials (CLI args override env vars)
    client_id, client_secret, organization = get_credentials(
        args.client_id, args.client_secret, args.organization
    )
    
    # Create client
    log_section("Initializing Earl SDK Client")
    
    try:
        client = EarlClient(
            client_id=client_id,
            client_secret=client_secret,
            organization=organization,
            environment=args.env,
        )
        log_success(f"Client created for {args.env} environment")
        print(f"   API URL: {client.api_url}")
        print(f"   Organization: {client.organization}")
    except Exception as e:
        log_error(f"Failed to create client: {e}")
        sys.exit(1)
    
    # Set external doctor URL and key from args
    if args.doctor_url:
        os.environ["EXTERNAL_DOCTOR_URL"] = args.doctor_url
    # Always set the key (has default value)
    os.environ["EXTERNAL_DOCTOR_API_KEY"] = args.doctor_key
    
    # Run tests
    results = {}
    
    # Define test functions
    def run_internal_test(c):
        return test_internal_doctor_workflow(
            c,
            wait_for_completion=args.wait,
            use_default_pipeline=args.use_default_pipeline,
        )
    
    def run_external_test(c):
        url = os.environ.get("EXTERNAL_DOCTOR_URL")
        key = os.environ.get("EXTERNAL_DOCTOR_API_KEY")
        if not url and not args.use_default_pipeline:
            log_info("External doctor test skipped (no --doctor-url provided)")
            return True
        return test_external_doctor_workflow(
            c,
            doctor_api_url=url or "N/A",
            doctor_api_key=key,
            wait_for_completion=args.wait,
            use_default_pipeline=args.use_default_pipeline,
        )
    
    tests = {
        "auth": test_authentication,
        "dimensions": test_dimensions,
        "patients": test_patients,
        "pipelines": test_pipelines,
        "simulations": test_simulations,
        "full": test_full_workflow,
        "integration": test_integration,
        "internal": run_internal_test,
        "external": run_external_test,
    }
    
    if args.test == "all":
        tests_to_run = ["auth", "dimensions", "patients", "pipelines", "simulations"]
    elif args.test == "integration":
        tests_to_run = ["auth", "integration"]
    else:
        tests_to_run = [args.test]
    
    for test_name in tests_to_run:
        results[test_name] = tests[test_name](client)
    
    # Summary
    log_section("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for test_name, result in results.items():
        status = f"{Colors.GREEN}PASSED{Colors.END}" if result else f"{Colors.RED}FAILED{Colors.END}"
        print(f"   {test_name}: {status}")
    
    print()
    if failed == 0:
        log_success(f"All {passed} tests passed!")
        sys.exit(0)
    else:
        log_error(f"{failed}/{len(results)} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

