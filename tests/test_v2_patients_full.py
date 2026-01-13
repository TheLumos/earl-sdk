#!/usr/bin/env python3
"""
Earl SDK - V2 Patients Full Integration Test with Dialogue Export

Runs simulations with V2 patients using Earl's internal doctor,
waits for completion, and exports full dialogues, scores, and summaries.

All sensitive parameters are configurable via CLI or environment variables.

Usage:
    # Using environment variables
    export EARL_CLIENT_ID="your_client_id"
    export EARL_CLIENT_SECRET="your_client_secret"
    python3 test_v2_patients_full.py --env test

    # Using CLI parameters
    python3 test_v2_patients_full.py --env test \\
        --client-id "your_id" \\
        --client-secret "your_secret"

    # Test specific subset
    python3 test_v2_patients_full.py --env test --subset anxiety

    # Custom output file
    python3 test_v2_patients_full.py --env test --output results.json
"""

import os
import sys
import argparse
import time
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

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
    },
    "Anxiety_Medication_Reconciliation": {
        "name": "Olen Hills",
        "age": 66,
        "condition": "Anxiety",
        "encounter_type": "telehealth",
    },
    "Anxiety_Pre_Visit_Intake_History": {
        "name": "Olen Hills",
        "age": 66,
        "condition": "Anxiety",
        "encounter_type": "telehealth",
    },
    
    # Anxiety Patients - Shemeka Gutmann (25 years old, impaired)
    "Anxiety_Impaired_Focused_Clinical_Encounter": {
        "name": "Shemeka Gutmann",
        "age": 25,
        "condition": "Anxiety (Impaired)",
        "encounter_type": "telehealth",
    },
    "Anxiety_Impaired_Pre_Visit_Intake_History": {
        "name": "Shemeka Gutmann",
        "age": 25,
        "condition": "Wellness Check (Anxiety Impaired)",
        "encounter_type": "medical screening",
    },
    "Anxiety_Impaired_Schedule_Appointment": {
        "name": "Shemeka Gutmann",
        "age": 25,
        "condition": "Anxiety (Impaired)",
        "encounter_type": "telehealth",
    },
    
    # Asthma Patients - Darleen Zulauf (54 years old)
    "Asthma_Chronic_Symptom_Monitoring": {
        "name": "Darleen Zulauf",
        "age": 54,
        "condition": "Asthma",
        "encounter_type": "telehealth",
    },
    "Asthma_Focused_Clinical_Encounter": {
        "name": "Darleen Zulauf",
        "age": 54,
        "condition": "Asthma",
        "encounter_type": "telehealth",
    },
    "Asthma_Medication_Adherence": {
        "name": "Darleen Zulauf",
        "age": 54,
        "condition": "Asthma",
        "encounter_type": "telehealth",
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


# =============================================================================
# COLORED OUTPUT
# =============================================================================
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


def log_success(msg): print(f"{Colors.GREEN}✓ {msg}{Colors.END}")
def log_error(msg): print(f"{Colors.RED}✗ {msg}{Colors.END}")
def log_info(msg): print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")
def log_warning(msg): print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")
def log_highlight(msg): print(f"{Colors.CYAN}{Colors.BOLD}{msg}{Colors.END}")
def log_section(title): print(f"\n{Colors.BOLD}{'='*70}\n{title}\n{'='*70}{Colors.END}")
def log_subsection(title): print(f"\n{Colors.MAGENTA}{Colors.BOLD}--- {title} ---{Colors.END}")


def get_credentials(cli_client_id=None, cli_client_secret=None, cli_organization=None):
    """Get credentials from CLI args or environment."""
    client_id = cli_client_id or os.environ.get("EARL_CLIENT_ID", "")
    client_secret = cli_client_secret or os.environ.get("EARL_CLIENT_SECRET", "")
    organization = cli_organization or os.environ.get("EARL_ORGANIZATION", "")
    
    if not client_id or not client_secret:
        log_error("Missing credentials!")
        print("\nProvide credentials via:")
        print("  1. Environment variables: EARL_CLIENT_ID, EARL_CLIENT_SECRET")
        print("  2. CLI args: --client-id, --client-secret")
        sys.exit(1)
    
    return client_id, client_secret, organization


def format_dialogue(dialogue: List[Dict[str, str]], indent: int = 4) -> str:
    """Format dialogue history for display."""
    if not dialogue:
        return f"{' ' * indent}(No dialogue)"
    
    lines = []
    prefix = " " * indent
    for msg in dialogue:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        
        # Color by role
        if role == "DOCTOR":
            role_color = Colors.CYAN
        elif role == "PATIENT":
            role_color = Colors.GREEN
        else:
            role_color = Colors.DIM
        
        lines.append(f"{prefix}{role_color}[{role}]{Colors.END}")
        # Wrap long lines
        for line in content.split("\n"):
            lines.append(f"{prefix}  {line}")
        lines.append("")
    
    return "\n".join(lines)


def run_v2_patients_test(
    client: EarlClient,
    patient_ids: List[str],
    max_turns: int = 20,
    timeout: int = 900,
    output_file: Optional[str] = None,
    show_dialogues: bool = True,
) -> Dict[str, Any]:
    """
    Run V2 patients test with full dialogue capture.
    
    Args:
        client: Earl SDK client
        patient_ids: List of patient IDs to test
        max_turns: Maximum conversation turns
        timeout: Timeout in seconds
        output_file: Optional file to save results (JSON)
        show_dialogues: Print dialogues to console
    
    Returns:
        Full results dictionary
    """
    log_section(f"V2 Patients Test ({len(patient_ids)} patients)")
    
    timestamp = datetime.now().isoformat()
    pipeline_name = f"v2-test-{int(time.time())}"
    
    results = {
        "test_info": {
            "timestamp": timestamp,
            "patients_count": len(patient_ids),
            "max_turns": max_turns,
            "timeout": timeout,
            "pipeline_name": pipeline_name,
        },
        "simulation_id": None,
        "status": "pending",
        "episodes": [],
        "summary": {
            "total_episodes": len(patient_ids),
            "completed": 0,
            "failed": 0,
            "average_score": None,
            "dimension_averages": {},
        },
        "error": None,
    }
    
    pipeline_created = False
    
    try:
        # Show patient info
        log_info("Patients to simulate:")
        for pid in patient_ids:
            info = V2_PATIENTS.get(pid, {})
            name = info.get("name", "Unknown")
            condition = info.get("condition", "Unknown")
            print(f"   • {pid}")
            print(f"     {name} ({info.get('age', '?')} yo) - {condition}")
        
        # Get dimensions
        log_info("Fetching available dimensions...")
        dimensions = client.dimensions.list()
        dimension_ids = [d.id for d in dimensions]
        log_success(f"Found {len(dimension_ids)} dimensions")
        results["test_info"]["dimensions"] = dimension_ids
        
        # Create pipeline
        log_info(f"Creating pipeline: {pipeline_name}")
        log_info(f"   Doctor: Internal (Gemini 3 Pro)")
        log_info(f"   Max turns: {max_turns}")
        
        pipeline = client.pipelines.create(
            name=pipeline_name,
            doctor_config=DoctorApiConfig.internal(),
            patient_ids=patient_ids,
            dimension_ids=dimension_ids,
            description=f"V2 Patients Full Test - {timestamp}",
            conversation_initiator="doctor",
            max_turns=max_turns,
        )
        pipeline_created = True
        log_success(f"Pipeline created: {pipeline.name}")
        
        # Start simulation
        num_episodes = len(patient_ids)
        log_info(f"Starting simulation with {num_episodes} episodes...")
        
        simulation = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=num_episodes,
            parallel_count=min(3, num_episodes),
        )
        results["simulation_id"] = simulation.id
        log_success(f"Simulation started: {simulation.id}")
        
        # Wait for completion
        log_info(f"Waiting for completion (timeout: {timeout}s)...")
        start_time = time.time()
        last_status = ""
        
        while time.time() - start_time < timeout:
            sim = client.simulations.get(simulation.id)
            
            # Get episode progress
            try:
                episodes = client.simulations.get_episodes(simulation.id)
                completed = sum(1 for e in episodes if e.get("status") in ["completed", "judged"])
                running = sum(1 for e in episodes if e.get("status") in ["running", "awaiting_doctor_response"])
                failed = sum(1 for e in episodes if e.get("status") == "failed")
            except:
                completed, running, failed = 0, 0, 0
            
            status_msg = f"{sim.status.value} | {completed}/{num_episodes} done, {running} running, {failed} failed"
            
            if status_msg != last_status:
                elapsed = int(time.time() - start_time)
                print(f"\r   [{elapsed:3d}s] {status_msg}          ", end="", flush=True)
                last_status = status_msg
            
            if sim.status.value in ["completed", "failed"]:
                break
            
            time.sleep(5)
        
        print()
        
        # Get final simulation status
        final_sim = client.simulations.get(simulation.id)
        results["status"] = final_sim.status.value
        
        if final_sim.status.value == "completed":
            log_success("Simulation completed!")
        elif final_sim.status.value == "failed":
            log_error(f"Simulation failed")
            results["error"] = getattr(final_sim, 'error', 'Unknown error')
        else:
            log_warning(f"Simulation timed out: {final_sim.status.value}")
        
        # Collect detailed episode results
        log_section("Episode Results")
        
        episodes = client.simulations.get_episodes(simulation.id)
        all_scores = []
        dimension_scores = {}
        
        for ep in episodes:
            episode_id = ep.get("episode_id")
            patient_id = ep.get("patient_id", "Unknown")
            status = ep.get("status", "unknown")
            
            patient_info = V2_PATIENTS.get(patient_id, {})
            
            episode_data = {
                "episode_id": episode_id,
                "patient_id": patient_id,
                "patient_name": patient_info.get("name", "Unknown"),
                "patient_condition": patient_info.get("condition", "Unknown"),
                "status": status,
                "dialogue_history": [],
                "turn_count": 0,
                "scores": {},
                "average_score": None,
                "dimension_results": [],
            }
            
            # Get full episode details
            try:
                full_ep = client.simulations.get_episode(simulation.id, episode_id)
                
                dialogue = full_ep.get("dialogue_history", [])
                episode_data["dialogue_history"] = dialogue
                episode_data["turn_count"] = sum(1 for m in dialogue if m.get("role") == "doctor")
                
                scores = full_ep.get("scores", {})
                episode_data["scores"] = scores
                
                avg_score = full_ep.get("average_score")
                episode_data["average_score"] = avg_score
                
                # Collect dimension results if available
                dim_results = full_ep.get("dimension_results", [])
                episode_data["dimension_results"] = dim_results
                
            except Exception as e:
                log_warning(f"Could not get episode details for {episode_id}: {e}")
            
            results["episodes"].append(episode_data)
            
            # Track for summary
            if status in ["completed", "judged"]:
                results["summary"]["completed"] += 1
                if episode_data["average_score"] is not None:
                    all_scores.append(episode_data["average_score"])
                for dim, score in episode_data["scores"].items():
                    if dim not in dimension_scores:
                        dimension_scores[dim] = []
                    dimension_scores[dim].append(score)
            elif status == "failed":
                results["summary"]["failed"] += 1
            
            # Display episode
            status_icon = "✓" if status in ["completed", "judged"] else "✗" if status == "failed" else "⋯"
            status_color = Colors.GREEN if status in ["completed", "judged"] else Colors.RED if status == "failed" else Colors.YELLOW
            
            log_subsection(f"{status_color}{status_icon}{Colors.END} {patient_id}")
            print(f"   Patient: {episode_data['patient_name']} ({patient_info.get('age', '?')} yo)")
            print(f"   Condition: {episode_data['patient_condition']}")
            print(f"   Status: {status}")
            print(f"   Turns: {episode_data['turn_count']}")
            
            if episode_data["scores"]:
                print(f"   Scores:")
                for dim, score in episode_data["scores"].items():
                    bar = "█" * int(score) + "░" * (4 - int(score))
                    print(f"      {dim}: {bar} {score:.2f}/4")
            
            if episode_data["average_score"] is not None:
                avg = episode_data["average_score"]
                bar = "█" * int(avg) + "░" * (4 - int(avg))
                print(f"   {Colors.BOLD}Average: {bar} {avg:.2f}/4{Colors.END}")
            
            # Show dialogue
            if show_dialogues and episode_data["dialogue_history"]:
                print(f"\n   {Colors.DIM}Dialogue:{Colors.END}")
                print(format_dialogue(episode_data["dialogue_history"], indent=6))
        
        # Calculate summary
        if all_scores:
            results["summary"]["average_score"] = sum(all_scores) / len(all_scores)
        
        for dim, scores in dimension_scores.items():
            results["summary"]["dimension_averages"][dim] = sum(scores) / len(scores)
        
        # Print summary
        log_section("Summary")
        
        print(f"   Simulation ID: {results['simulation_id']}")
        print(f"   Status: {results['status']}")
        print(f"   Episodes: {results['summary']['completed']}/{results['summary']['total_episodes']} completed")
        
        if results["summary"]["failed"] > 0:
            print(f"   Failed: {results['summary']['failed']}")
        
        if results["summary"]["average_score"] is not None:
            avg = results["summary"]["average_score"]
            log_highlight(f"\n   Overall Average Score: {avg:.2f}/4")
        
        if results["summary"]["dimension_averages"]:
            print(f"\n   Dimension Averages:")
            for dim, avg in sorted(results["summary"]["dimension_averages"].items()):
                bar = "█" * int(avg) + "░" * (4 - int(avg))
                print(f"      {dim}: {bar} {avg:.2f}/4")
        
        # Save to file
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            log_success(f"Results saved to: {output_file}")
            
            # Also save markdown report
            md_file = output_file.rsplit(".", 1)[0] + ".md"
            save_markdown_report(results, md_file)
            log_success(f"Markdown report saved to: {md_file}")
        
        return results
        
    except Exception as e:
        log_error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        results["status"] = "error"
        results["error"] = str(e)
        return results
        
    finally:
        # Cleanup pipeline
        if pipeline_created:
            try:
                client.pipelines.delete(pipeline_name)
                log_info(f"Cleaned up pipeline: {pipeline_name}")
            except:
                pass


def save_markdown_report(results: Dict[str, Any], filepath: str):
    """Save results as a markdown report."""
    lines = []
    
    # Header
    lines.append("# V2 Patients Test Report")
    lines.append("")
    lines.append(f"**Date:** {results['test_info']['timestamp']}")
    lines.append(f"**Simulation ID:** `{results['simulation_id']}`")
    lines.append(f"**Status:** {results['status']}")
    lines.append("")
    
    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Episodes | {results['summary']['total_episodes']} |")
    lines.append(f"| Completed | {results['summary']['completed']} |")
    lines.append(f"| Failed | {results['summary']['failed']} |")
    if results["summary"]["average_score"] is not None:
        lines.append(f"| **Average Score** | **{results['summary']['average_score']:.2f}/4** |")
    lines.append("")
    
    # Dimension averages
    if results["summary"]["dimension_averages"]:
        lines.append("### Dimension Averages")
        lines.append("")
        lines.append("| Dimension | Score |")
        lines.append("|-----------|-------|")
        for dim, avg in sorted(results["summary"]["dimension_averages"].items()):
            lines.append(f"| {dim} | {avg:.2f}/4 |")
        lines.append("")
    
    # Episode details
    lines.append("## Episode Details")
    lines.append("")
    
    for ep in results["episodes"]:
        status_emoji = "✅" if ep["status"] in ["completed", "judged"] else "❌" if ep["status"] == "failed" else "⏳"
        
        lines.append(f"### {status_emoji} {ep['patient_id']}")
        lines.append("")
        lines.append(f"- **Patient:** {ep['patient_name']}")
        lines.append(f"- **Condition:** {ep['patient_condition']}")
        lines.append(f"- **Status:** {ep['status']}")
        lines.append(f"- **Turns:** {ep['turn_count']}")
        
        if ep["average_score"] is not None:
            lines.append(f"- **Average Score:** {ep['average_score']:.2f}/4")
        
        if ep["scores"]:
            lines.append("")
            lines.append("**Scores:**")
            lines.append("")
            for dim, score in ep["scores"].items():
                lines.append(f"- {dim}: {score:.2f}/4")
        
        if ep["dialogue_history"]:
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>📝 Dialogue History</summary>")
            lines.append("")
            lines.append("```")
            for msg in ep["dialogue_history"]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                lines.append(f"[{role}]")
                lines.append(content)
                lines.append("")
            lines.append("```")
            lines.append("")
            lines.append("</details>")
        
        lines.append("")
    
    with open(filepath, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Earl SDK - V2 Patients Full Test with Dialogue Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using environment variables
  export EARL_CLIENT_ID="your_id"
  export EARL_CLIENT_SECRET="your_secret"
  python3 test_v2_patients_full.py --env test

  # Using CLI parameters
  python3 test_v2_patients_full.py --env test --client-id "id" --client-secret "secret"

  # Test specific subset
  python3 test_v2_patients_full.py --env test --subset anxiety

  # Save results to file
  python3 test_v2_patients_full.py --env test --output my_results.json
        """,
    )
    
    # Environment
    parser.add_argument(
        "--env", 
        choices=["dev", "test", "prod"], 
        default="test",
        help="Environment to test against (default: test)"
    )
    
    # Credentials (optional - can use env vars)
    parser.add_argument(
        "--client-id",
        help="Earl client ID (or set EARL_CLIENT_ID env var)"
    )
    parser.add_argument(
        "--client-secret", 
        help="Earl client secret (or set EARL_CLIENT_SECRET env var)"
    )
    parser.add_argument(
        "--organization",
        help="Earl organization (or set EARL_ORGANIZATION env var)"
    )
    
    # Test options
    parser.add_argument(
        "--subset",
        choices=["all", "anxiety", "asthma"],
        default="all",
        help="Which patient subset to test (default: all)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum conversation turns (default: 20)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds (default: 900)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON format, also creates .md)"
    )
    parser.add_argument(
        "--no-dialogues",
        action="store_true",
        help="Don't print dialogues to console (still saved to file)"
    )
    
    args = parser.parse_args()
    
    # Get credentials
    client_id, client_secret, organization = get_credentials(
        args.client_id, args.client_secret, args.organization
    )
    
    # Select patients
    if args.subset == "anxiety":
        patient_ids = ANXIETY_PATIENTS
    elif args.subset == "asthma":
        patient_ids = ASTHMA_PATIENTS
    else:
        patient_ids = ALL_V2_PATIENTS
    
    # Default output file
    if args.output is None:
        args.output = f"v2_test_results_{int(time.time())}.json"
    
    # Header
    log_section("Earl SDK - V2 Patients Full Test")
    log_info(f"Environment: {args.env}")
    log_info(f"Patient subset: {args.subset} ({len(patient_ids)} patients)")
    log_info(f"Max turns: {args.max_turns}")
    log_info(f"Timeout: {args.timeout}s")
    log_info(f"Output: {args.output}")
    
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
    results = run_v2_patients_test(
        client,
        patient_ids=patient_ids,
        max_turns=args.max_turns,
        timeout=args.timeout,
        output_file=args.output,
        show_dialogues=not args.no_dialogues,
    )
    
    # Final status
    log_section("FINAL RESULT")
    
    if results["status"] == "completed" and results["summary"]["completed"] > 0:
        log_success("TEST COMPLETED SUCCESSFULLY")
        if results["summary"]["average_score"] is not None:
            log_highlight(f"Average Score: {results['summary']['average_score']:.2f}/4")
        log_info(f"Results saved to: {args.output}")
        sys.exit(0)
    else:
        log_error("TEST FAILED")
        if results["error"]:
            log_error(f"Error: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
