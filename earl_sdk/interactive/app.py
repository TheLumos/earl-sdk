"""Main application loop for EARL Interactive UI.

Displays a branded banner, initializes the SDK client from stored credentials,
and presents the top-level menu dispatching to each flow.

Launch with: ``earl-ui`` or ``python -m earl_sdk.interactive``
"""

from __future__ import annotations

import sys

from .ui import banner, console, error, muted, select_one, success, warn
from .storage.config_store import ConfigStore
from .storage.run_store import RunStore
from .flows.config import build_client_from_store, flow_config


def main() -> None:
    """Main interactive loop."""
    store = ConfigStore()
    run_store = RunStore()
    client_ref = build_client_from_store(store)

    _print_banner(store, client_ref)

    while True:
        action = select_one("Main Menu", [
            ("chat",    "Chat with Patient      — be the doctor: live conversation with a simulated patient, then get judged"),
            ("run",     "Run Simulation         — evaluate a doctor against simulated patients and get scored results"),
            ("browse",  "Browse Simulations     — inspect past runs: episodes, dialogues, judge scores, and reports"),
            ("compare", "Compare Runs           — side-by-side delta view of 2-5 simulations across all dimensions"),
            ("explore", "Explore Catalog        — browse available dimensions, patients, and pipelines on the platform"),
            ("config",  "Configuration          — manage auth credentials, doctor API endpoints, and preferences"),
            ("exit",    "Exit"),
        ], allow_back=False)

        if action == "chat":
            client = _require_client(client_ref)
            if client:
                from .flows.chat import flow_chat
                flow_chat(client, store, run_store)
        elif action == "run":
            client = _require_client(client_ref)
            if client:
                from .flows.run import flow_run
                flow_run(client, store, run_store)
        elif action == "browse":
            client = _require_client(client_ref)
            if client:
                from .flows.browse import flow_browse
                flow_browse(client, run_store)
        elif action == "compare":
            client = _require_client(client_ref)
            if client:
                from .flows.compare import flow_compare
                flow_compare(client, run_store)
        elif action == "explore":
            client = _require_client(client_ref)
            if client:
                from .flows.explore import flow_explore
                flow_explore(client)
        elif action == "config":
            flow_config(store, client_ref)
            # Re-print banner after config changes
            _print_banner(store, client_ref)
        elif action == "exit" or action is None:
            muted("Goodbye.")
            break


def _print_banner(store: ConfigStore, client_ref: list) -> None:
    """Display branded header with connection status."""
    profile = store.get_active_profile()
    status_lines = []

    if profile:
        env_label = {"dev": "development", "test": "staging", "prod": "production"}.get(
            profile.environment, profile.environment
        )
        status_lines.append(f"[bold]Profile:[/]     {profile.name} ({env_label})")
        if profile.organization:
            status_lines.append(f"[bold]Org:[/]         {profile.organization}")

        if client_ref:
            status_lines.append(f"[bold]Status:[/]      [green]Connected ✓[/]")
        else:
            status_lines.append(f"[bold]Status:[/]      [yellow]Not connected — check credentials[/]")
    else:
        status_lines.append("[yellow]No auth profile configured — go to Configuration to add one[/]")

    # Count local runs
    runs = run_store_count()
    if runs > 0:
        status_lines.append(f"[bold]Local Runs:[/]  {runs} saved")

    banner(
        "EARL — Medical AI Evaluation Platform",
        "Interactive terminal UI for the EARL SDK",
        status_lines,
    )


def run_store_count() -> int:
    """Quick count of local runs without loading full run store."""
    from .storage.run_store import RUNS_DIR
    idx_path = RUNS_DIR / "index.json"
    if not idx_path.exists():
        return 0
    try:
        import json
        data = json.loads(idx_path.read_text())
        return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0


def _require_client(client_ref: list):
    """Return the active client or print error."""
    if client_ref:
        return client_ref[0]
    error("No active profile — go to Configuration → Auth Profiles to add credentials.")
    return None
