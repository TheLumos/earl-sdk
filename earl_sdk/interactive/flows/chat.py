"""Chat with Patient flow — be the doctor in a live conversation.

Launch a client-driven simulation with a single patient, then type
doctor responses in real time.  The patient is powered by EARL's
generative patient engine.  After the conversation, the judge scores
your performance across all selected dimensions.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

_log = logging.getLogger("earl.ui")

from rich.columns import Columns
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from ..ui import (
    ask_confirm,
    ask_int,
    ask_text,
    console,
    datatable,
    error,
    info_panel,
    muted,
    score_text,
    select_many,
    select_one,
    select_with_preview,
    success,
    warn,
)
from ..storage.config_store import ConfigStore
from ..storage.run_store import LocalRun, RunStore


# ── Main entry ────────────────────────────────────────────────────────────────


def flow_chat(client, store: ConfigStore, run_store: RunStore) -> None:
    """Full chat flow: pick patient → show details → converse → judge results."""

    console.print("\n[bold]Chat with Patient[/]")
    muted("You are the doctor.  A simulated patient will describe their symptoms.")
    muted("Type your responses.  After the conversation the judge scores you.\n")

    # ── 1. Select patient ─────────────────────────────────────────────────

    patient = _pick_patient(client)
    if not patient:
        return

    # ── 2. Show patient insight card ──────────────────────────────────────

    _render_patient_card(patient)

    # ── 3. Select evaluation dimensions ───────────────────────────────────

    dims = _pick_dimensions(client)
    if not dims:
        return

    # ── 4. Conversation settings ──────────────────────────────────────────

    max_turns = ask_int(
        "Max conversation turns",
        default=10, min_val=3, max_val=50,
    )
    if max_turns is None:
        return

    initiator = select_one("Who speaks first?", [
        ("patient", "Patient opens         — patient describes symptoms, you respond (typical flow)"),
        ("doctor",  "You open              — you greet the patient and lead the conversation"),
    ], allow_back=False)
    if not initiator:
        return

    # ── 5. Create pipeline + simulation ───────────────────────────────────

    if not ask_confirm("Start conversation?"):
        muted("Cancelled.")
        return

    console.print()
    pipeline_name, sim_id, episode_id = _setup_simulation(
        client, patient, dims, max_turns, initiator,
    )
    if not sim_id:
        return

    # ── 6. Conversation loop ──────────────────────────────────────────────

    end_reason = _conversation_loop(client, sim_id, episode_id, initiator, max_turns)

    # ── 7. Handle conversation ending ─────────────────────────────────────

    _handle_ending(client, sim_id, episode_id, end_reason, pipeline_name, store, run_store)

    # ── 8. Cleanup pipeline ───────────────────────────────────────────────

    try:
        client.pipelines.delete(pipeline_name)
    except Exception as e:
        _log.debug("Pipeline cleanup failed (best-effort): %s", e, exc_info=True)


# ── Patient selection ─────────────────────────────────────────────────────────


def _pick_patient(client):
    """Let user browse and pick one patient with live detail preview."""
    from .explore import _format_patient_preview

    console.print("  [dim]Loading patients...[/]")
    try:
        patients = client.patients.list(limit=200)
    except Exception as e:
        error(f"Failed to load patients: {e}")
        return None

    if not patients:
        warn("No patients found.")
        return None

    # Pre-fetch full details for preview panel
    console.print(f"  [dim]Loading patient details ({len(patients)})...[/]")
    details_map: dict[str, object] = {}
    for p in patients:
        try:
            details_map[p.id] = client.patients.get(p.id)
        except Exception:
            details_map[p.id] = p

    items: list[tuple[str, str]] = []
    previews: dict[str, str] = {}
    for p in patients:
        primary = p.condition or (p.conditions[0] if p.conditions else "")
        label = f"{p.name}"
        if p.age:
            label += f"  {p.age}y"
        if p.gender:
            label += f" {p.gender[0].upper()}"
        if primary:
            label += f"  — {primary}"
        items.append((p.id, label))
        previews[p.id] = _format_patient_preview(details_map.get(p.id, p))

    pick = select_with_preview(
        f"Select patient ({len(patients)} available)",
        items,
        previews,
        multi=False,
    )
    if not pick:
        return None

    # Return the full detail object
    detail = details_map.get(pick)
    if detail:
        return detail
    # Fallback
    return next((p for p in patients if p.id == pick), None)


# ── Patient card ──────────────────────────────────────────────────────────────


def _render_patient_card(patient) -> None:
    """Show a rich, detailed patient information card."""

    # Left column: demographics + clinical
    left_lines: list[str] = []
    left_lines.append(f"[bold cyan]{patient.name}[/]")
    if patient.age or patient.gender:
        parts = []
        if patient.age:
            parts.append(f"{patient.age}yo")
        if patient.gender:
            parts.append(patient.gender.capitalize())
        left_lines.append("  ".join(parts))
    if patient.occupation:
        left_lines.append(f"[dim]Occupation:[/]  {patient.occupation}")
    if patient.difficulty:
        diff_color = {"easy": "green", "medium": "yellow", "hard": "red"}.get(patient.difficulty, "dim")
        left_lines.append(f"[dim]Difficulty:[/]  [{diff_color}]{patient.difficulty}[/]")
    if patient.condition:
        left_lines.append(f"[dim]Condition:[/]   {patient.condition}")
    if patient.conditions:
        left_lines.append(f"[dim]Conditions:[/]  {', '.join(patient.conditions)}")
    if patient.medical_speciality:
        left_lines.append(f"[dim]Specialty:[/]   {patient.medical_speciality}")
    if patient.task_display or patient.task:
        left_lines.append(f"[dim]Task:[/]        {patient.task_display or patient.task}")

    # Medical details
    if patient.medical_history:
        left_lines.append("")
        left_lines.append("[bold]Medical History[/]")
        for item in patient.medical_history[:5]:
            left_lines.append(f"  • {item}")
    if patient.current_medications:
        left_lines.append("")
        left_lines.append("[bold]Current Medications[/]")
        for med in patient.current_medications[:5]:
            left_lines.append(f"  • {med}")
    if patient.allergies:
        left_lines.append("")
        left_lines.append("[bold]Allergies[/]")
        for allergy in patient.allergies:
            left_lines.append(f"  • {allergy}")

    left_panel = Panel(
        "\n".join(left_lines),
        title="[bold cyan]Patient Profile[/]",
        border_style="cyan",
        expand=True,
        padding=(1, 2),
    )

    # Right column: encounter details + personality
    right_lines: list[str] = []

    if patient.chief_complaint or patient.reason:
        right_lines.append("[bold yellow]Chief Complaint[/]")
        right_lines.append(f"  {patient.chief_complaint or patient.reason}")
        right_lines.append("")

    if patient.clinical_vignette or patient.vignette:
        vignette = patient.clinical_vignette or patient.vignette
        right_lines.append("[bold]Clinical Vignette[/]")
        right_lines.append(f"  {vignette[:300]}{'...' if len(vignette) > 300 else ''}")
        right_lines.append("")

    if patient.doctor_goal:
        right_lines.append("[bold green]Your Goal (as Doctor)[/]")
        right_lines.append(f"  {patient.doctor_goal}")
        right_lines.append("")

    if patient.patient_goal:
        right_lines.append("[bold]Patient's Goal[/]")
        right_lines.append(f"  {patient.patient_goal}")
        right_lines.append("")

    if patient.personality:
        right_lines.append("[bold]Personality[/]")
        right_lines.append(f"  {patient.personality[:200]}")
        right_lines.append("")

    if patient.reactive_traits:
        right_lines.append("[bold]Reactive Traits[/]")
        right_lines.append(f"  {patient.reactive_traits[:200]}")
        right_lines.append("")

    if patient.scenario:
        right_lines.append("[bold]Scenario[/]")
        right_lines.append(f"  {patient.scenario[:300]}")

    if patient.life_narrative:
        right_lines.append("")
        right_lines.append("[bold]Life Narrative[/]")
        right_lines.append(f"  {patient.life_narrative[:250]}{'...' if len(str(patient.life_narrative)) > 250 else ''}")

    # Goals evidence (what the judge looks for)
    if patient.goals_evidence:
        right_lines.append("")
        right_lines.append("[bold magenta]Evaluation Criteria[/]")
        for ge in patient.goals_evidence[:5]:
            if isinstance(ge, dict):
                name = ge.get("name", ge.get("goal", ""))
                desc = ge.get("description", ge.get("evidence", ""))
                if name:
                    right_lines.append(f"  • [bold]{name}[/]: {desc[:100]}" if desc else f"  • {name}")

    if not right_lines:
        right_lines.append("[dim]No additional encounter details available.[/]")

    right_panel = Panel(
        "\n".join(right_lines),
        title="[bold yellow]Encounter Details[/]",
        border_style="yellow",
        expand=True,
        padding=(1, 2),
    )

    console.print(Columns([left_panel, right_panel], equal=True, expand=True))

    # Behaviors (compact, below)
    if patient.behaviors and isinstance(patient.behaviors, dict):
        beh_parts = []
        for k, v in patient.behaviors.items():
            if isinstance(v, bool):
                beh_parts.append(f"{k}={'[green]yes[/]' if v else '[dim]no[/]'}")
            elif isinstance(v, (int, float)):
                beh_parts.append(f"{k}={v}")
            elif isinstance(v, str) and v:
                beh_parts.append(f"{k}={v[:30]}")
        if beh_parts:
            console.print(Panel(
                "  ".join(beh_parts),
                title="[dim]Behavioral Traits[/]",
                border_style="#555555",
                padding=(0, 2),
            ))


# ── Dimension selection ───────────────────────────────────────────────────────


def _pick_dimensions(client) -> list[str] | None:
    """Select evaluation dimensions for judging."""
    console.print("  [dim]Loading dimensions...[/]")
    try:
        dims = client.dimensions.list()
    except Exception as e:
        error(f"Failed to load dimensions: {e}")
        return None

    if not dims:
        error("No dimensions available.")
        return None

    choices = [(d.id, f"{d.name}  [{d.category}]") for d in dims]
    selected = select_many("Select dimensions to judge on (space to toggle)", choices)
    if not selected:
        warn("No dimensions selected — cancelled.")
        return None
    return selected


# ── Simulation setup ──────────────────────────────────────────────────────────


def _setup_simulation(
    client, patient, dim_ids: list[str], max_turns: int, initiator: str,
) -> tuple[str, str | None, str | None]:
    """Create a temporary client_driven pipeline + 1-episode simulation."""
    import uuid

    pipeline_name = f"chat-{uuid.uuid4().hex[:8]}"

    console.print("  [dim]Setting up conversation...[/]")
    try:
        from earl_sdk import DoctorApiConfig
        pipeline = client.pipelines.create(
            name=pipeline_name,
            dimension_ids=dim_ids,
            patient_ids=[patient.id],
            doctor_config=DoctorApiConfig.client_driven(),
            conversation_initiator=initiator,
            max_turns=max_turns,
            validate_doctor=False,
        )
    except Exception as e:
        error(f"Failed to create pipeline: {e}")
        return pipeline_name, None, None

    try:
        sim = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=1,
        )
        success(f"Simulation {sim.id[:8]} started")
    except Exception as e:
        error(f"Failed to start simulation: {e}")
        try:
            client.pipelines.delete(pipeline_name)
        except Exception as e:
            _log.debug("Pipeline cleanup after simulation failure: %s", e, exc_info=True)
        return pipeline_name, None, None

    # Wait for episode to become ready
    console.print("  [dim]Waiting for patient to connect...[/]")
    episode_id = None
    for _ in range(30):
        try:
            episodes = client.simulations.get_episodes(sim.id)
            if episodes:
                ep = episodes[0]
                episode_id = ep.get("episode_id")
                status = ep.get("status", "")
                if status in ("awaiting_doctor", "conversation", "dialogue"):
                    break
                if status in ("failed", "completed"):
                    error(f"Episode ended unexpectedly: {status}")
                    return pipeline_name, None, None
        except Exception as e:
            _log.debug("Episode poll during setup: %s", e, exc_info=True)
        time.sleep(1)

    if not episode_id:
        error("Timed out waiting for episode to start.")
        return pipeline_name, None, None

    return pipeline_name, sim.id, episode_id


# ── Conversation loop ─────────────────────────────────────────────────────────


def _conversation_loop(
    client, sim_id: str, episode_id: str, initiator: str, max_turns: int,
) -> str:
    """Interactive conversation: patient messages ↔ doctor responses.

    Returns an end-reason string:
      ``"completed"``  – normal completion or judging started
      ``"patient_quit"``  – patient left the conversation
      ``"doctor_quit"``  – user typed /quit or interrupted
      ``"failed"``  – episode errored
      ``"lost"``  – connection lost
    """

    console.print()
    console.print(Rule("[bold cyan]Conversation Started[/]", style="cyan"))
    console.print(
        "  [dim]Type your responses as the doctor.  Arrow keys work.  "
        "[bold]/quit[/bold] to end, [bold]/history[/bold] to review.[/]"
    )
    console.print()

    turn = 0
    last_patient_msg = ""

    while True:
        # ── Fetch current episode state ───────────────────────────────
        ep = _poll_episode(client, sim_id, episode_id)
        if not ep:
            error("Lost connection to episode.")
            return "lost"

        status = ep.get("status", "")
        dialogue = ep.get("dialogue_history", [])
        ep_error = ep.get("error", "")

        # ── Check terminal states ─────────────────────────────────────
        if status in ("completed", "judging"):
            _show_new_messages(dialogue, turn)
            console.print()
            console.print(Rule("[bold green]Conversation Complete[/]", style="green"))
            console.print(f"  [dim]{len(dialogue)} messages exchanged[/]")
            return "completed"

        if status == "failed":
            _show_new_messages(dialogue, turn)
            # Detect patient-quit vs real error
            if _is_patient_quit(dialogue, last_patient_msg):
                console.print()
                console.print(Rule("[bold yellow]Patient Left the Conversation[/]", style="yellow"))
                console.print(f"  [dim]{len(dialogue)} messages exchanged[/]")
                return "patient_quit"
            else:
                console.print()
                console.print(Rule("[bold red]Episode Error[/]", style="red"))
                if ep_error:
                    error(f"{ep_error}")
                console.print(f"  [dim]{len(dialogue)} messages exchanged[/]")
                return "failed"

        # ── Show new patient messages ─────────────────────────────────
        _show_new_messages(dialogue, turn)
        if dialogue:
            last_msg = dialogue[-1]
            if last_msg.get("role") == "patient":
                last_patient_msg = last_msg.get("content", "")
        turn = len(dialogue)

        # ── Check if waiting for doctor ───────────────────────────────
        if status != "awaiting_doctor":
            _show_status("Patient is typing...")
            time.sleep(1)
            _clear_status()
            continue

        # ── Get doctor input (with arrow key support) ─────────────────
        remaining = max_turns - (len(dialogue) // 2)
        suffix = f"  (~{remaining} turns left)" if remaining <= 5 else ""

        try:
            response = _doctor_input(suffix)
        except KeyboardInterrupt:
            console.print()
            warn("Interrupted.")
            return "doctor_quit"

        if response is None:
            return "doctor_quit"
        if response == "/history":
            _show_full_history(dialogue)
            continue
        if response == "/quit":
            console.print("  [dim]Ending conversation...[/]")
            return "doctor_quit"
        if not response.strip():
            continue  # silently ignore empty

        # ── Submit with retries ────────────────────────────────────────
        submitted = _submit_with_retry(client, sim_id, episode_id, response)
        if submitted is None:
            return "failed"
        if submitted == "already_sent":
            # Idempotency: our message was already delivered (timeout on response)
            pass
        else:
            turn = len(submitted.get("dialogue_history", dialogue))

        # Brief pause for patient to respond
        time.sleep(0.3)


def _is_patient_quit(dialogue: list[dict], last_patient_msg: str) -> bool:
    """Heuristic: did the patient leave/end the conversation rather than a system error?"""
    if not dialogue:
        return False
    last = dialogue[-1]
    if last.get("role") != "patient":
        return False
    content = (last.get("content", "") or "").lower()
    quit_signals = [
        "leave", "leaving", "going to leave", "i'm done", "i'm leaving",
        "goodbye", "good bye", "bye", "hanging up", "end this",
        "don't want to continue", "not continuing", "calling the clinic",
        "asking for a real", "want another", "different doctor",
        "i'm out", "done with this", "walking out", "unsafe",
    ]
    return any(sig in content for sig in quit_signals)


# ── Submit with retry + idempotency ──────────────────────────────────────────


_RETRYABLE_ERRORS = ("timed out", "timeout", "connection reset", "broken pipe",
                     "connection refused", "network is unreachable", "eof occurred",
                     "remote end closed", "urlopen error")


def _is_retryable(err: Exception) -> bool:
    msg = str(err).lower()
    return any(s in msg for s in _RETRYABLE_ERRORS)


def _submit_with_retry(client, sim_id: str, episode_id: str, message: str, max_retries: int = 3):
    """Submit a doctor response with automatic retries and idempotency check.

    Returns:
      - updated episode dict on success
      - ``"already_sent"`` if idempotency check shows it was already delivered
      - ``None`` on unrecoverable failure
    """
    last_error = None

    for attempt in range(max_retries):
        _show_status(f"Sending...{f' (retry {attempt})' if attempt > 0 else ''}")
        try:
            updated = client.simulations.submit_response(sim_id, episode_id, message)
            _clear_status()
            _show_status("✓ Sent — waiting for patient...")
            time.sleep(0.3)
            _clear_status()
            return updated
        except Exception as e:
            _clear_status()
            last_error = e
            err_str = str(e).lower()

            # If server says "not awaiting" it means our message already arrived
            # (we got a timeout on the response, but the server processed it)
            if "not awaiting" in err_str or "already" in err_str:
                success("Message was already delivered (network hiccup on response)")
                return "already_sent"

            if _is_retryable(e) and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s
                warn(f"Network error, retrying in {wait}s... ({attempt + 1}/{max_retries})")

                # Idempotency check: before retrying, verify our message wasn't
                # already stored (timeout may have occurred after server processed it)
                try:
                    ep = client.simulations.get_episode(sim_id, episode_id)
                    dialogue = ep.get("dialogue_history", [])
                    if dialogue and dialogue[-1].get("role") == "doctor":
                        last_content = dialogue[-1].get("content", "")
                        if last_content == message:
                            success("Message was already delivered (confirmed by server)")
                            return "already_sent"
                except Exception as e:
                    _log.debug("Idempotency check failed, will retry submit: %s", e, exc_info=True)

                time.sleep(wait)
                continue
            else:
                # Non-retryable error
                error(f"Failed to send: {e}")
                return None

    error(f"Failed after {max_retries} attempts: {last_error}")
    return None


# ── Input with arrow key support ──────────────────────────────────────────────


def _doctor_input(suffix: str = "") -> str | None:
    """Prompt for doctor input with full readline support (arrow keys, home/end, etc.)."""
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.formatted_text import HTML

    toolbar = HTML(f"<b>/quit</b> end  <b>/history</b> review{suffix}")

    try:
        result = pt_prompt(
            [("class:green bold", "  You › ")],
            bottom_toolbar=toolbar,
            style=_pt_style(),
        )
        return result
    except (KeyboardInterrupt, EOFError):
        return None


def _pt_style():
    """prompt_toolkit style matching the EARL theme."""
    from prompt_toolkit.styles import Style as PTStyle
    return PTStyle.from_dict({
        "": "#ffffff",
        "green": "#22c55e",
        "bottom-toolbar": "bg:#1a1a2e #858585",
        "bottom-toolbar.text": "#858585",
    })


# ── Status line helpers ───────────────────────────────────────────────────────

_LAST_STATUS_LEN = 0


def _show_status(msg: str) -> None:
    """Print a transient status line (overwritten by next status or cleared)."""
    global _LAST_STATUS_LEN
    text = f"  [dim]{msg}[/]"
    console.print(text, end="\r")
    _LAST_STATUS_LEN = len(msg) + 4


def _clear_status() -> None:
    """Clear the last status line."""
    global _LAST_STATUS_LEN
    if _LAST_STATUS_LEN > 0:
        # Move up and clear line
        print(f"\r{' ' * (_LAST_STATUS_LEN + 10)}\r", end="", flush=True)
        _LAST_STATUS_LEN = 0


# ── Episode polling & display ─────────────────────────────────────────────────


def _poll_episode(client, sim_id: str, episode_id: str, retries: int = 6) -> dict | None:
    """Fetch episode with exponential backoff retries (tolerates network switches)."""
    for attempt in range(retries):
        try:
            return client.simulations.get_episode(sim_id, episode_id)
        except Exception as e:
            _log.debug("Episode poll attempt %d: %s", attempt, e, exc_info=True)
            if attempt < retries - 1:
                wait = min(2 ** attempt, 8)  # 1, 2, 4, 8, 8, 8
                time.sleep(wait)
    return None


def _show_new_messages(dialogue: list[dict], already_shown: int) -> int:
    """Display messages the user hasn't seen yet. Returns count of new msgs shown."""
    new_msgs = dialogue[already_shown:]
    for msg in new_msgs:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "patient":
            console.print(f"  [bold cyan]Patient:[/] {content}")
            console.print()
        elif role == "doctor":
            # Don't re-echo — the user already typed it
            pass
        else:
            console.print(f"  [dim]{role}:[/] {content}")
    return len(new_msgs)


def _show_full_history(dialogue: list[dict]) -> None:
    """Print the complete dialogue so far."""
    console.print()
    console.print(Rule("[dim]Full Dialogue History[/]", style="#555555"))
    for i, msg in enumerate(dialogue, 1):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if role == "patient":
            console.print(f"  [cyan]{i}. Patient:[/] {content}")
        elif role == "doctor":
            console.print(f"  [green]{i}. Doctor:[/]  {content}")
        else:
            console.print(f"  [dim]{i}. {role}:[/]    {content}")
    console.print(Rule(style="#555555"))
    console.print()


# ── Results ───────────────────────────────────────────────────────────────────


def _handle_ending(
    client, sim_id: str, episode_id: str, end_reason: str,
    pipeline_name: str, store: ConfigStore, run_store: RunStore,
) -> None:
    """Handle conversation ending — always offer to proceed to judging."""
    console.print()

    # ── Summarize what happened ───────────────────────────────────────
    reason_labels = {
        "completed": ("green", "Conversation finished normally"),
        "patient_quit": ("yellow", "Patient left the conversation"),
        "doctor_quit": ("yellow", "You ended the conversation"),
        "failed": ("red", "Episode encountered an error"),
        "lost": ("red", "Lost connection to episode"),
    }
    color, label = reason_labels.get(end_reason, ("dim", "Conversation ended"))
    console.print(f"  [{color}]{label}[/]")

    # ── Check current episode status ──────────────────────────────────
    ep = _poll_episode(client, sim_id, episode_id)
    if not ep:
        error("Cannot reach episode. Check results later in Browse Simulations.")
        return

    status = ep.get("status", "")

    # If already judging or completed, just wait for results
    if status in ("judging", "completed"):
        _wait_for_judge_and_show(client, sim_id, episode_id, pipeline_name, store, run_store)
        return

    # ── Offer to proceed to judging ───────────────────────────────────
    dialogue = ep.get("dialogue_history", [])
    msg_count = len(dialogue)

    if msg_count == 0:
        muted("No messages were exchanged — nothing to judge.")
        return

    # Always offer judging regardless of how the conversation ended
    action = select_one("What would you like to do?", [
        ("judge",  f"Get Judged             — submit the {msg_count}-message conversation for evaluation"),
        ("save",   f"Save Without Judging   — save the conversation as-is without scores"),
        ("discard", "Discard                — don't save, just return to menu"),
    ], allow_back=False)

    if action == "judge":
        # Try to trigger judging by stopping the simulation (with retries)
        console.print("  [dim]Submitting conversation for judging...[/]")
        for attempt in range(3):
            try:
                client.simulations.stop(sim_id)
                break
            except Exception as e:
                if _is_retryable(e) and attempt < 2:
                    warn(f"Network error, retrying... ({attempt + 1}/3)")
                    time.sleep(2 ** attempt)
                else:
                    _log.debug("Stop simulation failed (may already be terminal): %s", e, exc_info=True)

        # Wait a moment for the stop to propagate
        time.sleep(1)
        _wait_for_judge_and_show(client, sim_id, episode_id, pipeline_name, store, run_store)

    elif action == "save":
        # Save without judging
        _save_without_judging(client, sim_id, episode_id, pipeline_name, store, run_store)

    # "discard" — just return


def _wait_for_judge_and_show(
    client, sim_id: str, episode_id: str,
    pipeline_name: str, store: ConfigStore, run_store: RunStore,
) -> None:
    """Wait for judging to complete and display scores."""

    console.print()
    console.print("  [dim]Waiting for judge to score your conversation...[/]")

    # Poll until judging is done (up to 3 minutes)
    final_ep = None
    for i in range(90):
        try:
            ep = client.simulations.get_episode(sim_id, episode_id)
            status = ep.get("status", "")
            if status == "completed":
                final_ep = ep
                break
            if status == "failed":
                ep_error = ep.get("error", "")
                # If it failed due to serialization or similar, still try to show what we have
                warn(f"Episode error: {ep_error}" if ep_error else "Episode failed during judging.")
                if ep.get("judge_feedback") or ep.get("total_score") is not None:
                    final_ep = ep  # has partial results
                    break
                # Wait a bit more in case it's being retried
                if i < 15:
                    time.sleep(2)
                    continue
                break
            if status not in ("judging", "dialogue", "conversation", "awaiting_doctor"):
                warn(f"Unexpected status: {status}")
                break
        except Exception as e:
            _log.debug("Episode poll during judge wait: %s", e, exc_info=True)
        time.sleep(2)

    if not final_ep:
        warn("Judge hasn't finished yet. You can check results later in Browse Simulations.")
        muted(f"Simulation ID: {sim_id}")
        return

    # ── Score summary ─────────────────────────────────────────────────

    total_score = final_ep.get("total_score")
    dialogue = final_ep.get("dialogue_history", [])

    console.print()
    if total_score is not None:
        score_color = "green" if total_score >= 3.0 else "yellow" if total_score >= 2.0 else "red"
        console.print(Panel(
            f"[bold]Overall Score: [{score_color}]{total_score:.2f} / 4.0[/][/]\n"
            f"[dim]Based on {len(dialogue)} messages[/]",
            title="[bold]Judge Results[/]",
            border_style=score_color,
            padding=(1, 2),
        ))
    else:
        muted("No score available (judging may have been partial).")

    # ── Dimension breakdown ───────────────────────────────────────────

    judge_fb = final_ep.get("judge_feedback") or {}
    dim_results = judge_fb.get("dimension_results", []) if isinstance(judge_fb, dict) else []

    if dim_results:
        rows = []
        for dr in dim_results:
            dim_name = dr.get("dimension_name", dr.get("dimension_id", "?"))
            jr = dr.get("judge_result", {})
            dim_score = jr.get("score") if isinstance(jr, dict) else dr.get("score")
            reasoning = jr.get("reasoning", "") if isinstance(jr, dict) else ""

            if dim_score is not None:
                rows.append([
                    dim_name,
                    score_text(dim_score),
                    reasoning[:80] + ("..." if len(reasoning) > 80 else "") if reasoning else "",
                ])

        if rows:
            datatable(
                columns=[
                    ("Dimension", "bold"),
                    ("Score", ""),
                    ("Reasoning", "dim"),
                ],
                rows=rows,
                title="Dimension Scores",
            )

    # ── Full reasoning (expandable) ───────────────────────────────────

    if dim_results and ask_confirm("Show full judge reasoning?", default=False):
        for dr in dim_results:
            dim_name = dr.get("dimension_name", dr.get("dimension_id", "?"))
            jr = dr.get("judge_result", {})
            dim_score = jr.get("score") if isinstance(jr, dict) else dr.get("score")
            reasoning = jr.get("reasoning", "") if isinstance(jr, dict) else ""

            if reasoning:
                score_display = f"{dim_score:.1f}/4" if dim_score is not None else "?"
                console.print(Panel(
                    reasoning,
                    title=f"[bold]{dim_name}[/] — {score_display}",
                    border_style="#555555",
                    padding=(1, 2),
                ))

    # ── Save locally ──────────────────────────────────────────────────

    if store.preferences.auto_save_runs or ask_confirm("Save this conversation locally?"):
        try:
            report = client.simulations.get_report(sim_id)
        except Exception as e:
            _log.debug("Report fetch failed, using episode fallback: %s", e, exc_info=True)
            report = {"episode": final_ep, "simulation_id": sim_id}

        meta = LocalRun(
            simulation_id=sim_id,
            pipeline_name=pipeline_name,
            status="completed" if total_score is not None else "stopped",
            total_episodes=1,
            completed_episodes=1 if total_score is not None else 0,
            average_score=total_score,
            doctor_type="chat",
            environment=getattr(client, "_environment", ""),
        )
        run_store.save_run(meta, report)
        success(f"Conversation saved to ~/.earl/runs/{meta.short_id}/")


def _save_without_judging(
    client, sim_id: str, episode_id: str,
    pipeline_name: str, store: ConfigStore, run_store: RunStore,
) -> None:
    """Save the conversation locally without waiting for judge scores."""
    ep = _poll_episode(client, sim_id, episode_id)
    dialogue = ep.get("dialogue_history", []) if ep else []

    report = {
        "simulation_id": sim_id,
        "episode": ep,
        "note": "Saved without judging",
    }
    meta = LocalRun(
        simulation_id=sim_id,
        pipeline_name=pipeline_name,
        status="stopped",
        total_episodes=1,
        completed_episodes=0,
        doctor_type="chat",
        environment=getattr(client, "_environment", ""),
    )
    run_store.save_run(meta, report)
    success(f"Conversation ({len(dialogue)} messages) saved to ~/.earl/runs/{meta.short_id}/")
