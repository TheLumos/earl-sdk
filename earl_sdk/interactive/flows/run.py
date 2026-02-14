"""Run Simulation flow — configure and launch an evaluation.

Guides the user through pipeline selection, doctor configuration, episode
count, and parallelism, then streams live progress until completion.
Results are automatically saved locally if auto-save is enabled.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

_log = logging.getLogger("earl.ui")

from rich.live import Live
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from ..ui import (
    ask_confirm,
    ask_int,
    console,
    datatable,
    error,
    info_panel,
    muted,
    score_text,
    select_one,
    success,
    warn,
)
from ..storage.config_store import ConfigStore, DoctorConfig
from ..storage.run_store import LocalRun, RunStore


def flow_run(client, store: ConfigStore, run_store: RunStore, *, pipeline_name: str | None = None) -> None:
    """Full run-simulation flow: configure → launch → track → save.

    Args:
        pipeline_name: If provided, skip selection and use this pipeline directly.
    """

    # ── Step 1: Select pipeline ───────────────────────────────────────────

    console.print("\n[bold]Run Simulation[/]")
    muted("Evaluate a doctor against simulated patients and get scored results.\n")

    if pipeline_name:
        # Fetch the specific pipeline
        console.print(f"  [dim]Loading pipeline '{pipeline_name}'...[/]")
        try:
            pipeline = client.pipelines.get(pipeline_name)
        except Exception as e:
            error(f"Failed to load pipeline: {e}")
            return
    else:
        console.print("  [dim]Loading pipelines...[/]")
        try:
            pipelines = client.pipelines.list()
        except Exception as e:
            error(f"Failed to load pipelines: {e}")
            return

        if not pipelines:
            warn("No pipelines found.")
            if not ask_confirm("Create a new pipeline now?"):
                return
            from .explore import create_pipeline_wizard
            pipeline_name = create_pipeline_wizard(client)
            if not pipeline_name:
                return
            try:
                pipeline = client.pipelines.get(pipeline_name)
            except Exception as e:
                error(f"Failed to load created pipeline: {e}")
                return
        else:
            default_pipeline = store.preferences.default_pipeline
            choices = []
            choices.append(("__create__", "Create New Pipeline    — build a new evaluation config, then run it"))
            for p in pipelines:
                doc_type = "Lumos's"
                if p.doctor_api:
                    type_map = {"external": "Client's", "internal": "Lumos's", "client_driven": "Client-Driven"}
                    doc_type = type_map.get(p.doctor_api.type, p.doctor_api.type)
                parts = [p.name, " —"]
                if p.dimension_ids:
                    parts.append(f" {len(p.dimension_ids)} dims,")
                if p.patient_ids:
                    parts.append(f" {len(p.patient_ids)} patients,")
                parts.append(f" {doc_type} doctor")
                hint = "".join(parts)
                choices.append((p.name, hint))

            selected = select_one("Select pipeline to run", choices)
            if not selected:
                return

            if selected == "__create__":
                from .explore import create_pipeline_wizard
                pipeline_name = create_pipeline_wizard(client)
                if not pipeline_name:
                    return
                try:
                    pipeline = client.pipelines.get(pipeline_name)
                except Exception as e:
                    error(f"Failed to load created pipeline: {e}")
                    return
            else:
                pipeline_name = selected
                pipeline = next((p for p in pipelines if p.name == pipeline_name), None)
                if not pipeline:
                    error("Pipeline not found.")
                    return

    # ── Step 2: Doctor override (optional) ────────────────────────────────

    current_doc = "Lumos's (built-in)"
    if pipeline.doctor_api:
        type_map = {"external": "Client's", "internal": "Lumos's", "client_driven": "Client-Driven"}
        current_doc = type_map.get(pipeline.doctor_api.type, pipeline.doctor_api.type)
        if pipeline.doctor_api.api_url:
            current_doc += f" → {pipeline.doctor_api.api_url}"

    muted(f"Current doctor: {current_doc}")
    # For now, use pipeline's default doctor — override can be added later

    # ── Step 3: Configure run parameters ──────────────────────────────────

    default_episodes = len(pipeline.patient_ids) if pipeline.patient_ids else 1
    num_episodes = ask_int(
        f"Number of episodes (pipeline has {len(pipeline.patient_ids)} patients)",
        default=default_episodes,
        min_val=1,
        max_val=100,
    )
    if num_episodes is None:
        return

    parallel_count = ask_int(
        "Parallel episodes (simultaneous evaluations, 1-10)",
        default=min(store.preferences.default_parallel, num_episodes),
        min_val=1,
        max_val=min(10, num_episodes),
    )
    if parallel_count is None:
        return

    # ── Step 4: Confirmation ──────────────────────────────────────────────

    console.print()
    info_panel("Run Configuration", [
        f"[bold]Pipeline:[/]      {pipeline_name}",
        f"[bold]Doctor:[/]        {current_doc}",
        f"[bold]Episodes:[/]      {num_episodes} ({parallel_count} parallel)",
        f"[bold]Dimensions:[/]    {len(pipeline.dimension_ids)}",
        f"[bold]Max Turns:[/]     {pipeline.conversation.max_turns if pipeline.conversation else 10}",
    ])

    if not ask_confirm("Start simulation?"):
        muted("Cancelled.")
        return

    # ── Step 5: Launch and track ──────────────────────────────────────────

    console.print()
    try:
        sim = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=num_episodes,
            parallel_count=parallel_count,
        )
        success(f"Simulation started: {sim.id[:8]}...")
    except Exception as e:
        error(f"Failed to start simulation: {e}")
        return

    # Live progress tracking
    console.print()
    final_sim = _track_progress(client, sim.id, num_episodes)

    if not final_sim:
        error("Lost connection while tracking simulation.")
        return

    # ── Step 6: Results ───────────────────────────────────────────────────

    _show_results(client, final_sim, pipeline_name, store, run_store)


def _track_progress(client, simulation_id: str, total: int):
    """Poll simulation until terminal state, showing live progress bar."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total} episodes"),
        TimeElapsedColumn(),
        console=console,
    )
    task = progress.add_task("Running", total=total)

    try:
        with Live(progress, console=console, refresh_per_second=4):
            while True:
                try:
                    sim = client.simulations.get(simulation_id)
                except Exception as e:
                    _log.debug("Simulation poll during tracking: %s", e, exc_info=True)
                    time.sleep(3)
                    continue

                completed = sim.completed_episodes
                status = sim.status.value if hasattr(sim.status, "value") else str(sim.status)

                progress.update(task, completed=completed)

                if status == "judging":
                    progress.update(task, description="Judging")
                elif status == "running":
                    progress.update(task, description="Running")

                if status in ("completed", "failed", "stopped"):
                    progress.update(task, completed=total if status == "completed" else completed)
                    progress.update(task, description=status.capitalize())
                    break

                time.sleep(2)

        return sim
    except KeyboardInterrupt:
        warn("Interrupted — simulation continues on server.")
        console.print(f"  [dim]Track it later with simulation ID: {simulation_id}[/]")
        return None


def _show_results(client, sim, pipeline_name: str, store: ConfigStore, run_store: RunStore) -> None:
    """Display final results and optionally save locally."""
    status = sim.status.value if hasattr(sim.status, "value") else str(sim.status)

    if status == "failed":
        error(f"Simulation failed: {sim.error_message or 'unknown error'}")
        return

    if status == "stopped":
        warn("Simulation was stopped before completing all episodes.")

    # Fetch episodes for summary
    try:
        episodes = client.simulations.get_episodes(sim.id, include_dialogue=False)
    except Exception as e:
        _log.debug("Episodes fetch failed: %s", e, exc_info=True)
        episodes = []

    avg_score = sim.summary.get("average_score") if sim.summary else None

    # Summary panel
    lines = [
        f"[bold]Simulation:[/]    {sim.id[:8]}",
        f"[bold]Pipeline:[/]      {pipeline_name}",
        f"[bold]Status:[/]        [{_status_color(status)}]{status}[/]",
        f"[bold]Episodes:[/]      {sim.completed_episodes}/{sim.total_episodes}",
    ]
    if avg_score is not None:
        lines.append(f"[bold]Avg Score:[/]     {avg_score:.2f} / 4.0")
    if sim.finished_at and sim.started_at:
        lines.append(f"[bold]Duration:[/]      {_format_duration(sim.started_at, sim.finished_at)}")

    info_panel("Results", lines, border="green" if status == "completed" else "yellow")

    # Episode table
    if episodes:
        rows = []
        for ep in episodes:
            ep_status = ep.get("status", "?")
            ep_score = ep.get("total_score")
            patient = ep.get("patient_name") or ep.get("patient_id") or "?"
            turns = ep.get("dialogue_turns", 0)
            rows.append([
                str(ep.get("episode_number", 0) + 1),
                patient[:20],
                Text(ep_status, style=_status_color(ep_status)),
                score_text(ep_score) if ep_score is not None else Text("-", style="dim"),
                str(turns),
            ])
        datatable(
            columns=[
                ("Ep", "dim"),
                ("Patient", ""),
                ("Status", ""),
                ("Score", ""),
                ("Turns", "dim"),
            ],
            rows=rows,
            title="Episodes",
        )

    # Auto-save
    if store.preferences.auto_save_runs:
        _save_run(client, sim, pipeline_name, run_store)
    else:
        if ask_confirm("Save results locally for future comparison?"):
            _save_run(client, sim, pipeline_name, run_store)


def _save_run(client, sim, pipeline_name: str, run_store: RunStore) -> None:
    """Download and save full report locally."""
    try:
        report = client.simulations.get_report(sim.id)
    except Exception as e:
        _log.debug("Report fetch failed: %s", e, exc_info=True)
        report = {"simulation_id": sim.id, "note": "Report fetch failed — summary only"}

    status = sim.status.value if hasattr(sim.status, "value") else str(sim.status)
    avg_score = sim.summary.get("average_score") if sim.summary else None

    meta = LocalRun(
        simulation_id=sim.id,
        pipeline_name=pipeline_name,
        status=status,
        total_episodes=sim.total_episodes,
        completed_episodes=sim.completed_episodes,
        average_score=avg_score,
        started_at=str(sim.started_at) if sim.started_at else "",
        finished_at=str(sim.finished_at) if sim.finished_at else "",
        environment=getattr(client, "_environment", ""),
    )
    run_store.save_run(meta, report)
    success(f"Run saved locally (~/.earl/runs/{meta.short_id}/)")


def _status_color(status: str) -> str:
    return {
        "completed": "green",
        "running": "cyan",
        "judging": "yellow",
        "failed": "red",
        "stopped": "red",
        "pending": "dim",
    }.get(status, "dim")


def _format_duration(start, end) -> str:
    """Format duration between two datetime-like values."""
    from datetime import datetime
    try:
        if isinstance(start, str):
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        if isinstance(end, str):
            end = datetime.fromisoformat(end.replace("Z", "+00:00"))
        delta = end - start
        secs = int(delta.total_seconds())
        if secs < 60:
            return f"{secs}s"
        mins = secs // 60
        secs = secs % 60
        return f"{mins}m {secs}s"
    except Exception as e:
        _log.debug("Duration format failed: %s", e, exc_info=True)
        return "?"
