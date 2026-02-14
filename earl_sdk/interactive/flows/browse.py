"""Browse Simulations flow — view past and current simulation results.

Merges remote simulations (from the EARL API) with locally saved runs.
Drill into episodes, view dialogue history, judge feedback, and scores.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

_log = logging.getLogger("earl.ui")

from rich.text import Text

from ..ui import (
    ask_confirm,
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
from ..storage.run_store import LocalRun, RunStore


def flow_browse(client, run_store: RunStore) -> None:
    """Top-level browse menu — pick source, then drill into simulations."""
    while True:
        source = select_one("Browse Simulations", [
            ("all",    "All (remote + local)   — latest from EARL API merged with locally saved results"),
            ("remote", "Remote Only            — live data from the EARL API (requires connection)"),
            ("local",  "Local Only             — previously saved runs stored on this machine"),
        ])
        if source is None:
            return

        sims = _load_simulations(client, run_store, source)
        if not sims:
            warn("No simulations found.")
            continue

        _browse_list(client, run_store, sims)


def _load_simulations(client, run_store: RunStore, source: str) -> list[dict]:
    """Load and merge simulations from selected sources."""
    results: list[dict] = []

    # Remote
    if source in ("all", "remote"):
        console.print("  [dim]Loading from EARL API...[/]")
        try:
            remote = client.simulations.list(limit=50)
            for s in remote:
                status = s.status.value if hasattr(s.status, "value") else str(s.status)
                avg = s.summary.get("average_score") if s.summary else None
                results.append({
                    "id": s.id,
                    "pipeline": s.pipeline_name,
                    "status": status,
                    "total": s.total_episodes,
                    "completed": s.completed_episodes,
                    "score": avg,
                    "started": str(s.started_at) if s.started_at else "",
                    "source": "remote",
                    "sim": s,
                })
        except Exception as e:
            warn(f"Failed to load remote simulations: {e}")

    # Local
    if source in ("all", "local"):
        local_runs = run_store.list_runs(limit=50)
        remote_ids = {r["id"] for r in results}
        for lr in local_runs:
            if lr.simulation_id in remote_ids:
                # Mark remote entry as also locally saved
                for r in results:
                    if r["id"] == lr.simulation_id:
                        r["source"] = "both"
                continue
            results.append({
                "id": lr.simulation_id,
                "pipeline": lr.pipeline_name,
                "status": lr.status,
                "total": lr.total_episodes,
                "completed": lr.completed_episodes,
                "score": lr.average_score,
                "started": lr.started_at,
                "source": "local",
                "local_run": lr,
            })

    # Sort by date descending
    results.sort(key=lambda r: r.get("started", ""), reverse=True)
    return results


def _browse_list(client, run_store: RunStore, sims: list[dict]) -> None:
    """Show simulation list table and handle selection."""
    while True:
        rows = []
        for i, s in enumerate(sims):
            status = s["status"]
            score = s.get("score")
            source_tag = {"remote": "☁", "local": "💾", "both": "☁💾"}.get(s["source"], "?")
            rows.append([
                str(i + 1),
                _format_date(s.get("started", "")),
                s["pipeline"][:20],
                f"{s['completed']}/{s['total']}",
                score_text(score) if score is not None else Text("-", style="dim"),
                Text(status, style=_status_color(status)),
                source_tag,
            ])

        datatable(
            columns=[
                ("#", "dim"),
                ("Date", ""),
                ("Pipeline", "bold"),
                ("Episodes", ""),
                ("Score", ""),
                ("Status", ""),
                ("Src", "dim"),
            ],
            rows=rows,
            title=f"Simulations ({len(sims)} total — ☁=remote, 💾=local)",
        )

        choices = [
            (str(i), f"#{i+1}  {s['pipeline'][:16]}  {s['id'][:8]}  [{s['status']}]  " +
             (f"score={s['score']:.1f}" if s.get('score') is not None else ""))
            for i, s in enumerate(sims)
        ]
        pick = select_one("Select simulation to inspect", choices)
        if pick is None:
            return

        idx = int(pick)
        _inspect_simulation(client, run_store, sims[idx])


def _inspect_simulation(client, run_store: RunStore, sim_data: dict) -> None:
    """Drill into a single simulation: summary, episodes, actions."""
    sim_id = sim_data["id"]

    # Refresh from remote if available
    sim_obj = sim_data.get("sim")
    if sim_obj:
        try:
            sim_obj = client.simulations.get(sim_id)
        except Exception as e:
            _log.debug("Simulation refresh failed: %s", e, exc_info=True)

    # Load episodes
    episodes: list[dict] = []
    try:
        episodes = client.simulations.get_episodes(sim_id, include_dialogue=False)
    except Exception as e:
        _log.debug("Episodes fetch failed, trying local: %s", e, exc_info=True)
        # Try local report
        report = run_store.load_report(sim_id)
        if report and "episodes" in report:
            episodes = report["episodes"]

    # Summary
    status = sim_data["status"]
    score = sim_data.get("score")
    lines = [
        f"[bold]Simulation ID:[/]   {sim_id}",
        f"[bold]Pipeline:[/]        {sim_data['pipeline']}",
        f"[bold]Status:[/]          [{_status_color(status)}]{status}[/]",
        f"[bold]Episodes:[/]        {sim_data['completed']}/{sim_data['total']}",
    ]
    if score is not None:
        lines.append(f"[bold]Avg Score:[/]       {score:.2f} / 4.0")
    lines.append(f"[bold]Started:[/]         {_format_date(sim_data.get('started', ''))}")
    lines.append(f"[bold]Source:[/]          {sim_data['source']}")

    info_panel(f"Simulation {sim_id[:8]}", lines)

    # Dimension averages from summary
    if sim_obj and sim_obj.summary:
        dim_avgs = sim_obj.summary.get("dimension_averages")
        if dim_avgs:
            dim_rows = []
            for dim_name, avg in sorted(dim_avgs.items()):
                dim_rows.append([dim_name, score_text(avg)])
            if dim_rows:
                datatable(
                    columns=[("Dimension", "bold"), ("Avg Score", "")],
                    rows=dim_rows,
                    title="Dimension Breakdown",
                )

    # Episodes table
    if episodes:
        ep_rows = []
        for ep in episodes:
            ep_status = ep.get("status", "?")
            ep_score = ep.get("total_score")
            patient = ep.get("patient_name") or ep.get("patient_id") or "?"
            turns = ep.get("dialogue_turns", 0)
            ep_rows.append([
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
            rows=ep_rows,
            title=f"Episodes ({len(episodes)})",
        )

    # Actions
    while True:
        action_choices: list[tuple[str, str]] = []
        if episodes:
            action_choices.append(("episode",    "View Episode Detail    — drill into a specific episode's dialogue and judge scores"))
            action_choices.append(("rationales", "View All Rationales    — committee rationale for every dimension across all episodes"))
        if sim_data["source"] in ("remote", "both"):
            action_choices.append(("save",    "Save Locally           — download full report to disk for offline comparison"))
        if sim_data["source"] in ("local", "both"):
            action_choices.append(("delete",  "Delete Local Copy      — remove the saved run from this machine"))
        action_choices.append(("report",  "View Full Report       — fetch and display the complete evaluation report"))

        action = select_one("Actions", action_choices)
        if action is None:
            return

        if action == "episode":
            _pick_episode(client, sim_id, episodes)
        elif action == "rationales":
            _show_all_rationales(client, sim_id, episodes)
        elif action == "save":
            _save_simulation(client, sim_data, run_store)
        elif action == "delete":
            if ask_confirm("Delete local copy?", default=False):
                run_store.delete_run(sim_id)
                success("Local copy deleted.")
                sim_data["source"] = "remote"
        elif action == "report":
            _show_report(client, sim_id, run_store)


def _pick_episode(client, sim_id: str, episodes: list[dict]) -> None:
    """Select and view a single episode in detail."""
    choices = []
    for ep in episodes:
        ep_id = ep.get("episode_id", "?")
        patient = ep.get("patient_name") or ep.get("patient_id") or "?"
        ep_num = ep.get("episode_number", 0) + 1
        status = ep.get("status", "?")
        score = ep.get("total_score")
        score_hint = f"  score={score:.1f}" if score is not None else ""
        choices.append((ep_id, f"Ep {ep_num}: {patient[:20]}  [{status}]{score_hint}"))

    pick = select_one("Select episode", choices)
    if not pick:
        return

    _view_episode(client, sim_id, pick)


def _view_episode(client, sim_id: str, episode_id: str) -> None:
    """View full episode detail: dialogue + judge feedback with committee rationale."""
    console.print("  [dim]Loading episode details...[/]")
    try:
        ep = client.simulations.get_episode(sim_id, episode_id)
    except Exception as e:
        error(f"Failed to load episode: {e}")
        return

    ep_num = ep.get("episode_number", 0) + 1
    patient = ep.get("patient_name") or ep.get("patient_id") or "?"
    status = ep.get("status", "?")
    total_score = ep.get("total_score")

    lines = [
        f"[bold]Episode:[/]       #{ep_num}",
        f"[bold]Patient:[/]       {patient}",
        f"[bold]Status:[/]        [{_status_color(status)}]{status}[/]",
    ]
    if total_score is not None:
        lines.append(f"[bold]Total Score:[/]   {total_score:.2f} / 4.0")
    lines.append(f"[bold]Turns:[/]         {ep.get('dialogue_turns', 0)}")

    info_panel(f"Episode {ep_num}", lines)

    # Judge feedback — dimension scores with rationale
    judge_fb = ep.get("judge_feedback", {})
    dim_results = judge_fb.get("dimension_results", [])
    if dim_results:
        console.print(f"\n[bold]Committee Scores & Rationale[/]")
        console.print()
        for dr in dim_results:
            dim_name = dr.get("dimension_name") or dr.get("dimension_id", "?")
            category = dr.get("category", "")
            jr = dr.get("judge_result", {}) or {}
            dim_score = jr.get("score") if isinstance(jr, dict) else dr.get("score")
            rationale = jr.get("rationale", "") if isinstance(jr, dict) else ""
            committee_insights = jr.get("committee_insights", "") if isinstance(jr, dict) else ""
            confidence = jr.get("confidence") if isinstance(jr, dict) else None
            dim_error = dr.get("error")

            if dim_error:
                console.print(f"  [red]✗[/] {dim_name}  [red]error: {dim_error}[/]")
                continue

            # Score line
            score_display = score_text(dim_score) if dim_score is not None else Text("-", style="dim")
            cat_tag = f"  [dim][{category}][/]" if category else ""
            conf_tag = f"  [dim](confidence {confidence}/10)[/]" if confidence else ""
            console.print(f"  {score_display}  [bold]{dim_name}[/]{cat_tag}{conf_tag}")

            # Rationale — full text, wrapped
            if rationale:
                console.print(f"       [italic]{rationale}[/]")

            # Committee insights
            if committee_insights:
                console.print(f"       [dim]Committee: {committee_insights}[/]")

            console.print()  # spacing between dimensions

        # Committee discussions (full deliberation per category)
        committee_disc = judge_fb.get("committee_discussions", {})
        if committee_disc:
            console.print(f"[bold]Committee Discussions[/]")
            for cat_name, discussion in committee_disc.items():
                console.print(f"\n  [bold cyan]{cat_name}[/]")
                # Wrap long discussion text
                for line in discussion.split("\n"):
                    if line.strip():
                        console.print(f"    [dim]{line.strip()}[/]")
    else:
        # Fallback: show judge_scores dict if no dimension_results
        judge_scores = ep.get("judge_scores", {})
        if judge_scores:
            console.print(f"\n[bold]Judge Scores[/]  [dim](no rationale available)[/]")
            for dim_name, dim_score in sorted(judge_scores.items()):
                console.print(f"  {score_text(dim_score)}  {dim_name}")

    # Dialogue history
    dialogue = ep.get("dialogue_history", [])
    if dialogue:
        console.print(f"\n[bold]Dialogue ({len(dialogue)} messages):[/]")
        for msg in dialogue:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if role == "patient":
                console.print(f"  [cyan]Patient:[/] {content}")
            elif role == "doctor":
                console.print(f"  [green]Doctor:[/]  {content}")
            else:
                console.print(f"  [dim]{role}:[/]    {content}")


def _show_all_rationales(client, sim_id: str, episodes: list[dict]) -> None:
    """Fetch full episode details and display all committee rationales grouped by dimension."""
    console.print("  [dim]Loading episode details for rationales...[/]")

    # Collect rationales: { dimension_id: [ {episode, score, rationale, ...}, ... ] }
    from collections import defaultdict
    dim_rationales: dict[str, list[dict]] = defaultdict(list)

    for ep in episodes:
        ep_id = ep.get("episode_id")
        ep_num = ep.get("episode_number", 0) + 1
        patient = ep.get("patient_name") or ep.get("patient_id") or "?"

        try:
            ep_full = client.simulations.get_episode(sim_id, ep_id)
        except Exception as e:
            _log.debug("Episode detail fetch failed: %s", e, exc_info=True)
            continue

        jf = ep_full.get("judge_feedback", {})
        for dr in jf.get("dimension_results", []):
            dim_id = dr.get("dimension_id", "?")
            jr = dr.get("judge_result", {}) or {}
            dim_rationales[dim_id].append({
                "episode": ep_num,
                "patient": patient,
                "score": jr.get("score"),
                "rationale": jr.get("rationale", ""),
                "committee_insights": jr.get("committee_insights", ""),
                "confidence": jr.get("confidence"),
                "category": dr.get("category", ""),
            })

    if not dim_rationales:
        warn("No committee rationales available for this simulation.")
        return

    # Display grouped by dimension
    console.print(f"\n[bold]Committee Rationales — All Episodes[/]")
    console.print()

    for dim_id in sorted(dim_rationales.keys()):
        entries = dim_rationales[dim_id]
        scores = [e["score"] for e in entries if e["score"] is not None]
        avg = sum(scores) / len(scores) if scores else None
        category = entries[0].get("category", "")
        cat_tag = f"  [dim][{category}][/]" if category else ""

        avg_display = f"avg {avg:.1f}/4" if avg is not None else ""
        console.print(f"  [bold]{dim_id}[/]{cat_tag}  [dim]{avg_display}[/]")
        console.print(f"  {'─' * 60}")

        for entry in entries:
            ep_label = f"Ep {entry['episode']}"
            patient = entry["patient"][:16]
            sc = score_text(entry["score"]) if entry["score"] is not None else Text("-", style="dim")
            console.print(f"    {sc}  [bold]{ep_label}[/] ({patient})")
            if entry["rationale"]:
                console.print(f"         [italic]{entry['rationale']}[/]")
            if entry["committee_insights"]:
                console.print(f"         [dim]Committee: {entry['committee_insights']}[/]")

        console.print()


def _save_simulation(client, sim_data: dict, run_store: RunStore) -> None:
    """Download and save a remote simulation locally."""
    sim_id = sim_data["id"]
    try:
        report = client.simulations.get_report(sim_id)
    except Exception as e:
        _log.debug("Report fetch failed: %s", e, exc_info=True)
        report = {"simulation_id": sim_id, "note": "Report fetch failed"}

    meta = LocalRun(
        simulation_id=sim_id,
        pipeline_name=sim_data["pipeline"],
        status=sim_data["status"],
        total_episodes=sim_data["total"],
        completed_episodes=sim_data["completed"],
        average_score=sim_data.get("score"),
        started_at=sim_data.get("started", ""),
    )
    run_store.save_run(meta, report)
    sim_data["source"] = "both"
    success(f"Saved to ~/.earl/runs/{meta.short_id}/")


def _show_report(client, sim_id: str, run_store: RunStore) -> None:
    """Fetch and display the full report."""
    console.print("  [dim]Loading report...[/]")

    report = run_store.load_report(sim_id)
    if not report:
        try:
            report = client.simulations.get_report(sim_id)
        except Exception as e:
            error(f"Failed to load report: {e}")
            return

    # Display summary section
    summary = report.get("summary", {})
    if summary:
        lines = []
        for k, v in summary.items():
            if isinstance(v, float):
                lines.append(f"[bold]{k}:[/]  {v:.2f}")
            elif isinstance(v, dict):
                lines.append(f"[bold]{k}:[/]  {v}")
            else:
                lines.append(f"[bold]{k}:[/]  {v}")
        info_panel("Report Summary", lines)
    else:
        muted("No summary data available in report.")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_date(dt_str: str) -> str:
    if not dt_str:
        return "-"
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%b %d %H:%M")
    except Exception as e:
        _log.debug("Date format failed: %s", e, exc_info=True)
        return dt_str[:16]


def _status_color(status: str) -> str:
    return {
        "completed": "green",
        "running": "cyan",
        "judging": "yellow",
        "dialogue": "cyan",
        "failed": "red",
        "stopped": "red",
        "pending": "dim",
    }.get(status, "dim")
