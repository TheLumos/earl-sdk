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

    # Structured verifier results (new format)
    hg = ep.get("hard_gates", [])
    sd = ep.get("scoring_dimensions", [])
    cv = ep.get("case_verifiers", [])

    if hg or sd or cv:
        if hg:
            console.print(f"\n[bold]Hard Gates ({len(hg)})[/]")
            for g in hg:
                status = "[green]PASS[/]" if g.get("passed") else "[red]FAIL[/]"
                console.print(f"  {status}  [bold]{g['id']}[/]  {g.get('score', '?')}/{g.get('max_score', '?')}")
                if g.get("rationale"):
                    console.print(f"       [italic]{g['rationale']}[/]")
                console.print()

        if sd:
            activated = [d for d in sd if d.get("activated", True)]
            skipped = [d for d in sd if not d.get("activated", True)]
            if activated:
                console.print(f"\n[bold]Scoring Dimensions ({len(activated)} evaluated)[/]")
                for d in activated:
                    s = score_text(d.get("score")) if d.get("score") is not None else Text("-", style="dim")
                    console.print(f"  {s}  [bold]{d['id']}[/]  {d.get('score', '?')}/{d.get('max_score', 4)}")
                    if d.get("rationale"):
                        console.print(f"       [italic]{d['rationale']}[/]")
                    console.print()
            if skipped:
                console.print(f"\n  [dim]{len(skipped)} dimensions not activated (not applicable to this conversation)[/]")
                for d in skipped:
                    console.print(f"    [dim]- {d['id']}[/]")

        if cv:
            triggered = [v for v in cv if v.get("triggered")]
            not_triggered = [v for v in cv if not v.get("triggered")]
            total_pts = sum(v.get("points_awarded", 0) for v in cv)
            console.print(f"\n[bold]Case Verifiers ({len(triggered)}/{len(cv)} triggered, {total_pts:+d} pts)[/]")
            for v in triggered:
                pts = v.get("points_awarded", 0)
                color = "green" if pts > 0 else "red"
                console.print(f"  [{color}]{pts:+d}[/{color}]  [bold]{v['id']}[/]")
                if v.get("rationale"):
                    console.print(f"       [italic]{v['rationale']}[/]")
                console.print()
            if not_triggered:
                console.print(f"\n  [dim]{len(not_triggered)} verifiers not triggered[/]")
    else:
        # Fallback: legacy format (dimension_results in judge_feedback or flat judge_scores)
        judge_fb = ep.get("judge_feedback", {})
        dim_results = judge_fb.get("dimension_results", []) if isinstance(judge_fb, dict) else []
        if dim_results:
            console.print(f"\n[bold]Scores & Rationale[/]")
            console.print()
            for dr in dim_results:
                dim_name = dr.get("dimension_name") or dr.get("dimension_id", "?")
                category = dr.get("category", "")
                jr = dr.get("judge_result", {}) or {}
                dim_score = jr.get("score") if isinstance(jr, dict) else dr.get("score")
                rationale = jr.get("rationale", "") or jr.get("reasoning", "")
                score_display = score_text(dim_score) if dim_score is not None else Text("-", style="dim")
                cat_tag = f"  [dim][{category}][/]" if category else ""
                console.print(f"  {score_display}  [bold]{dim_name}[/]{cat_tag}")
                if rationale:
                    console.print(f"       [italic]{rationale}[/]")
                console.print()
        else:
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
    """Fetch and display the full structured report."""
    console.print("  [dim]Loading report...[/]")

    report = run_store.load_report(sim_id)
    if not report:
        try:
            report = client.simulations.get_report(sim_id)
        except Exception as e:
            error(f"Failed to load report: {e}")
            return

    # Summary
    summary = report.get("summary", {})
    status = report.get("status", "unknown")
    pipeline = report.get("pipeline_name", "")
    duration = report.get("duration_seconds")
    dur_str = f"{int(duration)}s" if duration else "-"

    info_panel("Report", [
        f"[bold]Pipeline:[/]     {pipeline}",
        f"[bold]Status:[/]       {status}",
        f"[bold]Duration:[/]     {dur_str}",
        f"[bold]Episodes:[/]     {summary.get('completed', 0)}/{summary.get('total_episodes', 0)} completed, {summary.get('failed', 0)} failed",
        f"[bold]Avg Score:[/]    {summary.get('average_score', 'N/A')}",
    ])

    # Per-episode details
    episodes = report.get("episodes", [])
    for ep in episodes:
        ep_num = ep.get("episode_number", "?")
        patient = ep.get("patient_name") or ep.get("patient_id", "?")
        ep_status = ep.get("status", "?")
        ep_score = ep.get("total_score")
        turns = ep.get("dialogue_turns", 0)
        ep_error = ep.get("error")

        score_str = f"{ep_score:.2f}" if ep_score is not None else "-"
        console.print(f"\n[bold]Episode {ep_num}:[/] {patient}  — {ep_status}  score={score_str}  turns={turns}")

        if ep_error:
            console.print(f"  [red]Error: {ep_error[:200]}[/]")
            continue

        # Hard gates
        hg = ep.get("hard_gates", [])
        if hg:
            passed = sum(1 for g in hg if g.get("passed"))
            color = "green" if passed == len(hg) else "yellow"
            console.print(f"\n  [{color}]Hard Gates: {passed}/{len(hg)} passed[/{color}]")
            for g in hg:
                tag = "[green]PASS[/]" if g.get("passed") else "[red]FAIL[/]"
                console.print(f"    {tag}  {g.get('id', '?')}  ({g.get('score', '?')}/{g.get('max_score', '?')})")
                if g.get("rationale"):
                    console.print(f"          [dim]{g['rationale'][:150]}[/]")

        # Scoring dimensions
        sd = ep.get("scoring_dimensions", [])
        if sd:
            activated = [d for d in sd if d.get("activated", True)]
            skipped = len(sd) - len(activated)
            if activated:
                avg = sum(d.get("score", 0) for d in activated) / len(activated)
                console.print(f"\n  [bold]Scoring Dimensions: {avg:.1f}/4 avg ({len(activated)} evaluated, {skipped} skipped)[/]")
                for d in activated:
                    s = d.get("score", 0)
                    color = "green" if s >= 3 else "yellow" if s >= 2 else "red"
                    console.print(f"    [{color}]{s}/{d.get('max_score', 4)}[/{color}]  {d.get('id', '?')}")
                    if d.get("rationale"):
                        console.print(f"          [dim]{d['rationale'][:150]}[/]")
            if skipped:
                console.print(f"    [dim]{skipped} dimensions not activated (not applicable)[/]")

        # Case verifiers
        cv = ep.get("case_verifiers", [])
        if cv:
            triggered = [v for v in cv if v.get("triggered")]
            total_pts = sum(v.get("points_awarded", 0) for v in cv)
            console.print(f"\n  [bold]Case Verifiers: {len(triggered)}/{len(cv)} triggered ({total_pts:+d} pts)[/]")
            for v in triggered:
                pts = v.get("points_awarded", 0)
                color = "green" if pts > 0 else "red"
                console.print(f"    [{color}]{pts:+d}[/{color}]  {v.get('id', '?')}")
                if v.get("rationale"):
                    console.print(f"          [dim]{v['rationale'][:150]}[/]")
            not_triggered = len(cv) - len(triggered)
            if not_triggered:
                console.print(f"    [dim]{not_triggered} verifiers not triggered[/]")

        # Fallback: flat judge_scores if no structured sections
        if not hg and not sd and not cv:
            js = ep.get("judge_scores", {})
            if js:
                console.print(f"\n  [bold]Scores ({len(js)}):[/]")
                for dk, dv in sorted(js.items()):
                    console.print(f"    {score_text(dv) if isinstance(dv, (int, float)) else dv}  {dk}")

        # Judge report (markdown)
        jf = ep.get("judge_feedback", {})
        if isinstance(jf, dict) and jf.get("report"):
            if jf.get("duration_ms"):
                console.print(f"\n  [dim]Model: {jf.get('model', '?')}, Duration: {jf['duration_ms']}ms[/]")

    console.print()


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
