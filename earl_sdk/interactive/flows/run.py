"""Run Simulation flow — case-driven evaluation.

Pick a clinical case → configure doctor → set episodes →
run with live progress → view rich results → browse dialogues → export.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
import time
from datetime import datetime
from typing import Optional

_log = logging.getLogger("earl.ui")

from rich.live import Live
from rich.panel import Panel
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
    success,
    warn,
)
from ..storage.config_store import ConfigStore
from ..storage.run_store import LocalRun, RunStore


# =============================================================================
# Entry Point
# =============================================================================


def flow_run(client, store: ConfigStore, run_store: RunStore, *, pipeline_name: str | None = None) -> None:
    """Case-driven evaluation flow."""

    if pipeline_name:
        _run_existing_pipeline(client, store, run_store, pipeline_name)
        return

    console.print("\n[bold]Evaluate Doctor[/]")
    muted("Run a clinical case evaluation with your doctor API.\n")

    # ── Step 1: Select case ──────────────────────────────────────────────

    console.print("  [dim]Loading cases...[/]")
    try:
        cases = client.cases.list()
    except Exception as e:
        error(f"Failed to load cases: {e}")
        return

    if not cases:
        error("No cases available yet.")
        return

    case_choices = []
    for c in cases:
        cv = c.get("case_verifiers", 0)
        hg = c.get("hard_gates", 0)
        sd = c.get("scoring_dimensions", 0)
        label = c.get("name", "") or c.get("case_snapshot", {}).get("name", "") or c["case_id"]
        case_choices.append((
            c["case_id"],
            f"{label}  ({cv} verifiers, {hg} gates, {sd} scoring dims)"
        ))

    case_id = select_one("Select clinical case", case_choices)
    if not case_id:
        return

    try:
        case_detail = client.cases.get(case_id)
    except Exception as e:
        error(f"Failed to load case: {e}")
        return

    totals = case_detail.get("totals", {})

    info_panel(f"Case: {case_detail.get('name', case_id)}", [
        f"[bold]ID:[/]            {case_id}",
        f"[bold]Patient:[/]       {case_detail.get('patient_id', '')}",
        f"[bold]Specialty:[/]     {case_detail.get('medical_speciality', 'N/A')}",
        f"[bold]Encounter:[/]     {case_detail.get('encounter_type', 'N/A')}",
        "",
        f"[bold]Case Verifiers:[/]       {totals.get('case_verifiers', 0)}",
        f"[bold]Hard Gates:[/]           {totals.get('hard_gates', 0)}",
        f"[bold]Scoring Dimensions:[/]   {totals.get('scoring_dimensions', 0)}",
    ])
    if case_detail.get("description"):
        muted(f"  {case_detail['description'][:200]}")

    # ── Step 2: Doctor ───────────────────────────────────────────────────

    console.print()
    doctor_config, doctor_label = _pick_doctor(client, store)
    if doctor_config is None:
        return

    # ── Step 3: Extra verifiers (optional) ───────────────────────────────

    extra_verifiers = []
    if ask_confirm("Add extra verifiers on top of case defaults?", default=False):
        console.print("  [dim]Loading generic verifiers catalog (API)...[/]")
        from .explore import _build_additional_verifier_choices

        choices = _build_additional_verifier_choices(client, case_detail)
        if choices:
            extra = select_many("Select additional verifiers", choices)
            if extra:
                extra_verifiers = extra
                success(f"Added {len(extra)} extra verifiers")
        else:
            warn("No generic verifiers available to add (catalog empty or unreachable).")

    # ── Step 4: Run parameters ───────────────────────────────────────────

    num_episodes = ask_int("Number of episodes", default=5, min_val=1, max_val=100)
    if num_episodes is None:
        return

    parallel_count = ask_int(
        "Parallel episodes (1-10)",
        default=min(3, num_episodes), min_val=1, max_val=min(10, num_episodes),
    )
    if parallel_count is None:
        return

    max_turns = ask_int("Max turns per episode", default=50, min_val=1, max_val=50)
    if max_turns is None:
        return

    initiator = select_one("Who starts?", [
        ("doctor",  "Doctor initiates"),
        ("patient", "Patient initiates"),
    ])
    if not initiator:
        return

    # ── Step 5: Confirm ──────────────────────────────────────────────────

    console.print()
    lines = [
        f"[bold]Case:[/]          {case_detail.get('name', case_id)}",
        f"[bold]Doctor:[/]        {doctor_label}",
        f"[bold]Episodes:[/]      {num_episodes} ({parallel_count} parallel)",
        f"[bold]Max Turns:[/]     {max_turns}",
        f"[bold]Initiator:[/]     {initiator}",
        f"[bold]Verifiers:[/]     {totals.get('case_verifiers', 0)} case + {totals.get('hard_gates', 0)} gates + {totals.get('scoring_dimensions', 0)} scoring",
    ]
    if extra_verifiers:
        lines.append(f"[bold]Extra:[/]         +{len(extra_verifiers)} verifiers")

    info_panel("Run Configuration", lines)

    if not ask_confirm("Start simulation?"):
        muted("Cancelled.")
        return

    # ── Step 6: Create pipeline & launch ─────────────────────────────────

    console.print()
    pipeline_name = f"case-{case_id}-{int(time.time())}"
    try:
        client.pipelines.create(
            name=pipeline_name,
            doctor_config=doctor_config,
            verifier_ids=extra_verifiers or None,
            validate_doctor=False,
            conversation_initiator=initiator,
            max_turns=max_turns,
            verifiers="lumos",
            case_id=case_id,
        )
        success(f"Pipeline created: {pipeline_name}")
    except Exception as e:
        error(f"Failed to create pipeline: {e}")
        return

    try:
        sim = client.simulations.create(
            pipeline_name=pipeline_name,
            num_episodes=num_episodes,
            parallel_count=parallel_count,
        )
        success(f"Simulation started: {sim.id[:12]}...")
    except Exception as e:
        error(f"Failed to start simulation: {e}")
        return

    # ── Step 7: Live progress ────────────────────────────────────────────

    console.print()
    final_sim = _track_progress(client, sim.id, num_episodes)
    if not final_sim:
        error("Lost connection while tracking simulation.")
        return

    # ── Step 8: Results ──────────────────────────────────────────────────

    _show_results(client, final_sim, pipeline_name, case_detail, store, run_store)


# =============================================================================
# Doctor Selection
# =============================================================================


def _pick_doctor(client, store: ConfigStore):
    """Interactive doctor selection. Returns (DoctorApiConfig, label) or (None, None)."""
    from earl_sdk.models import DoctorApiConfig

    saved = store.list_doctor_configs()
    choices = [
        ("internal", "Lumos Internal Doctor  — built-in AI, no setup needed"),
    ]
    for dc in saved:
        url = f"  → {dc.api_url}" if dc.api_url else ""
        choices.append((f"saved:{dc.name}", f"{dc.name}{url}"))
    choices.append(("new", "Enter Doctor API URL   — provide endpoint + key now"))

    pick = select_one("Select doctor", choices)
    if not pick:
        return None, None

    if pick == "internal":
        return DoctorApiConfig.internal(), "Lumos Internal"

    if pick.startswith("saved:"):
        dc = next((d for d in saved if d.name == pick[6:]), None)
        if dc:
            cfg = DoctorApiConfig.external(api_url=dc.api_url, api_key=dc.key_clear() or None, auth_type=dc.auth_type)
            label = f"{dc.name} → {dc.api_url}"
            if not _warmup_check(cfg):
                return None, None
            return cfg, label

    if pick == "new":
        url = ask_text("Doctor API URL")
        if not url:
            return None, None
        key = ask_text("API Key (optional)", secret=True) or None
        cfg = DoctorApiConfig.external(api_url=url, api_key=key)
        if not _warmup_check(cfg):
            return None, None
        return cfg, url

    return None, None


def _warmup_check(cfg) -> bool:
    """Ping external doctor API. Returns True if reachable."""
    if not cfg.api_url:
        return True
    console.print(f"  [dim]Pinging {cfg.api_url}...[/]")
    try:
        import urllib.request, urllib.error
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        body = json.dumps({"model": "default", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}).encode()
        req = urllib.request.Request(cfg.api_url, data=body, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=10)
        success(f"Doctor API is up (HTTP {resp.status})")
        return True
    except urllib.error.HTTPError as e:
        if e.code < 500:
            success(f"Doctor API is up (HTTP {e.code})")
            return True
        error(f"Doctor API returned HTTP {e.code}. It may be cold-starting. Try again later.")
        return False
    except (urllib.error.URLError, TimeoutError, OSError):
        error("Doctor API did not respond within 10s. It may be cold-starting (up to 40 min). Try again later.")
        return False
    except Exception as e:
        warn(f"Could not verify doctor API: {e}")
        return True


# =============================================================================
# Shortcut: run an existing pipeline directly
# =============================================================================


def _run_existing_pipeline(client, store: ConfigStore, run_store: RunStore, pipeline_name: str) -> None:
    """Run a named pipeline (called from create_pipeline_wizard or other flows)."""
    console.print(f"\n  [dim]Loading pipeline '{pipeline_name}'...[/]")
    try:
        pipeline = client.pipelines.get(pipeline_name)
    except Exception as e:
        error(f"Failed to load pipeline: {e}")
        return

    default_episodes = len(pipeline.patient_ids) if pipeline.patient_ids else 1
    num_episodes = ask_int(f"Number of episodes", default=default_episodes, min_val=1, max_val=100)
    if num_episodes is None:
        return

    parallel = ask_int("Parallel episodes (1-10)", default=min(3, num_episodes), min_val=1, max_val=min(10, num_episodes))
    if parallel is None:
        return

    if not ask_confirm(f"Start {num_episodes} episodes on '{pipeline_name}'?"):
        return

    try:
        sim = client.simulations.create(pipeline_name=pipeline_name, num_episodes=num_episodes, parallel_count=parallel)
        success(f"Simulation started: {sim.id[:12]}...")
    except Exception as e:
        error(f"Failed to start simulation: {e}")
        return

    console.print()
    final_sim = _track_progress(client, sim.id, num_episodes)
    if not final_sim:
        return

    _show_results(client, final_sim, pipeline_name, {}, store, run_store)


# =============================================================================
# Live Progress
# =============================================================================

_ICON = {
    "pending": "⏳", "conversation": "💬", "awaiting_doctor": "🩺",
    "judging": "⚖️ ", "completed": "✅", "failed": "❌",
}


def _build_progress_table(sim_status: str, episodes: list, total: int, elapsed: int):
    from rich.table import Table

    by_status: dict[str, int] = {}
    for ep in episodes:
        s = ep.get("status", "pending")
        by_status[s] = by_status.get(s, 0) + 1

    counts = []
    for s in ("completed", "judging", "conversation", "awaiting_doctor", "pending", "failed"):
        n = by_status.get(s, 0)
        if n:
            counts.append(f"{_ICON.get(s, '?')} {n}")
    summary = "   ".join(counts) or "starting..."

    mins, secs = divmod(elapsed, 60)
    t = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    tbl = Table(
        show_header=False, show_edge=False, pad_edge=False,
        title=f"[bold]{sim_status.capitalize()}[/]  {len(episodes)}/{total} episodes  ({t})   {summary}",
        title_style="",
    )
    tbl.add_column("icon", width=3, no_wrap=True)
    tbl.add_column("ep", width=7)
    tbl.add_column("detail", style="dim")

    for ep in episodes:
        s = ep.get("status", "pending")
        icon = _ICON.get(s, "?")
        num = ep.get("episode_number", "?")
        detail = ""
        if s in ("conversation", "awaiting_doctor"):
            turns = ep.get("dialogue_turns") or 0
            detail = f"turn {turns}"
        elif s == "judging":
            cd = ep.get("categories_completed", 0)
            ct = ep.get("categories_queued") or ep.get("total_categories", 0)
            detail = f"verifiers {cd}/{ct}" if ct else "scoring..."
        elif s == "completed":
            sc = ep.get("total_score")
            detail = f"score {sc:.2f}" if sc is not None else "done"
        elif s == "failed":
            err = ep.get("error") or "failed"
            detail = (err[:40] + "...") if len(err) > 40 else err
        tbl.add_row(f" {icon}", f"[{_color(s)}]ep #{num}[/]", detail)

    return tbl


def _track_progress(client, sim_id: str, total: int):
    start = time.time()
    last_fetch = 0.0
    episodes: list = []

    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                try:
                    sim = client.simulations.get(sim_id)
                except Exception:
                    time.sleep(3)
                    continue

                status = sim.status.value if hasattr(sim.status, "value") else str(sim.status)
                elapsed = int(time.time() - start)

                now = time.time()
                if now - last_fetch >= 10:
                    last_fetch = now
                    try:
                        episodes = client.simulations.get_episodes(sim_id)
                    except Exception:
                        pass

                live.update(_build_progress_table(status, episodes, total, elapsed))

                if status in ("completed", "failed", "stopped"):
                    break
                time.sleep(3)

        return sim
    except KeyboardInterrupt:
        warn("Interrupted — simulation continues on server.")
        muted(f"Simulation ID: {sim_id}")
        return None


# =============================================================================
# Results
# =============================================================================


def _show_results(
    client, sim, pipeline_name: str, case_detail: dict,
    store: ConfigStore, run_store: RunStore,
) -> None:
    status = sim.status.value if hasattr(sim.status, "value") else str(sim.status)

    if status == "failed":
        error(f"Simulation failed: {sim.error_message or 'unknown'}")
    if status == "stopped":
        warn("Simulation stopped before completing all episodes.")

    report = None
    try:
        report = client.simulations.get_report(sim.id)
    except Exception as e:
        warn(f"Could not fetch report: {e}")

    episodes = report.get("episodes", []) if report else []
    summary = report.get("summary", {}) if report else (sim.summary or {})
    completed = [ep for ep in episodes if ep.get("status") == "completed"]
    failed = [ep for ep in episodes if ep.get("status") == "failed"]

    # ── Score banner ─────────────────────────────────────────────────────

    avg = summary.get("average_score")
    dur = report.get("duration_seconds") if report else None
    case_name = case_detail.get("name", pipeline_name) if case_detail else pipeline_name

    sc = "green" if avg and avg >= 0.7 else "yellow" if avg and avg >= 0.4 else "red" if avg else "dim"
    dur_str = f"{int(dur // 60)}m {int(dur % 60)}s" if dur else "?"

    console.print()
    console.print(Panel(
        f"[bold]Score: [{sc}]{avg:.2f}[/][/]   |   "
        f"Episodes: {len(completed)}/{len(episodes)}   |   "
        f"Duration: {dur_str}   |   "
        f"Case: {case_name}" if avg is not None else
        f"[bold]No scores yet[/]   |   Episodes: {len(completed)}/{len(episodes)}   |   Case: {case_name}",
        border_style=sc, padding=(1, 2),
    ))

    for ep in failed:
        err = ep.get("error") or "unknown"
        warn(f"Episode #{ep.get('episode_number', '?')} failed: {err[:80]}")

    # ── Episode table ────────────────────────────────────────────────────

    if episodes:
        rows = []
        for ep in episodes:
            s = ep.get("status", "?")
            sc_val = ep.get("total_score")
            turns = ep.get("dialogue_turns", 0) or len(ep.get("dialogue_history", []))
            hg = ep.get("hard_gates", [])
            sd = ep.get("scoring_dimensions", [])
            cv = ep.get("case_verifiers", [])

            hg_str = f"{sum(1 for g in hg if g.get('passed'))}/{len(hg)}" if hg else "-"
            sd_vals = [d.get("score", 0) for d in sd if d.get("activated", True) and d.get("score") is not None]
            sd_str = f"{sum(sd_vals)/len(sd_vals):.1f}" if sd_vals else "-"
            cv_trig = [v for v in cv if v.get("triggered")]
            cv_str = f"{len(cv_trig)}/{len(cv)}" if cv else "-"

            rows.append([
                str(ep.get("episode_number", 0)),
                Text(s, style=_color(s)),
                score_text(sc_val) if sc_val is not None else Text("-", style="dim"),
                str(turns),
                hg_str, sd_str, cv_str,
            ])
        datatable(
            columns=[("Ep", "dim"), ("Status", ""), ("Score", ""), ("Turns", "dim"), ("Gates", ""), ("Dims", ""), ("Verifiers", "")],
            rows=rows, title="Episodes",
        )

    # ── Aggregated breakdowns ────────────────────────────────────────────

    _agg_hard_gates(completed)
    _agg_scoring_dims(completed)
    _agg_case_verifiers(completed)

    # ── Post-results menu ────────────────────────────────────────────────

    while True:
        action = select_one("Results", [
            ("dialogues", "View Dialogues         — browse episode conversations"),
            ("rationale", "View Rationales        — per-dimension scoring rationale"),
            ("export",    "Export Report          — save full report as JSON"),
            ("done",      "Done"),
        ])
        if action == "dialogues":
            _view_dialogues(episodes)
        elif action == "rationale":
            _view_rationales(episodes)
        elif action == "export":
            _export_report(report, sim.id)
        elif action == "done" or action is None:
            break

    if report:
        if store.preferences.auto_save_runs or ask_confirm("Save results locally?"):
            _save_run(client, sim, pipeline_name, run_store, report)


# =============================================================================
# Aggregated Views
# =============================================================================


def _agg_hard_gates(episodes: list) -> None:
    stats: dict[str, dict] = {}
    for ep in episodes:
        for g in ep.get("hard_gates", []):
            gid = g.get("id", "?")
            stats.setdefault(gid, {"p": 0, "f": 0})
            stats[gid]["p" if g.get("passed") else "f"] += 1
    if not stats:
        return
    n = len(episodes)
    rows = []
    for gid in sorted(stats):
        s = stats[gid]
        rate = s["p"] / (s["p"] + s["f"]) if (s["p"] + s["f"]) else 0
        c = "green" if rate == 1 else "yellow" if rate >= 0.5 else "red"
        rows.append([gid.replace("hard-gates/", ""), Text(f"{s['p']}/{n}", style=c), Text("PASS" if rate == 1 else f"{rate:.0%}", style=c)])
    datatable(columns=[("Hard Gate", "bold"), ("Passed", ""), ("Rate", "")], rows=rows, title=f"Hard Gates ({n} episodes)")


def _agg_scoring_dims(episodes: list) -> None:
    scores: dict[str, list] = {}
    for ep in episodes:
        for d in ep.get("scoring_dimensions", []):
            if not d.get("activated", True):
                continue
            did = d.get("id", "?")
            sc = d.get("score")
            if sc is not None:
                scores.setdefault(did, []).append(sc)
    if not scores:
        return
    rows = []
    for did in sorted(scores):
        s = scores[did]
        avg = sum(s) / len(s)
        rows.append([did.replace("scoring-dimensions/", ""), score_text(avg), Text(f"{min(s):.1f}", style="dim"), Text(f"{max(s):.1f}", style="dim"), str(len(s))])
    datatable(columns=[("Scoring Dimension", "bold"), ("Avg", ""), ("Min", "dim"), ("Max", "dim"), ("N", "dim")], rows=rows, title=f"Scoring Dimensions ({len(episodes)} episodes)")


def _agg_case_verifiers(episodes: list) -> None:
    stats: dict[str, dict] = {}
    for ep in episodes:
        for v in ep.get("case_verifiers", []):
            vid = v.get("id") or v.get("name", "?")
            stats.setdefault(vid, {"triggered": 0, "total": 0, "pts": []})
            stats[vid]["total"] += 1
            if v.get("triggered"):
                stats[vid]["triggered"] += 1
            stats[vid]["pts"].append(v.get("points_awarded", 0))
    if not stats:
        return
    rows = []
    for vid in sorted(stats):
        s = stats[vid]
        avg = sum(s["pts"]) / len(s["pts"]) if s["pts"] else 0
        c = "green" if avg > 0 else "red" if avg < 0 else "dim"
        rows.append([vid[:40], f"{s['triggered']}/{s['total']}", Text(f"{avg:+.1f}", style=c)])
    datatable(columns=[("Case Verifier", "bold"), ("Triggered", ""), ("Avg Pts", "")], rows=rows, title=f"Case Verifiers ({len(episodes)} episodes)")


# =============================================================================
# Dialogue Viewer
# =============================================================================


def _view_dialogues(episodes: list) -> None:
    if not episodes:
        warn("No episodes.")
        return

    while True:
        choices = []
        for ep in episodes:
            n = ep.get("episode_number", "?")
            s = ep.get("status", "?")
            turns = len(ep.get("dialogue_history", []))
            sc = ep.get("total_score")
            label = f"Episode #{n} — {s}, {turns} turns"
            if sc is not None:
                label += f", score {sc:.2f}"
            choices.append((str(n), label))

        picked = select_one("Select episode", choices)
        if picked is None:
            return

        ep = next((e for e in episodes if str(e.get("episode_number")) == picked), None)
        if not ep:
            continue

        dialogue = ep.get("dialogue_history", [])
        if not dialogue:
            warn("No dialogue history.")
            continue

        console.print(f"\n[bold]Episode #{picked}[/] ({len(dialogue)} messages)\n")
        for i, turn in enumerate(dialogue):
            role = turn.get("role", "?")
            content = turn.get("content", "")
            style = "bold cyan" if role == "doctor" else "bold yellow"
            console.print(f"  [{style}]{role.capitalize()} ({i+1}):[/]")
            for line in content.split("\n"):
                for w in textwrap.wrap(line, 80) or [""]:
                    console.print(f"    {w}")
            console.print()

        sc = ep.get("total_score")
        if sc is not None:
            console.print(f"  [bold]Score:[/] {score_text(sc)}\n")


# =============================================================================
# Rationale Viewer
# =============================================================================


def _view_rationales(episodes: list) -> None:
    if not episodes:
        return
    console.print("\n[bold]Scoring Rationale[/]\n")

    for ep in episodes:
        if ep.get("status") != "completed":
            continue
        n = ep.get("episode_number", "?")
        sc = ep.get("total_score")
        console.print(f"  [bold]Episode #{n}[/]" + (f"  (score: {sc:.2f})" if sc else ""))

        for g in ep.get("hard_gates", []):
            icon = "[green]PASS[/]" if g.get("passed") else "[red]FAIL[/]"
            gid = g.get("id", "?").replace("hard-gates/", "")
            console.print(f"    {icon}  [bold]{gid}[/]")
            r = g.get("rationale", "")
            if r:
                console.print(f"          [italic dim]{r[:120]}[/]")

        for d in ep.get("scoring_dimensions", []):
            if not d.get("activated", True):
                continue
            ds = d.get("score")
            did = d.get("id", "?").replace("scoring-dimensions/", "")
            console.print(f"    {score_text(ds) if ds is not None else Text('-', style='dim')}  [bold]{did}[/]")
            r = d.get("rationale", "")
            if r:
                console.print(f"          [italic dim]{r[:120]}[/]")

        cv = ep.get("case_verifiers", [])
        triggered = [v for v in cv if v.get("triggered")]
        if triggered:
            console.print(f"    [bold]Case verifiers: {len(triggered)}/{len(cv)} triggered[/]")
            for v in triggered:
                pts = v.get("points_awarded", 0)
                c = "green" if pts > 0 else "red"
                console.print(f"      [{c}]{pts:+d}[/]  {v.get('name', v.get('id', '?'))}")
        console.print()


# =============================================================================
# Export
# =============================================================================


def _export_report(report: dict | None, sim_id: str) -> None:
    if not report:
        error("No report data.")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default = f"earl_report_{ts}_{sim_id[:12]}.json"
    filename = ask_text("Filename", default=default) or default
    try:
        with open(filename, "w") as f:
            json.dump(report, f, indent=2, default=str)
        success(f"Exported: {filename} ({os.path.getsize(filename) // 1024}KB)")
    except Exception as e:
        error(f"Export failed: {e}")


# =============================================================================
# Save Run
# =============================================================================


def _save_run(client, sim, pipeline_name: str, run_store: RunStore, report: dict | None = None) -> None:
    if not report:
        try:
            report = client.simulations.get_report(sim.id)
        except Exception:
            report = {"simulation_id": sim.id}

    status = sim.status.value if hasattr(sim.status, "value") else str(sim.status)
    avg = sim.summary.get("average_score") if sim.summary else None

    meta = LocalRun(
        simulation_id=sim.id, pipeline_name=pipeline_name, status=status,
        total_episodes=sim.total_episodes, completed_episodes=sim.completed_episodes,
        average_score=avg,
        started_at=str(sim.started_at) if sim.started_at else "",
        finished_at=str(sim.finished_at) if sim.finished_at else "",
        environment=getattr(client, "_environment", ""),
    )
    run_store.save_run(meta, report)
    success(f"Saved locally (~/.earl/runs/{meta.short_id}/)")


def _color(status: str) -> str:
    return {"completed": "green", "running": "cyan", "judging": "yellow", "failed": "red", "stopped": "red", "pending": "dim"}.get(status, "dim")
