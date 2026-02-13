"""Compare Runs flow — side-by-side comparison of simulation results.

Select 2-5 simulations and view a delta table showing score differences
across dimensions, episodes, and overall metrics.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

_log = logging.getLogger("earl.ui")

from rich.table import Table
from rich.text import Text

from ..ui import (
    console,
    datatable,
    error,
    info_panel,
    muted,
    score_text,
    select_many,
    select_one,
    warn,
)
from ..storage.run_store import RunStore


def flow_compare(client, run_store: RunStore) -> None:
    """Compare flow: select runs, then show side-by-side comparison."""
    while True:
        # Load all available simulations (remote + local)
        sims = _load_all(client, run_store)
        if len(sims) < 2:
            warn("Need at least 2 simulations to compare. Run more evaluations first.")
            return

        # Multi-select
        choices = []
        for s in sims:
            score_hint = f"  score={s['score']:.1f}" if s.get("score") is not None else ""
            choices.append((
                s["id"],
                f"{_format_date(s.get('started', ''))}  {s['pipeline'][:16]}  [{s['status']}]{score_hint}",
            ))

        selected = select_many(
            "Select simulations to compare (2-5, space to toggle)",
            choices,
            min_count=2,
        )
        if len(selected) < 2:
            return

        selected_sims = [s for s in sims if s["id"] in selected]
        _show_comparison(client, run_store, selected_sims)

        again = select_one("Compare", [
            ("again", "Compare different runs  — pick a new set of simulations to compare"),
        ])
        if again is None:
            return


def _load_all(client, run_store: RunStore) -> list[dict]:
    """Load simulations from both remote and local."""
    results: list[dict] = []
    seen_ids: set[str] = set()

    # Remote
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
            seen_ids.add(s.id)
    except Exception as e:
        _log.debug("Remote simulations load failed: %s", e, exc_info=True)

    # Local
    for lr in run_store.list_runs(limit=50):
        if lr.simulation_id in seen_ids:
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
        })

    results.sort(key=lambda r: r.get("started", ""), reverse=True)
    return results


def _show_comparison(client, run_store: RunStore, sims: list[dict]) -> None:
    """Render side-by-side comparison table."""
    console.print()

    # ── Overview table ────────────────────────────────────────────────────

    t = Table(title="Comparison", expand=False, border_style="#444444")
    t.add_column("Metric", style="bold")
    for i, s in enumerate(sims):
        label = f"Run {i+1}\n{s['id'][:8]}"
        t.add_column(label, justify="center")

    # Add delta column if exactly 2 runs
    has_delta = len(sims) == 2
    if has_delta:
        t.add_column("Delta", justify="center", style="bold")

    # Pipeline row
    values = [s["pipeline"][:16] for s in sims]
    row = ["Pipeline"] + values
    if has_delta:
        row.append("")
    t.add_row(*row)

    # Status row
    values = [Text(s["status"], style=_status_color(s["status"])) for s in sims]
    row_parts: list[str | Text] = ["Status"] + values
    if has_delta:
        row_parts.append("")
    t.add_row(*[str(x) if isinstance(x, str) else x for x in row_parts])

    # Episodes row
    values_str = [f"{s['completed']}/{s['total']}" for s in sims]
    row = ["Episodes"] + values_str
    if has_delta:
        row.append("")
    t.add_row(*row)

    # Score row
    score_vals: list[str | Text] = ["Avg Score"]
    scores = []
    for s in sims:
        sc = s.get("score")
        scores.append(sc)
        if sc is not None:
            score_vals.append(score_text(sc))
        else:
            score_vals.append(Text("-", style="dim"))
    if has_delta and scores[0] is not None and scores[1] is not None:
        delta = scores[1] - scores[0]
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        color = "green" if delta > 0 else "red" if delta < 0 else "dim"
        score_vals.append(Text(f"{delta:+.2f} {arrow}", style=color))
    elif has_delta:
        score_vals.append("")
    t.add_row(*[str(x) if isinstance(x, str) else x for x in score_vals])

    # Date row
    values_str = [_format_date(s.get("started", "")) for s in sims]
    row = ["Date"] + values_str
    if has_delta:
        row.append("")
    t.add_row(*row)

    console.print(t)

    # ── Dimension breakdown ───────────────────────────────────────────────

    # Try to get dimension data for each sim
    dim_data: list[dict[str, float]] = []
    for s in sims:
        sim_obj = s.get("sim")
        dims: dict[str, float] = {}
        if sim_obj and sim_obj.summary:
            dim_avgs = sim_obj.summary.get("dimension_averages", {})
            if dim_avgs:
                dims = {k: float(v) for k, v in dim_avgs.items() if v is not None}
        if not dims:
            # Try loading from local report
            report = run_store.load_report(s["id"])
            if report:
                summary = report.get("summary", {})
                dim_avgs = summary.get("dimension_averages", {})
                if dim_avgs:
                    dims = {k: float(v) for k, v in dim_avgs.items() if v is not None}
        dim_data.append(dims)

    # Find all dimension names across runs
    all_dims: set[str] = set()
    for dd in dim_data:
        all_dims.update(dd.keys())

    if all_dims:
        dt = Table(title="Dimension Scores", expand=False, border_style="#444444")
        dt.add_column("Dimension", style="bold")
        for i in range(len(sims)):
            dt.add_column(f"Run {i+1}", justify="center")
        if has_delta:
            dt.add_column("Delta", justify="center", style="bold")

        for dim_name in sorted(all_dims):
            row_parts = [dim_name]
            vals = []
            for dd in dim_data:
                v = dd.get(dim_name)
                vals.append(v)
                row_parts.append(score_text(v) if v is not None else Text("-", style="dim"))
            if has_delta and vals[0] is not None and vals[1] is not None:
                delta = vals[1] - vals[0]
                arrow = "↑" if delta > 0.05 else "↓" if delta < -0.05 else "="
                color = "green" if delta > 0.05 else "red" if delta < -0.05 else "dim"
                row_parts.append(Text(f"{delta:+.2f} {arrow}", style=color))
            elif has_delta:
                row_parts.append("")
            dt.add_row(*[str(x) if isinstance(x, str) else x for x in row_parts])

        console.print(dt)
    else:
        muted("No dimension-level data available for comparison.")


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
        "completed": "green", "running": "cyan", "judging": "yellow",
        "failed": "red", "stopped": "red", "pending": "dim",
    }.get(status, "dim")
