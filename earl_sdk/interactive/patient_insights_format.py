"""Format patient API ``insights`` blobs for terminal display (interactive chat).

Orchestrator stores these under each dialogue turn's ``metadata.insights`` for
client-driven episodes. Judge submission still uses only ``role`` + ``content``.
"""

from __future__ import annotations

import json
from typing import Any


def insights_from_message(msg: dict[str, Any]) -> dict[str, Any] | None:
    """Pull insights dict from a dialogue_history entry, if present."""
    meta = msg.get("metadata")
    if not isinstance(meta, dict):
        return None
    ins = meta.get("insights")
    return ins if isinstance(ins, dict) else None


def _as_text(value: Any, max_len: int = 220) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        s = value.strip()
        return s if len(s) <= max_len else s[: max_len - 3] + "..."
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        raw = json.dumps(value, ensure_ascii=False, default=str, indent=2)
    except (TypeError, ValueError):
        raw = str(value)
    if len(raw) > max_len:
        return raw[: max_len - 3] + "..."
    return raw


def format_patient_insights_markup(insights: dict[str, Any] | None) -> str | None:
    """
    Build Rich markup lines summarizing patient internal state for the CLI.

    Returns None if there is nothing meaningful to show.
    """
    if not insights:
        return None

    lines: list[str] = []

    tn = insights.get("turn_number")
    if tn is not None:
        lines.append(f"[dim]Turn[/] [bold]{tn}[/]")

    proc = insights.get("processing_time_ms")
    if proc is not None:
        try:
            sec = float(proc) / 1000.0
            lines.append(f"[dim]Processing[/] {sec:.1f}s")
        except (TypeError, ValueError):
            pass

    sc = insights.get("session_context")
    if isinstance(sc, dict):
        parts = []
        if "trust_level" in sc and sc["trust_level"] is not None:
            parts.append(f"[green]Trust[/] {sc['trust_level']}")
        if sc.get("emotional_damage"):
            parts.append(f"[red]Damage[/] {sc['emotional_damage']}")
        if sc.get("hostile_streak"):
            parts.append(f"[yellow]Hostile streak[/] {sc['hostile_streak']}")
        if parts:
            lines.append(" · ".join(parts))

    internal = insights.get("internal_state")
    if isinstance(internal, dict):
        eng = internal.get("engagement_level")
        if eng is not None:
            lines.append(f"[magenta]Engagement[/] {eng}/10")
        for key, label, style in (
            ("gut_reaction", "Gut reaction", "yellow"),
            ("memories_surfaced", "Memories", "blue"),
            ("after_memories", "After processing", "cyan"),
        ):
            val = internal.get(key)
            if val and not (
                key == "memories_surfaced"
                and isinstance(val, str)
                and "TRIVIAL UTTERANCE" in val
            ):
                txt = _as_text(val, 200)
                if txt:
                    lines.append(f"[{style}]{label}:[/] {txt}")

        emo = internal.get("emotional_state")
        if emo is not None:
            if isinstance(emo, str) and emo.strip():
                lines.append(f"[bright_magenta]Emotional state:[/] {emo.strip()[:200]}")
            elif isinstance(emo, dict):
                summ = emo.get("summary")
                if isinstance(summ, str) and summ.strip():
                    lines.append(f"[bright_magenta]Emotional state:[/] {summ.strip()[:200]}")
                else:
                    lines.append(f"[bright_magenta]Emotional state:[/]\n{_as_text(emo, 400)}")

    thoughts = insights.get("thoughts")
    if thoughts:
        items: list[Any]
        if isinstance(thoughts, list):
            items = thoughts
        else:
            items = [thoughts]
        preview: list[str] = []
        for t in items[:3]:
            preview.append(_as_text(t, 120))
        preview = [p for p in preview if p]
        if preview:
            lines.append("[purple]Thoughts:[/] " + " · ".join(f'"{p}"' for p in preview))
            if len(items) > 3:
                lines.append(f"[dim](+{len(items) - 3} more)[/]")

    if not lines:
        return None
    return "\n".join(lines)
