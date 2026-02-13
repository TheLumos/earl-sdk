"""Shared UI primitives: menus, prompts, tables, panels.

Wraps Rich + Questionary into a consistent API used by all flows.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import questionary
from questionary import Choice, Style

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# ── Questionary style (indigo/purple theme matching EARL branding) ────────────

STYLE = Style([
    ("qmark", "fg:#8b5cf6 bold"),       # purple
    ("question", "fg:white bold"),
    ("pointer", "fg:#8b5cf6 bold"),
    ("highlighted", "fg:#8b5cf6 bold"),
    ("selected", "fg:#22c55e bold"),      # green
    ("answer", "fg:#22c55e bold"),
    ("instruction", "fg:#858585"),
])

_BACK = "__back__"

# ── Colors ────────────────────────────────────────────────────────────────────

ACCENT = "bold #8b5cf6"        # purple
GOOD = "bold green"
BAD = "bold red"
WARN = "bold yellow"
MUTED = "dim"
INFO = "bold cyan"


# ── Selection helpers ─────────────────────────────────────────────────────────


def select_one(
    title: str,
    choices: list[tuple[str, str]],
    allow_back: bool = True,
) -> str | None:
    """Arrow-key single-select menu. Returns key or None for back/cancel.

    Choices are (key, label) tuples. Keys starting with ``__`` become
    disabled separator headers.
    """
    items: list[Choice | questionary.Separator] = []
    for key, label in choices:
        if key.startswith("__"):
            items.append(questionary.Separator(f"  {label}"))
        else:
            items.append(Choice(title=label, value=key))
    if allow_back:
        items.append(Choice(title="← Back", value=_BACK))

    try:
        result = questionary.select(
            title,
            choices=items,
            style=STYLE,
            qmark="›",
            instruction="(↑↓ navigate, enter select)",
        ).ask()
    except KeyboardInterrupt:
        return None

    if result is None or result == _BACK:
        return None
    return result


def select_many(
    title: str,
    choices: list[tuple[str, str]],
    min_count: int = 1,
) -> list[str]:
    """Arrow-key multi-select (space to toggle, enter to confirm)."""
    items = [Choice(title=label, value=key) for key, label in choices]

    try:
        result = questionary.checkbox(
            title,
            choices=items,
            style=STYLE,
            qmark="›",
            instruction="(↑↓ navigate, space toggle, enter confirm)",
            validate=lambda ans: len(ans) >= min_count or f"Select at least {min_count}",
        ).ask()
    except KeyboardInterrupt:
        return []

    return result if result else []


def ask_confirm(message: str, default: bool = True) -> bool:
    """Yes/no confirmation prompt."""
    try:
        result = questionary.confirm(
            message, default=default, style=STYLE, qmark="›",
        ).ask()
    except KeyboardInterrupt:
        return False
    return bool(result)


def ask_text(message: str, default: str = "", secret: bool = False) -> str | None:
    """Free-text input prompt."""
    fn = questionary.password if secret else questionary.text
    try:
        result = fn(message, default=default, style=STYLE, qmark="›").ask()
    except KeyboardInterrupt:
        return None
    return result


def ask_int(message: str, default: int = 1, min_val: int = 1, max_val: int = 100) -> int | None:
    """Integer input with validation."""
    try:
        result = questionary.text(
            message,
            default=str(default),
            style=STYLE,
            qmark="›",
            validate=lambda v: (
                v.isdigit() and min_val <= int(v) <= max_val
            ) or f"Enter a number between {min_val} and {max_val}",
        ).ask()
    except KeyboardInterrupt:
        return None
    return int(result) if result else None


# ── Rich rendering helpers ────────────────────────────────────────────────────


def banner(title: str, subtitle: str = "", status_lines: list[str] | None = None) -> None:
    """Print a branded banner panel."""
    lines = [f"[bold white]{title}[/]"]
    if subtitle:
        lines.append(f"[dim]{subtitle}[/]")
    if status_lines:
        lines.append("")
        lines.extend(status_lines)
    console.print(Panel(
        "\n".join(lines),
        border_style="#8b5cf6",
        padding=(1, 2),
    ))


def info_panel(title: str, lines: list[str], border: str = "cyan") -> None:
    """Print an info panel with key-value style lines."""
    console.print(Panel(
        "\n".join(lines),
        title=f"[bold {border}]{title}[/]",
        border_style=border,
        expand=False,
        padding=(1, 2),
    ))


def success(msg: str) -> None:
    console.print(f"  [green]✓[/] {msg}")


def error(msg: str) -> None:
    console.print(f"  [red]✗[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]![/] {msg}")


def muted(msg: str) -> None:
    console.print(f"  [dim]{msg}[/]")


def score_style(score: float) -> str:
    """Color style for a 1-4 score."""
    if score >= 3.0:
        return GOOD
    if score >= 2.0:
        return WARN
    return BAD


def score_text(score: float) -> Text:
    """Colored score text."""
    return Text(f"{score:.1f}", style=score_style(score))


def status_style(status: str) -> str:
    """Color style for simulation/episode status."""
    match status:
        case "completed":
            return "green"
        case "running" | "judging" | "dialogue":
            return "cyan"
        case "failed" | "stopped":
            return "red"
        case "pending":
            return "yellow"
        case _:
            return "dim"


def kvtable(rows: list[tuple[str, str]], title: str = "") -> None:
    """Render a key→value table (no header row)."""
    t = Table(show_header=False, expand=False, box=None, padding=(0, 2))
    t.add_column(style="bold")
    t.add_column()
    for k, v in rows:
        t.add_row(k, v)
    if title:
        console.print(f"\n[bold]{title}[/]")
    console.print(t)


def datatable(
    columns: list[tuple[str, str]],
    rows: list[list[str | Text]],
    title: str = "",
) -> None:
    """Render a data table with headers.

    *columns* is a list of ``(header, style)`` tuples.
    """
    t = Table(title=title, expand=False, border_style="#444444")
    for header, style in columns:
        t.add_column(header, style=style)
    for row in rows:
        t.add_row(*[str(c) if not isinstance(c, Text) else c for c in row])
    console.print(t)
