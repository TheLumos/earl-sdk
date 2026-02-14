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


def select_with_preview(
    title: str,
    items: list[tuple[str, str]],
    previews: dict[str, str],
    multi: bool = True,
) -> list[str] | str | None:
    """Interactive list selector with a live detail panel at the bottom.

    *items* are ``(value, label)`` tuples shown in the list.
    *previews* maps ``value → plain-text preview`` shown below the list
    as the user navigates with arrow keys.

    Returns:
        multi=True  → list of selected values (may be empty), or None on ESC
        multi=False → single selected value, or None on ESC
    """
    from prompt_toolkit import Application
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import (
        Dimension,
        FormattedTextControl,
        HSplit,
        Layout,
        VSplit,
        Window,
    )

    if not items:
        return [] if multi else None

    idx = [0]
    chosen: set[str] = set()

    # ── Key bindings ──────────────────────────────────────────────────────
    kb = KeyBindings()

    @kb.add("up")
    def _up(e):
        idx[0] = max(0, idx[0] - 1)

    @kb.add("down")
    def _down(e):
        idx[0] = min(len(items) - 1, idx[0] + 1)

    @kb.add("pageup")
    def _pgup(e):
        idx[0] = max(0, idx[0] - 8)

    @kb.add("pagedown")
    def _pgdn(e):
        idx[0] = min(len(items) - 1, idx[0] + 8)

    @kb.add("home")
    def _home(e):
        idx[0] = 0

    @kb.add("end")
    def _end(e):
        idx[0] = len(items) - 1

    if multi:
        @kb.add("space")
        def _toggle(e):
            v = items[idx[0]][0]
            chosen.symmetric_difference_update({v})

        @kb.add("enter")
        def _done(e):
            e.app.exit(result=list(chosen))
    else:
        @kb.add("enter")
        def _pick(e):
            e.app.exit(result=items[idx[0]][0])

    @kb.add("escape")
    def _esc(e):
        e.app.exit(result=None)

    @kb.add("c-c")
    def _cc(e):
        e.app.exit(result=None)

    # ── Render functions (re-evaluated on every keypress) ─────────────────
    def _render_list():
        fragments: list[tuple[str, str]] = []
        for i, (val, label) in enumerate(items):
            is_cur = i == idx[0]
            pointer = " › " if is_cur else "   "
            if multi:
                mark = " ● " if val in chosen else " ○ "
            else:
                mark = " "
            style = "reverse" if is_cur else ""
            fragments.append((style, f"{pointer}{mark}{label}\n"))
        return fragments

    def _render_preview():
        val = items[idx[0]][0]
        text = previews.get(val, "  No details available")
        return [("", text)]

    def _render_status():
        if multi:
            n = len(chosen)
            return [("fg:ansibrightblack",
                     f"  ↑↓ navigate │ SPACE toggle │ ENTER done ({n} selected) │ ESC cancel")]
        return [("fg:ansibrightblack",
                 "  ↑↓ navigate │ ENTER select │ ESC cancel")]

    # ── Layout: patient list (left) │ detail panel (right) ────────────────
    list_h = max(min(len(items), 20), 5)  # at least 5 rows
    # Panel height = enough for the tallest preview (or at least 18 rows)
    max_preview_lines = max(
        (text.count("\n") + 1 for text in previews.values()),
        default=10,
    )
    panel_h = max(list_h, min(max_preview_lines + 2, 28), 18)

    list_ctrl = FormattedTextControl(
        _render_list,
        focusable=True,
        show_cursor=False,
        get_cursor_position=lambda: Point(0, idx[0]),
    )
    preview_ctrl = FormattedTextControl(_render_preview)

    # Vertical separator
    def _render_separator():
        return [("fg:ansibrightblack", " │\n" * panel_h)]

    left_pane = HSplit([
        Window(
            FormattedTextControl([("bold fg:ansimagenta", f"  {title}")]),
            height=1,
        ),
        Window(
            list_ctrl,
            height=Dimension(min=3, preferred=panel_h - 1),
            width=Dimension(min=20, max=50, preferred=42),
        ),
    ])

    right_pane = HSplit([
        Window(
            FormattedTextControl(
                [("bold fg:ansicyan", "  Patient Details")]
            ),
            height=1,
        ),
        Window(
            preview_ctrl,
            height=Dimension(min=10, preferred=panel_h - 1),
            wrap_lines=True,
        ),
    ])

    body = VSplit([
        left_pane,
        Window(FormattedTextControl(_render_separator), width=2),
        right_pane,
    ])

    layout = Layout(HSplit([
        body,
        Window(FormattedTextControl(_render_status), height=1),
    ]))

    app = Application(layout=layout, key_bindings=kb, full_screen=False)
    try:
        return app.run()
    except KeyboardInterrupt:
        return None


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
