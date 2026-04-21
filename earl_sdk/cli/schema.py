"""Machine-readable description of the ``earl`` CLI.

Walks an :mod:`argparse` parser tree and emits a JSON document suitable for
LLM agents (and shell completions / docs generators) that need to reason
about every subcommand without running ``--help`` 40+ times.

Two output formats:

- ``json``: structured, stable schema.  See :func:`schema_dict`.
- ``markdown``: human-readable command reference; useful for docs sites.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from typing import Any

# ── Helpers ───────────────────────────────────────────────────────────────────


def _arg_type_name(action: argparse.Action) -> str:
    """Best-effort string name for an action's expected value type."""
    if isinstance(action, argparse._StoreTrueAction | argparse._StoreFalseAction):
        return "bool"
    if isinstance(action, argparse.BooleanOptionalAction):
        return "bool"
    if action.type is None:
        return "string"
    t = action.type
    name = getattr(t, "__name__", None) or str(t)
    # Normalise common aliases
    return {
        "int": "integer",
        "float": "number",
        "bool": "bool",
        "str": "string",
    }.get(name, name)


def _action_to_dict(action: argparse.Action) -> dict[str, Any] | None:
    """Convert a single :class:`argparse.Action` to a schema fragment.

    Returns ``None`` for actions that should not appear in the schema
    (help, subparser container, version).
    """
    if isinstance(action, argparse._HelpAction | argparse._SubParsersAction):
        return None
    if isinstance(action, argparse._VersionAction):
        return None

    flags = list(action.option_strings)
    entry: dict[str, Any] = {
        "name": action.dest,
        "flags": flags,  # empty for positionals
        "required": bool(getattr(action, "required", False)) and not flags,  # positionals
        "type": _arg_type_name(action),
        "help": action.help or "",
    }
    if flags:
        entry["required"] = bool(getattr(action, "required", False))
    if action.choices:
        entry["choices"] = list(action.choices) if isinstance(action.choices, (list, tuple, set)) else list(action.choices)
    if action.default not in (None, argparse.SUPPRESS, [], False):
        try:
            json.dumps(action.default)  # ensure JSON-serialisable
            entry["default"] = action.default
        except TypeError:
            entry["default"] = str(action.default)
    if isinstance(action, argparse.BooleanOptionalAction):
        entry["type"] = "bool"
        entry["boolean_style"] = "--flag / --no-flag"
    if action.nargs not in (None, 0):
        entry["nargs"] = str(action.nargs)
    return entry


def _iter_direct_actions(parser: argparse.ArgumentParser) -> Iterable[argparse.Action]:
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            continue
        yield a


def _parse_examples(epilog: str | None) -> list[str]:
    """Pull one-line examples out of an ``epilog`` block.

    Lines starting with the CLI name are treated as examples; everything else
    is returned in ``notes``.
    """
    if not epilog:
        return []
    examples: list[str] = []
    for raw in epilog.splitlines():
        line = raw.strip()
        if line.startswith("earl "):
            examples.append(line)
    return examples


def _parser_to_dict(parser: argparse.ArgumentParser) -> dict[str, Any]:
    node: dict[str, Any] = {
        "description": parser.description or "",
        "examples": _parse_examples(parser.epilog),
        "arguments": [],
        "subcommands": {},
    }
    for action in _iter_direct_actions(parser):
        entry = _action_to_dict(action)
        if entry is not None:
            node["arguments"].append(entry)

    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                # Skip aliases (they are duplicate entries in .choices pointing
                # to the same parser instance with a shorter name).
                if not isinstance(sub, argparse.ArgumentParser):
                    continue
                node["subcommands"][name] = _parser_to_dict(sub)
    return node


# ── Public API ────────────────────────────────────────────────────────────────


def schema_dict(parser: argparse.ArgumentParser, *, version: str = "") -> dict[str, Any]:
    """Return the full CLI schema as a JSON-serialisable dict."""
    root = _parser_to_dict(parser)
    return {
        "name": parser.prog,
        "version": version,
        "description": root["description"],
        "global_arguments": root["arguments"],
        "commands": root["subcommands"],
    }


def schema_json(parser: argparse.ArgumentParser, *, version: str = "", indent: int = 2) -> str:
    return json.dumps(schema_dict(parser, version=version), indent=indent, default=str)


# ── Markdown renderer ─────────────────────────────────────────────────────────


def _render_markdown(node: dict[str, Any], path: list[str], out: list[str]) -> None:
    heading_level = max(2, min(6, len(path) + 1))
    heading = "#" * heading_level
    title = "earl" if not path else "earl " + " ".join(path)
    out.append(f"{heading} `{title}`")
    out.append("")
    if node.get("description"):
        out.append(node["description"])
        out.append("")
    if node.get("arguments"):
        out.append("| Flag | Type | Required | Default | Description |")
        out.append("|------|------|----------|---------|-------------|")
        for arg in node["arguments"]:
            flags = ", ".join(f"`{f}`" for f in arg.get("flags", [])) or f"`{arg['name']}`"
            default = arg.get("default", "")
            choices = arg.get("choices")
            desc = arg.get("help") or ""
            if choices:
                desc = f"{desc} *(choices: {', '.join(map(str, choices))})*".strip()
            out.append(
                f"| {flags} | {arg['type']} | {'yes' if arg['required'] else 'no'} "
                f"| `{default}` | {desc} |"
            )
        out.append("")
    if node.get("examples"):
        out.append("**Examples:**")
        out.append("")
        out.append("```bash")
        for ex in node["examples"]:
            out.append(ex)
        out.append("```")
        out.append("")
    for sub_name, sub in sorted(node.get("subcommands", {}).items()):
        _render_markdown(sub, path + [sub_name], out)


def schema_markdown(parser: argparse.ArgumentParser, *, version: str = "") -> str:
    spec = schema_dict(parser, version=version)
    out: list[str] = []
    out.append(f"# `{spec['name']}` CLI reference")
    if version:
        out.append(f"_Version {version}_")
    out.append("")
    if spec.get("description"):
        out.append(spec["description"])
        out.append("")
    root_node = {
        "description": "",
        "arguments": spec.get("global_arguments", []),
        "examples": [],
        "subcommands": spec.get("commands", {}),
    }
    # Render global args as a top-level section.
    if root_node["arguments"]:
        out.append("## Global flags")
        out.append("")
        out.append("| Flag | Type | Required | Default | Description |")
        out.append("|------|------|----------|---------|-------------|")
        for arg in root_node["arguments"]:
            flags = ", ".join(f"`{f}`" for f in arg.get("flags", [])) or f"`{arg['name']}`"
            default = arg.get("default", "")
            choices = arg.get("choices")
            desc = arg.get("help") or ""
            if choices:
                desc = f"{desc} *(choices: {', '.join(map(str, choices))})*".strip()
            out.append(
                f"| {flags} | {arg['type']} | {'yes' if arg['required'] else 'no'} "
                f"| `{default}` | {desc} |"
            )
        out.append("")

    for sub_name, sub in sorted(root_node["subcommands"].items()):
        _render_markdown(sub, [sub_name], out)

    return "\n".join(out).rstrip() + "\n"
