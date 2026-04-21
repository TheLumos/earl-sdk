"""Rich CLI rendering for :class:`earl_sdk.exceptions.EarlError`.

Two output modes:

- **JSON** (``--json`` or ``--output json``): single-line JSON object on
  stderr with the canonical :meth:`EarlError.to_dict` envelope. Stable
  schema meant for scripts and LLM agents.
- **Human**: multi-line, indented summary with an actionable hint. Adding
  ``--debug`` appends the Python traceback and request/response metadata.

Exit codes follow a simple taxonomy so CI jobs and agents can branch:

- ``2``  — validation / bad input (``ValidationError``)
- ``3``  — authentication required (``AuthenticationError``)
- ``4``  — permission denied (``AuthorizationError``)
- ``5``  — not found (``NotFoundError``)
- ``6``  — rate limited (``RateLimitError``)
- ``7``  — upstream server error (``ServerError``)
- ``8``  — transport / network (``TransportError``)
- ``9``  — simulation failed (``SimulationError``)
- ``1``  — anything else
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

from earl_sdk.exceptions import (
    AuthenticationError,
    AuthorizationError,
    EarlError,
    NotFoundError,
    RateLimitError,
    ServerError,
    SimulationError,
    TransportError,
    ValidationError,
)

_EXIT_CODES: dict[type[BaseException], int] = {
    ValidationError: 2,
    AuthenticationError: 3,
    AuthorizationError: 4,
    NotFoundError: 5,
    RateLimitError: 6,
    ServerError: 7,
    TransportError: 8,
    SimulationError: 9,
}


def exit_code_for(exc: BaseException) -> int:
    for cls, code in _EXIT_CODES.items():
        if isinstance(exc, cls):
            return code
    return 1


def render_error(
    exc: BaseException,
    args: argparse.Namespace,
    *,
    stream: Any = None,
) -> int:
    """Render ``exc`` to ``stream`` and return the process exit code.

    ``stream`` defaults to :data:`sys.stderr` resolved at call time (so
    :func:`contextlib.redirect_stderr` inside tests is respected).
    """
    if stream is None:
        stream = sys.stderr
    json_mode = getattr(args, "output", None) == "json" or getattr(args, "json_output", False)
    debug = bool(getattr(args, "debug", False))
    verbose = bool(getattr(args, "verbose", False))

    if json_mode:
        _render_json(exc, stream=stream, debug=debug or verbose)
    else:
        _render_human(exc, stream=stream, debug=debug or verbose)

    return exit_code_for(exc)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _render_json(exc: BaseException, *, stream: Any, debug: bool) -> None:
    if isinstance(exc, EarlError):
        envelope: dict[str, Any] = {"error": exc.to_dict()}
    else:
        envelope = {
            "error": {
                "type": type(exc).__name__,
                "message": str(exc) or type(exc).__name__,
            }
        }
    if debug:
        envelope["error"]["traceback"] = traceback.format_exc().rstrip().splitlines()
    print(json.dumps(envelope, default=str), file=stream)


# ---------------------------------------------------------------------------
# Human
# ---------------------------------------------------------------------------


def _render_human(exc: BaseException, *, stream: Any, debug: bool) -> None:
    if not isinstance(exc, EarlError):
        print(f"error: {exc}", file=stream)
        if debug:
            print("", file=stream)
            traceback.print_exc(file=stream)
        return

    print(f"error: {exc.message}", file=stream)

    # Context block — aligned key/value pairs. Only emit keys that are set.
    rows: list[tuple[str, str]] = []
    if exc.status_code is not None:
        rows.append(("status", str(exc.status_code)))
    if exc.code:
        rows.append(("code", str(exc.code)))
    if exc.method and exc.url:
        rows.append(("request", f"{exc.method} {exc.url}"))
    if exc.request_id:
        rows.append(("request id", exc.request_id))
    if exc.retry_after is not None:
        rows.append(("retry after", f"{exc.retry_after:g}s"))

    if rows:
        width = max(len(k) for k, _ in rows)
        for k, v in rows:
            print(f"  {k.rjust(width)}: {v}", file=stream)

    if exc.hint:
        print(f"  hint: {exc.hint}", file=stream)

    if debug:
        if exc.details:
            print("", file=stream)
            print("details:", file=stream)
            try:
                body = json.dumps(exc.details, indent=2, default=str)
            except (TypeError, ValueError):
                body = repr(exc.details)
            for line in body.splitlines():
                print(f"  {line}", file=stream)
        print("", file=stream)
        traceback.print_exc(file=stream)
