"""Entry point for ``python -m earl_sdk.interactive`` and ``earl-ui`` CLI."""

from __future__ import annotations

import sys


def main() -> None:
    """Wrapper for console_scripts entry point."""
    from .app import main as _main
    try:
        _main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
