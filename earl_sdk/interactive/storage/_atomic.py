"""Atomic file write utilities for crash-safe persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path, data: str) -> None:
    """Write text to file atomically (write tmp then rename).

    Prevents data corruption on crash/kill by writing to a temporary file
    in the same directory, then atomically replacing the target via os.replace.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
