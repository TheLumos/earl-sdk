"""Local run history storage.

Saves simulation results as JSON files under ``~/.earl/runs/<sim_id_short>/``.
An index file (``~/.earl/runs/index.json``) enables fast listing without
reading every run file.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from ._atomic import atomic_write_text

RUNS_DIR = Path.home() / ".earl" / "runs"
INDEX_PATH = RUNS_DIR / "index.json"


@dataclass
class LocalRun:
    """Metadata for a locally saved simulation run."""

    simulation_id: str
    pipeline_name: str
    status: str
    total_episodes: int
    completed_episodes: int
    average_score: Optional[float] = None
    started_at: str = ""
    finished_at: str = ""
    doctor_type: str = ""  # "Lumos's" | "Client's" | "client_driven"
    environment: str = ""
    saved_at: float = 0.0  # unix timestamp

    @property
    def short_id(self) -> str:
        return self.simulation_id[:8]


class RunStore:
    """Manage locally saved simulation runs."""

    def __init__(self, runs_dir: Path = RUNS_DIR) -> None:
        self._dir = runs_dir
        self._index: list[LocalRun] | None = None

    @property
    def _index_path(self) -> Path:
        return self._dir / "index.json"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    # -- index --

    def _load_index(self) -> list[LocalRun]:
        if self._index is not None:
            return self._index
        if not self._index_path.exists():
            self._index = []
            return self._index
        try:
            raw = json.loads(self._index_path.read_text())
            self._index = [LocalRun(**entry) for entry in raw]
        except Exception:
            self._index = []
        return self._index

    def _save_index(self) -> None:
        self._ensure_dir()
        idx = self._load_index()
        atomic_write_text(self._index_path, json.dumps([asdict(r) for r in idx], indent=2) + "\n")

    # -- public api --

    def list_runs(self, limit: int = 50) -> list[LocalRun]:
        """Return locally saved runs, newest first."""
        idx = self._load_index()
        return sorted(idx, key=lambda r: r.saved_at, reverse=True)[:limit]

    def get_run(self, simulation_id: str) -> Optional[LocalRun]:
        """Find a run by full or short ID."""
        for r in self._load_index():
            if r.simulation_id == simulation_id or r.simulation_id.startswith(simulation_id):
                return r
        return None

    def has_run(self, simulation_id: str) -> bool:
        return self.get_run(simulation_id) is not None

    def save_run(self, meta: LocalRun, report: dict[str, Any]) -> None:
        """Save simulation report to disk and update index."""
        self._ensure_dir()
        run_dir = self._dir / meta.short_id
        run_dir.mkdir(exist_ok=True)

        meta.saved_at = time.time()

        # Write report
        atomic_write_text(run_dir / "report.json", json.dumps(report, indent=2, default=str) + "\n")

        # Write summary
        atomic_write_text(run_dir / "summary.json", json.dumps(asdict(meta), indent=2) + "\n")

        # Update index
        idx = self._load_index()
        idx = [r for r in idx if r.simulation_id != meta.simulation_id]
        idx.insert(0, meta)
        self._index = idx
        self._save_index()

    def load_report(self, simulation_id: str) -> dict[str, Any] | None:
        """Load full report JSON for a run."""
        run = self.get_run(simulation_id)
        if not run:
            return None
        report_path = self._dir / run.short_id / "report.json"
        if not report_path.exists():
            return None
        try:
            return json.loads(report_path.read_text())
        except Exception:
            return None

    def delete_run(self, simulation_id: str) -> bool:
        """Delete a locally saved run."""
        run = self.get_run(simulation_id)
        if not run:
            return False
        run_dir = self._dir / run.short_id
        if run_dir.exists():
            import shutil
            shutil.rmtree(run_dir)
        idx = self._load_index()
        self._index = [r for r in idx if r.simulation_id != simulation_id]
        self._save_index()
        return True

    def prune(self, max_runs: int = 50) -> int:
        """Delete oldest runs beyond max_runs. Returns count deleted."""
        idx = self._load_index()
        if len(idx) <= max_runs:
            return 0
        idx.sort(key=lambda r: r.saved_at, reverse=True)
        to_delete = idx[max_runs:]
        deleted = 0
        for run in to_delete:
            if self.delete_run(run.simulation_id):
                deleted += 1
        return deleted
