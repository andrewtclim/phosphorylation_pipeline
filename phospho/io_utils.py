"""I/O helpers for run directories and output files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def create_run_dir(base_dir: str = "runs") -> Path:
    """Create and return a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(records: list[dict], path: str) -> None:
    """Save records as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
