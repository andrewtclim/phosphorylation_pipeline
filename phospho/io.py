"""Input/output helpers for pipeline artifacts."""

from __future__ import annotations

# TODO: Define output directory conventions.
# TODO: Add helpers for writing CSV/JSONL outputs.
# TODO: Create run directory with timestamps.


def create_run_dir(base_dir: str) -> str:
    """Create and return a run directory path."""
    # TODO: Implement run directory creation.
    raise NotImplementedError


def save_interactions(records: list[dict], path: str) -> None:
    """Save normalized interaction records to disk."""
    # TODO: Implement writer for output records.
    raise NotImplementedError
