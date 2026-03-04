"""General I/O helpers for dark kinome dataset runs.

Planned contents:
- Timestamped run-directory creation for dark kinome dataset builds.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_dir(base_dir: str = "dark_kinome_runs") -> str:
    """Create and return a timestamped run directory path string.

    Planned behavior:
    - Create ``base_dir/YYYY-MM-DD_HHMM`` if it does not exist.
    - Return the created directory path as a string.
    """

    # Create timestamp in required folder format.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")

    # Compose full run directory path.
    run_dir = Path(base_dir) / timestamp

    # Ensure directory exists before returning.
    run_dir.mkdir(parents=True, exist_ok=True)

    return str(run_dir)
