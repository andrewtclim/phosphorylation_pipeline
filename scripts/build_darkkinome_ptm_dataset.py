"""CLI stub for building the dark kinome PTM weak-label dataset.

Planned contents:
- Parse input CSV path argument.
- Load and standardize dark kinome bait/prey pairs.
- Build PTM weak-label rows.
- Create timestamped run directory and write output CSV artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phospho.darkkinome_dataset import build_darkkinome_ptm_rows
from phospho.darkkinome_io import extract_pairs, load_darkkinome_ppi
from phospho.io import make_run_dir


def main() -> None:
    """Run the dark kinome PTM dataset build pipeline."""

    # Parse CLI arguments with a sensible default input path.
    parser = argparse.ArgumentParser(
        description="Build dark kinome PTM weak-label dataset."
    )
    parser.add_argument(
        "--input",
        default="data/dark_kinome_ppi.csv",
        help="Path to dark kinome PPI CSV input file.",
    )
    args = parser.parse_args()

    # Load raw dark kinome PPI input.
    raw_df = load_darkkinome_ppi(args.input)

    # Standardize bait/prey pairs to canonical internal schema.
    pairs_df = extract_pairs(raw_df)

    # Build long-form weak-label PTM rows using UniProt PTM comments.
    out_df = build_darkkinome_ptm_rows(pairs_df)

    # Create run directory and write output CSV artifact.
    run_dir = make_run_dir("dark_kinome_runs")
    output_path = Path(run_dir) / "dark_kinome_ptms.csv"
    out_df.to_csv(output_path, index=False)

    # Print concise run summary for verification.
    print(f"Input rows: {len(raw_df)}")
    print(f"Pair rows: {len(pairs_df)}")
    print(f"Output rows: {len(out_df)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
