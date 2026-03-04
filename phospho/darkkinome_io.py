"""Dark kinome PPI input-loading helpers.

Planned contents:
- Load dark kinome PPI CSV input.
- Validate required columns.
- Extract standardized bait/prey pair columns for downstream dataset building.
"""

from __future__ import annotations

import pandas as pd


def load_darkkinome_ppi(path: str) -> pd.DataFrame:
    """Load dark kinome PPI CSV from disk into a DataFrame.

    Planned behavior:
    - Read CSV from ``path``.
    - Validate required schema (at minimum: Experiment.ID, Bait, Prey).
    - Return the raw DataFrame for further transformation.
    """

    # Read the source CSV exactly as provided.
    df = pd.read_csv(path)

    # Validate minimal required columns from the observed dataset schema.
    required_cols = ["Experiment.ID", "Bait", "Prey"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Expected at least: {required_cols}"
        )

    # Return raw DataFrame; downstream function handles standardization.
    return df


def extract_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Extract standardized bait/prey pairs from the PPI DataFrame.

    Planned output columns:
    - experiment_id
    - bait_uniprot
    - prey_uniprot

    Planned behavior:
    - Select and rename source columns from the raw schema.
    - Clean whitespace/casing where needed.
    - Return a long-lived canonical pair table for downstream processing.
    """

    # Validate required input columns before selecting/renaming.
    required_cols = ["Experiment.ID", "Bait", "Prey"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame missing required column(s): {missing}. "
            f"Expected: {required_cols}"
        )

    # Select only the columns needed for pair construction.
    pairs_df = df[["Experiment.ID", "Bait", "Prey"]].copy()

    # Rename to canonical internal names used by the dataset builder.
    pairs_df = pairs_df.rename(
        columns={
            "Experiment.ID": "experiment_id",
            "Bait": "bait_uniprot",
            "Prey": "prey_uniprot",
        }
    )

    # Normalize accession text fields (strip whitespace and uppercase for consistency).
    pairs_df["bait_uniprot"] = pairs_df["bait_uniprot"].astype(
        str).str.strip().str.upper()
    pairs_df["prey_uniprot"] = pairs_df["prey_uniprot"].astype(
        str).str.strip().str.upper()

    # Treat empty/invalid placeholders as missing values and drop unusable rows.
    invalid_tokens = {"", "NAN", "NONE", "NULL"}
    pairs_df.loc[pairs_df["bait_uniprot"].isin(
        invalid_tokens), "bait_uniprot"] = pd.NA
    pairs_df.loc[pairs_df["prey_uniprot"].isin(
        invalid_tokens), "prey_uniprot"] = pd.NA
    pairs_df = pairs_df.dropna(subset=["bait_uniprot", "prey_uniprot"])

    # Deduplicate exact repeated triplets to keep pair table stable.
    pairs_df = pairs_df.drop_duplicates().reset_index(drop=True)

    return pairs_df
