"""Build an updated PTM dataset by adding PhosphoAtlas light-kinase background rows.

Hardcoded note:
- Expects the current dark kinome dataset at
  dark_kinome_runs/2026-03-03_1656/dark_kinome_ptms.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from phospho.uniprot import fetch_entry, get_ptm_texts


def main() -> None:
    """Append 100 sampled PhosphoAtlas kinases as class-0 background rows."""

    # Input paths for the existing dark-kinome dataset and the PhosphoAtlas workbook.
    dark_dataset_path = Path(
        "dark_kinome_runs/2026-03-03_1656/dark_kinome_ptms.csv")
    phosphoatlas_path = Path(
        "data/phosphoAtlas_data/2024_PhosphoAtlas 2.0_updated KSP network_withHTKAMconnections.xlsx"
    )

    # Output path for the updated merged dataset.
    output_path = Path("data/updated_PTM_dataset_03_10_2026.csv")

    # Load the existing dark-kinome dataset (classes 1 and 2).
    dark_df = pd.read_csv(dark_dataset_path)

    # Track all proteins already labeled in the current dataset.
    labeled_ids = set(
        dark_df["uniprot_id"].dropna().astype(str).str.strip().str.upper()
    )

    # Load the PhosphoAtlas workbook and use the first sheet.
    xls = pd.ExcelFile(phosphoatlas_path)
    pa_df = pd.read_excel(phosphoatlas_path, sheet_name=xls.sheet_names[0])

    # Extract unique kinase accessions from PhosphoAtlas.
    kinase_pool = (
        pa_df["KIN_ACC_ID"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
    )

    # Keep only strings that match common UniProt accession formats.
    uniprot_like = kinase_pool.str.match(
        r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])$"
    )
    kinase_pool = kinase_pool[uniprot_like]

    # Exclude any kinase already labeled in the dark-kinome dataset.
    candidate_background = kinase_pool[~kinase_pool.isin(labeled_ids)]

    # Sample 100 background kinases with a fixed seed for reproducibility.
    sampled_background = candidate_background.sample(n=100, random_state=42)

    # Fetch PTM texts once per sampled kinase and emit class-0 rows.
    background_rows: list[dict] = []
    for uniprot_id in sampled_background:
        entry = fetch_entry(uniprot_id)
        ptm_texts = get_ptm_texts(entry)

        # Skip proteins with no PTM comments.
        if not ptm_texts:
            continue

        for ptm_text in ptm_texts:
            background_rows.append(
                {
                    "uniprot_id": uniprot_id,
                    "role": "background",
                    "ptm_text": ptm_text,
                    "output_class": 0,
                    "experiment_id": pd.NA,
                    "bait_uniprot": pd.NA,
                    "prey_uniprot": pd.NA,
                    "source": "phosphoatlas_background",
                }
            )

    # Build background DataFrame with the same column order as the dark dataset.
    background_df = pd.DataFrame(background_rows)
    background_df = background_df.reindex(columns=dark_df.columns)

    # Append background rows to the existing dataset.
    updated_df = pd.concat([dark_df, background_df], ignore_index=True)

    # Save merged dataset.
    updated_df.to_csv(output_path, index=False)

    # Print concise run summary.
    print(f"Existing dark dataset rows: {len(dark_df)}")
    print(f"Existing labeled proteins: {len(labeled_ids)}")
    print(
        f"PhosphoAtlas unique kinase IDs (raw): {pa_df['KIN_ACC_ID'].nunique(dropna=True)}")
    print(f"PhosphoAtlas UniProt-like kinase IDs: {kinase_pool.nunique()}")
    print(f"Background candidate kinases: {len(candidate_background)}")
    print(f"Sampled background kinases: {len(sampled_background)}")
    print(f"Background PTM rows added: {len(background_df)}")
    print(f"Updated dataset rows: {len(updated_df)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
