"""Minimal end-to-end pipeline run (first runnable version)."""

from phospho.io_utils import create_run_dir, save_json
from phospho.llm_normalize import normalize_and_parse_ptm_texts
from phospho.uniprot import fetch_entry, get_gene_symbol, get_ptm_texts, search_uniprot


def build_final_table(records: list[dict]):
    """Convert parsed interaction records into a clean final DataFrame."""
    import pandas as pd

    # Create DataFrame from parsed records; handle empty runs safely.
    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Remove parser error rows from the final table.
    if "parse_error" in df.columns:
        df = df[df["parse_error"].isna()].copy()

    # Keep only expected columns if they exist.
    keep_cols = ["accession", "substrate_gene",
                 "kinase", "substrate", "site", "pmid", "raw"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Strip label suffixes to create cleaner columns (e.g., "AKT1(kinase)" -> "AKT1").
    for col, suffix in [("kinase", "(kinase)"), ("substrate", "(substrate)"), ("site", "(location)")]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                suffix, "", regex=False).str.strip()

    # Deduplicate exact repeated rows.
    return df.drop_duplicates().reset_index(drop=True)


def main() -> None:
    """Run a small UniProt batch through PTM normalization and save run artifacts."""
    from datetime import datetime
    import pandas as pd

    # Small first batch for initial pipeline runs (fast feedback, OG-like loop shape).
    query = "kinase"
    batch_size = 5

    # Capture run start time for summary logging.
    start_time = datetime.now()

    # Search UniProt and gather top accessions for this batch.
    search_result = search_uniprot(query, size=batch_size)
    accessions = [r["primaryAccession"]
                  for r in search_result.get("results", [])]

    # Create one run folder for all artifacts from this execution.
    run_dir = create_run_dir("runs")

    # Save raw search response as OG-style intermediate artifact.
    search_json_path = run_dir / "uniprot_search_batch.json"
    save_json([search_result], str(search_json_path))

    # Accumulate parsed interaction records across all proteins in this batch.
    all_records: list[dict] = []

    # Process each accession: fetch entry -> extract PTM text -> normalize+parse.
    for accession in accessions:
        entry = fetch_entry(accession)
        substrate_gene = get_gene_symbol(entry) or accession
        ptm_texts = get_ptm_texts(entry)

        # Save each raw entry for traceability/debug.
        entry_path = run_dir / f"uniprot_entry_{accession}.json"
        save_json([entry], str(entry_path))

        # Normalize and parse PTM text for this specific protein.
        parsed_records = normalize_and_parse_ptm_texts(
            ptm_texts, substrate_gene)

        # Add accession-level metadata to each parsed record before merging.
        for row in parsed_records:
            row["accession"] = accession
            row["substrate_gene"] = substrate_gene

        all_records.extend(parsed_records)

    # Save a cleaner final table (OG-style "final" artifact).
    final_df = build_final_table(all_records)
    final_csv_path = run_dir / "final_interactions.csv"
    final_df.to_csv(final_csv_path, index=False)

    # Save combined parsed interactions as JSON + CSV table outputs.
    parsed_json_path = run_dir / "parsed_interactions.json"
    save_json(all_records, str(parsed_json_path))

    parsed_csv_path = run_dir / "parsed_interactions.csv"
    pd.DataFrame(all_records).to_csv(parsed_csv_path, index=False)

    # Save compact run summary for quick auditing.
    end_time = datetime.now()
    summary = {
        "query": query,
        "batch_size_requested": batch_size,
        "accessions_processed": len(accessions),
        "parsed_record_count": len(all_records),
        "final_record_count": len(final_df),
        "started_at": start_time.isoformat(),
        "ended_at": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "search_json_path": str(search_json_path),
        "parsed_json_path": str(parsed_json_path),
        "parsed_csv_path": str(parsed_csv_path),
        "final_csv_path": str(final_csv_path),
    }
    summary_path = run_dir / "run_summary.json"
    save_json([summary], str(summary_path))

    # Print summary for terminal feedback.
    print(f"Query: {query}")
    print(f"Accessions processed: {len(accessions)}")
    print(f"Parsed records: {len(all_records)}")
    print(f"Final records: {len(final_df)}")
    print(f"Saved search JSON: {search_json_path}")
    print(f"Saved parsed JSON: {parsed_json_path}")
    print(f"Saved parsed CSV: {parsed_csv_path}")
    print(f"Saved final CSV: {final_csv_path}")
    print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
