"""Minimal pipeline entrypoint."""

from phospho.uniprot import fetch_entry, get_gene_symbol, search_uniprot


def main() -> None:
    """Run a minimal smoke workflow for the UniProt stage."""
    result = search_uniprot("p53", size=1)
    accession = result["results"][0]["primaryAccession"]
    entry = fetch_entry(accession)
    print(f"Accession: {accession}")
    print(f"Gene symbol: {get_gene_symbol(entry)}")


if __name__ == "__main__":
    main()
