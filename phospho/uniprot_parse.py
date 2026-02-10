"""Parsing helpers for UniProt entries."""

from __future__ import annotations

# TODO: Document expected UniProt entry schema.
# TODO: Normalize missing or unexpected fields.


def get_primary_accession(entry: dict) -> str:
    """Extract the primary accession from a UniProt entry."""
    return entry["primaryAccession"]


def get_gene_symbol(entry: dict) -> str | None:
    """Extract the preferred gene symbol from a UniProt entry."""
    # go to gene dict
    genes = entry.get("genes")
    if not genes:
        return None
    # under geneName return the value (aka its gene symbol)
    gene_name = genes[0].get("geneName", {})
    return gene_name.get('value')


def get_protein_name(entry: dict) -> str | None:
    """Extract the recommended protein name from a UniProt entry."""
    # TODO: Implement protein name extraction.
    raise NotImplementedError


def get_ptm_texts(entry: dict) -> list[str]:
    """Extract PTM comment texts from a UniProt entry."""
    # TODO: Implement PTM comment extraction.
    raise NotImplementedError
