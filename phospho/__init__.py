"""PhosphoAtlas package exports."""

from phospho.uniprot import (
    fetch_entry,
    get_gene_symbol,
    get_primary_accession,
    get_protein_name,
    get_ptm_texts,
    search_uniprot,
    uniprot_request_data,
)

__all__ = [
    "search_uniprot",
    "fetch_entry",
    "uniprot_request_data",
    "get_primary_accession",
    "get_gene_symbol",
    "get_protein_name",
    "get_ptm_texts",
]
