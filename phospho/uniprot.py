"""UniProt client + parsing helpers for the PhosphoAtlas pipeline."""

from __future__ import annotations

import requests

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb"


def search_uniprot(query: str, size: int = 500, cursor: str | None = None) -> dict:
    """Search reviewed human UniProt proteins."""
    full_query = f"({query}) AND reviewed:true AND organism_id:9606"
    params = {
        "query": full_query,
        "format": "json",
        "size": size,
    }
    if cursor:
        params["cursor"] = cursor

    response = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_entry(accession: str) -> dict:
    """Fetch a single UniProt entry by accession."""
    url = f"{UNIPROT_ENTRY_URL}/{accession}.json"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def uniprot_request_data(accession_id: str, subset: str | None = None) -> dict:
    """Search by accession and optionally return one top-level field."""
    params = {
        "query": f"accession:{accession_id}",
        "format": "json",
        "size": 1,
    }
    response = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not subset:
        return data

    first = data.get("results", [{}])[0]
    return first.get(subset)


def get_primary_accession(entry: dict) -> str:
    """Extract the primary accession from a UniProt entry."""
    return entry["primaryAccession"]


def get_gene_symbol(entry: dict) -> str | None:
    """Extract the preferred gene symbol from a UniProt entry."""
    genes = entry.get("genes")
    if not genes:
        return None
    gene_name = genes[0].get("geneName", {})
    return gene_name.get("value")


def get_protein_name(entry: dict) -> str | None:
    """Extract the recommended protein name from a UniProt entry."""
    protein = entry.get("proteinDescription", {})
    recommended = protein.get("recommendedName", {})
    full_name = recommended.get("fullName", {})
    return full_name.get("value")


def get_ptm_texts(entry: dict) -> list[str]:
    """Extract PTM comment text values from a UniProt entry."""
    comments = entry.get("comments", [])
    ptm_texts: list[str] = []

    for comment in comments:
        if comment.get("commentType") != "PTM":
            continue
        for text_obj in comment.get("texts", []):
            value = text_obj.get("value")
            if value:
                ptm_texts.append(value)

    return ptm_texts
