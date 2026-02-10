"""HTTP client helpers for UniProt data access."""

from __future__ import annotations
import requests
import json


# TODO: Add retry logic and error handling.
# TODO: Support pagination via cursor tokens.
# TODO: Centralize request configuration.


def search_uniprot(query: str, size: int = 500, cursor: str | None = None) -> dict:
    """Search UniProt for reviewed human proteins."""
    pass


def fetch_entry(accession: str) -> dict:
    """Fetch a single UniProt entry by accession."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
