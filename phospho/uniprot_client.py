"""HTTP client helpers for UniProt data access."""

from __future__ import annotations

# TODO: Add retry logic and error handling.
# TODO: Support pagination via cursor tokens.
# TODO: Centralize request configuration.


def search_uniprot(query: str, size: int = 500, cursor: str | None = None) -> dict:
    """Search UniProt for reviewed human proteins."""
    # TODO: Implement UniProt search request.
    raise NotImplementedError


def fetch_entry(accession: str) -> dict:
    """Fetch a single UniProt entry by accession."""
    # TODO: Implement UniProt entry fetch.
    raise NotImplementedError
