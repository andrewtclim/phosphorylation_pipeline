"""Schema definitions for pipeline outputs."""

from __future__ import annotations

from dataclasses import dataclass

# TODO: Align fields with downstream modeling requirements.
# TODO: Add optional metadata fields as needed.


@dataclass
class InteractionRecord:
    """Canonical interaction record for normalized phosphorylation data."""

    substrate_gene: str
    kinase_gene: str | None
    modification: str | None
    residue: str | None
    position: int | None
    evidence: str | None
