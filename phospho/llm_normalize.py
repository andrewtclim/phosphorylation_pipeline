"""LLM prompt/client helpers and PTM normalization stubs."""

from __future__ import annotations

import os

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:  # optional during early setup
    AzureChatOpenAI = None

PROTEIN_TO_GENE_PROMPT = """Map a protein full name to the canonical human gene symbol."""
PTM_TO_INTERACTIONS_PROMPT = """Convert PTM text into structured phosphorylation interactions."""


def get_llm_client():
    """Build an AzureChatOpenAI client from environment variables."""
    if AzureChatOpenAI is None:
        raise ImportError("langchain-openai is not installed")

    return AzureChatOpenAI(
        api_key=os.getenv("VERSA_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_endpoint=os.getenv(
            "AZURE_OPENAI_ENDPOINT", "https://unified-api.ucsf.edu/general"),
        deployment_name=os.getenv(
            "AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-2025-04-14"),
        temperature=0.1,
        model="gpt-4.1",
    )


def normalize_ptm_texts(ptm_texts: list[str], substrate_gene: str) -> list[str]:
    """Normalize PTM texts using the configured LLM (placeholder)."""
    # TODO: wire prompt + model invocation.
    # Keeping this explicit placeholder until API key is available.
    _ = (ptm_texts, substrate_gene)
    return []


def parse_interactions(raw_text: str) -> list[dict]:
    """Parse line-based LLM output into dictionaries (minimal parser)."""
    records: list[dict] = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.upper() == "N/A":
            continue
        records.append({"raw": line})
    return records
