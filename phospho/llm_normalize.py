"""LLM prompt/client helpers and PTM normalization stubs."""

from __future__ import annotations

import os

from langchain_openai import AzureChatOpenAI


PROTEIN_TO_GENE_PROMPT = """Map a protein full name to the canonical human gene symbol."""

PTM_TO_INTERACTIONS_PROMPT = """
You are an expert in extracting kinase-substrate phosphorylation interactions from scientific text.

Task:
- Extract only phosphorylation/dephosphorylation interactions.
- Focus on explicit phrases like "phosphorylated by" and "dephosphorylated by".
- Do not infer phosphatases unless "dephosphorylated by" is explicitly present.
- Exclude transcription-only effects, indirect regulation, or "phosphorylation enhanced" style statements.
- If phosphorylation is only a prerequisite for another process (e.g., ubiquitination), return N/A.
- If no valid interaction exists, return exactly: N/A.
- If autophosphorylation is stated, include {kinase_name}(kinase).
- If wording is uncertain ("probably"/"likely"), keep kinase extraction but mark uncertain in the kinase tag.

Output format rules (strict):
- Return ONLY plain text lines, no markdown, no bullets, no tables, no reasoning.
- One interaction per line in this format:
  KINASE(kinase), SUBSTRATE(substrate), SITE(location), PMID
- SITE examples: Ser-473, Thr-308, Tyr-15
- PMID format: PubMed:12345678 or N/A

Examples:
Input: "Phosphorylation at Thr-161 by CAK/CDK7 activates kinase activity (PubMed:20360007)."
Output: CAK/CDK7(kinase), {substrate_gene}(substrate), Thr-161(location), PubMed:20360007

Input: "Dephosphorylated in response to apoptotic stress (PubMed:27995898)."
Output: N/A

Input: "Autophosphorylated and phosphorylated during M-phase (PubMed:10518011)."
Output: {kinase_name}(kinase), {substrate_gene}(substrate), N/A, PubMed:10518011

Now process this PTM text exactly under the rules above.
"""


def get_llm_client():
    """Create and return the LLM client used for PTM normalization."""

    # Stop early with a clear message if the package is missing.
    if AzureChatOpenAI is None:
        raise ImportError("langchain-openai is not installed")

    # Read the required secret variables from env.
    versa_api_key = os.getenv("VERSA_API_KEY")
    if not versa_api_key:
        raise ValueError("Missing VERSA_API_KEY in environment")

    # Read optional config with safe defaults for the UCSF Versa setup from .env file
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT",
                         "https://unified-api.ucsf.edu/general")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-2025-04-14")

    # Build and return a configured client object.
    return AzureChatOpenAI(
        api_key=versa_api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
        deployment_name=deployment,
        temperature=0.1,
        model="gpt-4.1",
    )


def normalize_ptm_texts(ptm_texts: list[str], substrate_gene: str) -> list[str]:
    """Normalize PTM texts using the configured LLM."""
    # Build LLM client object
    client = get_llm_client()
    normalized_outputs = []

    # Iterate through inputs (raw PTM annotations)
    for ptm_text in ptm_texts:
        # Build a prompt with base normalization instructions w/ current substrate gene and PTM text
        prompt = f"{PTM_TO_INTERACTIONS_PROMPT}\n\nSubstrate gene: {substrate_gene}\nPTM text: {ptm_text}"
        # Send and store response to LLM
        response = client.invoke(prompt)
        # append the normalized text outputs in a list
        normalized_outputs.append(response.content if hasattr(
            response, "content") else str(response))

    return normalized_outputs


def parse_interactions(raw_text: str) -> list[dict]:
    """Parse line-based LLM output into dictionaries (minimal parser)."""
    # Hold one dict per usable output line
    records: list[dict] = []
    # Split full text into individual lines  "line1\nline2\n" -> ["line1", "line2"]
    for line in raw_text.splitlines():
        line = line.strip()  # remove leadind/training white spaces
        # skip all lines that are empty or N/A
        if not line or line.upper() == "N/A":
            continue
        # Keep each usable line as raw : text_outputs
        records.append({"raw": line})
    # return all kept records for downstream processing
    return records
