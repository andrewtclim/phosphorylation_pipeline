"""LLM prompt/client helpers and PTM normalization utilities."""

from __future__ import annotations

import os

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:
    AzureChatOpenAI = None


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
- If autophosphorylation is stated, include {substrate_gene}(kinase).
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
Output: {substrate_gene}(kinase), {substrate_gene}(substrate), N/A, PubMed:10518011

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


def build_ptm_prompt(ptm_text: str, substrate_gene: str) -> str:
    """Build the full PTM normalization prompt for one PTM text input."""

    # Start from the base instruction prompt and substitute template placeholders.
    prompt = PTM_TO_INTERACTIONS_PROMPT.replace(
        "{substrate_gene}", substrate_gene)

    # Add record-specific context required for extraction.
    prompt += f"\n\nSubstrate gene: {substrate_gene}"
    prompt += f"\nPTM text: {ptm_text}"

    return prompt


def normalize_ptm_texts(ptm_texts: list[str], substrate_gene: str) -> list[str]:
    """Normalize PTM texts using the configured LLM."""

    # Build one LLM client instance for this batch.
    client = get_llm_client()

    # Store one normalized output string per processed PTM input.
    normalized_outputs: list[str] = []

    # Iterate through raw PTM annotation texts.
    for ptm_text in ptm_texts:
        # Skip empty PTM strings so we do not waste model calls.
        if not ptm_text or not ptm_text.strip():
            continue

        # Build a record-specific prompt using base instructions + substrate context + PTM text.
        prompt = build_ptm_prompt(ptm_text, substrate_gene)

        # Send prompt to model and capture response object.
        response = client.invoke(prompt)

        # Append response content as text (fallback to str(response) if needed).
        normalized_outputs.append(
            response.content if hasattr(response, "content") else str(response)
        )

    # Return all normalized text outputs for downstream parsing.
    return normalized_outputs


def parse_interactions(raw_text: str) -> list[dict]:
    """Parse line-based LLM output into structured interaction dictionaries."""
    # Hold one dict per usable output line.
    records: list[dict] = []
    # Split full text into individual lines.
    for line in raw_text.splitlines():
        line = line.strip()
        # Skip empty lines or sentinel N/A lines.
        if not line or line.upper() == "N/A":
            continue

        # Parse one expected "KINASE, SUBSTRATE, SITE, PMID" line.
        parts = [p.strip() for p in line.split(",")]
        # Preserve malformed lines for debugging.
        if len(parts) != 4:
            records.append(
                {"raw": line, "parse_error": "expected 4 comma-separated fields"}
            )
            continue

        # Convert the parsed fields into a stable downstream schema.
        kinase, substrate, site, pmid = parts
        records.append({
            "kinase": kinase,
            "substrate": substrate,
            "site": site,
            "pmid": pmid,
            "raw": line,
        })

    # Return all parsed records for downstream processing.
    return records


def normalize_and_parse_ptm_texts(
    ptm_texts: list[str],
    substrate_gene: str,
) -> list[dict]:
    """Normalize PTM texts with the LLM and parse results into dictionaries."""

    # Accumulate parsed interaction records across all PTM text inputs.
    parsed_records: list[dict] = []

    # Normalize each PTM text (LLM output is raw text), then parse into structured dict records.
    for raw_output in normalize_ptm_texts(ptm_texts, substrate_gene):
        parsed_records.extend(parse_interactions(raw_output))

    # Return one combined list of structured interaction records.
    return parsed_records
