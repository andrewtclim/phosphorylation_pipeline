"""UniProt client + parsing helpers for the PhosphoAtlas pipeline."""

from __future__ import annotations
import requests

# search endpoint for UniProt
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb"


# UNIPROT CLIENT FUNCTIONS
def search_uniprot(query: str, size: int = 500, cursor: str | None = None) -> dict:
    """Search reviewed human UniProt proteins."""
    # add human + reviewed filters to input query
    full_query = f"({query}) AND reviewed:true AND organism_id:9606"
    # build URL query params for UniProt REST search
    params = {
        "query": full_query,  # search expression for UniProt
        "format": "json",    # return as JSON
        "size": size,        # max number of results in page
    }
    # if pagination cursor given, include it for next-page fetch
    if cursor:
        params["cursor"] = cursor

    # make HTTP GET call to UniProt search endpoint w/ 30s timeout
    response = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()  # raise exception if status is not 2xx
    return response.json()  # parse and return as Python dict


# NOTE: querying entry_url returns the entry object directly
# querying search_url gives a {results : object} structure
def fetch_entry(accession: str) -> dict:
    """Fetch a single UniProt entry by accession."""
    # build direct UniProt entry URL for one accession
    url = f"{UNIPROT_ENTRY_URL}/{accession}.json"
    # send HTTP GET request to entry URL
    response = requests.get(url, timeout=30)
    # raise excpetion for failed stauses
    response.raise_for_status()
    return response.json()


def uniprot_request_data(accession_id: str, subset: str | None = None) -> dict:
    """Search by accession and optionally return one top-level field."""
    # Build search params w/ accession lookup
    params = {
        "query": f"accession:{accession_id}",
        "format": "json",
        "size": 1,   # only first hit
    }
    # request to UniProt search endpoint
    response = requests.get(UNIPROT_SEARCH_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()  # parse response JSON to dict

    # when no subset requested, return full wrapped payload
    if not subset:
        return data

    # grab the first (main and only) result object
    first = data.get("results", [{}])[0]

    # Return requested top-level key (ex: "sequence") from first result (or None if missing)
    return first.get(subset)


# PARSING FUNCTIONS (CLEANING)
def get_primary_accession(entry: dict) -> str:
    """Extract the primary accession from a UniProt entry."""
    return entry["primaryAccession"]


def get_gene_symbol(entry: dict) -> str | None:
    """Extract the preferred gene symbol from a UniProt entry."""
    # lookup "genes" from entry
    genes = entry.get("genes")
    # if empty return None
    if not genes:
        return None
    # Take first geneObject and return its "geneName" block
    gene_name = genes[0].get("geneName", {})
    # Return the value of geneName dict
    return gene_name.get("value")


def get_protein_name(entry: dict) -> str | None:
    """Extract the recommended protein name from a UniProt entry."""
    # grab top level protein description block
    protein = entry.get("proteinDescription", {})
    # grab recommended name block
    recommended = protein.get("recommendedName", {})
    # grab the full name block
    full_name = recommended.get("fullName", {})
    return full_name.get("value")  # return the proteins full name


def get_ptm_texts(entry: dict) -> list[str]:
    """Extract PTM comment text values from a UniProt entry."""
    ptm_texts: list[str] = []

    # comments are stored in a list where each comment is a dict
    for comment in entry.get("comments", []):
        # each comment has a key called commentType (filter for PTM text)
        if comment.get("commentType") == "PTM":
            # access the comment texts
            for text_obj in comment.get("texts", []):
                # isolate value (where text is stored)
                value = text_obj.get("value")
                if value:
                    # append PTM text to running list
                    ptm_texts.append(value)

    return ptm_texts
