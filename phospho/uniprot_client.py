"""HTTP client helpers for UniProt data access."""

from __future__ import annotations
import requests
import json


# TODO: Add retry logic and error handling.
# TODO: Support pagination via cursor tokens.
# TODO: Centralize request configuration.

# search endpoiunt for UniProt
base_url = "https://rest.uniprot.org/uniprotkb/search"


def search_uniprot(query: str, size: int = 500, cursor: str | None = None) -> dict:
    """Search UniProt for reviewed human proteins. It returns a list of results and supports pagination."""
    # append human filters to user query
    full_query = f"({query}) AND reviewed:true AND organism_id:9606"
    # build basic query params
    params = {
        "query": full_query,
        "format": "json",
        "size": size,
    }
    # adds pagination to params when provided
    if cursor:
        params["cursor"] = cursor
    response = requests.get(base_url, params=params)  # http requests
    response.raise_for_status()  # throw error if request fails
    return response.json()


def uniprot_request_data(accession_id: str, subset: str | None = None) -> dict:
    """Search UniProt by accession and return either the full search response or one field from the first result."""
    # build params
    params = {
        "query": f"accession:{accession_id}",  # adjust query to accession_id
        "format": "json",                      # return data as JSON
    }

    # send a request to UniProt
    response = requests.get(base_url, params=params)

    # error handling
    if response.status_code == 200:
        print("Data successfully recieved")
    else:
        print(f"Error: {response.status_code}")

    # convert JSON response to a Python dictionary
    data = response.json()

    if subset:
        # return the subset of query such as the proteins sequence!
        return data["results"][0][subset]
    else:
        return data


def fetch_entry(accession: str) -> dict:
    """Fetch a single UniProt entry by accession ID and returns the protein data in json format"""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()
