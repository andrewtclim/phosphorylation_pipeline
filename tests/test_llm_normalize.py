"""Tests for LLM normalization module."""

from phospho.llm_normalize import parse_interactions


def test_parse_interactions_parses_valid_lines() -> None:
    raw_text = (
        "MTORC2(kinase), AKT1(substrate), Ser-473(location), N/A\n"
        "AKT1(kinase), AKT1(substrate), Thr-308(location), PubMed:12345678"
    )

    got = parse_interactions(raw_text)

    assert got == [
        {
            "kinase": "MTORC2(kinase)",
            "substrate": "AKT1(substrate)",
            "site": "Ser-473(location)",
            "pmid": "N/A",
            "raw": "MTORC2(kinase), AKT1(substrate), Ser-473(location), N/A",
        },
        {
            "kinase": "AKT1(kinase)",
            "substrate": "AKT1(substrate)",
            "site": "Thr-308(location)",
            "pmid": "PubMed:12345678",
            "raw": "AKT1(kinase), AKT1(substrate), Thr-308(location), PubMed:12345678",
        },
    ]


def test_parse_interactions_skips_na_and_empty_lines() -> None:
    raw_text = "\nN/A\n   \n"

    got = parse_interactions(raw_text)

    assert got == []


def test_parse_interactions_marks_malformed_line() -> None:
    raw_text = "Only one field"

    got = parse_interactions(raw_text)

    assert got == [
        {
            "raw": "Only one field",
            "parse_error": "expected 4 comma-separated fields",
        }
    ]
