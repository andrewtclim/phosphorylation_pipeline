# PhosphoAtlas Phosphorylation Pipeline

A modular pipeline for retrieving reviewed human proteins from UniProt, extracting PTM comments, normalizing phosphorylation interactions, and exporting analysis-ready tables.

## Folder Structure

```
phospho_pipeline/
├── notebooks/
│   └── (WIP notebook already exists here — do not touch)
│
├── phospho/
│   ├── __init__.py
│   ├── config.py
│   ├── uniprot_client.py
│   ├── uniprot_parse.py
│   ├── llm_client.py
│   ├── prompts.py
│   ├── normalize_ptm.py
│   ├── parse_llm_output.py
│   ├── schemas.py
│   ├── io.py
│   └── validation.py
│
├── scripts/
│   └── run_pipeline.py
│
├── tests/
│   ├── test_uniprot_parse.py
│   └── test_parse_llm_output.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## How To Run

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
```

## Notes