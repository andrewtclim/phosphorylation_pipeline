# PhosphoAtlas Phosphorylation Pipeline

Modular reconstruction of the original UniProt -> PTM -> LLM normalization workflow.

## Folder Structure

```text
notebooks/
  WIP_pipeline_.ipynb
  TEST_pipeline.ipynb

phospho/
  __init__.py
  uniprot.py        # UniProt search/fetch + parsing helpers
  llm_normalize.py  # LLM client + normalization/parsing stubs
  io_utils.py       # run directory + output write helpers

scripts/
  run_pipeline.py

tests/
  test_uniprot.py
  test_llm_normalize.py

requirements.txt
README.md
.gitignore
```

## How To Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
```

## Notes

- `OG_phosphorylation_pipeline_updated.py` is reference-only and ignored by git.
- Notebooks under `notebooks/` are for exploration/testing and are not pipeline source modules.
