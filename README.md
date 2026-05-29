# RAGAS-infused RAG pipeline

Observable hybrid-retrieval RAG over Microsoft/Azure docs, built on Microsoft Agent Framework + Azure AI Foundry + Azure AI Search, evaluated with RAGAS. See the design spec in `docs/superpowers/specs/` and the architecture diagram in `docs/pipeline.mmd`.

## Prerequisites

- Python 3.11
- Azure CLI — `az login`
- Azure Developer CLI — `azd`
- An active Azure subscription

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
azd up                       # provisions Foundry, Search, model deployments; runs ingestion + agent registration
cp .env.example .env         # then fill from azd outputs
```

## Run

```bash
streamlit run app/dashboard.py     # Run / Evaluation / Architecture tabs
```

## Evaluate

```bash
python -m ragpipe.eval.run         # offline RAGAS harness over data/testset.jsonl
```

Set `TESTSET_MODE=synthetic` in `.env` to generate the test set from the corpus instead.

## Test

```bash
pytest -q
```
