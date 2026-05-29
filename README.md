# RAGAS-infused RAG pipeline

Observable hybrid-retrieval RAG over Microsoft/Azure docs, built on Microsoft
Agent Framework + Azure AI Foundry + Azure AI Search, evaluated with RAGAS.

See the design spec in `docs/superpowers/specs/` and the diagram in `docs/pipeline.mmd`.

## Prerequisites

- Python 3.11, Azure CLI (`az login`), Azure Developer CLI (`azd`), an Azure subscription.

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

## Use an existing Foundry project (skip `azd`)

`azd up` is optional. The app reads all connection details from `.env`, so you can
point it at a Foundry project you already have. You need:

1. **A Foundry project** — copy its endpoint into `FOUNDRY_PROJECT_ENDPOINT`
   (`https://<resource>.services.ai.azure.com/api/projects/<project>`).
2. **A chat model deployment** (e.g. `gpt-4o`) — set `FOUNDRY_CHAT_MODEL` to its
   deployment name.
3. **An embedding deployment** — set `FOUNDRY_EMBEDDING_MODEL` (default
   `text-embedding-3-small`). Note `text-embedding-3-small` is **not** available in
   every region (notably not in `swedencentral`); deploy it in one of the supported
   regions below, or switch to `text-embedding-3-large` if you must use a region that
   only offers the large model. If the deployment does not exist yet, add it in the
   Foundry portal or via `az cognitiveservices account deployment create`.
4. **An Azure AI Search service with the semantic ranker enabled** (Basic tier or
   higher) — set `SEARCH_ENDPOINT` and `SEARCH_INDEX`. Create one if you don't have
   it; the index itself is created for you by the ingestion step.

Then run the two setup steps that `azd`'s post-provision hook would otherwise run:

```bash
cp .env.example .env            # fill in the values above
python -m ragpipe.ingest        # builds the Search index from data/corpus_sources.yaml
python scripts/setup_agents.py  # registers the generator agent (+ Code Interpreter tool)
```

> **Embedding dimensions:** `text-embedding-3-small` returns 1536-dim vectors. The
> Search index is sized automatically from the first vector, so switching the
> embedding model needs no code change. To shrink vectors for a larger corpus,
> reduce the model's output dimensions and rebuild the index.

## Supported regions

Provisioning is restricted (in `infra/main.bicep`) to regions where `gpt-4o`,
`text-embedding-3-small`, and Azure AI Search (with semantic ranker) are all
available on the Standard deployment type: **eastus2** (default), eastus, westus,
canadaeast, switzerlandnorth, australiaeast, japaneast, uaenorth.

`text-embedding-3-small` is not offered in most EU regions — `switzerlandnorth` is
the only EU option here. If you need another EU region, switch
`FOUNDRY_EMBEDDING_MODEL` (and the Bicep `embeddingModel` default) to
`text-embedding-3-large`, which is available in `swedencentral`, `francecentral`,
`norwayeast`, `uksouth`, and more.
```
