# ragas-infused-pipeline (`ragpipe`)

Observable hybrid-retrieval RAG over Microsoft/Azure docs, evaluated with RAGAS.
Every model call fills a model role (generator, planner, ingest writer, embedder, online
judge, offline judge, reranker) served by Azure AI Foundry or by an OpenAI-compatible URL
like vLLM on the DGX Spark (`model_roles.py`, ADR-0022); the plan is to move roles to the
Spark one at a time. Retrieval is pgvector in homelab's Postgres (ADR-0021). The `ragpipe`
package (`src/ragpipe/`) holds the pipeline; `app/` is the Streamlit UI + FastAPI
service. Read `README.md` for setup/run/eval commands and `docs/adr/` for the *why*
behind every significant design decision.

## Environment & commands

- **`uv` manages everything.** `uv sync` creates `.venv` and installs prod + dev deps
  from `uv.lock`. There is no manual activation, so **prefix every command with `uv run`**
  (e.g. `uv run python -m ragpipe.ingest`, `uv run streamlit run app/dashboard.py`).
- **Test:** `uv run pytest -q` (full suite). Single file / single test:
  `uv run pytest tests/test_config.py -q` ·
  `uv run pytest "tests/test_config.py::test_settings_loads_from_env" -q`.
- **Lint:** `uv run ruff check .` (line-length 100, target py311). Run it before committing. CI
  (`publish-image.yml`) runs ruff + pytest on every PR and push, and only publishes the
  image when they pass.
- `pytest` is configured with `asyncio_mode = "auto"` (async tests need no marker) and
  `pythonpath = ["src", "."]`. Tests run in deterministic order (no `pytest-randomly`).
- Python 3.11+ (`requires-python`); the API image is `python:3.11-slim`.

## Architecture (the big picture spans several files)

Three phases (see the diagram in `README.md` / `docs/pipeline*.svg`):

1. **Ingest** (`ragpipe.ingest`): crawl Microsoft Learn → extract markdown → heading-aware
   chunk → decorate each chunk with a breadcrumb + cached LLM "situating context" → embed →
   upload to Azure AI Search, then export → `ragpipe.pg_import` into Postgres (retrieval
   reads Postgres only).
2. **Query pipeline**: `app_wiring.build_pipeline_fn(settings, *, mode)` assembles a
   substrate + reranker (`build_reranker`: the Spark cross-encoder for `RERANK_MODES`,
   score-sort passthrough otherwise) + Foundry generator agent + RAGAS faithfulness judge into
   `PipelineDeps`, then `workflow.run_pipeline` executes: retrieve **once** → loop
   {rerank (window widens each retry) → generate → faithfulness score → `decide_next`}.
   On `PASS` it returns; on `EXHAUSTED` it sets `abstained` and replaces the answer with
   `ABSTENTION_ANSWER` (directive guardrail, ADR-0009).
3. **Evaluation** (`ragpipe.eval.run`): replays the pipeline over `data/testset.jsonl`,
   scoring deterministic per-stage retrieval metrics + the RAGAS suite vs. a frozen baseline.

**Retrieval substrate seam (ADR-0012), the central extension point.** Every strategy
implements the `RetrievalSubstrate` protocol (`name` + `async retrieve`). Strategies are
keyed by `RetrievalMode` in `retrieval/registry.py`; `registered_modes()` returns the **9
live modes** (`contextual`, `baseline`, `raptor_sac`, `graphrag`, `combined` + their
`*_agentic` wrappers; there is **no** `contextual_agentic`). Adding a strategy =
implement the protocol, add its table to `pgstore.py`, register it. Nothing downstream
of retrieval changes. `PipelineState.stages` is a **dynamic `dict[str, list[Chunk]]`**;
each substrate names its own stages and the final set is always mirrored under `"reranked"`.
Do not reintroduce fixed `dense`/`bm25` stage fields.

**Three-family judge split (ADR-0009/0011).** The generator (gpt), the online faithfulness
gate (Claude/Anthropic, `JUDGE_MODEL`), and the offline RAGAS judge (DeepSeek,
`OFFLINE_JUDGE_MODEL`) are deliberately different model families to avoid self-preference
bias. The rule is three model *families* by base-model lineage, not vendors (ADR-0022).
On Foundry, `model_roles.judge_provider()` routes Anthropic vs. the OpenAI-compatible
route. `JUDGE_MODEL` is required: builders raise rather than silently fall back to the generator.

**Two app surfaces.** `app/dashboard.py` (Streamlit: Run / Evaluation / Architecture tabs)
and `app/api.py` (FastAPI: `/run`, `/run/stream`, `/compare`, `/eval`, `/modes`, `/health`). The API reuses the
pipeline and imports the dashboard's *pure* helpers. `mode` is **caller-supplied and
required** everywhere: an omitted/unknown mode returns HTTP 422; there is no server default.

## Conventions specific to this repo

- **`from __future__ import annotations`** is at the top of every module (annotations are
  strings at runtime, which matters for pydantic/Agent-Framework decorators; see notes in
  `workflow.build_viz_workflow` and `faithfulness_gate.prewarm_ragas_imports`).
- **Config:** `Settings` is a frozen dataclass built via `Settings.from_env()` (uses
  `load_dotenv`). `RetrievalMode(str, Enum)` members serialize to their `.value` and work
  as plain string dict keys.
- **Testing pattern:** pure logic is unit-tested with fakes/dataclass stand-ins and **no
  network** (construct `Settings(...)` directly, or a minimal `_S` stub). Live Azure wiring
  (`main()`, `build_pipeline_fn`, the live judge builders) is marked `# pragma: no cover`.
- **RAGAS/langchain import hazard:** import `langchain_openai` *before* `ragas`;
  `faithfulness_gate.ensure_ragas_importable()` does this and installs a `vertexai` placeholder.
- **Model clients come from `model_roles`, never built inline.** Ask for a role
  (`openai_client`, `complete_fn`, `judge_chat_factory`, `langchain_embeddings`,
  `generator_agent`); it picks the host, credential, Foundry route, API version and the
  role's `timeout` + `max_retries` budget, and records usage for metered roles. A new kind
  of model call is a new role or a new function there, tested in `tests/test_model_roles.py`
  with SDK recorders. `<ROLE>_URL` moves a role to an OpenAI-compatible host (ADR-0022).
- **Usage reporting (ADR-0020):** every metered call records a `UsageEntry` through
  `ragpipe.usage.record_*` at the point the provider response is in hand; `run_pipeline`
  collects them into `PipelineState.usage`. A new paid call needs its own `record_*` line.
  Usage only, no prices.
- **The `context` column is retrieval-only.** Embeddings are computed from
  `context + "\n\n" + content`, but retrievers select only `content` into prompts/judging.
- **Docs discipline:** record significant decisions as an ADR in `docs/adr/NNNN-kebab.md`
  (Nygard format: Context / Decision / Consequences / **Sources**), written *in the same
  change set* as the decision. Design specs live in `docs/superpowers/specs/`, plans in
  `docs/superpowers/plans/` (date-prefixed).
- **Eval outputs:** each mode writes a committed, resumable `eval_results_<mode>.json`
  checkpoint; the combined `eval_results.json` (read by `/eval` and the dashboard) is rebuilt
  from them and stays gitignored.

## Provisioning & deploy

`azd up` provisions Foundry/Search/model deployments via `infra/` Bicep, then the
`postprovision` hook runs `ragpipe.ingest` + `scripts/setup_agents.py`. Default region is
**swedencentral** (only region with Claude + gpt-5.4 + embeddings, ADR-0008). Pushes to
`main` build the API container and publish it to GHCR (`.github/workflows/publish-image.yml`).

## Checking the live deploy

The site's `/demos/rag` (whoami.joranbergfeld.com) is the working example of a client. The
calls live in the personal-website repo; `docs/subsystems/rag-and-budget.md` there maps them:
our `/modes` feeds its `/rag/status`, our `/run/stream` sits behind `/api/rag/query/stream`.
A rerank mode is really reranking when the `reranked` stage has 0 to 1 cross-encoder scores,
not RRF's ~0.03.
