# Copilot instructions for `ragpipe`

Observable hybrid-retrieval RAG over Microsoft/Azure docs, built on Microsoft Agent
Framework + Azure AI Foundry + Azure AI Search, evaluated with RAGAS. The `ragpipe`
package (`src/ragpipe/`) holds the pipeline; `app/` is the Streamlit UI + FastAPI
service. Read `README.md` for setup/run/eval commands and `docs/adr/` for the *why*
behind every significant design decision.

## Environment & commands

- **`uv` manages everything.** `uv sync` creates `.venv` and installs prod + dev deps
  from `uv.lock`. There is no manual activation — **prefix every command with `uv run`**
  (e.g. `uv run python -m ragpipe.ingest`, `uv run streamlit run app/dashboard.py`).
- **Test:** `uv run pytest -q` (full suite). Single file / single test:
  `uv run pytest tests/test_config.py -q` ·
  `uv run pytest "tests/test_config.py::test_settings_loads_from_env" -q`.
- **Lint:** `uv run ruff check .` (line-length 100, target py311). Run it before committing.
- `pytest` is configured with `asyncio_mode = "auto"` (async tests need no marker) and
  `pythonpath = ["src", "."]`. Tests run in deterministic order (no `pytest-randomly`).
- Python 3.11+ (`requires-python`); the API image is `python:3.11-slim`.

## Architecture (the big picture spans several files)

Three phases (see the diagram in `README.md` / `docs/pipeline*.svg`):

1. **Ingest** (`ragpipe.ingest`): crawl Microsoft Learn → extract markdown → heading-aware
   chunk → decorate each chunk with a breadcrumb + cached LLM "situating context" → embed →
   upload to Azure AI Search.
2. **Query pipeline**: `app_wiring.build_pipeline_fn(settings, *, mode)` assembles a
   substrate + reranker + Foundry generator agent + RAGAS faithfulness judge into
   `PipelineDeps`, then `workflow.run_pipeline` executes: retrieve **once** → loop
   {rerank (window widens each retry) → generate → faithfulness score → `decide_next`}.
   On `PASS` it returns; on `EXHAUSTED` it sets `abstained` and replaces the answer with
   `ABSTENTION_ANSWER` (directive guardrail, ADR-0009).
3. **Evaluation** (`ragpipe.eval.run`): replays the pipeline over `data/testset.jsonl`,
   scoring deterministic per-stage retrieval metrics + the RAGAS suite vs. a frozen baseline.

**Retrieval substrate seam (ADR-0012) — the central extension point.** Every strategy
implements the `RetrievalSubstrate` protocol (`name` + `async retrieve`). Strategies are
keyed by `RetrievalMode` in `retrieval/registry.py`; `registered_modes()` returns the **9
live modes** (`contextual`, `baseline`, `raptor_sac`, `graphrag`, `combined` + their
`*_agentic` wrappers — note there is **no** `contextual_agentic`). Adding a strategy =
implement the protocol, add index config in `Settings`, register it — nothing downstream
of retrieval changes. `PipelineState.stages` is a **dynamic `dict[str, list[Chunk]]`**;
each substrate names its own stages and the final set is always mirrored under `"reranked"`.
Do not reintroduce fixed `dense`/`bm25` stage fields.

**Three-family judge split (ADR-0009/0011).** The generator (gpt), the online faithfulness
gate (Claude/Anthropic, `JUDGE_MODEL`), and the offline RAGAS judge (DeepSeek,
`OFFLINE_JUDGE_MODEL`) are deliberately different model families to avoid self-preference
bias. `foundry_judge.judge_provider()` routes Anthropic vs. the OpenAI-compatible Foundry
route. `JUDGE_MODEL` is required — builders raise rather than silently fall back to the generator.

**Two app surfaces.** `app/dashboard.py` (Streamlit: Run / Evaluation / Architecture tabs)
and `app/api.py` (FastAPI: `/run`, `/compare`, `/eval`, `/health`). The API reuses the
pipeline and imports the dashboard's *pure* helpers. `mode` is **caller-supplied and
required** everywhere — an omitted/unknown mode returns HTTP 422; there is no server default.

## Conventions specific to this repo

- **`from __future__ import annotations`** is at the top of every module (annotations are
  strings at runtime — relevant for pydantic/Agent-Framework decorators; see notes in
  `workflow.build_viz_workflow` and `guardrail.prewarm_ragas_imports`).
- **Config:** `Settings` is a frozen dataclass built via `Settings.from_env()` (uses
  `load_dotenv`). `RetrievalMode(str, Enum)` members serialize to their `.value` and work
  as plain string dict keys.
- **Testing pattern:** pure logic is unit-tested with fakes/dataclass stand-ins and **no
  network** (construct `Settings(...)` directly, or a minimal `_S` stub). Live Azure wiring
  (`main()`, `build_pipeline_fn`, the live judge builders) is marked `# pragma: no cover`.
- **RAGAS/langchain import hazard:** import `langchain_openai` *before* `ragas` —
  `guardrail._ensure_ragas_importable()` does this and installs a `vertexai` placeholder.
  For judges on the OpenAI-compatible Foundry route, build `AzureChatOpenAI(model=<deployment>)`
  (a null model 400s on the sglang-backed deployments). Every live LLM/judge client is built
  with explicit `timeout` + `max_retries` (shared `JUDGE_TIMEOUT`/`JUDGE_MAX_RETRIES`).
- **Search index `context` field is retrieval-only.** Embeddings are computed from
  `context + "\n\n" + content`, but retrievers select only `content` into prompts/judging.
- **Docs discipline:** record significant decisions as an ADR in `docs/adr/NNNN-kebab.md`
  (Nygard format: Context / Decision / Consequences / **Sources**), written *in the same
  change set* as the decision. Design specs live in `docs/superpowers/specs/`, plans in
  `docs/superpowers/plans/` (date-prefixed).
- **Eval outputs:** each mode writes a committed, resumable `eval_results_<mode>.json`
  checkpoint; the combined `eval_results.json` (read by `/eval` and the dashboard) is rebuilt
  from them and stays gitignored.
- **Commits:** include the trailer `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## Provisioning & deploy

`azd up` provisions Foundry/Search/model deployments via `infra/` Bicep, then the
`postprovision` hook runs `ragpipe.ingest` + `scripts/setup_agents.py`. Default region is
**swedencentral** (only region with Claude + gpt-5.4 + embeddings, ADR-0008). Pushes to
`main` build the API container and publish it to GHCR (`.github/workflows/publish-image.yml`).
