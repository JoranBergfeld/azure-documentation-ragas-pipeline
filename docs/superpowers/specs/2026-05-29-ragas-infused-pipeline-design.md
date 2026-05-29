# RAGAS-infused RAG pipeline — design

**Date:** 2026-05-29
**Status:** Draft for review
**Domain:** Microsoft / Azure documentation Q&A

## 1. Summary

Build an end-to-end, observable Retrieval-Augmented Generation system that implements the
hybrid-retrieval + rerank + faithfulness-guardrail pipeline from the source diagram, on a
Microsoft technology stack, and uses **RAGAS** both *online* (an inline faithfulness guardrail)
and *offline* (an evaluation harness). The system answers questions about Microsoft/Azure
documentation and lets a user **see** the pipeline run and **measure** its quality.

The system is a **single decomposed RAG pipeline** — the literal diagram, built as a Microsoft
Agent Framework *Workflow* where every box is its own executor: we control dense retrieval, BM25,
RRF fusion, semantic reranking, generation, and the faithfulness loop. The generation step is a
**Foundry-registered Prompt Agent that carries a Foundry-defined tool (Code Interpreter)**,
satisfying the requirement that agents registered in Foundry leverage Foundry tools. RAGAS scores
the pipeline on a test set (per-stage where decomposition allows) and the dashboard renders both
the live run and the evaluation results.

## 2. Goals / Non-goals

### Goals
- Faithful implementation of the diagram (dense + BM25 → RRF → cross-encoder rerank → generate →
  faithfulness check → loop on low faithfulness).
- Use Microsoft Agent Framework (Python) Workflows as the orchestration backbone.
- Use Azure AI Foundry for models *and* for at least one **agent registered in Foundry that
  leverages a Foundry-defined tool** (the user requirement).
- Use Azure AI Search for hybrid retrieval and the semantic (L2) reranker.
- Use RAGAS online (guardrail) and offline (evaluation report, per-stage where possible).
- Provide a Streamlit dashboard to watch a query flow through the pipeline and view eval results.
- Reproducible infrastructure via `azd` + Bicep.

### Non-goals (YAGNI)
- Production hardening (autoscaling, multi-tenant auth, private networking) — demo-grade only.
- A custom-trained reranker or embedding model.
- Multi-turn conversational memory (single-turn Q&A is enough to demonstrate the pipeline).
- A non-Azure / fully-local fallback (the project is committed to the Foundry stack).
- A bespoke test-set authoring UI — the two sources (hand-authored / synthetic) are selected by
  config, not built/edited in the dashboard.
- A second "managed agent" baseline pipeline. (An earlier draft proposed a Strategy B where a
  single Foundry agent + the Foundry Azure AI Search tool did everything; it was dropped as
  redundant — it shared A's index and also just registered a Foundry agent. The Foundry-tool
  requirement is instead met by attaching Code Interpreter to A's generator. See §4.2.)

## 3. Domain & corpus

- **Corpus:** a curated, fixed list of Microsoft Learn article URLs (e.g. Azure AI Search hybrid
  search, Foundry overview, Agent Framework workflows, RAG concepts). Public content, no privacy
  or governance concerns.
- **Ingestion:** a script fetches each page, converts HTML → text, chunks it (heading-aware,
  ~500–800 tokens with overlap), embeds each chunk with the Foundry embedding deployment, and
  uploads documents to an Azure AI Search index.
- **Index schema (Azure AI Search):**
  - `id` (key), `title`, `url`, `chunk_id`, `content` (searchable, BM25),
    `content_vector` (vector field, HNSW), plus a semantic configuration over `title`/`content`.
- The corpus URL list lives in `data/corpus_sources.yaml` so it is easy to extend.

## 4. Architecture

### 4.1 Strategy A — the decomposed Workflow

Orchestrated as a Microsoft Agent Framework **Workflow** (graph API: `WorkflowBuilder`,
executors, edges, a conditional edge for the loop). Each executor has a single responsibility and
a typed input/output message, and emits a trace event consumed by the dashboard.

```
User query (start executor)
   ├──> Dense retrieval executor   (Azure AI Search vector-only query)
   └──> BM25 retrieval executor    (Azure AI Search full-text-only query)
            │
            └──> RRF fusion executor (our own reciprocal-rank-fusion over the two lists)
                     │
                     └──> Rerank executor (Azure AI Search semantic ranker over fused top-N)
                              │
                              └──> Generate executor (Foundry-registered Prompt Agent)
                                       │
                                       └──> Faithfulness executor (RAGAS faithfulness)
                                                │ pass            ↑ low faithfulness
                                                ▼                 │ (conditional edge,
                                            Answer to user   ─────┘  capped retries)
```

**Executor responsibilities & interfaces** (each is a small, independently testable unit):

| Executor | Input | Output | Backed by |
|---|---|---|---|
| `DenseRetriever` | query | ranked `[Chunk]` + vector scores | `azure-search-documents` vector query; `FoundryEmbeddingClient` for the query embedding |
| `BM25Retriever` | query | ranked `[Chunk]` + BM25 scores | `azure-search-documents` full-text query |
| `RRFFusion` | two `[Chunk]` lists | fused `[Chunk]` with RRF scores | our own code (`score = Σ 1/(k+rank)`) |
| `SemanticReranker` | query + fused top-N | reranked `[Chunk]` with `rerankerScore` | Azure AI Search semantic query **filtered to the fused doc IDs** (see 4.3) |
| `Generator` | query + top reranked chunks | answer text + citations | **Foundry Prompt Agent** (registered) carrying the **Code Interpreter** Foundry tool, consumed via `FoundryAgent`; chunks passed as message context (see §4.2) |
| `FaithfulnessGuard` | answer + context | faithfulness score + pass/fail | RAGAS `faithfulness`, judged by the Foundry chat model |

**The loop:** if faithfulness < threshold and retries remain, a conditional edge routes back to
`RRFFusion` with a widened `top_n` (and/or a flag to pull more candidates), then regenerates.
After max retries, the best-scoring attempt is returned, flagged as "low confidence".

A `PipelineState` object (query, per-stage results, scores, attempt count, trace events) threads
through the workflow and is what the dashboard renders.

### 4.2 The generator agent and its Foundry tool

The `Generator` executor is backed by a **Prompt Agent registered in Azure AI Foundry** (created
via `azure-ai-projects`, consumed via `FoundryAgent`). Its definition attaches the hosted
**Code Interpreter** Foundry tool. This is what satisfies the requirement that a Foundry-registered
agent leverage a Foundry-defined tool.

- **Why Code Interpreter (and not the AI Search tool):** the pipeline already does its own
  retrieval — handing the same job to a Foundry search tool would duplicate the diagram's stages
  and hide them. Code Interpreter instead adds a capability the pipeline *doesn't* have: when an
  Azure-docs question needs a small computation over the retrieved facts (counts, a comparison
  table, pricing/limit arithmetic), the model can run code on demand. The answer still grounds in
  the retrieved chunks, so **RAGAS faithfulness stays meaningful** — the tool transforms/organises
  grounded facts rather than introducing ungrounded ones.
- **How it appears:** tool invocations surface in the agent run steps and are captured into the
  `PipelineState` trace, so the dashboard shows "Code Interpreter ran" alongside the generated
  answer.
- **Behavioural note:** because the tool lives on the *Foundry agent definition*, it is configured
  at agent-registration time (setup script / azd post-provision), not passed in client code. The
  model decides per-query whether to call it; many doc questions won't need it, and that's fine.

### 4.3 The semantic reranker mechanism (risk note)

Azure AI Search's semantic ranker (L2) reranks the results of a query it executes; it is not a
free-standing "rank this arbitrary list" API. To rerank **our** RRF-fused candidate set we:

1. Take the top-N fused doc IDs from `RRFFusion`.
2. Issue a semantic query (`query_type=semantic`) with `search=<user query>` and a filter
   `search.in(id, '<id1,id2,...>')` restricting results to those N docs.
3. Read `@search.rerankerScore` to produce the reranked order.

This genuinely uses the Azure semantic ranker as the "cross-encoder" box over our fused set.
**Fallback if this proves awkward** (e.g. filter + semantic interactions): swap the reranker
executor's internals for a local cross-encoder (`sentence-transformers` bge-reranker) behind the
same interface. The executor boundary makes this a one-file change. *(Decision: try the
Azure-native approach first; the fallback is documented, not built.)*

## 5. RAGAS integration

- **LLM/embeddings wiring:** RAGAS is configured to use the Foundry chat deployment (LLM judge)
  and Foundry embedding deployment, via the Azure OpenAI-compatible client (`langchain-openai`
  `AzureChatOpenAI` / `AzureOpenAIEmbeddings`, or RAGAS's wrappers). One config module builds
  these so both the guardrail and the harness share the same judged model.
- **Online (guardrail):** `faithfulness` only, on the single generated answer + its context. Fast
  enough to gate each attempt.
- **Offline (harness):** the full set — `faithfulness`, `answer_relevancy`, `context_precision`,
  `context_recall` — run over the test set. Because the pipeline is decomposed, we additionally
  attribute the context metrics to retrieval stages (e.g. context precision after RRF vs after
  rerank), which is the main analytical payoff of building it stage-by-stage.
- **Cross-check:** optionally also run Foundry's own `azure-ai-evaluation` RAG evaluators
  (Groundedness, Relevance, Retrieval) for a second opinion. Behind a flag.

## 6. Test set

- **Format:** JSONL with `question`, `ground_truth` (reference answer), and `ground_truth_context`
  (the source chunk/URL the answer came from) in `data/testset.jsonl`.
- **Config switch:** a `TESTSET_MODE` setting selects the source behind a single
  `load_testset()` interface, so the eval harness is agnostic to which is active:
  - `handauthored` (default) — load the curated `data/testset.jsonl` (~15–25 items written by us).
  - `synthetic` — generate the set from the indexed corpus with RAGAS `TestsetGenerator` (using
    the Foundry chat + embedding deployments), cache it to `data/testset.synthetic.jsonl`, and
    load that. Questions are LLM-written but grounded in the *real* indexed documents — facts
    come from the corpus, not invented. A `--regenerate` flag forces a fresh build.
- Both code paths are implemented and exercised; the switch just chooses which `load_testset()`
  returns. The synthetic path is spot-checkable (cached JSONL is human-readable) before a run.

## 7. Visualization

- **Structure diagram:** `WorkflowViz` exports the Strategy-A workflow to Mermaid/Graphviz; the
  rendered diagram is committed to `docs/` and shown in the dashboard's "Architecture" view.
- **Streamlit dashboard** (`app/dashboard.py`):
  - *Run* tab: enter a query → stages light up in sequence showing each stage's chunks and scores
    (dense hits, BM25 hits, fused order, reranked order with `rerankerScore`), the generated
    answer, the faithfulness score, and whether the loop fired (with attempt count).
  - *Evaluation* tab: run/load the RAGAS harness and show per-metric scores (table + bar chart),
    plus the per-stage context metrics (after RRF vs after rerank).
  - *Architecture* tab: the WorkflowViz diagram + a short legend.
- The dashboard reads `PipelineState` (live) and the harness's saved results JSON (eval).

## 8. Provisioning (azd + Bicep)

- `azd up` provisions: a Foundry project/resource, a chat model deployment and an embedding model
  deployment, an Azure AI Search service (tier with semantic ranker enabled), and the Search index.
- A post-provision step (script invoked by azd hooks) runs ingestion (build the index) and
  **registers the Foundry Prompt Agent** (the generator, with the Code Interpreter tool attached)
  via `azure-ai-projects`, writing its name/version into the app config.
- `azd down` tears everything down. Endpoints/deployment names are surfaced as outputs and written
  to `.env` for local runs (`AzureCliCredential` / `DefaultAzureCredential` for auth — no keys in
  code where avoidable).

## 9. Configuration

A single typed settings module loads from environment / `.env`:
`FOUNDRY_PROJECT_ENDPOINT`, `FOUNDRY_CHAT_MODEL`, `FOUNDRY_EMBEDDING_MODEL` (+ models endpoint),
`SEARCH_ENDPOINT`, `SEARCH_INDEX`, `GENERATOR_AGENT_NAME`/`_VERSION`,
`FAITHFULNESS_THRESHOLD`, `MAX_RETRIES`, `TOP_K`, `RRF_K`,
`TESTSET_MODE` (`handauthored` | `synthetic`).
`load_dotenv()` is called explicitly (Agent Framework does not auto-load `.env`).

## 10. Error handling

- **Retrieval:** empty results → short-circuit to a graceful "no relevant docs found" answer
  (never fabricate); transient Search/Foundry errors → bounded retry with backoff.
- **Guardrail loop:** strictly capped by `MAX_RETRIES`; on exhaustion return best attempt flagged
  low-confidence rather than looping forever.
- **RAGAS judge failures:** a failed faithfulness computation is treated as "fail-closed" for the
  guardrail (does not silently pass) and logged.
- **Provisioning/agents:** setup scripts are idempotent (safe to re-run); missing agent
  registration fails fast with a clear message pointing at `azd`/setup.

## 11. Testing strategy

- **Unit:** `RRFFusion` (deterministic given inputs), chunking, config loading, trace assembly,
  test-set loading — pure logic, no network.
- **Component (mocked Azure/Foundry):** each retriever and the reranker against a fake Search
  client; the guardrail against a stubbed RAGAS judge; the loop logic (passes through, loops then
  passes, loops then exhausts).
- **Integration (opt-in, requires Azure):** a smoke test that runs one query end-to-end through
  Strategy A and Strategy B and asserts a well-formed answer + trace.
- **Eval as test:** the RAGAS harness on the hand-authored test set, with non-blocking thresholds
  reported (not hard CI gates initially).

## 12. Proposed project structure

```
ragas-infused-pipeline/
  infra/                     # Bicep + azd (azure.yaml)
  data/
    corpus_sources.yaml      # MS Learn URLs
    testset.jsonl            # hand-authored Q/A/ground-truth
  src/ragpipe/
    config.py                # settings + Foundry/Search/RAGAS clients
    ingest.py                # fetch → chunk → embed → index
    retrieval/
      dense.py  bm25.py  rrf.py  rerank.py
    generate.py              # FoundryAgent generator (carries Code Interpreter tool)
    guardrail.py             # RAGAS faithfulness + loop policy
    workflow.py              # WorkflowBuilder wiring of the pipeline + WorkflowViz export
    state.py                 # PipelineState + trace events
    eval/
      harness.py             # RAGAS suite over testset (overall + per-stage context metrics)
      testset.py             # load_testset(): TESTSET_MODE switch (handauthored | synthetic)
  app/dashboard.py           # Streamlit
  scripts/
    setup_agents.py          # register the Foundry generator agent (+ Code Interpreter tool)
  tests/
  README.md
```

## 13. Milestones (suggested build order)

1. Infra (azd+Bicep) + ingestion → a populated Azure AI Search index.
2. Strategy A retrieval executors (dense, BM25, RRF, rerank) + trace, verified standalone.
3. Generator agent (registered in Foundry, with Code Interpreter tool) + workflow wiring +
   WorkflowViz export.
4. RAGAS faithfulness guardrail + loop.
5. RAGAS offline harness (overall + per-stage context metrics) + report.
6. Streamlit dashboard.

## 14. Open questions / risks

- **Semantic-rerank-over-filtered-set** (4.3) is the main technical risk; mitigated by the
  documented local-cross-encoder fallback behind the same interface.
- **Foundry preview surfaces:** some Foundry agent/tool APIs are preview/experimental and may
  shift; pin SDK versions and isolate Foundry calls behind thin adapters.
- **Cost:** ingestion embeddings + RAGAS LLM-judge calls consume Foundry tokens; keep the corpus
  and test set modest for the demo.
