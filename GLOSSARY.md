# ragpipe

Retrieval-augmented answers over Microsoft Learn docs, with a faithfulness gate in front of
the user and an offline RAGAS evaluation behind it. The point of the project is comparing
retrieval modes head to head.

## Corpus

**Corpus**:
The set of Microsoft Learn pages the system answers from.
_Avoid_: dataset, knowledge base

**Chunk**:
A heading-aware slice of one corpus page; the thing retrieval returns and the generator cites.
_Avoid_: passage, document, snippet

**Breadcrumb**:
The heading path of a chunk inside its page.

**Situating context**:
A short model-written note that places a chunk within its page. Used to find the chunk, never
shown to the generator or the judges.
_Avoid_: context (on its own, it collides with RAGAS "contexts"), SAC

**Decoration**:
Adding the breadcrumb and situating context to a chunk before it is indexed.

## Retrieval

**Retrieval mode**:
A named, caller-chosen way of retrieving chunks for a query. Every request names one; there is no
default.
_Avoid_: strategy, pipeline type

**Substrate**:
The retrieval approach behind a mode: contextual hybrid, baseline hybrid, RAPTOR over contextual
leaves, GraphRAG, or Combined.
_Avoid_: retriever, backend, index

**Baseline mode**:
The naive mode: plain chunks with no decoration and plain hybrid retrieval, no tricks. Every
other mode is measured against it.
_Avoid_: naive mode, vanilla

**Agentic mode**:
A mode that plans sub-queries for a query and runs its substrate once per sub-query.
_Avoid_: agent mode

**Leg**:
One retrieval path inside a substrate (dense, keyword, graph-local, graph-global) whose results
get fused with the others.

**Stage**:
A named list of chunks recorded along the way to the final set. The final set is always the
**reranked** stage.

**Candidate pool**:
How many chunks each leg brings back before fusion.

**Rerank**:
Reordering the fused candidates and keeping the top of the list. In rerank modes it's a
cross-encoder; elsewhere it keeps the substrate's own order.

**Rerank mode**:
A mode whose last retrieval step is a cross-encoder rerank over one table, so it can only run
while the reranker answers.

**Rerank window**:
How many reranked chunks the generator sees on an attempt. It widens on each retry.

**Query class**:
Whether a query asks for a specific fact (local) or a broad overview (global). Decides whether
GraphRAG runs its global leg.
_Avoid_: intent

**Mode availability**:
Whether a mode can run right now, and if not, why (for example, the reranker is offline).

**Experimental mode**:
A mode exposed without committed evaluation evidence.

## Answering

**Generator**:
The model that writes the answer from the numbered chunks it is given.

**Attempt**:
One rerank, generate and score pass. A query gets the first attempt plus up to the retry limit.

**Faithfulness gate**:
The online check that scores how well an answer is supported by its chunks and decides pass,
retry, or give up.
_Avoid_: guardrail (that's the whole retry-then-abstain policy), validator

**Claim verdict**:
The judge's call on one statement from an answer: supported by the chunks or not. A faithfulness
score is computed from these.

**Abstention**:
Replacing the answer with a fixed "I don't know" when every attempt fails the gate.
_Avoid_: refusal, fallback

**Guardrail**:
The policy around the gate: retry with a wider rerank window, abstain when attempts run out, and
fail closed when the judge can't score.

## Judging

**Judge**:
A model that scores answers rather than writing them. Always a different model family from the
generator.

**Online judge**:
The judge behind the faithfulness gate, called on every attempt.

**Offline judge**:
The judge that scores the RAGAS suite during evaluation.

**Model family**:
A base model and everything derived from it. A fine-tune or distillation belongs to its base's
family, whatever brand it ships under (a Llama-based Nemotron is Llama family).
_Avoid_: vendor, provider

**Judge family split**:
Keeping generator, online judge and offline judge in three different model families so no judge
grades its own family's work. Which families they are doesn't matter.

**Drift canary**:
A frozen set of obviously faithful and obviously unfaithful answers, re-scored on a schedule to
catch a judge quietly changing its mind.

**Gate threshold**:
The faithfulness score an answer needs to pass the gate, calibrated against labelled items.

## Evaluation

**Testset**:
The fixed questions, with expected sources and reference answers, that every mode is replayed
over.

**Cohort**:
A tagged slice of the testset (original, paraphrase, lookalike, synthetic, multi-hop, global
sensemaking) reported on its own.
_Avoid_: subset, split

**Eval checkpoint**:
The committed, resumable results of evaluating one mode.

**Eval results**:
All modes' checkpoints combined into one view for the dashboard and the API.

**Frozen baseline**:
A committed run kept unchanged so later runs of the same mode can be compared with it.
_Avoid_: baseline (on its own, it means the baseline mode)

**Paired comparison**:
Each mode's per-question scores compared with the baseline mode's scores on the same questions.

## Cost

**Usage entry**:
One metered model call's reported usage (tokens or requests), tagged with the stage that made it.
Usage only, never prices.

## Where things run

**Model role**:
A job a model does here: generator, planner, situating-context writer, embedder, online judge,
offline judge, or reranker. Each role is served by exactly one inference host at a time.
_Avoid_: deployment (that's the host's name for a model), client

**Inference host**:
Where a model role is served: the Foundry account or the DGX Spark.
_Avoid_: provider, backend

**Foundry account**:
The Azure AI Foundry account that serves model roles in the cloud.

**DGX Spark**:
The homelab GPU box that serves model roles locally, shared with the other demos.
_Avoid_: Spark (on its own, it reads as Apache Spark)
