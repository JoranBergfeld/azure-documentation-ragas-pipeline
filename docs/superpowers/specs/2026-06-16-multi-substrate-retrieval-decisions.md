# Autonomous build decision log: multi-substrate retrieval

Joran went to bed and asked me to proceed without involving them, recording decisions for
morning review. This log captures every non-trivial call I made that wasn't explicitly
decided during the brainstorm. Newest entries at the bottom of each section.

## Decided together during the brainstorm (for reference)

- Purpose: research demonstrator, both benchmarking and live serving on the website.
- SAC = the existing contextual decoration; Baseline is plainer than today (no decoration).
- 8 modes = 4 substrates (Baseline, SAC+RAPTOR, GraphRAG, Combined) × agentic on/off.
- Agentic is an orthogonal wrapper over a common `retrieve` interface.
- Hand-roll RAPTOR and GraphRAG, Azure-native, no external RAG frameworks.
- GraphRAG graph stored as flat rows in Azure AI Search (no graph DB).
- Agentic loop built on Microsoft Agent Framework, bounded iterations.
- One spec covering the whole expansion, phased build.
- API: `/query?mode=` plus a `/compare` endpoint for side-by-side.

## Decisions I made autonomously while building

(Filled in as I go. Each entry: what I chose, why, and what the alternative was so you can
overrule it cheaply.)

- **`.superpowers/` gitignore:** already present, no action needed.
- **RAPTOR+SAC gets its own `raptor-sac` index** rather than mutating the Foundry-bound
  contextual index. Why: keeps RAPTOR summary nodes out of the live generator's knowledge
  source. Cost: re-uploads SAC leaves (cheap, decoration is cache-hit). Alternative: add a
  `level` filter on the knowledge source and reuse the existing index.
- (more to come during implementation)

## Open questions for you (morning)

- (Filled in if I hit anything I'd genuinely want your call on but had to pick a default
  for.)
