# 0006 — Baseline-before-treatment evaluation with tagged test subsets

**Status:** Accepted (2026-06-10)
**Spec:** `docs/superpowers/specs/2026-06-10-preprocessing-contextual-decoration-design.md`

## Context

The 16-item hand-authored test set shows recall = 1.0 at every retrieval stage:
retrieval never fails on it, so no retrieval improvement can be demonstrated (or
falsified) against it. Worse, if eval changes and pipeline changes land together,
any metric movement is unattributable.

## Decision

1. **Expand the test set** to ~40–50 items with a `tags` field: the original 16
   (`original`), ~10 hand-authored hard items (`paraphrase`: low lexical overlap;
   `lookalike`: answer lives on one of several near-identical service pages — the
   DRM case), and screened synthetic items (`synthetic`). Aggregation reports per
   tag group.
2. **Sequence measurement around the change**: metrics + test set land first with
   no pipeline changes; a baseline run against the current index is committed
   (`eval_baseline.json`); only then do extraction/decoration/re-ingest land,
   followed by the post-change run.
3. **Success criteria fixed in advance**: `hit_rate`/`mrr` improve on `paraphrase`
   and `lookalike`; no regression on `original`; RAGAS faithfulness mean does not
   degrade. No gain on the hard subsets ⇒ revisit ADR-0001 (SAC / breadcrumb-only
   fallbacks).

## Alternatives rejected

- **Measure on the existing 16 items**: saturated; "1.0 before, 1.0 after" teaches
  nothing.
- **Fully synthetic expansion**: cheap but biased toward questions the corpus
  vocabulary already answers verbatim — exactly the cases decoration doesn't need
  to fix. Hand-authored hard cases target the actual failure mode; synthetic items
  are screened before commit.
- **Single combined change-set**: conflates measurement changes with treatment
  changes; attribution impossible.

## Consequences

- The spec's implementation order is constrained (eval work strictly before
  pipeline work) — slightly slower to the "fun" part, but every later retrieval
  spec (rerank fix, candidate-pool widening) inherits a ready-made baseline
  protocol and a harder test set.
- `eval_baseline.json` is committed as the comparison anchor.

## Sources

- Saad-Falcon et al., *ARES* (small-sample, judge-noise pitfalls in RAG eval;
  motivates fixed protocols and deterministic anchors) —
  https://arxiv.org/abs/2311.09476
- *Towards Reliable Retrieval in RAG Systems for Large Legal Datasets*
  (DRM-targeted evaluation design) — https://arxiv.org/abs/2510.06999

## Addendum (2026-06-11)

ADR-0009 re-anchors all LLM-judged metrics (judge models changed) and adds
`abstained` to every report; ADR-0010 sets the synthetic-data policy and size
targets for the testset expansion in §1. The baseline protocol in §2 is
unchanged and still pending execution: `eval_baseline.json` must be produced
(on the work machine) from `main` *before* the ADR-0009 branch's first eval
run, judged with the same three-family configuration so the pair is comparable.
