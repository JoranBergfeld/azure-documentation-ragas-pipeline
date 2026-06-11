# 0010 — Synthetic test data policy

**Status:** Accepted (2026-06-11)

## Context

The testset must grow (n=33 supports no statistically meaningful subset
comparison: binary hit_rate on a 7-item tag group has a 95% CI of roughly
±35pp), but synthetic generation has two failure modes: (1) distributional
bias — questions generated FROM chunks lexically echo them, trivially easy for
BM25/embeddings, understating exactly the vocabulary-mismatch failures the
decoration treatment targets; (2) family self-affinity — questions phrased in
the system-under-test's own idiom. Previous generation also recovered gold URLs
by fragile 200-char substring matching.

## Decision

1. **A different family authors questions:** Claude (`JUDGE_MODEL`) writes
   candidates; the generator (gpt) and embeddings (OpenAI) never author test
   items. Committed items come only from the curated path below.
2. **Provenance gold labels:** candidates are generated from explicitly named
   corpus pages (`scripts/generate_synthetic_testset.py <url> ...`), so the
   gold URL is known by construction, never recovered. (The legacy
   `TESTSET_MODE=synthetic` path inherits the DeepSeek offline-judge model as
   author — also family-separated — and now maps chunks back to provenance
   URLs, dropping unrecoverable rows.)
3. **Mechanical lexical screening:** candidates whose question shares >60% of
   content words with the source page are rejected (`content_word_overlap`),
   plus the existing manual screen before anything is committed.
4. **Human/synthetic monitoring:** items keep the `synthetic` tag; a
   persistent score gap between `synthetic` and `original` items IS the bias,
   made visible per release. Hard subsets (`paraphrase`, `lookalike`) remain
   preferentially hand-authored.
5. **Size target:** grow toward ~30 items per tag group (CI ≈ ±18pp), roughly
   40% human-written / 60% screened synthetic; report per-item paired flips
   (McNemar-style) when comparing runs, not just aggregate deltas.

## Alternatives rejected

- **Hand-authoring everything:** does not scale past ~50 items.
- **RAGAS TestsetGenerator over a corpus sample:** generator and gold-URL
  recovery are both biased/fragile (the previous approach; kept only as the
  legacy ad-hoc mode).
- **Embedding-similarity screening:** uses the system-under-test's own
  embedding space to judge distance — circular; word overlap is crude but
  independent and testable.

## Sources

- ARES (synthetic eval data + judge noise): https://arxiv.org/abs/2311.09476
- BEIR (lexical-overlap bias in IR benchmarks): https://arxiv.org/abs/2104.08663
- ADR-0002 (URL-match metrics), ADR-0006 (tagged subsets, baseline protocol)
