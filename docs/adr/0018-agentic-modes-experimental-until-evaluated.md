# 0018 - Agentic retrieval modes are experimental until evaluated

**Status:** Accepted (2026-06-24)

## Context

The pipeline exposes **9 retrieval modes** (ADR-0012 substrate seam). Five of them ship
with committed, head-to-head eval evidence: `baseline`, `contextual`, `raptor_sac`,
`graphrag`, `combined` each have an `eval_results_<mode>.json` checkpoint that the
dashboard and `/eval` render (ADR-0016).

The four `*_agentic` wrappers (`baseline_agentic`, `raptor_sac_agentic`,
`graphrag_agentic`, `combined_agentic`) do **not**. There is no
`eval_results_*_agentic.json`, yet the wrappers are first-class everywhere else: they are
registered in `retrieval/registry.py`, selectable in the dashboard, and 422-validated by
`/run`, `/run/stream`, and `/compare` exactly like the evaluated modes. The exposed
surface area therefore overstates the evidence.

This matters because the "agentic" label oversells the mechanism. Per ADR-0015 the wrapper
is a single-shot LLM planner plus a fixed, bounded fan-out (`agentic_max_iterations`, no
sufficiency-based early stop), not an agent plan/act/observe loop. The literature is clear
that planned multi-query / iterative retrieval has **workload-dependent** value -- it helps
mainly on multi-hop queries and can add cost without gains otherwise -- so its benefit must
be measured per regime, not assumed (see Sources).

Two honest options: (a) run the eval over the four wrappers so they earn their first-class
status, or (b) gate them as experimental/unevaluated until that eval exists. A full eval
run requires live Azure Foundry + Search and the judge deployments, and takes hours; it
cannot be produced as part of this change. So we choose (b) now and leave (a) as the
follow-up that flips the flag.

## Decision

1. **Mark, don't remove.** The four `*_agentic` modes stay registered and runnable -- they
   already emit `iter_*` / `fused` stages and work end-to-end. We only label them
   experimental/unevaluated so the surface matches the evidence; we do not break callers or
   the dashboard.

2. **One structural source of truth.** `config.EXPERIMENTAL_MODES` is derived as every
   `RetrievalMode` whose value ends in `_agentic`, with an `is_experimental_mode()` helper.
   Deriving it structurally (rather than hand-listing four names) means any future wrapper
   is experimental by default until it earns committed eval evidence -- the safe default.

3. **Surface it in the API.** `/run`, `/run/stream`, and `/compare` payloads carry an
   `experimental` boolean per result, and a new `GET /modes` returns every runnable mode
   with its flag plus an `experimental` list. Clients can discover, at runtime, which modes
   are benchmarked.

4. **Surface it in the UI and docs.** The dashboard Run tab warns when an experimental mode
   is selected. The README mode list, API reference, and evaluation-results section all
   label the four wrappers experimental/unevaluated and point back to issue #11 and this ADR.

5. **Exit criterion.** When an eval run produces committed `eval_results_*_agentic.json`
   files for these modes -- ideally on the multi-hop cohort where planning should actually
   help -- this ADR is superseded and the flag is dropped (the wrappers no longer match the
   `EXPERIMENTAL_MODES` predicate only after we also stop deriving it purely from the
   suffix; until then the eval evidence is what reclassifies them in the docs/UI narrative).

## Alternatives rejected

- **Remove the agentic modes from the registry/API.** Throws away working, composable code
  (ADR-0015) and a genuine research axis. The honest gap is *evaluation*, not capability;
  hiding the modes would also delete the `iter_*` stage visibility the harness needs to
  eventually score them.
- **Hand-list the four mode names as experimental.** Brittle: a fifth wrapper added later
  would silently ship as "evaluated". The suffix-derived predicate fails safe instead.
- **Claim eval coverage by running it here.** Not possible without live Azure resources and
  hours of judge calls; fabricating or stubbing results would defeat the entire point of
  the evidence-matching exercise.

## Consequences

- The exposed surface now matches the evidence: anything labeled experimental has no
  committed eval, and `/modes` makes that machine-readable.
- A new `*_agentic` wrapper is automatically flagged experimental until evaluated -- no code
  change needed to keep the invariant true.
- The flag is descriptive, not a gate: experimental modes still run, so existing callers and
  the `/compare` flow are unaffected. Teams that want a hard block can branch on
  `experimental` themselves.
- A follow-up eval run (issue #11) is still required to actually learn whether planned
  multi-query retrieval helps on this corpus; this ADR only stops the surface from implying
  that answer is already known.

## Sources

- Issue #11 (this repo): "Eval: evaluate the four `*_agentic` modes or mark them
  experimental."
- ADR-0012 (retrieval substrate seam, the 9 modes), ADR-0015 (agentic wrapper: single-shot
  planner + bounded fan-out, no early stop), ADR-0016 (multi-mode evaluation axis and the
  committed per-mode eval files).
- Implementation: `src/ragpipe/config.py` (`EXPERIMENTAL_MODES`, `is_experimental_mode`),
  `app/api.py` (`experimental` payload field, `GET /modes`), `app/dashboard.py` (Run-tab
  warning), `README.md` (mode list / API reference / evaluation section).
- Azure AI Search agentic (knowledge-agent) retrieval overview -- planned/multi-query
  retrieval is a real but workload-dependent technique:
  https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-overview
- Gutierrez et al., *HippoRAG*, NeurIPS 2024 -- multi-step retrieval gains are largely
  multi-hop-specific: https://arxiv.org/abs/2405.14831
- Li et al., *LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs*,
  ICML 2025 -- iterative/agentic retrieval is "no silver bullet" and can underperform
  without the right query regime: https://proceedings.mlr.press/v267/li25dv.html
- Li et al., EMNLP 2024 (Industry) -- routing/when-to-retrieve analysis:
  https://aclanthology.org/2024.emnlp-industry.66/
