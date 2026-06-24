"""Confidence intervals and paired significance tests for the eval harness.

The multi-mode comparison (ADR-0016) decides between up to 9 retrieval modes on a
small test set (n in the tens). At that scale the per-item variance of both the
LLM-judged RAGAS metrics and the deterministic URL-match metrics swamps most
mode-to-mode mean differences, so a bare difference of means is not evidence of a
real effect. This module adds the two reporting primitives the IR/RAG-eval
literature recommends for exactly this situation:

- **Percentile bootstrap confidence intervals** on each per-mode mean. Calibrated
  estimation *with intervals* is the recommended fix for LLM-judge noise at small n
  (ARES, Saad-Falcon et al., NAACL 2024). Overlapping intervals between two modes
  are reported as "no measurable difference".
- **A paired randomization (sign-flip) test** of each mode against the baseline,
  the standard significance test for IR evaluation (Smucker, Allan & Carterette,
  CIKM 2007), complemented by a bootstrap CI on the paired per-item effect.

Everything here is pure, deterministic (seeded RNG), and network-free, so it is
unit-tested directly with no Azure/RAGAS wiring. The functions operate on
per-item *metric dicts* (``{metric_name: score}``) — the shape produced by both
``EvalRecord.metrics`` and the ``records`` entries persisted in
``eval_results_<mode>.json`` — so the same code serves the live run and a re-run
that resumes from committed checkpoints.
"""
from __future__ import annotations

import math
import random
from statistics import mean

# Defaults tuned for the small-n eval: 10k resamples is plenty stable for n in the
# tens yet finishes in well under a second per metric, and a fixed seed makes every
# reported interval and p-value reproducible across runs (ADR-0016's reproducibility
# goal). 95% is the conventional confidence level.
DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260616

MetricsRow = dict[str, float]


def _valid(value: object) -> bool:
    """A usable score: a real, finite number (mirrors harness._is_valid)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list (q in [0, 1])."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def bootstrap_mean_ci(
    values,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict | None:
    """Percentile-bootstrap CI for the mean of ``values``.

    Non-finite/NaN scores are dropped first (RAGAS occasionally emits NaN for an
    item; see harness.aggregate). Returns ``{"mean", "low", "high", "n"}`` or
    ``None`` when no valid scores remain. With a single valid score the interval
    degenerates to that point (a CI is undefined for n=1, but a point keeps the
    output uniform).
    """
    clean = [float(v) for v in values if _valid(v)]
    if not clean:
        return None
    point = mean(clean)
    if len(clean) == 1:
        return {"mean": point, "low": point, "high": point, "n": 1}
    n = len(clean)
    rng = random.Random(seed)
    resampled: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += clean[rng.randrange(n)]
        resampled.append(total / n)
    resampled.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": point,
        "low": _percentile(resampled, alpha),
        "high": _percentile(resampled, 1.0 - alpha),
        "n": n,
    }


def _aligned_diffs(treatment, baseline) -> list[float]:
    """Per-item (treatment - baseline) differences over index-aligned pairs.

    Inputs are positionally aligned (same test item at the same index, the order
    the harness evaluates them). A pair is dropped when either side is missing or
    non-finite, so the test only counts items both modes actually scored.
    """
    pairs = zip(treatment, baseline)
    return [float(t) - float(b) for t, b in pairs if _valid(t) and _valid(b)]


def paired_randomization_test(
    treatment,
    baseline,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict | None:
    """Paired sign-flip randomization test of treatment vs. baseline.

    Computes per-item differences ``d_i = treatment_i - baseline_i`` over aligned,
    jointly-valid pairs, then estimates a two-sided p-value by randomly flipping the
    sign of each difference (the exchangeable-under-H0 permutation for paired data;
    Smucker et al., CIKM 2007) and counting how often the resampled mean is at least
    as extreme as the observed one. Also returns a percentile-bootstrap CI on the
    mean difference (the effect size). Returns ``None`` if fewer than two valid pairs
    remain (no variance to test). Keys: ``n``, ``mean_diff``, ``ci_low``,
    ``ci_high``, ``p_value``, ``significant``.
    """
    diffs = _aligned_diffs(treatment, baseline)
    if len(diffs) < 2:
        return None
    n = len(diffs)
    observed = mean(diffs)
    rng = random.Random(seed)

    at_least_as_extreme = 0
    abs_observed = abs(observed)
    for _ in range(n_resamples):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        # Tolerance guards against float noise when observed is effectively zero.
        if abs(total / n) >= abs_observed - 1e-12:
            at_least_as_extreme += 1
    # Add-one smoothing: a randomization p-value is never exactly 0.
    p_value = (at_least_as_extreme + 1) / (n_resamples + 1)

    boot_seed = seed + 1
    boot_rng = random.Random(boot_seed)
    resampled: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[boot_rng.randrange(n)]
        resampled.append(total / n)
    resampled.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": n,
        "mean_diff": observed,
        "ci_low": _percentile(resampled, alpha),
        "ci_high": _percentile(resampled, 1.0 - alpha),
        "p_value": p_value,
        "significant": p_value < (1.0 - confidence),
    }


def _per_metric_values(rows: list[MetricsRow]) -> dict[str, list]:
    """Column-major view: {metric: [score per row]} preserving row order/length."""
    keys = sorted({k for row in rows for k in row})
    return {k: [row.get(k) for row in rows] for k in keys}


def mode_confidence_intervals(
    rows: list[MetricsRow],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict]:
    """Bootstrap CI per metric across a mode's per-item metric rows.

    ``rows`` is one dict of ``{metric: score}`` per evaluated item. Returns
    ``{metric: {"mean", "low", "high", "n"}}``, omitting metrics with no valid
    scores. Keeps per-stage keys (``metric@stage``) as-is, so deterministic
    retrieval metrics get intervals too.
    """
    out: dict[str, dict] = {}
    for metric, values in _per_metric_values(rows).items():
        ci = bootstrap_mean_ci(
            values, confidence=confidence, n_resamples=n_resamples, seed=seed
        )
        if ci is not None:
            out[metric] = ci
    return out


def paired_tests_vs_baseline(
    rows_by_mode: dict[str, list[MetricsRow]],
    baseline_mode: str,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, dict]]:
    """Paired significance test of every other mode against the baseline, per metric.

    ``rows_by_mode`` maps mode name -> per-item metric rows, with items in the same
    order across modes (the harness evaluates the same test set per mode, so index i
    is the same item everywhere). Returns ``{mode: {metric: <paired result>}}`` for
    each non-baseline mode. Returns ``{}`` if the baseline mode is absent. A metric a
    mode shares with the baseline but with fewer than two jointly-valid pairs is
    skipped.
    """
    if baseline_mode not in rows_by_mode:
        return {}
    base_rows = rows_by_mode[baseline_mode]
    base_cols = _per_metric_values(base_rows)
    out: dict[str, dict[str, dict]] = {}
    for mode, rows in rows_by_mode.items():
        if mode == baseline_mode:
            continue
        cols = _per_metric_values(rows)
        per_metric: dict[str, dict] = {}
        for metric, treatment in cols.items():
            if metric not in base_cols:
                continue
            result = paired_randomization_test(
                treatment,
                base_cols[metric],
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed,
            )
            if result is not None:
                per_metric[metric] = result
        if per_metric:
            out[mode] = per_metric
    return out
