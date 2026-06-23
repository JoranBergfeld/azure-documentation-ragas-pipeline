from __future__ import annotations

import math
import random
from statistics import mean
from typing import Iterable


def _finite_values(values: Iterable[object]) -> list[float]:
    return [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _mean_bootstrap(values: list[float], n_resamples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    return [mean(rng.choice(values) for _ in range(n)) for _ in range(n_resamples)]


def bootstrap_ci_mean(
    values,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> dict:
    finite = _finite_values(values)
    n = len(finite)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    observed = mean(finite)
    if n == 1:
        return {"mean": observed, "lo": observed, "hi": observed, "n": 1}

    boot = sorted(_mean_bootstrap(finite, n_resamples, seed))
    alpha = 1 - confidence
    return {
        "mean": observed,
        "lo": _percentile(boot, alpha / 2),
        "hi": _percentile(boot, 1 - alpha / 2),
        "n": n,
    }


def paired_diff_test(
    treatment,
    baseline,
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 12345,
) -> dict:
    diffs = [
        float(t) - float(b)
        for t, b in zip(treatment, baseline)
        if isinstance(t, (int, float))
        and isinstance(b, (int, float))
        and math.isfinite(t)
        and math.isfinite(b)
    ]
    n = len(diffs)
    if n == 0:
        return {
            "mean_diff": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "p_value": float("nan"),
            "n": 0,
        }
    observed = mean(diffs)
    if all(diff == 0 for diff in diffs):
        return {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0, "n": n}
    if n == 1:
        return {
            "mean_diff": observed,
            "lo": observed,
            "hi": observed,
            "p_value": float("nan"),
            "n": 1,
        }

    boot = sorted(_mean_bootstrap(diffs, n_resamples, seed))
    alpha = 1 - confidence
    frac_le_zero = sum(1 for value in boot if value <= 0) / len(boot)
    frac_ge_zero = sum(1 for value in boot if value >= 0) / len(boot)
    return {
        "mean_diff": observed,
        "lo": _percentile(boot, alpha / 2),
        "hi": _percentile(boot, 1 - alpha / 2),
        "p_value": min(1.0, 2 * min(frac_le_zero, frac_ge_zero)),
        "n": n,
    }
