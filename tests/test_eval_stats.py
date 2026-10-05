from __future__ import annotations

import math

from ragpipe.eval.stats import (
    bootstrap_mean_ci,
    mode_confidence_intervals,
    paired_randomization_test,
    paired_tests_vs_baseline,
)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    a = bootstrap_mean_ci(values, n_resamples=2000, seed=7)
    b = bootstrap_mean_ci(values, n_resamples=2000, seed=7)
    assert a == b  # same seed -> identical interval
    assert math.isclose(a["mean"], 0.6)
    assert a["low"] < a["mean"] < a["high"]
    assert a["n"] == 5


def test_bootstrap_ci_drops_nan_and_none():
    ci = bootstrap_mean_ci([1.0, float("nan"), None, 1.0, 1.0], n_resamples=500)
    assert ci["n"] == 3
    assert math.isclose(ci["mean"], 1.0)
    # zero-variance sample -> degenerate interval at the point
    assert math.isclose(ci["low"], 1.0) and math.isclose(ci["high"], 1.0)


def test_bootstrap_ci_none_when_no_valid_scores():
    assert bootstrap_mean_ci([float("nan"), None]) is None
    assert bootstrap_mean_ci([]) is None


def test_bootstrap_ci_single_value_is_a_point():
    assert bootstrap_mean_ci([0.5]) == {"mean": 0.5, "low": 0.5, "high": 0.5, "n": 1}


def test_bootstrap_ci_ignores_bool():
    # bools are not real scores even though isinstance(True, int)
    assert bootstrap_mean_ci([True, False]) is None


def test_paired_test_detects_a_real_constant_shift():
    base = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    treat = [b + 0.2 for b in base]
    r = paired_randomization_test(treat, base, n_resamples=5000, seed=1)
    assert r["n"] == 8
    assert math.isclose(r["mean_diff"], 0.2)
    assert r["p_value"] < 0.05
    assert r["significant"] is True
    assert r["ci_low"] <= 0.2 <= r["ci_high"]


def test_paired_test_finds_no_difference_for_identical_modes():
    vals = [0.3, 0.5, 0.7, 0.9, 0.1, 0.4]
    r = paired_randomization_test(vals, vals, n_resamples=5000, seed=2)
    assert math.isclose(r["mean_diff"], 0.0)
    assert r["p_value"] == 1.0
    assert r["significant"] is False


def test_paired_test_aligns_and_drops_incomplete_pairs():
    treat = [1.0, 1.0, float("nan"), 1.0]
    base = [0.0, None, 0.0, 0.0]
    r = paired_randomization_test(treat, base, n_resamples=1000, seed=3)
    # only indices 0 and 3 are jointly valid
    assert r["n"] == 2
    assert math.isclose(r["mean_diff"], 1.0)


def test_paired_test_none_with_fewer_than_two_pairs():
    assert paired_randomization_test([1.0], [0.0]) is None
    assert paired_randomization_test([1.0, None], [None, 0.0]) is None


def test_mode_confidence_intervals_keys_each_metric_including_stage_keys():
    rows = [
        {"faithfulness": 0.9, "hit_rate@reranked": 1.0},
        {"faithfulness": 0.7, "hit_rate@reranked": 0.0},
        {"faithfulness": 0.8, "hit_rate@reranked": 1.0},
    ]
    cis = mode_confidence_intervals(rows, n_resamples=1000, seed=5)
    assert set(cis) == {"faithfulness", "hit_rate@reranked"}
    assert math.isclose(cis["faithfulness"]["mean"], 0.8)
    assert cis["hit_rate@reranked"]["n"] == 3


def test_paired_tests_vs_baseline_skips_baseline_and_keys_by_mode():
    rows_by_mode = {
        "baseline": [{"faithfulness": 0.5} for _ in range(8)],
        "contextual": [{"faithfulness": 0.9} for _ in range(8)],
    }
    out = paired_tests_vs_baseline(rows_by_mode, "baseline", n_resamples=2000, seed=4)
    assert set(out) == {"contextual"}
    res = out["contextual"]["faithfulness"]
    assert math.isclose(res["mean_diff"], 0.4)
    assert res["significant"] is True


def test_paired_tests_vs_baseline_empty_when_baseline_absent():
    assert paired_tests_vs_baseline({"contextual": [{"x": 1.0}]}, "baseline") == {}
