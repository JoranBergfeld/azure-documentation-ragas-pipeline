from __future__ import annotations

import math

import pytest

from ragpipe.eval.stats import bootstrap_ci_mean, paired_diff_test


def test_bootstrap_ci_mean_bounds_mean_and_is_reproducible():
    first = bootstrap_ci_mean([0.1, 0.4, 0.8, 0.9], n_resamples=500, seed=7)
    second = bootstrap_ci_mean([0.1, 0.4, 0.8, 0.9], n_resamples=500, seed=7)

    assert first == second
    assert first["n"] == 4
    assert first["lo"] <= first["mean"] <= first["hi"]
    assert first["mean"] == pytest.approx(0.55)


def test_bootstrap_ci_mean_drops_non_finite_values():
    result = bootstrap_ci_mean([1.0, float("nan"), float("inf"), "x", 0.0], n_resamples=100)

    assert result["n"] == 2
    assert result["mean"] == pytest.approx(0.5)
    assert result["lo"] <= result["mean"] <= result["hi"]


def test_bootstrap_ci_mean_single_value_is_degenerate():
    result = bootstrap_ci_mean([0.75])

    assert result == {"mean": 0.75, "lo": 0.75, "hi": 0.75, "n": 1}


def test_bootstrap_ci_mean_empty_finite_values_returns_nan():
    result = bootstrap_ci_mean([float("nan"), float("inf"), object()])

    assert result["n"] == 0
    assert math.isnan(result["mean"])
    assert math.isnan(result["lo"])
    assert math.isnan(result["hi"])


def test_paired_diff_test_identical_inputs_have_zero_diff_and_p_one():
    result = paired_diff_test([0.2, 0.5, 0.9], [0.2, 0.5, 0.9], n_resamples=300)

    assert result["n"] == 3
    assert result["mean_diff"] == 0.0
    assert result["lo"] == 0.0
    assert result["hi"] == 0.0
    assert result["p_value"] == 1.0


def test_paired_diff_test_separated_pairs_exclude_zero_with_small_p_value():
    result = paired_diff_test([0.88, 0.9, 0.92, 0.94], [0.1, 0.12, 0.08, 0.11], n_resamples=500)

    assert result["n"] == 4
    assert result["mean_diff"] > 0.75
    assert result["lo"] > 0.0
    assert result["hi"] > 0.0
    assert result["p_value"] < 0.05


def test_paired_diff_test_drops_pairs_where_either_side_is_non_finite():
    result = paired_diff_test([1.0, float("nan"), 0.7, 0.3], [0.5, 0.2, float("inf"), 0.1])

    assert result["n"] == 2
    assert result["mean_diff"] == pytest.approx(0.35)


def test_paired_diff_test_is_reproducible_with_seed():
    first = paired_diff_test([0.8, 0.3, 0.9], [0.2, 0.4, 0.6], n_resamples=400, seed=99)
    second = paired_diff_test([0.8, 0.3, 0.9], [0.2, 0.4, 0.6], n_resamples=400, seed=99)

    assert first == second


def test_paired_diff_test_single_pair_has_point_ci_and_nan_p_value():
    result = paired_diff_test([0.9], [0.1])

    assert result["n"] == 1
    assert result["mean_diff"] == pytest.approx(0.8)
    assert result["lo"] == pytest.approx(0.8)
    assert result["hi"] == pytest.approx(0.8)
    assert math.isnan(result["p_value"])


def test_paired_diff_test_empty_valid_pairs_returns_nan():
    result = paired_diff_test([float("nan")], [0.1])

    assert result["n"] == 0
    assert math.isnan(result["mean_diff"])
    assert math.isnan(result["lo"])
    assert math.isnan(result["hi"])
    assert math.isnan(result["p_value"])
