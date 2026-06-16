from __future__ import annotations

import math

import numpy as np
import pytest

from factor_analysis import (
    factor_ic_series,
    factor_quantile_returns,
    ic_information_ratio,
    information_coefficient,
    summarize_factor,
)


def test_information_coefficient_perfect_alignment() -> None:
    factor = [1.0, 2.0, 3.0, 4.0]

    assert information_coefficient(factor, [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)
    assert information_coefficient(factor, [0.4, 0.3, 0.2, 0.1]) == pytest.approx(-1.0)


def test_spearman_captures_monotonic_nonlinear_relationship() -> None:
    factor = [1.0, 2.0, 3.0, 4.0]
    forward = [1.0, 4.0, 9.0, 16.0]

    pearson = information_coefficient(factor, forward, method="pearson")
    spearman = information_coefficient(factor, forward, method="spearman")

    assert spearman == pytest.approx(1.0)
    assert pearson < 0.99


def test_spearman_handles_ties_with_average_ranks() -> None:
    # Factor ranks become [0.5, 0.5, 2, 3]; worked out by hand the rank IC is
    # 4.5 / sqrt(4.5 * 5.0) = 0.9486833.
    factor = [10.0, 10.0, 20.0, 30.0]
    forward = [1.0, 2.0, 3.0, 4.0]

    assert information_coefficient(factor, forward, method="spearman") == pytest.approx(0.9486833)


def test_information_coefficient_drops_non_finite_pairs() -> None:
    result = information_coefficient([1.0, 2.0, 3.0, float("nan")], [1.0, 2.0, 3.0, 99.0])

    assert result == pytest.approx(1.0)


def test_information_coefficient_zero_variance_returns_zero() -> None:
    assert information_coefficient([1.0, 1.0, 1.0], [0.1, 0.2, 0.3]) == 0.0


def test_information_coefficient_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        information_coefficient([1.0, 2.0, 3.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="one-dimensional"):
        information_coefficient([[1.0, 2.0]], [[1.0, 2.0]])

    with pytest.raises(ValueError, match="pearson"):
        information_coefficient([1.0, 2.0], [1.0, 2.0], method="kendall")


def test_factor_ic_series_skips_thin_rows() -> None:
    factor_panel = [
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [float("nan"), float("nan"), 1.0],  # only one valid pair, skipped
    ]
    forward_panel = [
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
    ]

    ic_series = factor_ic_series(factor_panel, forward_panel, method="pearson")

    assert ic_series.tolist() == pytest.approx([1.0, -1.0])


def test_factor_ic_series_validates_shapes() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        factor_ic_series([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="same shape"):
        factor_ic_series([[1.0, 2.0]], [[1.0, 2.0, 3.0]])


def test_ic_information_ratio_matches_mean_over_std() -> None:
    # mean 0.2, sample std 0.1 -> ICIR 2.0.
    assert ic_information_ratio([0.1, 0.2, 0.3]) == pytest.approx(2.0)
    assert ic_information_ratio([0.2]) == 0.0
    assert ic_information_ratio([0.2, 0.2, 0.2]) == 0.0


def test_factor_quantile_returns_are_monotonic() -> None:
    factor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    forward = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]

    buckets = factor_quantile_returns(factor, forward, quantiles=3)

    assert buckets.tolist() == pytest.approx([15.0, 35.0, 55.0])
    assert buckets[-1] > buckets[0]


def test_factor_quantile_returns_validates_quantiles() -> None:
    with pytest.raises(ValueError, match="positive"):
        factor_quantile_returns([1.0, 2.0], [0.1, 0.2], quantiles=0)

    with pytest.raises(ValueError, match="exceed"):
        factor_quantile_returns([1.0, 2.0], [0.1, 0.2], quantiles=3)


def test_summarize_factor_reports_strong_signal() -> None:
    factor_panel = [[1.0, 2.0, 3.0]] * 3
    forward_panel = [[1.0, 2.0, 3.0]] * 3

    summary = summarize_factor(factor_panel, forward_panel, method="pearson")

    assert summary["mean_ic"] == pytest.approx(1.0)
    assert summary["ic_std"] == pytest.approx(0.0)
    assert summary["ic_ir"] == 0.0  # zero dispersion leaves the ratio undefined -> 0
    assert summary["hit_rate"] == pytest.approx(1.0)
    assert summary["periods"] == 3


def test_summarize_factor_reports_no_edge_for_mixed_signs() -> None:
    factor_panel = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    forward_panel = [[2.0, 4.0, 6.0], [6.0, 4.0, 2.0]]

    summary = summarize_factor(factor_panel, forward_panel, method="pearson")

    assert summary["mean_ic"] == pytest.approx(0.0)
    assert summary["ic_std"] == pytest.approx(math.sqrt(2.0))
    assert summary["hit_rate"] == pytest.approx(0.5)
    assert summary["t_stat"] == pytest.approx(0.0)
    assert summary["periods"] == 2


def test_summarize_factor_annualizes_information_ratio() -> None:
    rng = np.random.default_rng(0)
    # Factor that genuinely leads returns plus mild noise across many dates.
    factor_panel = rng.normal(size=(40, 30))
    forward_panel = 0.5 * factor_panel + rng.normal(scale=0.5, size=(40, 30))

    summary = summarize_factor(factor_panel, forward_panel, method="pearson", periods_per_year=252)

    assert summary["mean_ic"] > 0
    assert summary["ic_ir"] > 0
    assert summary["ic_ir_annualized"] == pytest.approx(summary["ic_ir"] * math.sqrt(252))
    assert summary["t_stat"] > 0
    assert summary["periods"] == 40


def test_summarize_factor_empty_panel_is_safe() -> None:
    summary = summarize_factor(np.empty((0, 3)), np.empty((0, 3)))

    assert summary["periods"] == 0
    assert all(value == 0 for value in summary.values())
