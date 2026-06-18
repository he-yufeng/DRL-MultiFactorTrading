from __future__ import annotations

import numpy as np
import pytest

from strategy_metrics import (
    annualized_return,
    annualized_volatility,
    benchmark_comparison,
    calmar_ratio,
    conditional_value_at_risk,
    downside_deviation,
    drawdown_analysis,
    max_drawdown,
    returns_from_equity,
    sharpe_ratio,
    summarize_equity_curve,
    summarize_vs_benchmark,
    value_at_risk,
)


def test_returns_from_equity_curve() -> None:
    returns = returns_from_equity([100.0, 110.0, 99.0])

    assert returns.tolist() == pytest.approx([0.10, -0.10])


def test_max_drawdown_is_positive_fraction() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25)


def test_drawdown_analysis_tracks_recovery_and_multiple_episodes() -> None:
    analysis = drawdown_analysis([100.0, 120.0, 90.0, 120.0, 110.0, 125.0])

    assert analysis["max_drawdown"] == pytest.approx(0.25)
    assert analysis["peak_index"] == 1
    assert analysis["trough_index"] == 2
    assert analysis["recovery_index"] == 3
    assert analysis["drawdown_duration"] == 1
    assert analysis["recovery_duration"] == 1
    assert analysis["underwater_periods"] == 2
    assert analysis["ulcer_index"] > 0


def test_drawdown_analysis_marks_unrecovered_trough() -> None:
    analysis = drawdown_analysis([100.0, 120.0, 90.0, 100.0])

    assert analysis["recovery_index"] == -1
    assert analysis["recovery_duration"] == -1


def test_volatility_and_sharpe_are_finite() -> None:
    returns = [0.01, -0.005, 0.012, 0.002, -0.001]

    assert annualized_volatility(returns) > 0
    assert downside_deviation(returns) > 0
    assert sharpe_ratio(returns) != 0


def test_sortino_and_calmar_metrics_are_reported() -> None:
    equity = [100.0, 104.0, 102.0, 112.0, 118.0]

    summary = summarize_equity_curve(equity)

    assert annualized_return(equity) > 0
    assert calmar_ratio(equity) > 0
    assert summary["sortino"] > 0
    assert summary["calmar"] > 0
    assert summary["downside_deviation"] > 0


def test_summarize_equity_curve() -> None:
    summary = summarize_equity_curve([100.0, 105.0, 98.0, 112.0])

    assert summary["total_return"] == pytest.approx(0.12)
    assert summary["max_drawdown"] == pytest.approx((105.0 - 98.0) / 105.0)
    assert summary["annualized_return"] > 0
    assert summary["annualized_volatility"] > 0
    assert "sharpe" in summary
    assert "sortino" in summary
    assert "calmar" in summary
    assert summary["max_drawdown_duration"] == 1
    assert summary["recovery_duration"] == 1
    assert summary["underwater_periods"] == 1
    assert summary["ulcer_index"] > 0


def test_rejects_multidimensional_equity() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        returns_from_equity([[100.0, 101.0]])

    with pytest.raises(ValueError, match="one-dimensional"):
        summarize_equity_curve([[100.0, 101.0]])


def test_benchmark_comparison_recovers_known_beta_and_alpha() -> None:
    # Strategy return is 2x the benchmark every period plus a steady 1% edge.
    benchmark = [100.0, 105.0, 99.75, 109.725]
    strategy = [100.0, 111.0, 101.01, 122.2221]

    result = benchmark_comparison(strategy, benchmark, periods_per_year=1)

    assert result["beta"] == pytest.approx(2.0)
    assert result["alpha"] == pytest.approx(0.01)
    assert result["win_rate"] == pytest.approx(2 / 3)
    assert result["excess_annualized_return"] > 0


def test_benchmark_comparison_pure_leverage_has_no_alpha() -> None:
    benchmark = [100.0, 105.0, 99.75, 109.725]
    strategy = [100.0, 110.0, 99.0, 118.8]

    result = benchmark_comparison(strategy, benchmark, periods_per_year=1)

    assert result["beta"] == pytest.approx(2.0)
    assert result["alpha"] == pytest.approx(0.0, abs=1e-9)


def test_benchmark_comparison_information_ratio_and_tracking_error() -> None:
    # Active return is [-1%, +1%, +3%]: mean 1%, sample std 2%.
    benchmark = [100.0, 102.0, 103.02, 100.9596]
    strategy = [100.0, 101.0, 103.02, 104.0502]

    result = benchmark_comparison(strategy, benchmark, periods_per_year=1)

    assert result["tracking_error"] == pytest.approx(0.02)
    assert result["information_ratio"] == pytest.approx(0.5)
    assert result["win_rate"] == pytest.approx(2 / 3)


def test_benchmark_comparison_identical_curves() -> None:
    curve = [100.0, 110.0, 99.0, 105.0]

    result = benchmark_comparison(curve, curve)

    assert result["beta"] == pytest.approx(1.0)
    assert result["alpha"] == pytest.approx(0.0)
    assert result["excess_annualized_return"] == pytest.approx(0.0)
    assert result["information_ratio"] == 0.0
    assert result["tracking_error"] == 0.0
    assert result["win_rate"] == 0.0


def test_benchmark_comparison_handles_flat_benchmark() -> None:
    result = benchmark_comparison([100.0, 105.0, 103.0], [100.0, 100.0, 100.0])

    assert result["beta"] == 0.0
    assert all(np.isfinite(value) for value in result.values())


def test_benchmark_comparison_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        benchmark_comparison([100.0, 101.0, 102.0], [100.0, 101.0])

    with pytest.raises(ValueError, match="one-dimensional"):
        benchmark_comparison([[100.0, 101.0]], [[100.0, 101.0]])

    empty = benchmark_comparison([100.0], [100.0])
    assert empty == {
        "excess_annualized_return": 0.0,
        "information_ratio": 0.0,
        "tracking_error": 0.0,
        "beta": 0.0,
        "alpha": 0.0,
        "win_rate": 0.0,
    }


def test_summarize_vs_benchmark_bundles_sections() -> None:
    strategy = [100.0, 108.0, 104.0, 119.0]
    benchmark = [100.0, 103.0, 101.0, 106.0]

    summary = summarize_vs_benchmark(strategy, benchmark)

    assert set(summary) == {"strategy", "benchmark", "relative"}
    assert summary["strategy"]["total_return"] > summary["benchmark"]["total_return"]
    assert summary["relative"]["win_rate"] > 0
    assert summary["relative"]["excess_annualized_return"] > 0


def test_value_at_risk_reports_positive_tail_loss() -> None:
    returns = [-0.08, -0.04, -0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.05]
    var = value_at_risk(returns, confidence=0.90)
    # the 10th-percentile return is a loss, surfaced as a positive fraction
    assert var > 0.0
    assert var == pytest.approx(max(-float(np.percentile(returns, 10.0)), 0.0))


def test_cvar_is_at_least_as_severe_as_var() -> None:
    returns = [-0.08, -0.04, -0.01, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.05]
    var = value_at_risk(returns, confidence=0.90)
    cvar = conditional_value_at_risk(returns, confidence=0.90)
    # expected shortfall averages the worst tail, so it never reports milder than VaR
    assert cvar >= var > 0.0


def test_var_and_cvar_are_zero_when_no_tail_loss() -> None:
    gains = [0.01, 0.02, 0.03, 0.04, 0.05]
    assert value_at_risk(gains) == 0.0
    assert conditional_value_at_risk(gains) == 0.0


def test_value_at_risk_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        value_at_risk([0.01, -0.02], confidence=1.5)
