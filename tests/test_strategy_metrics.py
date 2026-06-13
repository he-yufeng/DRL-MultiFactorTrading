from __future__ import annotations

import pytest

from strategy_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    downside_deviation,
    max_drawdown,
    returns_from_equity,
    sharpe_ratio,
    summarize_equity_curve,
)


def test_returns_from_equity_curve() -> None:
    returns = returns_from_equity([100.0, 110.0, 99.0])

    assert returns.tolist() == pytest.approx([0.10, -0.10])


def test_max_drawdown_is_positive_fraction() -> None:
    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25)


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


def test_rejects_multidimensional_equity() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        returns_from_equity([[100.0, 101.0]])

    with pytest.raises(ValueError, match="one-dimensional"):
        summarize_equity_curve([[100.0, 101.0]])
