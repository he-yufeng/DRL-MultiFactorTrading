"""Small evaluation helpers for strategy equity curves."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def returns_from_equity(equity: Sequence[float]) -> np.ndarray:
    """Convert an equity curve to simple period returns."""
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1:
        raise ValueError("equity must be a one-dimensional sequence")
    if len(values) < 2:
        return np.array([], dtype=float)
    previous = values[:-1]
    current = values[1:]
    return (current - previous) / np.maximum(np.abs(previous), 1e-12)


def max_drawdown(equity: Sequence[float]) -> float:
    """Return the worst peak-to-trough drawdown as a positive fraction."""
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1:
        raise ValueError("equity must be a one-dimensional sequence")
    if len(values) == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / np.maximum(np.abs(peaks), 1e-12)
    return float(np.max(drawdowns))


def drawdown_analysis(equity: Sequence[float]) -> dict[str, float | int]:
    """Describe the worst drawdown, recovery path, and underwater severity."""
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1:
        raise ValueError("equity must be a one-dimensional sequence")
    if len(values) == 0:
        return {
            "max_drawdown": 0.0,
            "peak_index": -1,
            "trough_index": -1,
            "recovery_index": -1,
            "drawdown_duration": 0,
            "recovery_duration": -1,
            "underwater_periods": 0,
            "ulcer_index": 0.0,
        }

    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / np.maximum(np.abs(peaks), 1e-12)
    trough_index = int(np.argmax(drawdowns))
    peak_index = int(np.argmax(values[: trough_index + 1]))
    recovery_index = -1
    for index in range(trough_index + 1, len(values)):
        if values[index] >= values[peak_index]:
            recovery_index = index
            break

    underwater = drawdowns > 0
    underwater_periods = int(
        sum(
            is_underwater and (index == 0 or not underwater[index - 1])
            for index, is_underwater in enumerate(underwater)
        )
    )
    return {
        "max_drawdown": float(drawdowns[trough_index]),
        "peak_index": peak_index,
        "trough_index": trough_index,
        "recovery_index": recovery_index,
        "drawdown_duration": trough_index - peak_index,
        "recovery_duration": recovery_index - trough_index if recovery_index >= 0 else -1,
        "underwater_periods": underwater_periods,
        "ulcer_index": float(math.sqrt(np.mean(np.square(drawdowns)))),
    }


def annualized_volatility(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized volatility from simple period returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) * math.sqrt(periods_per_year))


def downside_deviation(
    returns: Sequence[float],
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized downside deviation below a minimum acceptable return."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return 0.0
    target = minimum_acceptable_return / periods_per_year
    downside = np.minimum(values - target, 0.0)
    if not np.any(downside):
        return 0.0
    return float(math.sqrt(np.mean(np.square(downside))) * math.sqrt(periods_per_year))


def sharpe_ratio(
    returns: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio from simple period returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return 0.0
    excess = values - risk_free_rate / periods_per_year
    vol = np.std(excess, ddof=1)
    if vol == 0:
        return 0.0
    return float(np.mean(excess) / vol * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: Sequence[float],
    minimum_acceptable_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio using downside volatility only."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return 0.0
    target = minimum_acceptable_return / periods_per_year
    downside = downside_deviation(values, minimum_acceptable_return, periods_per_year)
    if downside == 0:
        return 0.0
    return float(np.mean(values - target) * periods_per_year / downside)


def annualized_return(equity: Sequence[float], periods_per_year: int = 252) -> float:
    """Compound annual growth rate implied by an equity curve."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1:
        raise ValueError("equity must be a one-dimensional sequence")
    if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
        return 0.0
    periods = len(values) - 1
    return float((values[-1] / values[0]) ** (periods_per_year / periods) - 1)


def calmar_ratio(equity: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized return divided by maximum drawdown."""
    dd = max_drawdown(equity)
    if dd == 0:
        return 0.0
    return annualized_return(equity, periods_per_year) / dd


def summarize_equity_curve(equity: Sequence[float], periods_per_year: int = 252) -> dict[str, float]:
    """Return common headline metrics for a backtest equity curve."""
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1:
        raise ValueError("equity must be a one-dimensional sequence")
    if len(values) == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "downside_deviation": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown_duration": 0.0,
            "recovery_duration": -1.0,
            "underwater_periods": 0.0,
            "ulcer_index": 0.0,
        }
    rets = returns_from_equity(values)
    drawdown = drawdown_analysis(values)
    total_return = (values[-1] - values[0]) / max(abs(values[0]), 1e-12)
    return {
        "total_return": float(total_return),
        "max_drawdown": max_drawdown(values),
        "annualized_return": annualized_return(values, periods_per_year),
        "annualized_volatility": annualized_volatility(rets, periods_per_year),
        "downside_deviation": downside_deviation(rets, periods_per_year=periods_per_year),
        "sharpe": sharpe_ratio(rets, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(rets, periods_per_year=periods_per_year),
        "calmar": calmar_ratio(values, periods_per_year=periods_per_year),
        "max_drawdown_duration": float(drawdown["drawdown_duration"]),
        "recovery_duration": float(drawdown["recovery_duration"]),
        "underwater_periods": float(drawdown["underwater_periods"]),
        "ulcer_index": float(drawdown["ulcer_index"]),
    }


def benchmark_comparison(
    strategy_equity: Sequence[float],
    benchmark_equity: Sequence[float],
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compare a strategy equity curve against a benchmark such as buy-and-hold.

    ``win_rate`` is the fraction of periods the strategy return strictly beats
    the benchmark; ``alpha`` and ``excess_annualized_return`` are annualized.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    strategy = np.asarray(strategy_equity, dtype=float)
    benchmark = np.asarray(benchmark_equity, dtype=float)
    if strategy.ndim != 1 or benchmark.ndim != 1:
        raise ValueError("equity curves must be one-dimensional sequences")
    if len(strategy) != len(benchmark):
        raise ValueError("strategy and benchmark must be the same length")

    if len(strategy) < 2:
        return {
            "excess_annualized_return": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
            "beta": 0.0,
            "alpha": 0.0,
            "win_rate": 0.0,
        }

    strategy_returns = returns_from_equity(strategy)
    benchmark_returns = returns_from_equity(benchmark)
    active = strategy_returns - benchmark_returns

    tracking_error = annualized_volatility(active, periods_per_year)
    information_ratio = (
        float(np.mean(active) * periods_per_year / tracking_error)
        if tracking_error > 0
        else 0.0
    )

    if strategy_returns.size < 2 or np.var(benchmark_returns, ddof=1) == 0:
        beta = 0.0
    else:
        covariance = np.cov(strategy_returns, benchmark_returns, ddof=1)[0, 1]
        beta = float(covariance / np.var(benchmark_returns, ddof=1))
    alpha = float(
        (np.mean(strategy_returns) - beta * np.mean(benchmark_returns)) * periods_per_year
    )

    return {
        "excess_annualized_return": annualized_return(strategy, periods_per_year)
        - annualized_return(benchmark, periods_per_year),
        "information_ratio": information_ratio,
        "tracking_error": tracking_error,
        "beta": beta,
        "alpha": alpha,
        "win_rate": float(np.mean(strategy_returns > benchmark_returns)),
    }


def summarize_vs_benchmark(
    strategy_equity: Sequence[float],
    benchmark_equity: Sequence[float],
    periods_per_year: int = 252,
) -> dict[str, dict[str, float]]:
    """Headline metrics for the strategy and benchmark plus their relative comparison."""
    return {
        "strategy": summarize_equity_curve(strategy_equity, periods_per_year),
        "benchmark": summarize_equity_curve(benchmark_equity, periods_per_year),
        "relative": benchmark_comparison(strategy_equity, benchmark_equity, periods_per_year),
    }
