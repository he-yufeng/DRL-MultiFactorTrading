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


def annualized_volatility(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualized volatility from simple period returns."""
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    values = np.asarray(returns, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1) * math.sqrt(periods_per_year))


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


def summarize_equity_curve(equity: Sequence[float], periods_per_year: int = 252) -> dict[str, float]:
    """Return common headline metrics for a backtest equity curve."""
    values = np.asarray(equity, dtype=float)
    if values.ndim != 1:
        raise ValueError("equity must be a one-dimensional sequence")
    if len(values) == 0:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
        }
    rets = returns_from_equity(values)
    total_return = (values[-1] - values[0]) / max(abs(values[0]), 1e-12)
    return {
        "total_return": float(total_return),
        "max_drawdown": max_drawdown(values),
        "annualized_volatility": annualized_volatility(rets, periods_per_year),
        "sharpe": sharpe_ratio(rets, periods_per_year=periods_per_year),
    }
