"""Factor evaluation helpers: information coefficient, ICIR, and quantile spread.

These functions answer a question the equity-curve metrics in ``strategy_metrics``
cannot: does a factor actually predict forward returns before it is traded? They
work on plain NumPy arrays so they can be dropped into a research notebook or a
backtest harness without pulling in pandas or scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# IC values are correlations in [-1, 1], so any genuine dispersion dwarfs this.
# The guard catches std that is only non-zero because of floating-point noise.
_ZERO_DISPERSION = 1e-12


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of two equal-length 1-D arrays, 0.0 when degenerate."""
    if x.size < 2:
        return 0.0
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = math.sqrt(float(np.dot(x_centered, x_centered)) * float(np.dot(y_centered, y_centered)))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank values from 0, assigning the average rank to ties (Spearman-style)."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    index = 0
    while index < values.size:
        end = index
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        average_rank = (index + end) / 2.0
        ranks[order[index : end + 1]] = average_rank
        index = end + 1
    return ranks


def _clean_pair(factor: Sequence[float], forward_returns: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Validate two 1-D sequences and drop positions where either value is non-finite."""
    factor_values = np.asarray(factor, dtype=float)
    return_values = np.asarray(forward_returns, dtype=float)
    if factor_values.ndim != 1 or return_values.ndim != 1:
        raise ValueError("factor and forward_returns must be one-dimensional sequences")
    if factor_values.size != return_values.size:
        raise ValueError("factor and forward_returns must be the same length")
    mask = np.isfinite(factor_values) & np.isfinite(return_values)
    return factor_values[mask], return_values[mask]


def information_coefficient(
    factor: Sequence[float],
    forward_returns: Sequence[float],
    method: str = "pearson",
) -> float:
    """Correlation between a single cross-section of factor exposures and forward returns.

    ``method="spearman"`` ranks both series first, giving the rank IC that is robust
    to outliers and monotonic-but-nonlinear relationships. Pairs containing NaN or
    inf are dropped; fewer than two valid pairs yields 0.0.
    """
    factor_values, return_values = _clean_pair(factor, forward_returns)
    if factor_values.size < 2:
        return 0.0
    if method == "pearson":
        return _pearson(factor_values, return_values)
    if method == "spearman":
        return _pearson(_average_ranks(factor_values), _average_ranks(return_values))
    raise ValueError("method must be 'pearson' or 'spearman'")


def factor_ic_series(
    factor_panel: Sequence[Sequence[float]],
    forward_return_panel: Sequence[Sequence[float]],
    method: str = "spearman",
) -> np.ndarray:
    """Per-period IC for a panel shaped ``(periods, assets)``.

    Each row is one rebalance date: the factor exposures across the cross-section
    of assets and the returns realised over the following period. Rows with fewer
    than two valid (finite) pairs are skipped rather than reported as zero.
    """
    factor_values = np.asarray(factor_panel, dtype=float)
    return_values = np.asarray(forward_return_panel, dtype=float)
    if factor_values.ndim != 2 or return_values.ndim != 2:
        raise ValueError("panels must be two-dimensional (periods, assets)")
    if factor_values.shape != return_values.shape:
        raise ValueError("factor and forward-return panels must have the same shape")
    coefficients = []
    for period in range(factor_values.shape[0]):
        row_factor = factor_values[period]
        row_returns = return_values[period]
        valid = int(np.sum(np.isfinite(row_factor) & np.isfinite(row_returns)))
        if valid < 2:
            continue
        coefficients.append(information_coefficient(row_factor, row_returns, method))
    return np.asarray(coefficients, dtype=float)


def ic_information_ratio(ic_series: Sequence[float]) -> float:
    """Mean IC divided by its standard deviation (the ICIR), 0.0 when undefined."""
    coefficients = np.asarray(ic_series, dtype=float)
    if coefficients.ndim != 1:
        raise ValueError("ic_series must be a one-dimensional sequence")
    if coefficients.size < 2:
        return 0.0
    std = np.std(coefficients, ddof=1)
    if std < _ZERO_DISPERSION:
        return 0.0
    return float(np.mean(coefficients) / std)


def factor_quantile_returns(
    factor: Sequence[float],
    forward_returns: Sequence[float],
    quantiles: int = 5,
) -> np.ndarray:
    """Mean forward return per factor quantile, ordered from lowest to highest exposure.

    A monotonic profile and a positive top-minus-bottom spread are the usual sign
    that a factor carries tradeable information. NaN/inf pairs are dropped first.
    """
    if quantiles < 1:
        raise ValueError("quantiles must be positive")
    factor_values, return_values = _clean_pair(factor, forward_returns)
    if factor_values.size == 0:
        return np.zeros(quantiles, dtype=float)
    if quantiles > factor_values.size:
        raise ValueError("quantiles cannot exceed the number of observations")
    order = np.argsort(factor_values, kind="mergesort")
    buckets = np.array_split(order, quantiles)
    return np.asarray([float(np.mean(return_values[bucket])) for bucket in buckets], dtype=float)


def summarize_factor(
    factor_panel: Sequence[Sequence[float]],
    forward_return_panel: Sequence[Sequence[float]],
    method: str = "spearman",
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Headline predictive-power statistics for a factor across many rebalance dates.

    ``hit_rate`` is the fraction of periods with a positive IC and ``t_stat`` tests
    whether the mean IC differs from zero. ``ic_ir_annualized`` scales the ICIR by
    ``sqrt(periods_per_year)`` so factors sampled at different frequencies compare.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    ic_series = factor_ic_series(factor_panel, forward_return_panel, method)
    if ic_series.size == 0:
        return {
            "mean_ic": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_ir_annualized": 0.0,
            "hit_rate": 0.0,
            "t_stat": 0.0,
            "periods": 0,
        }
    mean_ic = float(np.mean(ic_series))
    ic_std = float(np.std(ic_series, ddof=1)) if ic_series.size > 1 else 0.0
    information_ratio = ic_information_ratio(ic_series)
    t_stat = (
        float(mean_ic / (ic_std / math.sqrt(ic_series.size)))
        if ic_std > _ZERO_DISPERSION
        else 0.0
    )
    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_ir": information_ratio,
        "ic_ir_annualized": float(information_ratio * math.sqrt(periods_per_year)),
        "hit_rate": float(np.mean(ic_series > 0)),
        "t_stat": t_stat,
        "periods": int(ic_series.size),
    }
