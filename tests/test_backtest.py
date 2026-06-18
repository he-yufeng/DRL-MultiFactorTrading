"""Tests for the self-contained offline backtest harness."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest  # noqa: E402


def test_synthetic_bars_are_deterministic() -> None:
    a = backtest.synthetic_bars(n=50, seed=1)
    b = backtest.synthetic_bars(n=50, seed=1)
    assert [bar.price for bar in a] == [bar.price for bar in b]
    assert len(a) == 50


def test_conservative_backtest_runs_and_is_deterministic() -> None:
    bars = backtest.synthetic_bars(n=300, seed=3)
    first = backtest.run_backtest("conservative", bars)
    second = backtest.run_backtest("conservative", bars)

    assert len(first.equity) == len(bars)
    assert first.trades > 0
    assert "sharpe" in first.metrics
    # the conservative strategy is deterministic, so the same bars reproduce
    # exactly — this guards the harness against accidental hidden state.
    assert first.equity == second.equity
    assert first.trades == second.trades


def test_radical_backtest_runs_to_completion() -> None:
    np.random.seed(0)
    bars = backtest.synthetic_bars(n=250, seed=5)
    result = backtest.run_backtest("radical", bars)

    assert len(result.equity) == len(bars)
    assert result.trades >= 0
    assert all(np.isfinite(value) for value in result.equity)


def test_csv_bars_reads_ohlcv(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["close", "high", "low", "volume"])
        writer.writeheader()
        for i in range(5):
            writer.writerow({"close": 100 + i, "high": 101 + i, "low": 99 + i, "volume": 1000})

    bars = backtest.csv_bars(str(path))
    assert len(bars) == 5
    assert bars[0].price == 100.0
    assert bars[4].high == 105.0
