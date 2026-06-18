"""Run the trading strategies offline with the bundled backtest engine.

The Conservative and Radical strategies are event-driven and depend only on the
local :mod:`engine` (no trading account or third-party platform). This module
adds data sources and a small CLI on top of :class:`engine.BacktestEngine`.

Usage::

    python backtest.py                       # Conservative on synthetic data
    python backtest.py --strategy radical    # Radical (numpy Double-DQN)
    python backtest.py --csv prices.csv      # your own OHLCV CSV
    python backtest.py --ticker 0700.HK      # real data via yfinance (optional)
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import Bar, BacktestEngine, BacktestResult  # noqa: E402
from strategy_metrics import summarize_equity_curve  # noqa: E402

STRATEGY_MODULES = {
    "conservative": "Conservative_strategy_clean",
    "radical": "Radical_strategy_clean",
}


def load_strategy(strategy: str):
    """Import a strategy module and return its ``AlgoEvent`` class."""
    module = importlib.import_module(STRATEGY_MODULES[strategy])
    return module.AlgoEvent


def run_backtest(
    strategy: str,
    bars: list[Bar],
    *,
    initial_capital: float = 100_000_000.0,
) -> BacktestResult:
    """Run ``strategy`` ('conservative'|'radical') over ``bars``."""
    algo = load_strategy(strategy)()
    return BacktestEngine(initial_capital).run(algo, bars)


def run_ensemble(
    bars: list[Bar],
    *,
    weights: dict[str, float] | None = None,
    initial_capital: float = 100_000_000.0,
) -> tuple[BacktestResult, dict[str, BacktestResult]]:
    """Run a capital-weighted portfolio of the two strategies and combine equity.

    Allocating capital across an uncorrelated stable strategy (Conservative) and
    an aggressive one (Radical) diversifies the equity curve. Each sub-strategy
    trades its own slice independently; the combined curve is their sum.
    """
    weights = weights or {"conservative": 0.5, "radical": 0.5}
    subs: dict[str, BacktestResult] = {}
    combined = np.zeros(len(bars))
    for strategy, weight in weights.items():
        res = run_backtest(strategy, bars, initial_capital=initial_capital * weight)
        subs[strategy] = res
        combined += np.asarray(res.equity, dtype=float)

    result = BacktestResult(
        equity=[float(value) for value in combined],
        trades=sum(res.trades for res in subs.values()),
    )
    if len(result.equity) > 2:
        result.metrics = summarize_equity_curve(result.equity)
    return result, subs


def synthetic_bars(n: int = 400, seed: int = 7) -> list[Bar]:
    """A deterministic trending-with-noise price path for offline runs/tests."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.0, 0.6, n)
    noise = np.cumsum(rng.normal(0.0, 0.01, n))
    wave = 0.08 * np.sin(np.linspace(0.0, 12.0, n))
    prices = 100.0 * np.exp(drift + noise + wave)
    bars: list[Bar] = []
    for i, price in enumerate(prices):
        spread = price * 0.004
        vol = 1_000_000.0 * (1.0 + 0.3 * abs(np.sin(i / 9.0)))
        bars.append(Bar(price=float(price), high=float(price + spread),
                        low=float(price - spread), volume=float(vol)))
    return bars


def csv_bars(path: str) -> list[Bar]:
    """Load OHLCV bars from a CSV with close/high/low/volume columns."""
    bars: list[Bar] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lower = {k.lower().strip(): v for k, v in row.items()}
            price = float(lower.get("close") or lower.get("price"))
            bars.append(Bar(
                price=price,
                high=float(lower.get("high") or price),
                low=float(lower.get("low") or price),
                volume=float(lower.get("volume") or 1_000_000.0),
            ))
    return bars


def yfinance_bars(ticker: str, period: str = "2y", interval: str = "1d") -> list[Bar]:
    """Fetch real OHLCV bars via yfinance (optional dependency)."""
    import yfinance  # noqa: PLC0415 — optional, only when --ticker is used

    frame = yfinance.download(ticker, period=period, interval=interval, progress=False)
    if frame.empty:
        raise SystemExit(f"No data returned for ticker {ticker!r}")
    bars: list[Bar] = []
    for _, row in frame.iterrows():
        close = float(row["Close"])
        bars.append(Bar(
            price=close,
            high=float(row.get("High", close)),
            low=float(row.get("Low", close)),
            volume=float(row.get("Volume", 1_000_000.0)),
        ))
    return bars


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest the strategies offline.")
    parser.add_argument(
        "--strategy", choices=sorted([*STRATEGY_MODULES, "ensemble"]), default="conservative",
        help="conservative, radical, or ensemble (a 50/50 portfolio of both).",
    )
    parser.add_argument("--csv", help="Path to an OHLCV CSV (close/high/low/volume columns).")
    parser.add_argument("--ticker", help="Fetch real data via yfinance, e.g. 0700.HK.")
    parser.add_argument("--capital", type=float, default=100_000_000.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.csv:
        bars = csv_bars(args.csv)
        source = f"csv:{args.csv}"
    elif args.ticker:
        bars = yfinance_bars(args.ticker)
        source = f"yfinance:{args.ticker}"
    else:
        bars = synthetic_bars()
        source = "synthetic"

    subs: dict[str, BacktestResult] = {}
    if args.strategy == "ensemble":
        result, subs = run_ensemble(bars, initial_capital=args.capital)
    else:
        result = run_backtest(args.strategy, bars, initial_capital=args.capital)

    print(f"Strategy : {args.strategy}")
    print(f"Data     : {source} ({len(bars)} bars)")
    if subs:
        for name, res in subs.items():
            sharpe = res.metrics.get("sharpe", float("nan"))
            ret = res.metrics.get("total_return", float("nan"))
            print(f"  - {name:13s}: return {ret:+.4f}  sharpe {sharpe:+.4f}  trades {res.trades}")
    print(f"Trades   : {result.trades}")
    for key, value in result.metrics.items():
        print(f"{key:24s}: {value:.4f}")


if __name__ == "__main__":
    main()
