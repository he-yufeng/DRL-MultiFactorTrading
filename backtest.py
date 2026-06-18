"""Self-contained, pip-installable backtest for the AlgoGene strategies.

The ``Conservative_strategy_clean`` and ``Radical_strategy_clean`` modules are
written against AlgoGene's ``AlgoEvent`` API, so they can only run on that
platform. This harness drives the *unchanged* strategy code with a local
simulated broker: it feeds historical OHLCV bars into ``on_marketdatafeed`` and
fills the orders the strategy sends, tracking an equity curve. The decision
logic is identical to the platform version — only the data feed and order
execution are replaced — so anyone can ``pip install -r requirements.txt`` and
actually run and evaluate the strategies offline.

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
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategy_metrics import summarize_equity_curve  # noqa: E402

STRATEGY_MODULES = {
    "conservative": "Conservative_strategy_clean",
    "radical": "Radical_strategy_clean",
}


class _Order:
    """Stand-in for ``AlgoAPIUtil.OrderObject`` (a plain attribute bag)."""

    instrument: str = ""
    orderRef: str = ""
    volume: int = 0
    openclose: str = "open"
    buysell: int = 0
    ordertype: int = 0


class _Broker:
    """Captures the orders a strategy sends so the harness can fill them."""

    def __init__(self) -> None:
        self.pending: list[_Order] = []

    def sendOrder(self, order: _Order) -> None:
        self.pending.append(order)

    def start(self) -> None:  # the strategy may call evt.start()
        pass


def _install_algoapi_stub(broker: _Broker) -> None:
    """Inject a fake ``AlgoAPI`` package so the strategy modules import."""
    sys.modules["AlgoAPI"] = types.SimpleNamespace(
        AlgoAPIUtil=types.SimpleNamespace(OrderObject=_Order),
        AlgoAPI_Backtest=types.SimpleNamespace(AlgoEvtHandler=lambda *a, **k: broker),
    )


def load_strategy(module_name: str, broker: _Broker):
    """Import a strategy module against the stubbed AlgoAPI and return it."""
    _install_algoapi_stub(broker)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@dataclass
class Bar:
    price: float
    high: float
    low: float
    volume: float


@dataclass
class BacktestResult:
    equity: list[float] = field(default_factory=list)
    trades: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


def run_backtest(
    strategy: str,
    bars: list[Bar],
    *,
    initial_capital: float = 100_000_000.0,
) -> BacktestResult:
    """Run ``strategy`` ('conservative'|'radical') over ``bars`` with a local broker."""
    module_name = STRATEGY_MODULES[strategy]
    broker = _Broker()
    module = load_strategy(module_name, broker)

    algo = module.AlgoEvent()
    algo.instrument = "BACKTEST"
    algo.evt = broker

    cash = initial_capital
    holding = 0  # signed share count
    result = BacktestResult()

    for bar in bars:
        md = types.SimpleNamespace(
            lastPrice=bar.price, high=bar.high, low=bar.low, volume=bar.volume
        )
        ab = types.SimpleNamespace(availableBalance=cash)
        broker.pending.clear()
        algo.on_marketdatafeed(md, ab)
        for order in broker.pending:
            signed = int(order.volume) * (1 if order.buysell > 0 else -1)
            cash -= signed * bar.price  # buy spends cash, sell returns it
            holding += signed
            result.trades += 1
        result.equity.append(cash + holding * bar.price)

    if len(result.equity) > 2:
        result.metrics = summarize_equity_curve(result.equity)
    return result


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
    parser = argparse.ArgumentParser(description="Backtest the AlgoGene strategies offline.")
    parser.add_argument("--strategy", choices=sorted(STRATEGY_MODULES), default="conservative")
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

    result = run_backtest(args.strategy, bars, initial_capital=args.capital)
    print(f"Strategy : {args.strategy}")
    print(f"Data     : {source} ({len(bars)} bars)")
    print(f"Trades   : {result.trades}")
    if result.metrics:
        for key, value in result.metrics.items():
            print(f"{key:24s}: {value:.4f}")


if __name__ == "__main__":
    main()
