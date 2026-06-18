"""A small, self-contained backtest engine for the trading strategies.

The strategies are event-driven: each implements ``on_marketdatafeed(md, ab)``
and emits :class:`Order` objects through ``self.evt.sendOrder(...)``.
:class:`BacktestEngine` plays the role of both the broker and the event loop —
it feeds OHLCV bars to a strategy and fills the orders it sends, tracking an
equity curve. No third-party trading platform or account is required, so the
strategies can be run and evaluated anywhere with just ``pip install numpy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Iterable

from strategy_metrics import summarize_equity_curve


class Order:
    """A market order — the strategy fills these fields and calls ``sendOrder``."""

    instrument: str = ""
    orderRef: str = ""
    volume: int = 0
    openclose: str = "open"
    buysell: int = 0
    ordertype: int = 0


@dataclass
class Bar:
    """A single OHLCV bar fed to the strategy."""

    price: float
    high: float
    low: float
    volume: float


@dataclass
class BacktestResult:
    equity: list[float] = field(default_factory=list)
    trades: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


class BacktestEngine:
    """Drive a strategy over OHLCV bars with a local simulated broker.

    Attaching itself as ``strategy.evt`` means the strategy's
    ``self.evt.sendOrder(order)`` calls land here and get filled at the current
    bar price. The strategy keeps its own ``self.position`` state; the engine
    independently tracks cash, holdings, and the resulting equity curve.
    """

    def __init__(self, initial_capital: float = 100_000_000.0) -> None:
        self.initial_capital = initial_capital
        self._pending: list[Order] = []

    def run(self, strategy, bars: Iterable[Bar]) -> BacktestResult:
        strategy.evt = self
        if not getattr(strategy, "instrument", None):
            strategy.instrument = "BACKTEST"

        cash = self.initial_capital
        holding = 0
        result = BacktestResult()

        for bar in bars:
            md = SimpleNamespace(
                lastPrice=bar.price, high=bar.high, low=bar.low, volume=bar.volume
            )
            ab = SimpleNamespace(availableBalance=cash)
            self._pending = []
            strategy.on_marketdatafeed(md, ab)
            for order in self._pending:
                signed = int(order.volume) * (1 if order.buysell > 0 else -1)
                cash -= signed * bar.price  # buy spends cash, sell returns it
                holding += signed
                result.trades += 1
            result.equity.append(cash + holding * bar.price)

        if len(result.equity) > 2:
            result.metrics = summarize_equity_curve(result.equity)
        return result

    # ---- broker hooks the strategy calls through ``self.evt`` ----
    def sendOrder(self, order: Order) -> None:
        self._pending.append(order)

    def start(self) -> None:  # pragma: no cover - compatibility no-op
        pass
