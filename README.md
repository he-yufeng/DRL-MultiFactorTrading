<div align="center">

# DRL-MultiFactorTrading

[![CI](https://github.com/universeplayer/DRL-MultiFactorTrading/actions/workflows/ci.yml/badge.svg)](https://github.com/universeplayer/DRL-MultiFactorTrading/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Deep Reinforcement Learning trading strategies combining Double DQN with Transformer Attention and Multi-Factor Models inspired by Fama-French. Features adaptive risk management and volatility targeting.**

**[English](README.md) | [中文](README_CN.md)**

</div>

## 📊 Performance Visualizations
*Excellent adaptability on growth stocks with strong momentum characteristics*

### Xiaomi Corporation (01810.HK) - DRL Learning in Action
![Radical Strategy - Xiaomi](radical-01810HK.png)

### Tencent Holdings (00700.HK) - High Returns, Higher Volatility
![Radical Strategy - Tencent](radical-00700HK.png)

### Meituan (03690.HK) - Growth Stock Performance
![Radical Strategy - Meituan](radical-03690HK.png)

## Evaluation Helpers

The repository includes pure Python helpers for reviewing an equity curve before showing results:

```python
from strategy_metrics import drawdown_analysis, summarize_equity_curve

summary = summarize_equity_curve([100_000, 103_000, 98_500, 110_000])
drawdown = drawdown_analysis([100_000, 103_000, 98_500, 110_000])
print(summary["total_return"], summary["max_drawdown"], summary["sharpe"])
print(drawdown["drawdown_duration"], drawdown["recovery_duration"])
```

This keeps headline metrics such as total return, annualized return, max drawdown, drawdown and recovery duration, underwater episodes, Ulcer Index, annualized volatility, downside deviation, Sharpe, Sortino, and Calmar ratio consistent across the conservative and radical strategies.

### Benchmarking against buy-and-hold

A trading strategy is only worth running if it beats a passive hold. `benchmark_comparison`
takes the strategy and benchmark equity curves (same length) and reports excess annualized
return, information ratio, tracking error, beta, CAPM-style alpha, and the fraction of periods
the strategy beats the benchmark:

```python
from strategy_metrics import benchmark_comparison, summarize_vs_benchmark

strategy = [100_000, 108_000, 104_000, 119_000]
buy_and_hold = [100_000, 103_000, 101_000, 106_000]

rel = benchmark_comparison(strategy, buy_and_hold)
print(rel["excess_annualized_return"], rel["information_ratio"], rel["beta"], rel["alpha"])

# Strategy, benchmark, and relative metrics in one bundle for a side-by-side table.
report = summarize_vs_benchmark(strategy, buy_and_hold)
print(report["strategy"]["sharpe"], report["benchmark"]["sharpe"], report["relative"]["win_rate"])
```

Beta and alpha fall back to safe values when the benchmark has no variance, and a flat or
single-point curve returns zeros instead of raising.

### Measuring factor predictive power

Equity-curve metrics only tell you how a strategy did after the fact. Before a factor is
worth trading it has to predict forward returns, so `factor_analysis` reports the
information coefficient (IC), its stability over time (ICIR), and the spread between
factor quantiles:

```python
from factor_analysis import information_coefficient, summarize_factor, factor_quantile_returns

# One rebalance date: factor exposures across the cross-section vs. next-period returns.
rank_ic = information_coefficient(factor_today, forward_returns, method="spearman")

# A panel shaped (periods, assets): one row per rebalance date.
report = summarize_factor(factor_panel, forward_return_panel, method="spearman")
print(report["mean_ic"], report["ic_ir"], report["hit_rate"], report["t_stat"])

# Sanity-check monotonicity: mean forward return from the lowest to highest factor bucket.
buckets = factor_quantile_returns(factor_today, forward_returns, quantiles=5)
print(buckets[-1] - buckets[0])  # top-minus-bottom spread
```

Rank IC (Spearman) is the default because it is robust to outliers and to monotonic but
nonlinear relationships. Non-finite pairs are dropped, thin cross-sections are skipped
rather than reported as zero IC, and degenerate inputs return zeros instead of raising.

## 📋 Overview

This repository contains two sophisticated algorithmic trading strategies designed for quantitative trading:

| Strategy | Approach | Risk Profile | Key Technology |
|----------|----------|--------------|----------------|
| **Conservative** | Multi-Factor Model | Low-Medium | Weighted Signal Aggregation |
| **Radical** | Deep Reinforcement Learning | Medium-High | Double DQN + Transformer |

## 🏗️ Architecture

### Strategy 1: Conservative Multi-Factor Model

```
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL GENERATION                        │
├─────────────────────────────────────────────────────────────┤
│  Trend Analysis ──────────────────────────────── 35%        │
│  Momentum Indicators ─────────────────────────── 25%        │
│  RSI (Relative Strength Index) ───────────────── 20%        │
│  MACD (Moving Average Convergence Divergence) ── 15%        │
│  Bollinger Bands ─────────────────────────────── 5%         │
├─────────────────────────────────────────────────────────────┤
│                 WEIGHTED AGGREGATION                        │
│                        ↓                                    │
│              FINAL TRADING SIGNAL                           │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Volatility Targeting**: Dynamically adjusts position size based on 15% annualized volatility target
- **Drawdown Protection**: Reduces exposure when drawdown exceeds 10%
- **ATR-based Stops**: Stop-loss at 2x ATR, take-profit at 4x ATR
- **Time-based Exit**: Maximum holding period of 150 bars

### Strategy 2: Radical Deep Reinforcement Learning

```
┌─────────────────────────────────────────────────────────────┐
│              24-DIMENSIONAL STATE VECTOR                    │
├─────────────────────────────────────────────────────────────┤
│  [1-6]   Multi-timeframe Momentum (2,3,5,8,13,21 periods)   │
│  [7-10]  Moving Average Position (5,10,20,40 periods)       │
│  [11-14] Technical Indicators (Vol, RSI, MACD, CCI)         │
│  [15-18] Volume Features (ratio, trend, correlation, vol)   │
│  [19-21] Breakout & Trend Strength                          │
│  [22-24] Acceleration, Volatility Change, Position PnL      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              TRANSFORMER SELF-ATTENTION                     │
│                                                             │
│         Q = X·Wq    K = X·Wk    V = X·Wv                    │
│                                                             │
│         Attention(Q,K,V) = softmax(QK^T/√d)·V               │
│                                                             │
│         Output = X + 0.5 × Attention(Q,K,V)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 DOUBLE DQN NETWORK                          │
│                                                             │
│    Input(24) → Dense(128) → Dense(64) → Dense(32) → (9)     │
│                    ↓           ↓           ↓                │
│                  tanh        tanh        tanh               │
│                                                             │
│    Actions: [-4, -3, -2, -1, 0, +1, +2, +3, +4]             │
│             (Short)     (Hold)      (Long)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│           PRIORITIZED EXPERIENCE REPLAY                     │
│                                                             │
│    Priority = |TD-error|^α        (α = 0.6)                 │
│    Sampling = Priority / Σ(Priority)                        │
│    IS Weight = (N × P(i))^(-β)    (β → 1.0)                 │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Double DQN**: Reduces Q-value overestimation using separate target network
- **Transformer Attention**: Enhances feature representation with self-attention mechanism
- **Prioritized Replay**: Samples important experiences more frequently (α=0.6, β=0.4→1.0)
- **ε-greedy Exploration**: Starts at 25%, decays to 5% minimum
- **Dynamic Trailing Stop**: 1.8x ATR with profit lock-in at 70%

## 📁 Project Structure

```
DRL-MultiFactorTrading/
├── Conservative_strategy_clean.py  # Multi-Factor strategy (streamlined)
├── Radical_strategy_clean.py       # DRL strategy (streamlined)
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
├── .flake8                          # Linting configuration
├── .github/
│   └── workflows/
│       └── ci.yml                   # CI pipeline (Python 3.9-3.12)
│
├── radical-01810HK.png             # Performance: Xiaomi (01810.HK)
├── radical-00700HK.png             # Performance: Tencent (00700.HK)
└── radical-03690HK.png             # Performance: Meituan (03690.HK)
```

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run a backtest locally (no AlgoGene account)

The strategies were written for AlgoGene's `AlgoEvent` API, but you don't need an
AlgoGene account to run them. `backtest.py` drives the **unchanged** strategy code
with a local simulated broker — it feeds OHLCV bars into `on_marketdatafeed` and
fills the orders the strategy sends, so the decision logic is identical to the
platform version while the data feed and order execution are replaced:

```bash
python backtest.py                       # Conservative on synthetic data
python backtest.py --strategy radical    # Radical (numpy Double-DQN)
python backtest.py --csv prices.csv      # your own OHLCV CSV (close/high/low/volume)
python backtest.py --ticker 0700.HK      # real data via yfinance (pip install yfinance)
```

It prints the trade count plus a full metrics bundle (return, Sharpe, Sortino,
max drawdown, Calmar, ...) computed by `strategy_metrics`. You can also call it
programmatically:

```python
import backtest

bars = backtest.synthetic_bars(n=400)            # or backtest.csv_bars("prices.csv")
result = backtest.run_backtest("conservative", bars)
print(result.trades, result.metrics["sharpe"])
```

### Usage on AlgoGene

Both strategies also run unchanged on the AlgoGene `AlgoEvent` framework:

```python
from Radical_strategy_clean import AlgoEvent

# Initialize strategy
strategy = AlgoEvent()

# Configure with market event
mEvt = {
    'subscribeList': ['01810HK']  # Hong Kong Xiaomi stock
}
strategy.start(mEvt)
```

### Strategy Parameters

#### Conservative Strategy
| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_position_pct` | 0.35 | Base position size (35% of capital) |
| `max_position_pct` | 0.55 | Maximum position size |
| `target_volatility` | 0.15 | Target annualized volatility (15%) |
| `stop_loss_atr` | 2.0 | Stop-loss in ATR multiples |
| `take_profit_atr` | 4.0 | Take-profit in ATR multiples |
| `min_gap` | 8 | Minimum bars between trades |

#### Radical Strategy
| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_position_pct` | 0.40 | Base position size (40% of capital) |
| `max_position_pct` | 0.70 | Maximum position size |
| `epsilon` | 0.25 | Initial exploration rate |
| `epsilon_min` | 0.05 | Minimum exploration rate |
| `gamma` | 0.97 | Discount factor |
| `learning_rate` | 0.005 | Network learning rate |
| `buffer_size` | 2000 | Replay buffer capacity |
| `batch_size` | 64 | Training batch size |

## 📊 Signal Generation

### Multi-Factor Model (Conservative)

The signal is computed as a weighted sum of five independent factors:

```
Final_Signal = Σ(Factor_i × Weight_i × Strength_i)

where:
  - Trend:     Weight = 0.35, based on MA crossovers (8/20/40)
  - Momentum:  Weight = 0.25, based on 5/10-bar returns
  - RSI:       Weight = 0.20, oversold (<35) / overbought (>65)
  - MACD:      Weight = 0.15, histogram direction
  - Bollinger: Weight = 0.05, band breakouts
```

### DQN Action Space (Radical)

| Action | Signal | Strength | Interpretation |
|--------|--------|----------|----------------|
| 0 | -4 | 0.55 | Strong Short |
| 1 | -3 | 0.45 | Medium Short |
| 2 | -2 | 0.35 | Weak Short |
| 3 | -1 | 0.25 | Very Weak Short |
| 4 | 0 | 0.00 | Hold |
| 5 | +1 | 0.25 | Very Weak Long |
| 6 | +2 | 0.35 | Weak Long |
| 7 | +3 | 0.45 | Medium Long |
| 8 | +4 | 0.55 | Strong Long |

## 🛡️ Risk Management

Both strategies implement comprehensive risk controls:

### Position Sizing
```python
# Volatility-adjusted position sizing
if realized_volatility > target_volatility:
    position_size *= target_volatility / realized_volatility

# Drawdown protection
if drawdown > 0.10:
    position_size *= (1 - drawdown * 0.6)
```

### Exit Conditions
1. **Stop-Loss**: ATR-based dynamic stop (2.0x for Conservative, 1.8x for Radical)
2. **Take-Profit**: ATR-based target (4.0x for Conservative, 5.0x for Radical)
3. **Trailing Stop**: Locks in 50-70% of maximum profit
4. **Time Stop**: Maximum holding period (150 bars Conservative, 60 bars Radical)

## 🔬 Research Methodology

### Development Process

- **600+ iterations** on Conservative Strategy (parameter optimization, factor weight tuning)
- **400+ experiments** on Radical Strategy (network architecture search, hyperparameter tuning)
- **1000+ total backtests** across multiple assets and timeframes
- **4+ years** of historical data (2020-2024) covering multiple market regimes

### Testing Period Coverage

- ✅ COVID-19 crash and recovery (2020)
- ✅ Bull market conditions (2021)
- ✅ Bear market stress test (2022)
- ✅ Recovery rally (2023)
- ✅ Recent market conditions (2024)

### Instruments Tested

- **Hong Kong Equities**: Tencent (00700.HK), Xiaomi (01810.HK), Meituan (03690.HK)

## 📚 References

### Academic Papers

1. **Fama, E. F., & French, K. R.** (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3-56.

2. **Mnih, V., et al.** (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

3. **Van Hasselt, H., Guez, A., & Silver, D.** (2016). Deep reinforcement learning with double Q-learning. *AAAI Conference on Artificial Intelligence*.

4. **Vaswani, A., et al.** (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

5. **Schaul, T., et al.** (2015). Prioritized experience replay. *arXiv preprint arXiv:1511.05952*.

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- The authors are not responsible for any financial losses
- Always conduct thorough backtesting before live trading
- Consult with a qualified financial advisor

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<p align="center">
  Made with ❤️ for Quantitative Trading Research
</p>
