<div align="center">

# DRL-MultiFactorTrading

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **深度强化学习量化交易策略：Double DQN + Transformer 注意力机制 + Fama-French 多因子模型，配合自适应风险管理和波动率目标控制。**

**[English](README.md) | [中文](README_CN.md)**

</div>

## 实盘效果

*在具有强动量特征的成长股上表现出色*

### 小米集团 (01810.HK)
![Radical Strategy - Xiaomi](radical-01810HK.png)

### 腾讯控股 (00700.HK) - 高收益高波动
![Radical Strategy - Tencent](radical-00700HK.png)

### 美团 (03690.HK) - 成长股表现
![Radical Strategy - Meituan](radical-03690HK.png)

## 策略评估工具

仓库现在包含一组纯 Python 绩效指标 helper，可在展示回测结果前统一计算核心指标：

```python
from strategy_metrics import drawdown_analysis, summarize_equity_curve

summary = summarize_equity_curve([100_000, 103_000, 98_500, 110_000])
drawdown = drawdown_analysis([100_000, 103_000, 98_500, 110_000])
print(summary["total_return"], summary["max_drawdown"], summary["sharpe"])
print(drawdown["drawdown_duration"], drawdown["recovery_duration"])
```

这样保守策略和激进策略都可以用同一套口径输出总收益、年化收益、最大回撤、回撤与恢复时长、水下阶段、Ulcer Index、年化波动率、下行波动率、Sharpe、Sortino 和 Calmar ratio，避免 README 图表和代码评估口径脱节。

### 与 buy-and-hold 基准对比

策略真正的价值在于能否跑赢被动持有。`benchmark_comparison` 接收策略和基准两条等长净值曲线，
输出超额年化收益、信息比率、跟踪误差、beta、CAPM 口径的 alpha，以及策略跑赢基准的周期占比：

```python
from strategy_metrics import benchmark_comparison, summarize_vs_benchmark

strategy = [100_000, 108_000, 104_000, 119_000]
buy_and_hold = [100_000, 103_000, 101_000, 106_000]

rel = benchmark_comparison(strategy, buy_and_hold)
print(rel["excess_annualized_return"], rel["information_ratio"], rel["beta"], rel["alpha"])

# 策略、基准、相对指标三段一起返回，方便并排出表。
report = summarize_vs_benchmark(strategy, buy_and_hold)
print(report["strategy"]["sharpe"], report["benchmark"]["sharpe"], report["relative"]["win_rate"])
```

基准没有波动时 beta、alpha 会退回安全取值，曲线过短或全平则返回全零而不会抛异常。

## 概览

本仓库包含两套算法交易策略：

| 策略 | 方法 | 风险等级 | 核心技术 |
|------|------|----------|----------|
| **保守型** | 多因子模型 | 中低 | 加权信号聚合 |
| **激进型** | 深度强化学习 | 中高 | Double DQN + Transformer |

## 架构

### 策略 1：保守型多因子模型

信号由五个独立因子的加权和计算得出：

| 因子 | 权重 | 信号来源 |
|------|------|----------|
| 趋势 | 35% | MA 交叉 (8/20/40 周期) |
| 动量 | 25% | 5/10 bar 收益率 |
| RSI | 20% | 超卖 (<35) / 超买 (>65) |
| MACD | 15% | 柱状图方向 |
| 布林带 | 5% | 突破信号 |

**核心特性：**
- **波动率目标**：基于 15% 年化波动率目标动态调整仓位
- **回撤保护**：回撤超 10% 自动降低敞口
- **ATR 止损**：止损 2x ATR，止盈 4x ATR
- **时间止损**：最长持仓 150 根 K 线

### 策略 2：激进型深度强化学习

**24 维状态向量：**

| 维度 | 特征描述 |
|------|----------|
| 1-6 | 多时间框架动量 (2,3,5,8,13,21 周期) |
| 7-10 | 均线位置 (5,10,20,40 周期) |
| 11-14 | 技术指标 (波动率, RSI, MACD, CCI) |
| 15-18 | 成交量特征 (比率, 趋势, 相关性, 波动) |
| 19-21 | 突破与趋势强度 |
| 22-24 | 加速度, 波动率变化, 持仓盈亏 |

**网络结构：** Input(24) → Dense(128) → Dense(64) → Dense(32) → 9 个动作

**核心特性：**
- **Double DQN**：独立 target network 减少 Q 值过估计
- **Transformer 注意力**：自注意力机制增强特征表示
- **优先经验回放**：按 TD-error 优先采样重要经验 (α=0.6, β=0.4→1.0)
- **ε-greedy 探索**：从 25% 衰减到 5%
- **动态追踪止损**：1.8x ATR，利润锁定 70%

### DQN 动作空间

| 动作 | 信号 | 强度 | 含义 |
|------|------|------|------|
| 0 | -4 | 0.55 | 强做空 |
| 1 | -3 | 0.45 | 中等做空 |
| 2 | -2 | 0.35 | 弱做空 |
| 3 | -1 | 0.25 | 极弱做空 |
| 4 | 0 | 0.00 | 持有 |
| 5 | +1 | 0.25 | 极弱做多 |
| 6 | +2 | 0.35 | 弱做多 |
| 7 | +3 | 0.45 | 中等做多 |
| 8 | +4 | 0.55 | 强做多 |

## 项目结构

```
DRL-MultiFactorTrading/
├── Conservative_strategy_clean.py  # 多因子策略（精简版）
├── Radical_strategy_clean.py       # DRL 策略（精简版）
├── README.md                       # English README
├── README_CN.md                    # 中文 README
└── *.png                           # 回测可视化
```

## 快速开始

### 安装依赖

```bash
pip install numpy
```

### 使用方法

两套策略都基于 AlgoAPI 回测框架：

```python
from Radical_strategy_clean import AlgoEvent

strategy = AlgoEvent()
mEvt = {'subscribeList': ['01810HK']}  # 港股小米
strategy.start(mEvt)
```

### 策略参数

#### 保守策略
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_position_pct` | 0.35 | 基础仓位（资金的 35%）|
| `max_position_pct` | 0.55 | 最大仓位 |
| `target_volatility` | 0.15 | 目标年化波动率（15%）|
| `stop_loss_atr` | 2.0 | ATR 倍数止损 |
| `take_profit_atr` | 4.0 | ATR 倍数止盈 |

#### 激进策略
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `base_position_pct` | 0.40 | 基础仓位（资金的 40%）|
| `max_position_pct` | 0.70 | 最大仓位 |
| `epsilon` | 0.25 | 初始探索率 |
| `gamma` | 0.97 | 折扣因子 |
| `learning_rate` | 0.005 | 学习率 |
| `buffer_size` | 2000 | 经验回放缓冲区 |
| `batch_size` | 64 | 批大小 |

## 风险管理

### 仓位调整
```python
# 波动率调整
if realized_volatility > target_volatility:
    position_size *= target_volatility / realized_volatility

# 回撤保护
if drawdown > 0.10:
    position_size *= (1 - drawdown * 0.6)
```

### 退出条件
1. **止损**：ATR 动态止损（保守 2.0x，激进 1.8x）
2. **止盈**：ATR 目标（保守 4.0x，激进 5.0x）
3. **追踪止损**：锁定最大利润的 50-70%
4. **时间止损**：最长持仓周期（保守 150 bar，激进 60 bar）

## 研究方法

- 保守策略 **600+ 次迭代**（参数优化、因子权重调优）
- 激进策略 **400+ 次实验**（网络结构搜索、超参调优）
- 合计 **1000+ 次回测**，跨多标的、多周期
- **4 年以上历史数据** (2020-2024)，覆盖多种市场行情

### 测试周期覆盖
- COVID-19 暴跌与修复 (2020)
- 牛市行情 (2021)
- 熊市压力测试 (2022)
- 反弹行情 (2023)
- 近期市场 (2024)

### 测试标的
港股：腾讯 (00700.HK)、小米 (01810.HK)、美团 (03690.HK)

## 参考文献

1. **Fama & French** (1993). Common risk factors in the returns on stocks and bonds. *JFE*.
2. **Mnih et al.** (2015). Human-level control through deep reinforcement learning. *Nature*.
3. **Van Hasselt et al.** (2016). Deep reinforcement learning with double Q-learning. *AAAI*.
4. **Vaswani et al.** (2017). Attention is all you need. *NeurIPS*.
5. **Schaul et al.** (2015). Prioritized experience replay. *arXiv:1511.05952*.

## 免责声明

**本软件仅用于教育和研究目的。**

- 过往表现不代表未来收益
- 交易涉及重大亏损风险
- 作者不对任何财务损失负责
- 实盘前请务必充分回测
- 请咨询专业的财务顾问

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 贡献

欢迎提交 Pull Request！

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request
