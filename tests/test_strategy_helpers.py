from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyOrder:
    pass


class DummyBacktest:
    class AlgoEvtHandler:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass


def _load_strategy(module_name: str):
    sys.modules["AlgoAPI"] = types.SimpleNamespace(
        AlgoAPIUtil=types.SimpleNamespace(OrderObject=DummyOrder),
        AlgoAPI_Backtest=DummyBacktest,
    )
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_conservative_indicators_and_signal_are_stable() -> None:
    module = _load_strategy("Conservative_strategy_clean")
    algo = module.AlgoEvent()

    algo.prices = [float(i) for i in range(100, 160)]
    algo.highs = [p + 1.0 for p in algo.prices]
    algo.lows = [p - 1.0 for p in algo.prices]
    algo.returns = [0.01] * 59

    assert algo.calc_atr(period=3) == 2.0
    signal, strength = algo.generate_signal()

    assert signal == 1
    assert 0.25 <= strength <= 1.0


def test_radical_action_reward_and_features_are_stable() -> None:
    module = _load_strategy("Radical_strategy_clean")
    np.random.seed(0)
    algo = module.AlgoEvent()

    assert algo.action_to_signal(0) == (-4, 0.55)
    assert algo.action_to_signal(4) == (0, 0.0)
    assert algo.action_to_signal(8) == (4, 0.55)

    algo.returns = [0.01] * 10
    assert algo.compute_reward(price=102.0, prev_price=100.0, action=8, position=1) > 0
    assert algo.compute_reward(price=102.0, prev_price=100.0, action=0, position=1) < 0

    algo.prices = [100.0 + i * 0.2 for i in range(60)]
    algo.highs = [p + 0.5 for p in algo.prices]
    algo.lows = [p - 0.5 for p in algo.prices]
    algo.volumes = [1_000_000 + i * 1000 for i in range(60)]
    algo.returns = [0.002] * 59

    features = algo.extract_features()

    assert features.shape == (algo.state_dim,)
    assert np.isfinite(features).all()
    assert algo.calc_atr(period=3) > 0
