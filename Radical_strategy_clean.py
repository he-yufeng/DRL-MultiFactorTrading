"""
Radical Trading Strategy with Deep Reinforcement Learning
激进交易策略 - 深度强化学习驱动

Architecture: Double DQN (4-layer: 24->128->64->32->9) + Transformer Attention + Prioritized ER
References: Mnih et al. (2015), Van Hasselt et al. (2016), Vaswani et al. (2017)
"""

from engine import Order
import numpy as np
import random

class AlgoEvent:
    """Radical Trading Strategy with Deep Reinforcement Learning"""
    
    def __init__(self):
        # Data Storage
        self.prices = []
        self.returns = []
        self.volumes = []
        self.highs = []
        self.lows = []
        self.bar_count = 0
        
        # Position Management
        self.position = 0
        self.entry_price = 0
        self.entry_bar = 0
        self.trailing_stop = 0
        self.max_profit = 0
        
        # Strategy Parameters
        self.initial_capital = 100000000
        self.base_position_pct = 0.40
        self.max_position_pct = 0.70
        self.min_gap = 3
        self.min_bars = 30
        self.lot_size = 100
        
        # DQN Parameters
        self.state_dim = 24
        self.action_dim = 9  # [-4,-3,-2,-1,0,1,2,3,4] signal strength
        self.learning_rate = 0.005
        self.gamma = 0.97
        self.epsilon = 0.25
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        
        # Neural Networks
        self.q_network = self._init_network()
        self.target_network = self._init_network()
        self.update_target_freq = 50
        self.update_count = 0
        
        # Prioritized Experience Replay
        self.replay_buffer = []
        self.priorities = []
        self.buffer_size = 2000
        self.batch_size = 64
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 0.0001
        
        # Transformer Attention Weights
        self.attention_q = np.random.randn(self.state_dim, self.state_dim) * 0.01
        self.attention_k = np.random.randn(self.state_dim, self.state_dim) * 0.01
        self.attention_v = np.random.randn(self.state_dim, self.state_dim) * 0.01
        
        # Risk Control
        self.peak_equity = self.initial_capital
        self.last_trade_time = None
        self.last_state = None
        self.last_action = None
        
    def start(self, mEvt):
        # Record the instrument to trade. The backtest engine attaches itself as
        # ``self.evt`` and drives ``on_marketdatafeed`` bar by bar.
        self.instrument = mEvt['subscribeList'][0]
    
    # ==================== Neural Network ====================
    
    def _init_network(self) -> dict:
        """Initialize 4-layer Deep Q-Network"""
        return {
            'w1': np.random.randn(self.state_dim, 128) * 0.01,
            'w2': np.random.randn(128, 64) * 0.01,
            'w3': np.random.randn(64, 32) * 0.01,
            'w4': np.random.randn(32, self.action_dim) * 0.01
        }
    
    def _forward(self, state: np.ndarray, network: dict) -> np.ndarray:
        """Forward pass through network"""
        h1 = np.tanh(np.dot(state, network['w1']))
        h2 = np.tanh(np.dot(h1, network['w2']))
        h3 = np.tanh(np.dot(h2, network['w3']))
        return np.dot(h3, network['w4'])
    
    def _apply_attention(self, features: np.ndarray) -> np.ndarray:
        """Transformer Self-Attention Mechanism"""
        Q = np.dot(features, self.attention_q)
        K = np.dot(features, self.attention_k)
        V = np.dot(features, self.attention_v)
        
        scores = np.dot(Q, K.T) / np.sqrt(self.state_dim)
        scores = np.tanh(scores)
        exp_scores = np.exp(scores - np.max(scores))
        attn_probs = exp_scores / (np.sum(exp_scores) + 1e-8)
        
        return features + np.dot(attn_probs, V) * 0.5
    
    # ==================== Feature Extraction ====================
    
    def extract_features(self) -> np.ndarray:
        """Extract 24-dimensional state vector with attention enhancement"""
        if len(self.prices) < 60:
            return np.zeros(self.state_dim)
        
        features = []
        p = np.array(self.prices[-60:])
        r = np.array(self.returns[-40:]) if len(self.returns) >= 40 else np.array([0])
        v = np.array(self.volumes[-30:]) if len(self.volumes) >= 30 else np.array([1])
        
        # [1-6] Multi-timeframe Momentum
        for period in [2, 3, 5, 8, 13, 21]:
            if len(p) >= period:
                features.append(np.tanh((p[-1] - p[-period]) / (p[-period] + 1e-8) * 20))
            else:
                features.append(0)
        
        # [7-10] Moving Average Position
        for period in [5, 10, 20, 40]:
            if len(p) >= period:
                ma = np.mean(p[-period:])
                features.append((p[-1] - ma) / (ma + 1e-8))
            else:
                features.append(0)
        
        # [11-14] Technical Indicators (Volatility, RSI, MACD, CCI)
        if len(r) >= 20:
            features.append(np.std(r) * 60)  # Volatility
            
            gains = [x for x in r if x > 0]
            losses = [-x for x in r if x < 0]
            avg_gain = np.mean(gains) if gains else 0.001
            avg_loss = np.mean(losses) if losses else 0.001
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            features.append((rsi - 50) / 50)
            
            if len(p) >= 26:
                macd = (np.mean(p[-12:]) - np.mean(p[-26:])) / (np.mean(p[-26:]) + 1e-8)
                features.append(macd * 30)
            else:
                features.append(0)
            
            if len(self.highs) >= 20:
                tp = (np.array(self.highs[-20:]) + np.array(self.lows[-20:]) + p[-20:]) / 3
                sma = np.mean(tp)
                mad = np.mean(np.abs(tp - sma))
                features.append(np.tanh((tp[-1] - sma) / (0.015 * mad + 1e-8) / 100))
            else:
                features.append(0)
        else:
            features.extend([0.02, 0, 0, 0])
        
        # [15-18] Volume Features
        if len(v) >= 15:
            vol_ratio = v[-1] / (np.mean(v[-15:]) + 1e-8)
            features.append(min(vol_ratio - 1, 3))
            features.append(min((np.mean(v[-5:]) - np.mean(v[-10:-5])) / (np.mean(v[-10:-5]) + 1e-8), 2))
            features.append((1 if (r[-1] > 0 and vol_ratio > 1) else -0.5) * min(vol_ratio, 1.5))
            features.append(min(np.std(v[-10:]) / (np.mean(v[-10:]) + 1e-8), 1))
        else:
            features.extend([0, 0, 0, 0])
        
        # [19-21] Breakout and Trend Strength
        if len(p) >= 30:
            features.append((p[-1] - max(p[-30:])) / (max(p[-30:]) + 1e-8))
            features.append((p[-1] - min(p[-30:])) / (min(p[-30:]) + 1e-8))
            features.append(abs(np.mean(p[-10:]) - np.mean(p[-20:])) / (np.mean(p[-20:]) + 1e-8) * 15)
        else:
            features.extend([0, 0, 0])
        
        # [22-24] Acceleration, Volatility Change, Position PnL
        if len(r) >= 5:
            features.append((r[-1] - r[-3]) * 150 if len(r) >= 3 else 0)
            features.append((np.std(r[-5:]) - np.std(r[-10:-5])) * 80 if len(r) >= 10 else 0)
        else:
            features.extend([0, 0])
        
        if self.position != 0:
            pnl = (p[-1] - self.entry_price) / (self.entry_price + 1e-8)
            if self.position < 0:
                pnl = -pnl
            features.append(np.tanh(pnl * 10))
        else:
            features.append(0)
        
        return self._apply_attention(np.array(features))
    
    # ==================== DQN Core ====================
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy Action Selection"""
        if training and random.random() < self.epsilon:
            weights = [0.10, 0.10, 0.08, 0.07, 0.30, 0.07, 0.08, 0.10, 0.10]
            return np.random.choice(self.action_dim, p=weights)
        q_values = self._forward(state, self.q_network).flatten()
        return np.argmax(q_values) if len(q_values) == self.action_dim else 4
    
    def action_to_signal(self, action: int) -> tuple:
        """Map action to trading signal and strength"""
        mapping = {
            0: (-4, 0.55), 1: (-3, 0.45), 2: (-2, 0.35), 3: (-1, 0.25),
            4: (0, 0.0),   5: (1, 0.25),  6: (2, 0.35),  7: (3, 0.45), 8: (4, 0.55)
        }
        return mapping.get(action, (0, 0.0))
    
    def compute_reward(self, price: float, prev_price: float, action: int, position: int) -> float:
        """Compute reward signal"""
        if action == 4:
            return 0
        
        pct_change = (price - prev_price) / (prev_price + 1e-8)
        reward = (-pct_change if action < 4 else pct_change) * 300
        
        # Position reward
        if position != 0:
            if (position > 0 and pct_change > 0) or (position < 0 and pct_change < 0):
                reward += abs(pct_change) * 150
        
        # Risk penalty
        if len(self.returns) >= 10:
            reward -= np.std(self.returns[-10:]) * 30
        
        return reward
    
    def train_network(self) -> None:
        """Train with Double DQN + Prioritized Experience Replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        priorities = np.power(np.array(self.priorities) + 1e-6, self.alpha)
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
        batch = [self.replay_buffer[i] for i in indices]
        
        weights = np.power(len(self.replay_buffer) * probs[indices], -self.beta)
        weights = weights / np.max(weights)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for idx, (state, action, reward, next_state, done) in enumerate(batch):
            current_q = self._forward(state, self.q_network).flatten()
            
            if done:
                target_q = reward
            else:
                next_q = self._forward(next_state, self.q_network).flatten()
                target_q_vals = self._forward(next_state, self.target_network).flatten()
                target_q = reward + self.gamma * target_q_vals[np.argmax(next_q)]
            
            td_error = abs(target_q - current_q[action])
            self.priorities[indices[idx]] = td_error
            
            if action < len(current_q):
                error = (target_q - current_q[action]) * weights[idx] * self.learning_rate
                h1 = np.tanh(np.dot(state, self.q_network['w1']))
                h2 = np.tanh(np.dot(h1, self.q_network['w2']))
                h3 = np.tanh(np.dot(h2, self.q_network['w3']))
                grad = np.outer(h3, np.zeros(self.action_dim))
                grad[:, action] = error
                self.q_network['w4'] += grad * 0.01
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.update_target_freq == 0:
            self.target_network = {k: v.copy() for k, v in self.q_network.items()}
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # ==================== Risk Management ====================
    
    def calc_atr(self, period: int = 14) -> float:
        """Calculate Average True Range as a percentage of current price."""
        if len(self.highs) < period + 1:
            return 0.025
        trs = []
        for i in range(len(self.highs) - period, len(self.highs)):
            tr = max(
                self.highs[i] - self.lows[i],
                abs(self.highs[i] - self.prices[i-1]) if i > 0 else 0,
                abs(self.lows[i] - self.prices[i-1]) if i > 0 else 0
            )
            trs.append(tr)
        return np.mean(trs) / (self.prices[-1] + 1e-8)
    
    def calc_position_size(self, price: float, available: float, signal_strength: float) -> int:
        """Dynamic Position Sizing with Risk Control"""
        target_pct = self.base_position_pct * (0.9 + signal_strength * 0.8)
        
        # Volatility adjustment
        if len(self.returns) >= 20:
            vol = np.std(self.returns[-20:]) * np.sqrt(252 * 24)
            if vol > 0.35:
                target_pct *= 0.35 / vol
        
        # Drawdown protection
        if available < self.peak_equity:
            drawdown = (self.peak_equity - available) / self.peak_equity
            if drawdown > 0.20:
                target_pct *= 0.4
            elif drawdown > 0.15:
                target_pct *= 0.6
        else:
            self.peak_equity = available
        
        target_pct = max(0.25, min(target_pct, self.max_position_pct))
        shares = max(1, int(available * target_pct / (price * self.lot_size))) * self.lot_size
        
        if shares * price * 1.25 <= available * 0.90:
            return shares
        return max(1, int(available * 0.7 / (price * self.lot_size))) * self.lot_size
    
    # ==================== Trading Logic ====================
    
    def on_marketdatafeed(self, md, ab):
        """Process incoming market data, manage positions, and train the DQN agent."""
        price = md.lastPrice
        high = getattr(md, 'high', price)
        low = getattr(md, 'low', price)
        volume = getattr(md, 'volume', 1000000)
        
        try:
            available = float(getattr(ab, 'availableBalance', self.initial_capital * 0.9))
            if available <= 0:
                available = self.initial_capital * 0.9
        except (AttributeError, TypeError, ValueError):
            available = self.initial_capital * 0.9
        available *= 0.90
        
        prev_price = self.prices[-1] if self.prices else price
        self.prices.append(price)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        self.bar_count += 1
        
        if len(self.prices) > 1:
            self.returns.append((price - self.prices[-2]) / (self.prices[-2] + 1e-8))
        
        for arr in [self.prices, self.highs, self.lows, self.returns, self.volumes]:
            if len(arr) > 500:
                arr[:] = arr[-500:]
        
        if len(self.prices) < self.min_bars:
            return
        
        state = self.extract_features()
        atr = self.calc_atr(14)
        
        # Position Management
        if self.position != 0:
            bars_held = self.bar_count - self.entry_bar
            pnl_pct = (price - self.entry_price) / (self.entry_price + 1e-8)
            if self.position < 0:
                pnl_pct = -pnl_pct
            self.max_profit = max(self.max_profit, pnl_pct)
            
            should_exit = False
            
            # Dynamic Trailing Stop
            if self.position > 0:
                new_stop = price * (1 - 1.8 * max(atr, 0.02))
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                if pnl_pct > 0.025:
                    self.trailing_stop = max(self.trailing_stop, self.entry_price * (1 + self.max_profit * 0.7))
                if pnl_pct > 5.0 * max(atr, 0.02) or price <= self.trailing_stop or bars_held >= 60:
                    should_exit = True
            else:
                new_stop = price * (1 + 1.8 * max(atr, 0.02))
                if new_stop < self.trailing_stop or self.trailing_stop == 0:
                    self.trailing_stop = new_stop
                if pnl_pct > 0.025:
                    self.trailing_stop = min(self.trailing_stop, self.entry_price * (1 - self.max_profit * 0.7))
                if pnl_pct > 5.0 * max(atr, 0.02) or price >= self.trailing_stop or bars_held >= 60:
                    should_exit = True
            
            if should_exit:
                self.close_position()
            
            # Store experience
            if self.last_state is not None and self.last_action is not None:
                reward = self.compute_reward(price, prev_price, self.last_action, self.position)
                if len(self.replay_buffer) >= self.buffer_size:
                    self.replay_buffer.pop(0)
                    self.priorities.pop(0)
                self.replay_buffer.append((self.last_state, self.last_action, reward, state, self.position == 0))
                self.priorities.append(abs(reward) + 0.1)
                
                if len(self.replay_buffer) >= self.batch_size:
                    self.train_network()
        
        # Entry Logic
        else:
            if self.last_trade_time and self.bar_count - self.last_trade_time < self.min_gap:
                return
            
            action = self.select_action(state, training=True)
            signal, strength = self.action_to_signal(action)
            
            if signal == 0 or strength < 0.30:
                return
            
            # Volatility Filter
            if len(self.returns) >= 20:
                vol = np.std(self.returns[-20:])
                if vol > 0.25 or vol < 0.003:
                    return
            
            # Volume Filter
            if len(self.volumes) >= 15:
                if self.volumes[-1] < np.mean(self.volumes[-15:]) * 0.45:
                    return
            
            size = self.calc_position_size(price, available, strength)
            if size > 0:
                self.open_position(signal, size, atr)
                self.last_action = action
                self.last_state = state
                self.last_trade_time = self.bar_count
    
    def open_position(self, direction: int, size: int, atr: float) -> None:
        """Open a new position with ATR-based trailing stop initialization."""
        if size == 0:
            return
        direction = 1 if direction > 0 else -1
        
        order = Order()
        order.instrument = self.instrument
        order.orderRef = f"R_{self.bar_count}"
        order.volume = int(size)
        order.openclose = 'open'
        order.buysell = direction
        order.ordertype = 0
        
        try:
            self.evt.sendOrder(order)
            self.position = size if direction > 0 else -size
            self.entry_price = self.prices[-1]
            self.entry_bar = self.bar_count
            self.max_profit = 0
            atr_adj = max(atr, 0.02)
            self.trailing_stop = self.prices[-1] * (1 - 1.8 * atr_adj if direction > 0 else 1 + 1.8 * atr_adj)
        except (AttributeError, TypeError, ValueError):
            pass
    
    def close_position(self) -> None:
        """Close the current position and reset tracking state."""
        if self.position == 0:
            return
        order = Order()
        order.instrument = self.instrument
        order.orderRef = f"RX_{self.bar_count}"
        order.volume = int(abs(self.position))
        order.openclose = 'open'
        order.buysell = -1 if self.position > 0 else 1
        order.ordertype = 0
        
        try:
            self.evt.sendOrder(order)
            self.position = 0
            self.entry_price = 0
            self.trailing_stop = 0
            self.max_profit = 0
        except (AttributeError, TypeError, ValueError):
            pass
    
    def on_bulkdatafeed(self, isSync, bd, ab): pass
    def on_orderfeed(self, of): pass
    def on_dailyPLfeed(self, pl): pass
    def on_openPositionfeed(self, op, oo, uo): pass

