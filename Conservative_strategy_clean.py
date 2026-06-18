"""
Conservative Trading Strategy with Multi-Factor Model
保守交易策略 - 多因子模型驱动

Architecture: Multi-Factor Model (Trend 35%, Momentum 25%, RSI 20%, MACD 15%, Bollinger 5%)
References: Fama & French (1993), Kim et al. (2016), Ang & Timmermann (2012)
"""

from engine import Order
import numpy as np

class AlgoEvent:
    """Conservative Trading Strategy with Multi-Factor Model"""
    
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
        self.max_profit_pct = 0
        
        # Strategy Parameters
        self.initial_capital = 100000000
        self.base_position_pct = 0.35
        self.max_position_pct = 0.55
        self.min_gap = 8
        self.min_bars = 50
        self.lot_size = 100
        
        # Risk Control
        self.target_volatility = 0.15
        self.atr_period = 14
        self.stop_loss_atr = 2.0
        self.take_profit_atr = 4.0
        self.peak_equity = self.initial_capital
        
        # Performance Tracking
        self.recent_pnls = []
        self.last_trade_time = None
        
    def start(self, mEvt):
        # Record the instrument to trade. The backtest engine attaches itself as
        # ``self.evt`` and drives ``on_marketdatafeed`` bar by bar.
        self.instrument = mEvt['subscribeList'][0]
    
    # ==================== Technical Indicators ====================
    
    def calc_atr(self, period: int = 14) -> float:
        """Calculate Average True Range"""
        if len(self.highs) < period + 1:
            return 0.02
        trs = []
        for i in range(len(self.highs) - period, len(self.highs)):
            tr = max(
                self.highs[i] - self.lows[i],
                abs(self.highs[i] - self.prices[i-1]) if i > 0 else 0,
                abs(self.lows[i] - self.prices[i-1]) if i > 0 else 0
            )
            trs.append(tr)
        return np.mean(trs)
    
    def calc_rsi(self, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(self.returns) < period:
            return 50.0
        gains = [r for r in self.returns[-period:] if r > 0]
        losses = [-r for r in self.returns[-period:] if r < 0]
        avg_gain = np.mean(gains) if gains else 0.001
        avg_loss = np.mean(losses) if losses else 0.001
        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def calc_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD"""
        if len(self.prices) < slow:
            return 0.0, 0.0, 0.0
        p = np.array(self.prices[-slow:])
        ema_fast = np.mean(p[-fast:])
        ema_slow = np.mean(p)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line * 0.8
        return macd_line, signal_line, macd_line - signal_line
    
    def calc_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        if len(self.prices) < period:
            return self.prices[-1], self.prices[-1], self.prices[-1]
        p = np.array(self.prices[-period:])
        ma = np.mean(p)
        std = np.std(p)
        return ma + std_dev * std, ma, ma - std_dev * std
    
    # ==================== Signal Generation ====================
    
    def generate_signal(self) -> tuple:
        """
        Multi-Factor Signal Generation
        Weights: Trend(35%) + Momentum(25%) + RSI(20%) + MACD(15%) + BB(5%)
        """
        if len(self.prices) < self.min_bars:
            return 0, 0.0
        
        price = self.prices[-1]
        p = np.array(self.prices[-50:])
        signals, strengths = [], []
        
        # 1. Trend Signal (35%)
        ma_fast = np.mean(p[-8:])
        ma_medium = np.mean(p[-20:])
        ma_slow = np.mean(p[-40:])
        if ma_fast > ma_medium > ma_slow:
            signals.append(1)
            strengths.append(min((ma_fast - ma_slow) / (ma_slow + 1e-8) * 15, 1.0) * 0.35)
        elif ma_fast < ma_medium < ma_slow:
            signals.append(-1)
            strengths.append(min((ma_slow - ma_fast) / (ma_fast + 1e-8) * 15, 1.0) * 0.35)
        
        # 2. Momentum Signal (25%)
        mom_5 = (p[-1] - p[-5]) / (p[-5] + 1e-8)
        mom_10 = (p[-1] - p[-10]) / (p[-10] + 1e-8)
        momentum = mom_5 * 0.6 + mom_10 * 0.4
        if abs(momentum) > 0.008:
            signals.append(1 if momentum > 0 else -1)
            strengths.append(min(abs(momentum) * 50, 1.0) * 0.25)
        
        # 3. RSI Signal (20%)
        rsi = self.calc_rsi(14)
        if rsi < 35:
            signals.append(1)
            strengths.append((35 - rsi) / 35 * 0.20)
        elif rsi > 65:
            signals.append(-1)
            strengths.append((rsi - 65) / 35 * 0.20)
        
        # 4. MACD Signal (15%)
        macd, signal_line, histogram = self.calc_macd()
        if histogram != 0:
            signals.append(1 if histogram > 0 else -1)
            strengths.append(min(abs(histogram) * 100, 1.0) * 0.15)
        
        # 5. Bollinger Bands Signal (5%)
        upper, middle, lower = self.calc_bollinger_bands()
        if price < lower:
            signals.append(1)
            strengths.append(min((lower - price) / (lower + 1e-8), 1.0) * 0.05)
        elif price > upper:
            signals.append(-1)
            strengths.append(min((price - upper) / (upper + 1e-8), 1.0) * 0.05)
        
        if not signals:
            return 0, 0.0
        
        # Weighted Aggregation
        signal_sum = sum(s * st for s, st in zip(signals, strengths))
        strength_sum = sum(strengths)
        
        if strength_sum < 0.25:
            return 0, 0.0
        
        return (1 if signal_sum > 0.15 else (-1 if signal_sum < -0.15 else 0)), min(strength_sum, 1.0)
    
    # ==================== Position Sizing ====================
    
    def calc_position_size(self, price: float, available: float, signal_strength: float) -> int:
        """Volatility-Adjusted Position Sizing"""
        target_pct = self.base_position_pct * (0.85 + signal_strength * 0.5)
        
        # Volatility adjustment
        if len(self.returns) >= 20:
            vol = np.std(self.returns[-20:]) * np.sqrt(252 * 24)
            if vol > 0:
                target_pct *= min(self.target_volatility / vol, 1.2)
        
        # Drawdown protection
        if available < self.peak_equity:
            drawdown = (self.peak_equity - available) / self.peak_equity
            if drawdown > 0.10:
                target_pct *= (1 - drawdown * 0.6)
        else:
            self.peak_equity = available
        
        target_pct = max(0.20, min(target_pct, self.max_position_pct))
        shares = max(1, int(available * target_pct / (price * self.lot_size))) * self.lot_size
        
        if shares * price * 1.2 <= available * 0.85:
            return shares
        return max(1, int(available * 0.8 / (price * self.lot_size))) * self.lot_size
    
    # ==================== Trading Logic ====================
    
    def on_marketdatafeed(self, md, ab):
        """Process incoming market data and execute multi-factor trading logic."""
        price = md.lastPrice
        high = getattr(md, 'high', price)
        low = getattr(md, 'low', price)
        volume = getattr(md, 'volume', 1000000)
        
        # Get available capital
        try:
            available = float(getattr(ab, 'availableBalance', self.initial_capital * 0.9))
            if available <= 0:
                available = self.initial_capital * 0.9
        except (AttributeError, TypeError, ValueError):
            available = self.initial_capital * 0.9
        available *= 0.90
        
        # Store data
        self.prices.append(price)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        self.bar_count += 1
        
        if len(self.prices) > 1:
            self.returns.append((price - self.prices[-2]) / (self.prices[-2] + 1e-8))
        
        # Limit data length
        for arr in [self.prices, self.highs, self.lows, self.returns, self.volumes]:
            if len(arr) > 500:
                arr[:] = arr[-500:]
        
        if len(self.prices) < self.min_bars:
            return
        
        atr = self.calc_atr()
        atr_pct = atr / (price + 1e-8)
        
        # Position Management
        if self.position != 0:
            pnl_pct = (price - self.entry_price) / (self.entry_price + 1e-8)
            if self.position < 0:
                pnl_pct = -pnl_pct
            self.max_profit_pct = max(self.max_profit_pct, pnl_pct)
            
            bars_held = self.bar_count - self.entry_bar
            should_exit = False
            
            # Stop Loss / Take Profit
            if self.position > 0:
                if price <= self.entry_price * (1 - self.stop_loss_atr * atr_pct):
                    should_exit = True
                if price >= self.entry_price * (1 + self.take_profit_atr * atr_pct):
                    should_exit = True
            else:
                if price >= self.entry_price * (1 + self.stop_loss_atr * atr_pct):
                    should_exit = True
                if price <= self.entry_price * (1 - self.take_profit_atr * atr_pct):
                    should_exit = True
            
            # Trailing Stop
            if pnl_pct > 0.02:
                trail = self.max_profit_pct * 0.5
                if (self.position > 0 and price < self.entry_price * (1 + trail)) or \
                   (self.position < 0 and price > self.entry_price * (1 - trail)):
                    should_exit = True
            
            # Time Stop
            if bars_held >= 150:
                should_exit = True
            
            if should_exit:
                self.close_position()
        
        # Entry Logic
        else:
            if self.last_trade_time and self.bar_count - self.last_trade_time < self.min_gap:
                return
            
            signal_dir, signal_strength = self.generate_signal()
            if signal_dir == 0 or signal_strength < 0.22:
                return
            
            # Volatility Filter
            if len(self.returns) >= 20:
                vol = np.std(self.returns[-20:])
                if vol > 0.20 or vol < 0.002:
                    return
            
            # Volume Filter
            if len(self.volumes) >= 20:
                if self.volumes[-1] < np.mean(self.volumes[-20:]) * 0.6:
                    return
            
            size = self.calc_position_size(price, available, signal_strength)
            if size > 0:
                self.open_position(signal_dir, size)
                self.last_trade_time = self.bar_count
    
    def open_position(self, direction: int, size: int) -> None:
        order = Order()
        order.instrument = self.instrument
        order.orderRef = f"C_{self.bar_count}"
        order.volume = int(size)
        order.openclose = 'open'
        order.buysell = 1 if direction > 0 else -1
        order.ordertype = 0
        
        try:
            self.evt.sendOrder(order)
            self.position = size if direction > 0 else -size
            self.entry_price = self.prices[-1]
            self.entry_bar = self.bar_count
            self.max_profit_pct = 0
        except (AttributeError, TypeError, ValueError):
            pass
    
    def close_position(self) -> None:
        if self.position == 0:
            return
        order = Order()
        order.instrument = self.instrument
        order.orderRef = f"CX_{self.bar_count}"
        order.volume = int(abs(self.position))
        order.openclose = 'open'
        order.buysell = -1 if self.position > 0 else 1
        order.ordertype = 0
        
        try:
            self.evt.sendOrder(order)
            self.position = 0
            self.entry_price = 0
            self.max_profit_pct = 0
        except (AttributeError, TypeError, ValueError):
            pass
    
    def on_bulkdatafeed(self, isSync, bd, ab): pass
    def on_orderfeed(self, of): pass
    def on_dailyPLfeed(self, pl): pass
    def on_openPositionfeed(self, op, oo, uo): pass

