"""
Risk Manager
Handles position sizing, stop-loss, take-profit, and risk controls
"""

import numpy as np
from typing import Dict, Tuple, Optional
from config import backtest_config as cfg


class RiskManager:
    """
    Manages all risk-related calculations:
    - Position sizing
    - Stop-loss and take-profit levels
    - Drawdown monitoring
    - Daily loss limits
    """
    
    def __init__(self, initial_capital: float = cfg.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.max_drawdown_hit = False
        self.max_daily_loss_hit = False
        
    def calculate_position_size(
        self,
        current_price: float,
        signal_confidence: float = 0.8,
        volatility: float = 0.02
    ) -> Tuple[float, float]:
        """
        Calculate position size based on risk parameters
        
        Args:
            current_price: Current asset price
            signal_confidence: Confidence in signal (0-1)
            volatility: Estimated volatility (optional)
            
        Returns:
            (shares_to_buy, capital_to_allocate)
        """
        
        # Check if trading is allowed
        if self.max_drawdown_hit or self.max_daily_loss_hit:
            return 0, 0
        
        # Base position size: % of capital at risk
        capital_at_risk = self.current_capital * cfg.RISK_PER_TRADE
        
        # Adjust by confidence
        capital_at_risk *= signal_confidence
        
        # Calculate shares based on stop-loss
        # If stop-loss is 3%, and we want to risk 2% of capital:
        # Position size = (Capital * Risk%) / Stop-loss%
        stop_loss_distance = cfg.STOP_LOSS_PCT
        max_position_value = capital_at_risk / stop_loss_distance
        
        # Ensure we don't exceed max position size
        max_allowed = self.current_capital * cfg.MAX_POSITION_SIZE
        position_value = min(max_position_value, max_allowed)
        
        # Convert to shares
        shares = position_value / current_price
        
        # Ensure we keep minimum cash reserve
        required_cash = shares * current_price * (1 + cfg.COMMISSION_RATE + cfg.SLIPPAGE_RATE)
        available_cash = self.current_capital * (1 - cfg.MIN_CASH_RESERVE)
        
        if required_cash > available_cash:
            # Reduce position to fit available cash
            shares = available_cash / (current_price * (1 + cfg.COMMISSION_RATE + cfg.SLIPPAGE_RATE))
            position_value = shares * current_price
        
        return shares, position_value
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_type: str = "LONG"
    ) -> float:
        """Calculate stop-loss price"""
        if position_type == "LONG":
            return entry_price * (1 - cfg.STOP_LOSS_PCT)
        else:
            return entry_price * (1 + cfg.STOP_LOSS_PCT)
    
    def calculate_take_profit(
        self,
        entry_price: float,
        position_type: str = "LONG"
    ) -> float:
        """Calculate take-profit price"""
        if position_type == "LONG":
            return entry_price * (1 + cfg.TAKE_PROFIT_PCT)
        else:
            return entry_price * (1 - cfg.TAKE_PROFIT_PCT)
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float
    ) -> float:
        """
        Calculate trailing stop price
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            current_stop: Current stop-loss price
            
        Returns:
            Updated stop-loss price
        """
        if not cfg.TRAILING_STOP:
            return current_stop
        
        # Check if profit is enough to activate trailing stop
        profit_pct = (current_price - entry_price) / entry_price
        
        if profit_pct < cfg.TRAILING_STOP_ACTIVATION:
            return current_stop
        
        # Calculate new trailing stop
        new_stop = current_price * (1 - cfg.TRAILING_STOP_DISTANCE)
        
        # Only move stop up, never down
        return max(new_stop, current_stop)
    
    def update_capital(self, new_capital: float):
        """Update current capital and track peak"""
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital
    
    def calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if self.peak_capital == 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital
    
    def check_drawdown_limit(self) -> bool:
        """Check if max drawdown exceeded"""
        current_dd = self.calculate_drawdown()
        if current_dd >= cfg.MAX_DRAWDOWN:
            self.max_drawdown_hit = True
            return True
        return False
    
    def calculate_daily_loss(self) -> float:
        """Calculate loss since start of day"""
        if self.daily_start_capital == 0:
            return 0.0
        return (self.daily_start_capital - self.current_capital) / self.daily_start_capital
    
    def check_daily_loss_limit(self) -> bool:
        """Check if max daily loss exceeded"""
        daily_loss = self.calculate_daily_loss()
        if daily_loss >= cfg.MAX_DAILY_LOSS:
            self.max_daily_loss_hit = True
            return True
        return False
    
    def reset_daily_limits(self):
        """Reset daily limits at start of new day"""
        self.daily_start_capital = self.current_capital
        self.max_daily_loss_hit = False
    
    def get_risk_status(self) -> Dict:
        """Get current risk metrics"""
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown': self.calculate_drawdown(),
            'daily_loss': self.calculate_daily_loss(),
            'max_drawdown_hit': self.max_drawdown_hit,
            'max_daily_loss_hit': self.max_daily_loss_hit,
            'trading_allowed': not (self.max_drawdown_hit or self.max_daily_loss_hit)
        }
    
    def calculate_transaction_costs(
        self,
        shares: float,
        price: float,
        action: str = "BUY"
    ) -> Dict[str, float]:
        """
        Calculate all transaction costs
        
        Args:
            shares: Number of shares
            price: Execution price
            action: 'BUY' or 'SELL'
            
        Returns:
            Dictionary with cost breakdown
        """
        gross_value = shares * price
        
        # Commission
        commission = gross_value * cfg.COMMISSION_RATE
        
        # Slippage (worse execution price)
        if action == "BUY":
            slippage = gross_value * cfg.SLIPPAGE_RATE
            net_value = gross_value + commission + slippage
        else:  # SELL
            slippage = gross_value * cfg.SLIPPAGE_RATE
            net_value = gross_value - commission - slippage
        
        return {
            'gross_value': gross_value,
            'commission': commission,
            'slippage': slippage,
            'net_value': net_value,
            'total_costs': commission + slippage
        }


class PortfolioManager:
    """
    Tracks portfolio state:
    - Current positions
    - Cash balance
    - Portfolio value
    """
    
    def __init__(self, initial_cash: float = cfg.INITIAL_CAPITAL):
        self.cash = initial_cash
        self.positions = {}  # {asset: {'shares': X, 'entry_price': Y, 'entry_date': Z}}
        self.portfolio_history = []
        
    def get_position(self, asset: str = "Gold") -> Optional[Dict]:
        """Get current position in asset"""
        return self.positions.get(asset)
    
    def has_position(self, asset: str = "Gold") -> bool:
        """Check if we have a position"""
        return asset in self.positions and self.positions[asset]['shares'] > 0
    
    def open_position(
        self,
        asset: str,
        shares: float,
        entry_price: float,
        entry_date: str,
        costs: float
    ):
        """Open new position"""
        self.positions[asset] = {
            'shares': shares,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'stop_loss': entry_price * (1 - cfg.STOP_LOSS_PCT),
            'take_profit': entry_price * (1 + cfg.TAKE_PROFIT_PCT),
            'total_cost': shares * entry_price + costs
        }
        self.cash -= (shares * entry_price + costs)
    
    def close_position(
        self,
        asset: str,
        exit_price: float,
        costs: float
    ) -> float:
        """Close position and return P&L"""
        if asset not in self.positions:
            return 0.0
        
        position = self.positions[asset]
        shares = position['shares']
        
        # Calculate P&L
        gross_proceeds = shares * exit_price
        net_proceeds = gross_proceeds - costs
        cost_basis = position['total_cost']
        pnl = net_proceeds - cost_basis
        pnl_pct = pnl / cost_basis
        
        # Update cash
        self.cash += net_proceeds
        
        # Remove position
        del self.positions[asset]
        
        return pnl
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        position_value = sum(
            pos['shares'] * current_prices.get(asset, pos['entry_price'])
            for asset, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get detailed portfolio summary"""
        total_value = self.get_portfolio_value(current_prices)
        
        return {
            'cash': self.cash,
            'positions_value': total_value - self.cash,
            'total_value': total_value,
            'num_positions': len(self.positions),
            'positions': self.positions.copy()
        }


if __name__ == "__main__":
    # Test risk manager
    print("✅ Risk Manager Module Loaded")
    
    rm = RiskManager(initial_capital=1_000_000_000)
    
    # Test position sizing
    shares, capital = rm.calculate_position_size(
        current_price=11_000_000,
        signal_confidence=0.8
    )
    
    print(f"\n📊 Position Sizing Test:")
    print(f"Price: 11,000,000 IRR")
    print(f"Shares: {shares:,.0f}")
    print(f"Capital: {capital:,.0f} IRR")
    
    # Test stop-loss
    stop = rm.calculate_stop_loss(entry_price=11_000_000)
    print(f"\n🛑 Stop-Loss: {stop:,.0f} IRR ({cfg.STOP_LOSS_PCT*100}% below entry)")
    
    # Test take-profit
    tp = rm.calculate_take_profit(entry_price=11_000_000)
    print(f"🎯 Take-Profit: {tp:,.0f} IRR ({cfg.TAKE_PROFIT_PCT*100}% above entry)")
