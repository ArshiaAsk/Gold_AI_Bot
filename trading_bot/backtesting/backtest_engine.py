"""
Backtest Engine - Main Trading Simulation
Day-by-day simulation of trading strategy
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import backtest_config as cfg
from strategy.signal_generator import get_strategy
from risk.risk_manager import RiskManager, PortfolioManager
from backtesting.performance_metrics import PerformanceMetrics


class BacktestEngine:
    """
    Main backtesting engine that simulates trading day-by-day
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        strategy_name: str = "momentum",
        initial_capital: float = cfg.INITIAL_CAPITAL
    ):
        """
        Initialize backtest engine
        
        Args:
            data: Historical OHLCV + features data
            strategy_name: Strategy to use ('momentum', 'mean_reversion')
            initial_capital: Starting capital
        """
        self.data = data.copy()
        self.strategy = get_strategy(strategy_name)
        self.risk_manager = RiskManager(initial_capital)
        self.portfolio = PortfolioManager(initial_capital)
        
        # Results storage
        self.trade_log = []
        self.portfolio_history = []
        self.signal_history = []
        
        # Current state
        self.current_trade = None
        self.trade_counter = 0
        
        print(f"✅ Backtest Engine Initialized")
        print(f"   Strategy: {self.strategy.name}")
        print(f"   Initial Capital: {initial_capital:,.0f} IRR")
        print(f"   Data Period: {self.data['Date'].iloc[0]} to {self.data['Date'].iloc[-1]}")
        print(f"   Total Days: {len(self.data)}")
    
    def run(self) -> Dict:
        """
        Run the complete backtest
        
        Returns:
            Dictionary with results
        """
        print("\n🚀 Starting Backtest...")
        
        for idx, row in self.data.iterrows():
            self._process_day(row)
        
        # Close any open positions at the end
        if self.current_trade is not None:
            self._close_position(
                exit_date=self.data['Date'].iloc[-1],
                exit_price=self.data['Gold_IRR'].iloc[-1],
                exit_reason="End of backtest"
            )
        
        print("\n✅ Backtest Complete!")
        
        # Calculate performance metrics
        results = self._generate_results()
        
        return results
    
    def _process_day(self, row: pd.Series):
        """
        Process a single trading day
        
        Args:
            row: Current day's data
        """
        current_date = row['Date']
        current_price = row['Gold_IRR']
        
        # Reset daily limits
        self.risk_manager.reset_daily_limits()
        
        # Get current position
        has_position = self.portfolio.has_position("Gold")
        position = self.portfolio.get_position("Gold") if has_position else None
        
        # Generate signal
        signal_info = self.strategy.generate_signal(
            row=row,
            current_position=1.0 if has_position else 0.0,
            entry_price=position['entry_price'] if position else None
        )
        
        # Store signal
        self.signal_history.append({
            'Date': current_date,
            'Signal': signal_info['signal'],
            'Confidence': signal_info['confidence'],
            'Price': current_price,
            'Has_Position': has_position
        })
        
        # Execute trade logic
        if has_position:
            self._manage_existing_position(row, signal_info, position)
        else:
            self._check_entry_signal(row, signal_info)
        
        # Update portfolio value
        portfolio_value = self.portfolio.get_portfolio_value({"Gold": current_price})
        self.risk_manager.update_capital(portfolio_value)
        
        # Check risk limits
        self.risk_manager.check_drawdown_limit()
        self.risk_manager.check_daily_loss_limit()
        
        # Record daily portfolio state
        self.portfolio_history.append({
            'Date': current_date,
            'Portfolio_Value': portfolio_value,
            'Cash': self.portfolio.cash,
            'Position_Value': portfolio_value - self.portfolio.cash,
            'Has_Position': has_position,
            'Drawdown': self.risk_manager.calculate_drawdown(),
            'Signal': signal_info['signal']
        })
    
    def _check_entry_signal(self, row: pd.Series, signal_info: Dict):
        """Check if we should enter a new position"""
        
        if signal_info['signal'] != 'BUY':
            return
        
        # Check if trading is allowed
        if not self.risk_manager.get_risk_status()['trading_allowed']:
            return
        
        current_date = row['Date']
        current_price = row['Gold_IRR']
        
        # Calculate position size
        shares, position_value = self.risk_manager.calculate_position_size(
            current_price=current_price,
            signal_confidence=signal_info['confidence']
        )
        
        if shares <= 0:
            return
        
        # Calculate transaction costs
        costs = self.risk_manager.calculate_transaction_costs(
            shares=shares,
            price=current_price,
            action="BUY"
        )
        
        # Check if we have enough cash
        total_cost = costs['net_value']
        if total_cost > self.portfolio.cash:
            return  # Not enough cash
        
        # Execute buy
        self.portfolio.open_position(
            asset="Gold",
            shares=shares,
            entry_price=current_price,
            entry_date=current_date,
            costs=costs['total_costs']
        )
        
        # Record trade
        self.trade_counter += 1
        self.current_trade = {
            'Trade_ID': self.trade_counter,
            'Entry_Date': current_date,
            'Entry_Price': current_price,
            'Shares': shares,
            'Entry_Value': shares * current_price,
            'Entry_Costs': costs['total_costs'],
            'Signal_Confidence': signal_info['confidence'],
            'Predicted_Return': signal_info.get('predicted_return', 0),
            'Entry_RSI': signal_info.get('rsi', 50)
        }
    
    def _manage_existing_position(self, row: pd.Series, signal_info: Dict, position: Dict):
        """Manage existing position (check exits, trailing stops, etc.)"""
        
        current_date = row['Date']
        current_price = row['Gold_IRR']
        
        # Update trailing stop
        if cfg.TRAILING_STOP:
            new_stop = self.risk_manager.calculate_trailing_stop(
                entry_price=position['entry_price'],
                current_price=current_price,
                current_stop=position['stop_loss']
            )
            position['stop_loss'] = new_stop
        
        # Check exit conditions
        should_exit = False
        exit_reason = ""
        
        # 1. Stop-loss hit
        if current_price <= position['stop_loss']:
            should_exit = True
            exit_reason = "Stop-loss"
        
        # 2. Take-profit hit
        elif current_price >= position['take_profit']:
            should_exit = True
            exit_reason = "Take-profit"
        
        # 3. Signal says EXIT
        elif signal_info['signal'] == 'EXIT':
            should_exit = True
            exit_reason = signal_info['reasons'][0] if signal_info['reasons'] else "Signal exit"
        
        # 4. Max holding period
        entry_date = pd.to_datetime(position['entry_date'])
        current_date_dt = pd.to_datetime(current_date)
        holding_days = (current_date_dt - entry_date).days
        
        if holding_days >= cfg.MAX_HOLDING_DAYS:
            should_exit = True
            exit_reason = f"Max holding period ({holding_days} days)"
        
        # Execute exit if needed
        if should_exit:
            self._close_position(
                exit_date=current_date,
                exit_price=current_price,
                exit_reason=exit_reason
            )
    
    def _close_position(self, exit_date: str, exit_price: float, exit_reason: str):
        """Close current position"""
        
        if self.current_trade is None:
            return
        
        position = self.portfolio.get_position("Gold")
        if position is None:
            return
        
        shares = position['shares']
        
        # Calculate transaction costs
        costs = self.risk_manager.calculate_transaction_costs(
            shares=shares,
            price=exit_price,
            action="SELL"
        )
        
        # Close position
        pnl = self.portfolio.close_position(
            asset="Gold",
            exit_price=exit_price,
            costs=costs['total_costs']
        )
        
        # Calculate metrics
        entry_price = self.current_trade['Entry_Price']
        pnl_pct = (exit_price - entry_price) / entry_price
        
        entry_date = pd.to_datetime(self.current_trade['Entry_Date'])
        exit_date_dt = pd.to_datetime(exit_date)
        holding_days = (exit_date_dt - entry_date).days
        
        # Record completed trade
        trade_record = {
            **self.current_trade,
            'Exit_Date': exit_date,
            'Exit_Price': exit_price,
            'Exit_Value': shares * exit_price,
            'Exit_Costs': costs['total_costs'],
            'PnL': pnl,
            'PnL_Pct': pnl_pct,
            'Holding_Days': holding_days,
            'Exit_Reason': exit_reason
        }
        
        self.trade_log.append(trade_record)
        self.current_trade = None
    
    def _generate_results(self) -> Dict:
        """Generate final results and metrics"""
        
        # Convert to DataFrames
        portfolio_df = pd.DataFrame(self.portfolio_history)
        trade_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()
        signal_df = pd.DataFrame(self.signal_history)
        
        # Calculate benchmark (buy-and-hold)
        if cfg.BENCHMARK_ENABLED:
            initial_price = self.data['Gold_IRR'].iloc[0]
            benchmark_shares = cfg.INITIAL_CAPITAL / initial_price
            benchmark_returns = self.data['Gold_IRR'] * benchmark_shares
        else:
            benchmark_returns = None
        
        # Calculate performance metrics
        metrics_calc = PerformanceMetrics()
        metrics = metrics_calc.calculate_all_metrics(
            portfolio_history=portfolio_df,
            trade_log=trade_df,
            benchmark_returns=benchmark_returns
        )
        
        # Print summary
        metrics_calc.print_summary()
        
        return {
            'metrics': metrics,
            'portfolio_history': portfolio_df,
            'trade_log': trade_df,
            'signal_history': signal_df,
            'metrics_calculator': metrics_calc
        }
    
    def save_results(self, results: Dict, output_dir: str = cfg.OUTPUT_DIR):
        """Save results to files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio history
        portfolio_file = os.path.join(output_dir, cfg.PORTFOLIO_LOG_FILE)
        results['portfolio_history'].to_csv(portfolio_file, index=False)
        print(f"💾 Portfolio history saved: {portfolio_file}")
        
        # Save trade log
        if len(results['trade_log']) > 0:
            trade_file = os.path.join(output_dir, cfg.TRADE_LOG_FILE)
            results['trade_log'].to_csv(trade_file, index=False)
            print(f"💾 Trade log saved: {trade_file}")
        
        # Save metrics
        import json
        metrics_file = os.path.join(output_dir, cfg.METRICS_FILE)
        with open(metrics_file, 'w') as f:
            json.dump(results['metrics'], f, indent=2, default=str)
        print(f"💾 Metrics saved: {metrics_file}")
        
        print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    print("✅ Backtest Engine Module Loaded")
