"""
Performance Metrics Calculator
Calculates comprehensive trading performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from config import backtest_config as cfg


class PerformanceMetrics:
    """
    Calculate all performance metrics:
    - Returns (total, annualized, CAGR)
    - Risk metrics (volatility, Sharpe, Sortino, max drawdown)
    - Trade metrics (win rate, avg P&L, profit factor)
    """
    
    def __init__(self):
        self.metrics = {}
        
    def calculate_all_metrics(
        self,
        portfolio_history: pd.DataFrame,
        trade_log: pd.DataFrame,
        benchmark_returns: pd.Series = None
    ) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio_history: Daily portfolio values
            trade_log: Individual trade records
            benchmark_returns: Buy-and-hold returns for comparison
            
        Returns:
            Dictionary of all metrics
        """
        
        metrics = {}
        
        # ====================================================================
        # RETURNS METRICS
        # ====================================================================
        
        initial_capital = portfolio_history['Portfolio_Value'].iloc[0]
        final_capital = portfolio_history['Portfolio_Value'].iloc[-1]
        
        # Total return
        total_return = (final_capital - initial_capital) / initial_capital
        metrics['total_return'] = total_return
        metrics['total_return_pct'] = total_return * 100
        
        # Number of trading days
        num_days = len(portfolio_history)
        num_years = num_days / 252  # Approximate trading days per year
        
        # CAGR (Compound Annual Growth Rate)
        if num_years > 0:
            cagr = (final_capital / initial_capital) ** (1 / num_years) - 1
            metrics['cagr'] = cagr
            metrics['cagr_pct'] = cagr * 100
        else:
            metrics['cagr'] = 0
            metrics['cagr_pct'] = 0
        
        # Daily returns
        portfolio_history['Daily_Return'] = portfolio_history['Portfolio_Value'].pct_change()
        daily_returns = portfolio_history['Daily_Return'].dropna()
        
        # ====================================================================
        # RISK METRICS
        # ====================================================================
        
        # Volatility (annualized)
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        metrics['volatility'] = annual_vol
        metrics['volatility_pct'] = annual_vol * 100
        
        # Sharpe Ratio
        # (Portfolio Return - Risk-Free Rate) / Volatility
        if annual_vol > 0:
            excess_return = metrics['cagr'] - cfg.RISK_FREE_RATE
            sharpe_ratio = excess_return / annual_vol
            metrics['sharpe_ratio'] = sharpe_ratio
        else:
            metrics['sharpe_ratio'] = 0
        
        # Sortino Ratio (uses only downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                sortino_ratio = (metrics['cagr'] - cfg.RISK_FREE_RATE) / downside_deviation
                metrics['sortino_ratio'] = sortino_ratio
            else:
                metrics['sortino_ratio'] = 0
        else:
            metrics['sortino_ratio'] = metrics['sharpe_ratio']  # No negative returns
        
        # Maximum Drawdown
        portfolio_values = portfolio_history['Portfolio_Value']
        cumulative_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        metrics['max_drawdown'] = abs(max_drawdown)
        metrics['max_drawdown_pct'] = abs(max_drawdown) * 100
        
        # Calmar Ratio (CAGR / Max Drawdown)
        if abs(max_drawdown) > 0:
            calmar_ratio = metrics['cagr'] / abs(max_drawdown)
            metrics['calmar_ratio'] = calmar_ratio
        else:
            metrics['calmar_ratio'] = 0
        
        # ====================================================================
        # TRADE METRICS
        # ====================================================================
        
        if len(trade_log) > 0:
            # Filter only closed trades
            closed_trades = trade_log[trade_log['Exit_Date'].notna()].copy()
            
            if len(closed_trades) > 0:
                # Total number of trades
                metrics['total_trades'] = len(closed_trades)
                
                # Winning trades
                winning_trades = closed_trades[closed_trades['PnL'] > 0]
                losing_trades = closed_trades[closed_trades['PnL'] < 0]
                
                metrics['winning_trades'] = len(winning_trades)
                metrics['losing_trades'] = len(losing_trades)
                
                # Win rate
                if len(closed_trades) > 0:
                    win_rate = len(winning_trades) / len(closed_trades)
                    metrics['win_rate'] = win_rate
                    metrics['win_rate_pct'] = win_rate * 100
                else:
                    metrics['win_rate'] = 0
                    metrics['win_rate_pct'] = 0
                
                # Average P&L
                metrics['avg_pnl'] = closed_trades['PnL'].mean()
                metrics['avg_pnl_pct'] = closed_trades['PnL_Pct'].mean() * 100
                
                # Average win/loss
                if len(winning_trades) > 0:
                    metrics['avg_win'] = winning_trades['PnL'].mean()
                    metrics['avg_win_pct'] = winning_trades['PnL_Pct'].mean() * 100
                else:
                    metrics['avg_win'] = 0
                    metrics['avg_win_pct'] = 0
                
                if len(losing_trades) > 0:
                    metrics['avg_loss'] = losing_trades['PnL'].mean()
                    metrics['avg_loss_pct'] = losing_trades['PnL_Pct'].mean() * 100
                else:
                    metrics['avg_loss'] = 0
                    metrics['avg_loss_pct'] = 0
                
                # Profit factor (total wins / total losses)
                total_wins = winning_trades['PnL'].sum() if len(winning_trades) > 0 else 0
                total_losses = abs(losing_trades['PnL'].sum()) if len(losing_trades) > 0 else 1
                if total_losses > 0:
                    metrics['profit_factor'] = total_wins / total_losses
                else:
                    metrics['profit_factor'] = 0 if total_wins == 0 else float('inf')
                
                # Expectancy (average $ per trade)
                metrics['expectancy'] = closed_trades['PnL'].mean()
                
                # Average holding period
                closed_trades['Holding_Days'] = (
                    pd.to_datetime(closed_trades['Exit_Date']) - 
                    pd.to_datetime(closed_trades['Entry_Date'])
                ).dt.days
                metrics['avg_holding_days'] = closed_trades['Holding_Days'].mean()
                
                # Best and worst trades
                metrics['best_trade'] = closed_trades['PnL'].max()
                metrics['best_trade_pct'] = closed_trades['PnL_Pct'].max() * 100
                metrics['worst_trade'] = closed_trades['PnL'].min()
                metrics['worst_trade_pct'] = closed_trades['PnL_Pct'].min() * 100
            else:
                # No closed trades
                self._set_zero_trade_metrics(metrics)
        else:
            self._set_zero_trade_metrics(metrics)
        
        # ====================================================================
        # BENCHMARK COMPARISON
        # ====================================================================
        
        if benchmark_returns is not None:
            # Benchmark metrics
            benchmark_total_return = (benchmark_returns.iloc[-1] / benchmark_returns.iloc[0]) - 1
            metrics['benchmark_return'] = benchmark_total_return
            metrics['benchmark_return_pct'] = benchmark_total_return * 100
            
            # Alpha (excess return over benchmark)
            metrics['alpha'] = total_return - benchmark_total_return
            metrics['alpha_pct'] = metrics['alpha'] * 100
            
            # Benchmark CAGR
            if num_years > 0:
                benchmark_cagr = (benchmark_returns.iloc[-1] / benchmark_returns.iloc[0]) ** (1 / num_years) - 1
                metrics['benchmark_cagr'] = benchmark_cagr
                metrics['benchmark_cagr_pct'] = benchmark_cagr * 100
            
        # ====================================================================
        # TIME METRICS
        # ====================================================================
        
        metrics['start_date'] = portfolio_history['Date'].iloc[0]
        metrics['end_date'] = portfolio_history['Date'].iloc[-1]
        metrics['num_days'] = num_days
        metrics['num_years'] = num_years
        
        # Capital metrics
        metrics['initial_capital'] = initial_capital
        metrics['final_capital'] = final_capital
        metrics['peak_capital'] = portfolio_values.max()
        
        self.metrics = metrics
        return metrics
    
    def _set_zero_trade_metrics(self, metrics: Dict):
        """Set trade metrics to zero when no trades"""
        metrics['total_trades'] = 0
        metrics['winning_trades'] = 0
        metrics['losing_trades'] = 0
        metrics['win_rate'] = 0
        metrics['win_rate_pct'] = 0
        metrics['avg_pnl'] = 0
        metrics['avg_pnl_pct'] = 0
        metrics['avg_win'] = 0
        metrics['avg_win_pct'] = 0
        metrics['avg_loss'] = 0
        metrics['avg_loss_pct'] = 0
        metrics['profit_factor'] = 0
        metrics['expectancy'] = 0
        metrics['avg_holding_days'] = 0
        metrics['best_trade'] = 0
        metrics['best_trade_pct'] = 0
        metrics['worst_trade'] = 0
        metrics['worst_trade_pct'] = 0
    
    def print_summary(self):
        """Print formatted performance summary"""
        if not self.metrics:
            print("No metrics calculated yet")
            return
        
        m = self.metrics
        
        print("\n" + "="*70)
        print("📊 BACKTEST PERFORMANCE SUMMARY")
        print("="*70)
        
        # Period
        print(f"\n📅 Period: {m['start_date']} to {m['end_date']}")
        print(f"   Duration: {m['num_days']} days ({m['num_years']:.2f} years)")
        
        # Returns
        print(f"\n💰 Returns:")
        print(f"   Initial Capital:     {m['initial_capital']:>15,.0f} IRR")
        print(f"   Final Capital:       {m['final_capital']:>15,.0f} IRR")
        print(f"   Total Return:        {m['total_return_pct']:>14.2f} %")
        print(f"   CAGR:                {m['cagr_pct']:>14.2f} %")
        
        # Risk
        print(f"\n📉 Risk Metrics:")
        print(f"   Volatility (Annual): {m['volatility_pct']:>14.2f} %")
        print(f"   Max Drawdown:        {m['max_drawdown_pct']:>14.2f} %")
        print(f"   Sharpe Ratio:        {m['sharpe_ratio']:>14.2f}")
        print(f"   Sortino Ratio:       {m['sortino_ratio']:>14.2f}")
        print(f"   Calmar Ratio:        {m['calmar_ratio']:>14.2f}")
        
        # Trades
        print(f"\n📈 Trade Statistics:")
        print(f"   Total Trades:        {m['total_trades']:>14.0f}")
        print(f"   Winning Trades:      {m['winning_trades']:>14.0f}")
        print(f"   Losing Trades:       {m['losing_trades']:>14.0f}")
        print(f"   Win Rate:            {m['win_rate_pct']:>14.2f} %")
        print(f"   Avg P&L per Trade:   {m['avg_pnl']:>15,.0f} IRR")
        print(f"   Avg Win:             {m['avg_win']:>15,.0f} IRR ({m['avg_win_pct']:.2f}%)")
        print(f"   Avg Loss:            {m['avg_loss']:>15,.0f} IRR ({m['avg_loss_pct']:.2f}%)")
        print(f"   Profit Factor:       {m['profit_factor']:>14.2f}")
        print(f"   Avg Holding Period:  {m['avg_holding_days']:>14.1f} days")
        
        # Best/Worst
        print(f"\n🎯 Best/Worst Trades:")
        print(f"   Best Trade:          {m['best_trade']:>15,.0f} IRR ({m['best_trade_pct']:.2f}%)")
        print(f"   Worst Trade:         {m['worst_trade']:>15,.0f} IRR ({m['worst_trade_pct']:.2f}%)")
        
        # Benchmark
        if 'benchmark_return' in m:
            print(f"\n📊 vs. Buy & Hold:")
            print(f"   Strategy Return:     {m['total_return_pct']:>14.2f} %")
            print(f"   Benchmark Return:    {m['benchmark_return_pct']:>14.2f} %")
            print(f"   Alpha:               {m['alpha_pct']:>14.2f} %")
        
        print("\n" + "="*70)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to DataFrame"""
        return pd.DataFrame([self.metrics])


if __name__ == "__main__":
    print("✅ Performance Metrics Module Loaded")
