"""
Visualization Module
Creates performance plots and charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import os

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)


class BacktestVisualizer:
    """
    Create visualizations for backtest results
    """
    
    def __init__(self, results: Dict):
        """
        Initialize with backtest results
        
        Args:
            results: Dictionary from BacktestEngine
        """
        self.portfolio_history = results['portfolio_history']
        self.trade_log = results['trade_log']
        self.signal_history = results['signal_history']
        self.metrics = results['metrics']
        
    def plot_all(self, output_dir: str = "/home/claude/trading_bot/outputs/plots"):
        """Generate all plots"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n📊 Generating Plots...")
        
        # 1. Portfolio value over time
        self.plot_portfolio_value(output_dir)
        
        # 2. Drawdown chart
        self.plot_drawdown(output_dir)
        
        # 3. Trade distribution
        if len(self.trade_log) > 0:
            self.plot_trade_distribution(output_dir)
            self.plot_trade_timeline(output_dir)
        
        # 4. Monthly returns
        self.plot_monthly_returns(output_dir)
        
        print(f"✅ All plots saved to: {output_dir}")
    
    def plot_portfolio_value(self, output_dir: str):
        """Plot portfolio value over time"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        # Convert date to datetime
        dates = pd.to_datetime(self.portfolio_history['Date'])
        values = self.portfolio_history['Portfolio_Value'] / 1_000_000  # Convert to millions
        
        # Plot portfolio value
        ax1.plot(dates, values, linewidth=2, label='Strategy', color='#2E86AB')
        
        # Add initial capital line
        initial_capital = self.metrics['initial_capital'] / 1_000_000
        ax1.axhline(y=initial_capital, color='gray', linestyle='--', 
                    alpha=0.5, label='Initial Capital')
        
        # Highlight trades
        if len(self.trade_log) > 0:
            entry_dates = pd.to_datetime(self.trade_log['Entry_Date'])
            entry_values = []
            for date in entry_dates:
                val = values[dates == date].values
                if len(val) > 0:
                    entry_values.append(val[0])
                else:
                    entry_values.append(np.nan)
            
            exit_dates = pd.to_datetime(self.trade_log['Exit_Date'])
            exit_values = []
            for date in exit_dates:
                val = values[dates == date].values
                if len(val) > 0:
                    exit_values.append(val[0])
                else:
                    exit_values.append(np.nan)
            
            # Mark entries and exits
            ax1.scatter(entry_dates, entry_values, color='green', marker='^', 
                       s=100, alpha=0.6, label='Buy', zorder=5)
            ax1.scatter(exit_dates, exit_values, color='red', marker='v', 
                       s=100, alpha=0.6, label='Sell', zorder=5)
        
        ax1.set_title(f'📈 Portfolio Performance\n'
                     f'Total Return: {self.metrics["total_return_pct"]:.2f}% | '
                     f'CAGR: {self.metrics["cagr_pct"]:.2f}% | '
                     f'Sharpe: {self.metrics["sharpe_ratio"]:.2f}',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (Million IRR)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Position indicator (in or out of market)
        positions = self.portfolio_history['Has_Position'].astype(int)
        ax2.fill_between(dates, 0, positions, alpha=0.3, color='blue', label='In Position')
        ax2.set_ylabel('Position', fontsize=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Out', 'In'])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'portfolio_value.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Portfolio value chart saved")
    
    def plot_drawdown(self, output_dir: str):
        """Plot drawdown over time"""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        dates = pd.to_datetime(self.portfolio_history['Date'])
        drawdown = self.portfolio_history['Drawdown'] * 100  # Convert to percentage
        
        ax.fill_between(dates, 0, -drawdown, alpha=0.3, color='red')
        ax.plot(dates, -drawdown, color='darkred', linewidth=2)
        
        # Mark max drawdown
        max_dd_idx = drawdown.idxmax()
        max_dd_date = dates.iloc[max_dd_idx]
        max_dd_value = drawdown.iloc[max_dd_idx]
        
        ax.scatter([max_dd_date], [-max_dd_value], color='red', s=200, 
                  marker='v', zorder=5, label=f'Max DD: {max_dd_value:.2f}%')
        
        ax.set_title(f'📉 Drawdown Over Time\n'
                    f'Maximum Drawdown: {self.metrics["max_drawdown_pct"]:.2f}%',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Drawdown chart saved")
    
    def plot_trade_distribution(self, output_dir: str):
        """Plot distribution of trade P&L"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        pnl_pct = self.trade_log['PnL_Pct'] * 100
        
        # 1. Histogram of returns
        ax1.hist(pnl_pct, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
        ax1.axvline(x=pnl_pct.mean(), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {pnl_pct.mean():.2f}%')
        ax1.set_title('Distribution of Trade Returns (%)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Win/Loss pie chart
        wins = (self.trade_log['PnL'] > 0).sum()
        losses = (self.trade_log['PnL'] < 0).sum()
        
        colors = ['#2ECC71', '#E74C3C']
        explode = (0.05, 0.05)
        
        ax2.pie([wins, losses], labels=[f'Wins\n({wins})', f'Losses\n({losses})'],
               autopct='%1.1f%%', colors=colors, explode=explode, startangle=90)
        ax2.set_title(f'Win/Loss Ratio\nWin Rate: {self.metrics["win_rate_pct"]:.1f}%',
                     fontsize=12, fontweight='bold')
        
        # 3. Trade P&L over time
        trade_dates = pd.to_datetime(self.trade_log['Exit_Date'])
        colors_list = ['green' if x > 0 else 'red' for x in self.trade_log['PnL']]
        
        ax3.bar(range(len(self.trade_log)), pnl_pct, color=colors_list, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_title('Trade Returns Over Time', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Return (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Holding period vs Return
        holding_days = self.trade_log['Holding_Days']
        colors_scatter = ['green' if x > 0 else 'red' for x in pnl_pct]
        
        ax4.scatter(holding_days, pnl_pct, c=colors_scatter, alpha=0.6, s=100)
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax4.set_title('Holding Period vs Return', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Holding Days')
        ax4.set_ylabel('Return (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trade_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Trade distribution chart saved")
    
    def plot_trade_timeline(self, output_dir: str):
        """Plot timeline of trades"""
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for idx, trade in self.trade_log.iterrows():
            entry = pd.to_datetime(trade['Entry_Date'])
            exit_date = pd.to_datetime(trade['Exit_Date'])
            
            color = 'green' if trade['PnL'] > 0 else 'red'
            ax.plot([entry, exit_date], [idx, idx], color=color, linewidth=3, alpha=0.7)
            ax.scatter([entry], [idx], color='blue', s=50, zorder=5)
            ax.scatter([exit_date], [idx], color=color, s=50, zorder=5)
        
        ax.set_title('Trade Timeline', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Trade Number', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trade_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Trade timeline chart saved")
    
    def plot_monthly_returns(self, output_dir: str):
        """Plot monthly returns heatmap"""
        
        # Calculate monthly returns
        df = self.portfolio_history.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Get last day of each month
        monthly_values = df.groupby(['Year', 'Month'])['Portfolio_Value'].last()
        monthly_returns = monthly_values.pct_change() * 100
        
        # Reshape for heatmap
        monthly_returns_df = monthly_returns.reset_index()
        monthly_returns_df.columns = ['Year', 'Month', 'Return']
        pivot_table = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Return (%)'}, ax=ax)
        
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_returns.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Monthly returns heatmap saved")


if __name__ == "__main__":
    print("✅ Visualization Module Loaded")
