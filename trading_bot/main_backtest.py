"""
Main Backtest Script
Loads data and runs complete backtest
"""

import pandas as pd
import sys
import os
from datetime import datetime

# Add module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import backtest_config as cfg
from backtesting.backtest_engine import BacktestEngine
from backtesting.visualizer import BacktestVisualizer


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare data for backtesting
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Prepared DataFrame
    """
    print(f"📂 Loading data from: {file_path}")
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Ensure Date column
    if 'Date' not in df.columns:
        raise ValueError("CSV must have 'Date' column")
    
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"✅ Data loaded: {len(df)} rows")
    print(f"   Period: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"   Features: {', '.join(df.columns[:10])}...")
    
    return df


def run_backtest(
    data_file: str,
    strategy: str = "momentum",
    save_results: bool = True,
    create_plots: bool = True
):
    """
    Run complete backtest
    
    Args:
        data_file: Path to data CSV
        strategy: Strategy name ('momentum' or 'mean_reversion')
        save_results: Whether to save results to files
        create_plots: Whether to create visualization plots
    """
    
    print("\n" + "="*70)
    print("🚀 GOLD TRADING BOT - BACKTEST")
    print("="*70)
    
    # Validate config
    print("\n📋 Validating Configuration...")
    cfg.validate_config()
    
    # Load data
    print("\n📊 Loading Data...")
    data = load_data(data_file)
    
    # Initialize backtest engine
    print("\n⚙️  Initializing Backtest Engine...")
    engine = BacktestEngine(
        data=data,
        strategy_name=strategy,
        initial_capital=cfg.INITIAL_CAPITAL
    )
    
    # Run backtest
    print("\n" + "-"*70)
    results = engine.run()
    print("-"*70)
    
    # Save results
    if save_results:
        print("\n💾 Saving Results...")
        engine.save_results(results)
    
    # Create visualizations
    if create_plots:
        print("\n🎨 Creating Visualizations...")
        visualizer = BacktestVisualizer(results)
        visualizer.plot_all()
    
    print("\n" + "="*70)
    print("✅ BACKTEST COMPLETE!")
    print("="*70)
    
    return results


def quick_comparison(data_file: str):
    """
    Quick comparison of multiple strategies
    
    Args:
        data_file: Path to data CSV
    """
    
    print("\n" + "="*70)
    print("📊 STRATEGY COMPARISON")
    print("="*70)
    
    data = load_data(data_file)
    
    strategies = ["momentum"]  # Can add "mean_reversion" later
    comparison = []
    
    for strategy in strategies:
        print(f"\n\n{'='*70}")
        print(f"Testing Strategy: {strategy.upper()}")
        print('='*70)
        
        engine = BacktestEngine(
            data=data,
            strategy_name=strategy,
            initial_capital=cfg.INITIAL_CAPITAL
        )
        
        results = engine.run()
        
        comparison.append({
            'Strategy': strategy,
            'Total Return (%)': results['metrics']['total_return_pct'],
            'CAGR (%)': results['metrics']['cagr_pct'],
            'Sharpe Ratio': results['metrics']['sharpe_ratio'],
            'Max Drawdown (%)': results['metrics']['max_drawdown_pct'],
            'Win Rate (%)': results['metrics']['win_rate_pct'],
            'Total Trades': results['metrics']['total_trades']
        })
    
    # Print comparison table
    print("\n\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70 + "\n")
    
    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))
    print()
    
    return comparison_df


if __name__ == "__main__":
    """
    Main execution
    
    Usage:
        python main_backtest.py
    """
    
    # Configuration
    DATA_FILE = "../data/processed/advanced_gold_features.csv"
    
    # Check if data file exists
    if not os.path.exists(DATA_FILE):
        # Try alternate path (the uploaded file in the system)
        print(f"⚠️  Data file not found at: {DATA_FILE}")
        print("Looking for data in system paths...")
        
        # Create sample data file from the CSV content we know exists
        # This is a workaround since we need to create the data path
        os.makedirs("/home/claude/trading_bot/data", exist_ok=True)
        
        # Let user know they need to provide the CSV
        print("\n" + "="*70)
        print("❌ ERROR: Data file not found!")
        print("="*70)
        print(f"\nPlease ensure the CSV file is at: {DATA_FILE}")
        print("\nExpected format:")
        print("  - Date column (required)")
        print("  - Gold_IRR (required)")
        print("  - Technical indicators (RSI_14, SMA_7, MACD, etc.)")
        print("  - Target_Next_LogRet (for simulating predictions)")
        print("\n" + "="*70)
        sys.exit(1)
    
    try:
        # Run single backtest
        results = run_backtest(
            data_file=DATA_FILE,
            strategy="momentum",
            save_results=True,
            create_plots=True
        )
        
        # Optional: Run comparison of multiple strategies
        # comparison = quick_comparison(DATA_FILE)
        
    except Exception as e:
        print(f"\n❌ Error during backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
