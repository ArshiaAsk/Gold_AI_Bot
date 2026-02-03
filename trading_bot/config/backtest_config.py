"""
Backtesting Configuration
All parameters for the trading bot backtesting engine
"""

# ============================================================================
# CAPITAL & RISK MANAGEMENT
# ============================================================================

INITIAL_CAPITAL = 1_000_000_000  # IRR (1 billion Toman)
RISK_PER_TRADE = 0.02  # 2% of capital per trade
MAX_POSITION_SIZE = 0.95  # Max 95% of capital in single position
MIN_CASH_RESERVE = 0.05  # Keep 5% cash minimum

# ============================================================================
# TRANSACTION COSTS
# ============================================================================

COMMISSION_RATE = 0.005  # 0.5% per trade
SLIPPAGE_RATE = 0.002  # 0.2% average slippage

# ============================================================================
# STRATEGY PARAMETERS (MOMENTUM)
# ============================================================================

# Signal generation thresholds
BUY_THRESHOLD = 0.008  # +0.8% predicted return minimum
SELL_THRESHOLD = -0.008  # -0.8% predicted return to exit
HOLD_THRESHOLD_UPPER = 0.006  # Above +0.6% but below buy threshold
HOLD_THRESHOLD_LOWER = -0.006  # Above sell threshold but below 0

# Technical filters
RSI_OVERBOUGHT = 75  # Don't buy if RSI > 75
RSI_OVERSOLD = 30  # Consider buying on oversold
RSI_EXIT = 75  # Exit long positions if RSI > 75

# Trend confirmation
USE_SMA_FILTER = True  # Require price above SMA_7 to buy
USE_MACD_FILTER = True  # Require MACD > Signal to buy

# Confidence filters
MAX_PREDICTION_STD = 0.025  # 2.5% max uncertainty to enter trade
MIN_CONFIDENCE_SCORE = 0.6  # Minimum confidence (if available)

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Stop-loss & Take-profit
STOP_LOSS_PCT = 0.03  # 3% stop-loss from entry
TAKE_PROFIT_PCT = 0.05  # 5% take-profit from entry
TRAILING_STOP = True  # Use trailing stop
TRAILING_STOP_ACTIVATION = 0.02  # Activate after 2% profit
TRAILING_STOP_DISTANCE = 0.015  # Trail by 1.5%

# Position management
MAX_HOLDING_DAYS = 30  # Force exit after 30 days
SCALE_OUT_ENABLED = True  # Take partial profits
SCALE_OUT_LEVELS = [0.03, 0.04, 0.05]  # At 3%, 4%, 5% profit
SCALE_OUT_PORTIONS = [0.33, 0.33, 0.34]  # Sell 33%, 33%, 34%

# Drawdown protection
MAX_DAILY_LOSS = 0.05  # Stop trading if daily loss > 5%
MAX_DRAWDOWN = 0.15  # Stop trading if drawdown > 15%

# ============================================================================
# BACKTESTING SETTINGS
# ============================================================================

BACKTEST_START_DATE = "2021-01-27"  # First available date
BACKTEST_END_DATE = "2024-12-31"  # Last available date
WALK_FORWARD_VALIDATION = False  # Not implemented yet

# Data requirements
WARMUP_PERIOD = 30  # Days needed for technical indicators
MIN_DATA_POINTS = 50  # Minimum historical data required

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

# Risk-free rate for Sharpe ratio (annual)
RISK_FREE_RATE = 0.20  # 20% (typical Iranian bank rate)

# Benchmark (buy-and-hold gold)
BENCHMARK_ENABLED = True

# ============================================================================
# LOGGING & OUTPUT
# ============================================================================

LOG_TRADES = True  # Save trade log
LOG_DAILY_PORTFOLIO = True  # Save daily portfolio value
PLOT_RESULTS = True  # Generate performance plots
SAVE_METRICS = True  # Save metrics to JSON

OUTPUT_DIR = ".../outputs"
TRADE_LOG_FILE = "../trade_log.csv"
PORTFOLIO_LOG_FILE = "../portfolio_log.csv"
METRICS_FILE = "../backtest_metrics.json"

# ============================================================================
# PREDICTION SETTINGS (for future API integration)
# ============================================================================

USE_REAL_PREDICTIONS = False  # If True, use model predictions; else use actual returns
MODEL_API_URL = "http://localhost:8000/predict"
PREDICTION_CONFIDENCE_THRESHOLD = 0.6

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration parameters"""
    assert 0 < RISK_PER_TRADE <= 0.10, "Risk per trade should be 0.1-10%"
    assert 0 < COMMISSION_RATE < 0.05, "Commission rate should be 0-5%"
    assert 0 < SLIPPAGE_RATE < 0.02, "Slippage should be 0-2%"
    assert STOP_LOSS_PCT > 0, "Stop-loss must be positive"
    assert TAKE_PROFIT_PCT > STOP_LOSS_PCT, "Take-profit should be > stop-loss"
    assert sum(SCALE_OUT_PORTIONS) == 1.0, "Scale-out portions must sum to 1.0"
    print("✅ Configuration validated successfully")

if __name__ == "__main__":
    validate_config()
    print("\n📊 Current Configuration:")
    print(f"Initial Capital: {INITIAL_CAPITAL:,} IRR")
    print(f"Risk per Trade: {RISK_PER_TRADE*100}%")
    print(f"Commission: {COMMISSION_RATE*100}%")
    print(f"Slippage: {SLIPPAGE_RATE*100}%")
    print(f"Stop-Loss: {STOP_LOSS_PCT*100}%")
    print(f"Take-Profit: {TAKE_PROFIT_PCT*100}%")
