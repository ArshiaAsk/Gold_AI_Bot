"""
API Configuration for Real-Time Trading System
Aligned with main project structure and config/settings.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import main project config
from config.settings import config as main_config

# ==================== API ENDPOINTS ====================
TGJU_BASE_URL = "https://api.tgju.org/v1"

# Endpoint for fetching current prices (same as data_loader.py)
TGJU_SUMMARY_ENDPOINT = f"{TGJU_BASE_URL}/market/indicator/summary-table-data"

# Item slugs (from config/settings.py)
TICKER_GOLD = main_config.data.TICKER_GOLD  # "geram18"
TICKER_USD = main_config.data.TICKER_USD    # "price_dollar_rl"

# ==================== RATE LIMITING ====================
MAX_REQUESTS_PER_MINUTE = 50
REQUEST_DELAY_SECONDS = 1.5  # Conservative: 40 requests/minute

# ==================== RETRY CONFIGURATION ====================
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
TIMEOUT = 10  # seconds per request

# ==================== DATA REFRESH RATES ====================
PRICE_UPDATE_INTERVAL = 60  # Every 1 minute during market hours
FEATURE_CALCULATION_INTERVAL = 300  # Every 5 minutes
SIGNAL_GENERATION_INTERVAL = 300  # Every 5 minutes

# ==================== MARKET HOURS (Tehran Time) ====================
MARKET_OPEN_HOUR = 8
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 30
TRADING_DAYS = [5, 6, 0, 1, 2]  # Saturday to Wednesday

# ==================== FILE PATHS (from main config) ====================
# Use paths from main project config
MODEL_PATH = Path(main_config.paths.model_path)
SCALER_PATH = Path(main_config.paths.scaler_path)

# Historical data for feature engineering
HISTORICAL_DATA_PATH = Path(main_config.paths.processed_data_path)

# API-specific cache directory
CACHE_DIR = PROJECT_ROOT / "api_layer" / "cache"
CACHE_DIR.mkdir(exist_ok=True)

LATEST_PRICES_FILE = CACHE_DIR / "latest_prices.json"
LATEST_FEATURES_FILE = CACHE_DIR / "latest_features.json"
LATEST_SIGNAL_FILE = CACHE_DIR / "latest_signal.json"

# ==================== LOGGING ====================
LOG_DIR = PROJECT_ROOT / "api_layer" / "logs"
LOG_DIR.mkdir(exist_ok=True)

API_LOG_FILE = LOG_DIR / "api_requests.log"
SIGNAL_LOG_FILE = LOG_DIR / "signals.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"
TRADING_BOT_LOG = LOG_DIR / "trading_bot.log"

# ==================== FEATURE ENGINEERING ====================
# Use same lookback as training
LOOKBACK_DAYS = main_config.data.LOOKBACK_DAYS  # 30 days

# Feature columns (must match training exactly)
FEATURE_COLUMNS = main_config.data.FEATURE_COLUMNS

# ==================== SIGNAL THRESHOLDS (from backtest config) ====================
# Import from backtest if exists, otherwise use defaults
try:
    from trading_bot.config.backtest_config import BUY_THRESHOLD, SELL_THRESHOLD
except ImportError:
    BUY_THRESHOLD = 0.008  # 0.8% predicted return
    SELL_THRESHOLD = -0.005  # -0.5% predicted return

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# ==================== PAPER TRADING ====================
PAPER_TRADING_MODE = True  # Set to False for live trading (NOT RECOMMENDED YET)
try:
    from trading_bot.config.backtest_config import INITIAL_CAPITAL
    PAPER_INITIAL_CAPITAL = INITIAL_CAPITAL
except ImportError:
    PAPER_INITIAL_CAPITAL = 1_000_000_000  # 1B IRR

# ==================== NOTIFICATION SETTINGS ====================
ENABLE_NOTIFICATIONS = True
NOTIFICATION_METHODS = ['console', 'file']

# Email/Telegram settings
EMAIL_ENABLED = False
TELEGRAM_ENABLED = False

# ==================== SAFETY LIMITS ====================
MAX_POSITION_SIZE = 0.5
MAX_DAILY_TRADES = 5
MIN_SIGNAL_CONFIDENCE = 0.6

# ==================== API KEY ====================
TGJU_API_KEY = os.getenv('TGJU_API_KEY', None)

# ==================== DEBUG MODE ====================
DEBUG_MODE = False
SAVE_RAW_RESPONSES = DEBUG_MODE
