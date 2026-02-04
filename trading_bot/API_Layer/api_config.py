"""
API Configuration for Real-Time Trading System
Handles TGJU API settings, rate limits, and credentials
"""

import os
from pathlib import Path

# ==================== API ENDPOINTS ====================
TGJU_BASE_URL = "https://api.tgju.org/v1"

# Price endpoints
GOLD_IRR_ENDPOINT = f"{TGJU_BASE_URL}/market/indicator/summary-table-data/global-markets"
USD_IRR_ENDPOINT = f"{TGJU_BASE_URL}/market/indicator/summary-table-data/currency-chart"
GOLD_USD_ENDPOINT = f"{TGJU_BASE_URL}/market/indicator/summary-table-data/global-markets"

# Historical data endpoints
HISTORICAL_ENDPOINT = f"{TGJU_BASE_URL}/market/indicator/graph-data"

# ==================== RATE LIMITING ====================
# TGJU typically allows ~60 requests/minute (adjust based on actual limits)
MAX_REQUESTS_PER_MINUTE = 50
REQUEST_DELAY_SECONDS = 1.5  # Conservative: 40 requests/minute

# ==================== RETRY CONFIGURATION ====================
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
TIMEOUT = 10  # seconds per request

# ==================== DATA REFRESH RATES ====================
# How often to fetch new data (in seconds)
PRICE_UPDATE_INTERVAL = 60  # Every 1 minute during market hours
FEATURE_CALCULATION_INTERVAL = 300  # Every 5 minutes (more intensive)
SIGNAL_GENERATION_INTERVAL = 300  # Every 5 minutes

# ==================== MARKET HOURS (Tehran Time) ====================
# TGJU updates gold prices roughly 08:30-16:30 Tehran time
MARKET_OPEN_HOUR = 8
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 30

# Days of week (0=Monday, 6=Sunday)
TRADING_DAYS = [5, 6, 0, 1, 2]  # Saturday to Wednesday (Iranian business week)

# ==================== FILE PATHS ====================
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "gold_xgboost_model.pkl"
FEATURE_SCALER_PATH = PROJECT_ROOT / "models" / "feature_scaler.pkl"
PIPELINE_PATH = PROJECT_ROOT / "models" / "trained_pipeline.pkl"

# Cache directory for storing latest data
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

LATEST_PRICES_FILE = CACHE_DIR / "latest_prices.json"
LATEST_FEATURES_FILE = CACHE_DIR / "latest_features.json"
LATEST_SIGNAL_FILE = CACHE_DIR / "latest_signal.json"

# ==================== LOGGING ====================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

API_LOG_FILE = LOG_DIR / "api_requests.log"
SIGNAL_LOG_FILE = LOG_DIR / "signals.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"

# ==================== FEATURE ENGINEERING ====================
# Number of historical days needed to compute technical indicators
LOOKBACK_DAYS = 100  # Need enough data for SMA_30, MACD, etc.

# ==================== SIGNAL THRESHOLDS (from backtest) ====================
BUY_THRESHOLD = 0.008  # 0.8% predicted return
SELL_THRESHOLD = -0.005  # -0.5% predicted return

RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# ==================== PAPER TRADING ====================
PAPER_TRADING_MODE = True  # Set to False for live trading (NOT RECOMMENDED YET)
PAPER_INITIAL_CAPITAL = 1_000_000_000  # 1B IRR

# ==================== NOTIFICATION SETTINGS ====================
ENABLE_NOTIFICATIONS = True
NOTIFICATION_METHODS = ['console', 'file']  # Options: 'console', 'file', 'email', 'telegram'

# Email settings (if using email notifications)
EMAIL_ENABLED = False
EMAIL_FROM = os.getenv('TRADING_EMAIL_FROM', '')
EMAIL_TO = os.getenv('TRADING_EMAIL_TO', '')
EMAIL_PASSWORD = os.getenv('TRADING_EMAIL_PASSWORD', '')

# Telegram settings (if using telegram notifications)
TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ==================== SAFETY LIMITS ====================
MAX_POSITION_SIZE = 0.5  # Never use more than 50% of capital on single trade
MAX_DAILY_TRADES = 5  # Limit to prevent over-trading
MIN_SIGNAL_CONFIDENCE = 0.6  # Only act on high-confidence signals (if available)

# ==================== API KEY (if required in future) ====================
TGJU_API_KEY = os.getenv('TGJU_API_KEY', None)  # Currently not required

# ==================== DEBUG MODE ====================
DEBUG_MODE = False  # Set to True for verbose logging
SAVE_RAW_RESPONSES = DEBUG_MODE  # Save all API responses for debugging
