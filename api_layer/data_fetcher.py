"""
Real-Time Data Fetcher for TGJU API
Uses same logic as src/data_loader.py but for live real-time data
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import re
import time
import yfinance as yf
from curl_cffi import requests
from datetime import datetime, timedelta
from typing import Dict, Optional

from api_layer.api_config import (
    TGJU_SUMMARY_ENDPOINT, TICKER_GOLD, TICKER_USD,
    MAX_RETRIES, RETRY_DELAY, TIMEOUT,
    REQUEST_DELAY_SECONDS, CACHE_DIR, LATEST_PRICES_FILE,
    DEBUG_MODE, SAVE_RAW_RESPONSES
)


class TGJUDataFetcher:
    """
    Fetches real-time gold/currency data from TGJU API
    Aligned with src/data_loader.py cleaning logic
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.last_request_time = 0
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Configure logging"""
        logger = logging.getLogger('TGJUDataFetcher')
        logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
        
        handler = logging.FileHandler(CACHE_DIR.parent / 'logs' / 'api_requests.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        
        return logger
    
    @staticmethod
    def clean_value(x):
        """
        Clean numeric values (same as DataLoader.clean_value)
        """
        if x is None:
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            # Remove HTML tags
            x = re.sub(r'<[^>]+>', '', x)
            # Remove commas and percentage signs
            x = x.replace(',', '').replace('%', '')
            try:
                return float(x)
            except ValueError:
                return 0.0
        return 0.0
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            sleep_time = REQUEST_DELAY_SECONDS - elapsed
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str) -> Optional[Dict]:
        """
        Make HTTP request with retry logic using curl_cffi
        (same approach as DataLoader.get_tgju_history)
        """
        self._rate_limit()
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.tgju.org',
            'Origin': 'https://www.tgju.org',
            'Accept': 'application/json, text/plain, */*',
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                self.logger.debug(f"Request attempt {attempt + 1}/{MAX_RETRIES}: {url}")
                
                # Use curl_cffi to bypass Cloudflare (same as DataLoader)
                response = self.session.get(
                    url, 
                    headers=headers, 
                    timeout=TIMEOUT,
                    impersonate="chrome110"
                )
                response.raise_for_status()
                
                data = response.json()
                
                if SAVE_RAW_RESPONSES:
                    self._save_raw_response(url, data)
                
                self.logger.info(f"✓ Successfully fetched data from {url}")
                return data
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        self.logger.error(f"✗ Failed to fetch data from {url} after {MAX_RETRIES} attempts")
        return None
    
    def _save_raw_response(self, url: str, data: Dict):
        """Save raw API response for debugging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = CACHE_DIR / f"raw_response_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'url': url,
                'data': data
            }, f, ensure_ascii=False, indent=2)
    
    def get_latest_price(self, item_slug: str) -> Optional[float]:
        """
        Fetch latest price for a single item (gold or USD)
        
        Args:
            item_slug: 'geram18' for gold, 'price_dollar_rl' for USD
            
        Returns:
            Latest price or None
        """
        url = f"{TGJU_SUMMARY_ENDPOINT}/{item_slug}"
        response = self._make_request(url)
        
        if not response:
            return None
        
        try:
            # TGJU response structure: {'data': [{'Close': '...', ...}, ...]}
            if 'data' in response and len(response['data']) > 0:
                latest_row = response['data'][0]  # First row is latest
                price_str = latest_row[0] if latest_row else None


                if price_str:
                    price = self.clean_value(price_str)
                    self.logger.info(f"{item_slug}: {price:,.0f}")
                    return price
                

        except (KeyError, ValueError, TypeError, IndexError) as e:
            self.logger.error(f"Error parsing {item_slug} price: {e}")
            self.logger.debug(type(response['data'][0]))

        return None
    
    def get_latest_gold_irr(self) -> Optional[float]:
        """Fetch latest gold price in IRR"""
        return self.get_latest_price(TICKER_GOLD)
    
    def get_latest_usd_irr(self) -> Optional[float]:
        """Fetch latest USD/IRR exchange rate"""
        return self.get_latest_price(TICKER_USD)
    
    def get_latest_global_prices(self) -> Optional[Dict[str, float]]:
        """
        Fetch latest global prices (Gold Ounce, Oil) from Yahoo Finance
        (same as DataLoader.fetch_global_data but for latest only)
        """
        self.logger.info("Fetching latest global prices from Yahoo Finance...")
        
        try:
            # Gold Ounce (GC=F)
            gold = yf.Ticker("GC=F")
            gold_hist = gold.history(period="1d")
            ounce_usd = float(gold_hist['Close'].iloc[-1]) if not gold_hist.empty else None
            
            # Crude Oil (BZ=F)
            oil = yf.Ticker("BZ=F")
            oil_hist = oil.history(period="1d")
            oil_usd = float(oil_hist['Close'].iloc[-1]) if not oil_hist.empty else None
            
            if ounce_usd and oil_usd:
                self.logger.info(f"Gold: ${ounce_usd:.2f}/oz, Oil: ${oil_usd:.2f}/barrel")
                return {'Ounce_USD': ounce_usd, 'Oil_USD': oil_usd}
            else:
                self.logger.error("Failed to fetch global prices")
                return None
                
        except Exception as e:
            self.logger.error(f"Yahoo Finance error: {e}")
            return None
    
    def get_all_latest_prices(self) -> Optional[Dict[str, float]]:
        """
        Fetch all required prices in one batch
        
        Returns:
            Dictionary with Gold_IRR, USD_IRR, Ounce_USD, Oil_USD
        """
        self.logger.info("="*60)
        self.logger.info("Fetching all latest prices...")
        
        # Local prices (TGJU)
        gold_irr = self.get_latest_gold_irr()
        usd_irr = self.get_latest_usd_irr()
        
        # Global prices (Yahoo Finance)
        global_prices = self.get_latest_global_prices()
        
        # Combine
        if all([gold_irr, usd_irr, global_prices]):
            prices = {
                'Gold_IRR': gold_irr,
                'USD_IRR': usd_irr,
                'Ounce_USD': global_prices['Ounce_USD'],
                'Oil_USD': global_prices['Oil_USD'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info("✓ Successfully fetched all prices")
            self.logger.info(f"  Gold IRR: {gold_irr:,.0f}")
            self.logger.info(f"  USD/IRR: {usd_irr:,.0f}")
            self.logger.info(f"  Gold Ounce: ${global_prices['Ounce_USD']:,.2f}")
            self.logger.info(f"  Oil: ${global_prices['Oil_USD']:,.2f}")
            self.logger.info("="*60)
            
            # Cache the prices
            self._cache_prices(prices)
            
            return prices
        else:
            self.logger.error("✗ Failed to fetch some prices")
            return None
    
    def _cache_prices(self, prices: Dict):
        """Save latest prices to cache file"""
        with open(LATEST_PRICES_FILE, 'w') as f:
            json.dump(prices, f, indent=2)
        self.logger.debug(f"Cached prices to {LATEST_PRICES_FILE}")
    
    def get_cached_prices(self) -> Optional[Dict]:
        """Load cached prices (fallback when API fails)"""
        if LATEST_PRICES_FILE.exists():
            with open(LATEST_PRICES_FILE, 'r') as f:
                prices = json.load(f)
            
            # Check if cache is recent (< 1 hour old)
            cached_time = datetime.fromisoformat(prices['timestamp'])
            if datetime.now() - cached_time < timedelta(hours=1):
                self.logger.info("Using cached prices")
                return prices
            else:
                self.logger.warning("Cached prices are stale")
        
        return None


# ==================== UTILITY FUNCTIONS ====================

def test_api_connection():
    """Test if TGJU API is accessible"""
    fetcher = TGJUDataFetcher()
    prices = fetcher.get_all_latest_prices()
    
    if prices:
        print("\n✓ API connection successful!")
        print(f"Gold IRR: {prices['Gold_IRR']:,.0f}")
        print(f"USD/IRR: {prices['USD_IRR']:,.0f}")
        print(f"Gold USD: ${prices['Ounce_USD']:,.2f}")
        print(f"Oil USD: ${prices['Oil_USD']:,.2f}")
        return True
    else:
        print("\n✗ API connection failed")
        print("This is expected in development - will use historical data fallback")
        return False


if __name__ == "__main__":
    test_api_connection()
