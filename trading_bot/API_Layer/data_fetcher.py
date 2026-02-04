"""
Real-Time Data Fetcher for TGJU API
Handles all API requests with rate limiting, caching, and error handling
"""

import requests
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path
import pandas as pd

from api_config import (
    GOLD_IRR_ENDPOINT, USD_IRR_ENDPOINT, GOLD_USD_ENDPOINT,
    HISTORICAL_ENDPOINT, MAX_RETRIES, RETRY_DELAY, TIMEOUT,
    REQUEST_DELAY_SECONDS, CACHE_DIR, LATEST_PRICES_FILE,
    DEBUG_MODE, SAVE_RAW_RESPONSES, TGJU_API_KEY
)


class TGJUDataFetcher:
    """
    Fetches real-time and historical gold/currency data from TGJU API
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        
        if TGJU_API_KEY:
            self.session.headers.update({'Authorization': f'Bearer {TGJU_API_KEY}'})
        
        self.last_request_time = 0
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """Configure logging"""
        logger = logging.getLogger('TGJUDataFetcher')
        logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
        
        handler = logging.FileHandler('logs/api_requests.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        
        return logger
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            sleep_time = REQUEST_DELAY_SECONDS - elapsed
            self.logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """
        Make HTTP request with retry logic
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response or None on failure
        """
        self._rate_limit()
        
        for attempt in range(MAX_RETRIES):
            try:
                self.logger.debug(f"Request attempt {attempt + 1}/{MAX_RETRIES}: {url}")
                
                response = self.session.get(url, params=params, timeout=TIMEOUT)
                response.raise_for_status()
                
                data = response.json()
                
                if SAVE_RAW_RESPONSES:
                    self._save_raw_response(url, data)
                
                self.logger.info(f"✓ Successfully fetched data from {url}")
                return data
                
            except requests.exceptions.Timeout:
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decode error: {e}")
                return None
        
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
    
    def get_latest_gold_irr(self) -> Optional[float]:
        """
        Fetch latest gold price in IRR (per gram)
        
        Returns:
            Price in IRR or None
        """
        # TGJU API typically returns data in format:
        # {'geram18': {'p': '10500000', 'dt': '1404/11/15 14:30:00', ...}}
        
        response = self._make_request(GOLD_IRR_ENDPOINT)
        
        if not response:
            return None
        
        try:
            # Parse TGJU response structure (adjust based on actual API format)
            # This is a placeholder - adjust based on real API response
            gold_data = response.get('geram18', {})
            price_str = gold_data.get('p', None)
            
            if price_str:
                price = float(price_str.replace(',', ''))
                self.logger.info(f"Gold IRR: {price:,.0f}")
                return price
            
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing gold IRR price: {e}")
        
        return None
    
    def get_latest_usd_irr(self) -> Optional[float]:
        """
        Fetch latest USD/IRR exchange rate
        
        Returns:
            Exchange rate or None
        """
        response = self._make_request(USD_IRR_ENDPOINT)
        
        if not response:
            return None
        
        try:
            # Adjust based on actual TGJU response format
            usd_data = response.get('price_dollar_rl', {})
            price_str = usd_data.get('p', None)
            
            if price_str:
                price = float(price_str.replace(',', ''))
                self.logger.info(f"USD/IRR: {price:,.0f}")
                return price
                
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing USD/IRR rate: {e}")
        
        return None
    
    def get_latest_gold_usd(self) -> Optional[float]:
        """
        Fetch latest gold price in USD (per ounce)
        
        Returns:
            Price in USD or None
        """
        response = self._make_request(GOLD_USD_ENDPOINT)
        
        if not response:
            return None
        
        try:
            # Adjust based on actual TGJU response format
            gold_data = response.get('gold', {})
            price_str = gold_data.get('p', None)
            
            if price_str:
                price = float(price_str.replace(',', ''))
                self.logger.info(f"Gold USD: ${price:,.2f}/oz")
                return price
                
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing gold USD price: {e}")
        
        return None
    
    def get_latest_oil_usd(self) -> Optional[float]:
        """
        Fetch latest WTI oil price in USD
        
        Returns:
            Price in USD or None
        """
        response = self._make_request(GOLD_USD_ENDPOINT)  # Oil might be in same endpoint
        
        if not response:
            return None
        
        try:
            oil_data = response.get('oil', {})
            price_str = oil_data.get('p', None)
            
            if price_str:
                price = float(price_str.replace(',', ''))
                self.logger.info(f"Oil USD: ${price:,.2f}/barrel")
                return price
                
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Error parsing oil price: {e}")
        
        return None
    
    def get_all_latest_prices(self) -> Optional[Dict[str, float]]:
        """
        Fetch all required prices in one batch
        
        Returns:
            Dictionary with all prices or None
        """
        self.logger.info("Fetching all latest prices...")
        
        prices = {
            'Gold_IRR': self.get_latest_gold_irr(),
            'USD_IRR': self.get_latest_usd_irr(),
            'Ounce_USD': self.get_latest_gold_usd(),
            'Oil_USD': self.get_latest_oil_usd(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if all prices were fetched successfully
        if all(prices[k] is not None for k in ['Gold_IRR', 'USD_IRR', 'Ounce_USD', 'Oil_USD']):
            self.logger.info("✓ Successfully fetched all prices")
            
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
    
    def get_historical_data(self, 
                           symbol: str, 
                           days: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for technical indicator calculation
        
        Args:
            symbol: 'gold', 'usd', 'oil', etc.
            days: Number of days to fetch
            
        Returns:
            DataFrame with historical data or None
        """
        self.logger.info(f"Fetching {days} days of historical data for {symbol}")
        
        # TGJU historical endpoint typically requires symbol and date range
        params = {
            'symbol': symbol,
            'days': days
        }
        
        response = self._make_request(HISTORICAL_ENDPOINT, params=params)
        
        if not response:
            return None
        
        try:
            # Parse historical data (adjust based on actual API format)
            # Expected format: list of [timestamp, price] pairs
            data_points = response.get('data', [])
            
            df = pd.DataFrame(data_points, columns=['Date', 'Price'])
            df['Date'] = pd.to_datetime(df['Date'])
            df['Price'] = pd.to_numeric(df['Price'])
            
            self.logger.info(f"✓ Fetched {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing historical data: {e}")
            return None


# ==================== UTILITY FUNCTIONS ====================

def test_api_connection():
    """Test if TGJU API is accessible"""
    fetcher = TGJUDataFetcher()
    prices = fetcher.get_all_latest_prices()
    
    if prices:
        print("✓ API connection successful!")
        print(f"Gold IRR: {prices['Gold_IRR']:,.0f}")
        print(f"USD/IRR: {prices['USD_IRR']:,.0f}")
        print(f"Gold USD: ${prices['Ounce_USD']:,.2f}")
        print(f"Oil USD: ${prices['Oil_USD']:,.2f}")
        return True
    else:
        print("✗ API connection failed")
        return False


if __name__ == "__main__":
    # Test the data fetcher
    test_api_connection()
