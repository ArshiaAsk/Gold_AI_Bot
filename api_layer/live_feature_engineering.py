"""
Real-Time Feature Engineering
Computes the same technical indicators as training pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

from data_fetcher import TGJUDataFetcher
from api_config import LOOKBACK_DAYS, CACHE_DIR, LATEST_FEATURES_FILE


class LiveFeatureEngineer:
    """
    Computes technical indicators in real-time
    Must match exactly the features used during training
    """
    
    def __init__(self):
        self.fetcher = TGJUDataFetcher()
        self.logger = self._setup_logger()
        self.feature_columns = [
            'SMA_7', 'SMA_30', 'MACD', 'MACD_Signal', 'RSI_14',
            'Bollinger_Upper', 'Bollinger_Lower',
            'Gold_LogRet', 'USD_LogRet', 'Ounce_LogRet', 'Oil_LogRet',
            'Gold_LogRet_Lag_1', 'USD_LogRet_Lag_1', 'Ounce_LogRet_Lag_1',
            'Gold_LogRet_Lag_2', 'USD_LogRet_Lag_2', 'Ounce_LogRet_Lag_2',
            'Gold_LogRet_Lag_3', 'USD_LogRet_Lag_3', 'Ounce_LogRet_Lag_3',
            'Gold_LogRet_Lag_7', 'USD_LogRet_Lag_7', 'Ounce_LogRet_Lag_7'
        ]
    
    def _setup_logger(self):
        logger = logging.getLogger('LiveFeatureEngineer')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('logs/feature_engineering.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        
        return logger
    
    def calculate_sma(self, series: pd.Series, window: int) -> float:
        """Calculate Simple Moving Average"""
        if len(series) < window:
            self.logger.warning(f"Insufficient data for SMA_{window}")
            return series.iloc[-1]  # Fallback to last price
        return series.iloc[-window:].mean()
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(series) < period + 1:
            return 50.0  # Neutral
        
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.iloc[-period:].mean()
        avg_loss = loss.iloc[-period:].mean()
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, series: pd.Series, 
                       fast: int = 12, 
                       slow: int = 26, 
                       signal: int = 9) -> tuple:
        """Calculate MACD and Signal Line"""
        if len(series) < slow:
            return 0.0, 0.0
        
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return macd.iloc[-1], signal_line.iloc[-1]
    
    def calculate_bollinger_bands(self, series: pd.Series, 
                                  window: int = 20, 
                                  num_std: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        if len(series) < window:
            price = series.iloc[-1]
            return price * 1.02, price * 0.98  # Fallback
        
        sma = series.iloc[-window:].mean()
        std = series.iloc[-window:].std()
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return upper, lower
    
    def calculate_log_returns(self, series: pd.Series) -> pd.Series:
        """Calculate log returns"""
        return np.log(series / series.shift(1))
    
    def get_historical_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Fetch and combine historical data for all assets
        Returns DataFrame similar to training data structure
        """
        self.logger.info(f"Fetching {LOOKBACK_DAYS} days of historical data...")
        
        # Fetch historical data for each asset
        # NOTE: This assumes TGJU has historical endpoints
        # You may need to adjust based on actual API capabilities
        
        # For now, we'll create a simplified version
        # In production, fetch from TGJU or use cached CSV
        
        try:
            # Option 1: Load from saved CSV (most reliable for now)
            csv_path = '/home/claude/trading_bot/advanced_gold_features.csv'
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Get last LOOKBACK_DAYS
            df = df.tail(LOOKBACK_DAYS).copy()
            
            self.logger.info(f"✓ Loaded {len(df)} historical records")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            return None
    
    def compute_features_for_latest(self, 
                                    latest_prices: Dict[str, float]) -> Optional[Dict]:
        """
        Compute all features for the latest price point
        
        Args:
            latest_prices: Dict with Gold_IRR, USD_IRR, Ounce_USD, Oil_USD
            
        Returns:
            Dict with all feature values
        """
        self.logger.info("Computing features for latest prices...")
        
        # Get historical data
        hist_df = self.get_historical_dataframe()
        
        if hist_df is None:
            self.logger.error("Cannot compute features without historical data")
            return None
        
        # Append latest prices to historical data
        new_row = {
            'Date': datetime.now(),
            'Gold_IRR': latest_prices['Gold_IRR'],
            'USD_IRR': latest_prices['USD_IRR'],
            'Ounce_USD': latest_prices['Ounce_USD'],
            'Oil_USD': latest_prices['Oil_USD']
        }
        
        # Create temporary extended dataframe
        df = pd.concat([hist_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Calculate log returns
        for col in ['Gold_IRR', 'USD_IRR', 'Ounce_USD', 'Oil_USD']:
            df[f'{col}_LogRet'] = self.calculate_log_returns(df[col])
        
        # Get the last row (latest data point)
        latest_idx = len(df) - 1
        
        # Calculate technical indicators using all historical data
        gold_prices = df['Gold_IRR']
        
        features = {
            # Moving Averages
            'SMA_7': self.calculate_sma(gold_prices, 7),
            'SMA_30': self.calculate_sma(gold_prices, 30),
            
            # MACD
            'MACD': 0.0,
            'MACD_Signal': 0.0,
            
            # RSI
            'RSI_14': self.calculate_rsi(gold_prices, 14),
            
            # Bollinger Bands
            'Bollinger_Upper': 0.0,
            'Bollinger_Lower': 0.0,
            
            # Log Returns
            'Gold_LogRet': df['Gold_IRR_LogRet'].iloc[-1] if not pd.isna(df['Gold_IRR_LogRet'].iloc[-1]) else 0.0,
            'USD_LogRet': df['USD_IRR_LogRet'].iloc[-1] if not pd.isna(df['USD_IRR_LogRet'].iloc[-1]) else 0.0,
            'Ounce_LogRet': df['Ounce_USD_LogRet'].iloc[-1] if not pd.isna(df['Ounce_USD_LogRet'].iloc[-1]) else 0.0,
            'Oil_LogRet': df['Oil_USD_LogRet'].iloc[-1] if not pd.isna(df['Oil_USD_LogRet'].iloc[-1]) else 0.0,
        }
        
        # Calculate MACD
        features['MACD'], features['MACD_Signal'] = self.calculate_macd(gold_prices)
        
        # Calculate Bollinger Bands
        features['Bollinger_Upper'], features['Bollinger_Lower'] = \
            self.calculate_bollinger_bands(gold_prices)
        
        # Lagged returns
        for lag in [1, 2, 3, 7]:
            for asset in ['Gold', 'USD', 'Ounce']:
                lag_col = f'{asset}_LogRet_Lag_{lag}'
                ret_col = f'{asset}_IRR_LogRet' if asset != 'Ounce' else 'Ounce_USD_LogRet'
                
                if lag < len(df):
                    lag_val = df[ret_col].iloc[-(lag + 1)]
                    features[lag_col] = lag_val if not pd.isna(lag_val) else 0.0
                else:
                    features[lag_col] = 0.0
        
        # Add metadata
        features['timestamp'] = datetime.now().isoformat()
        features['Gold_IRR'] = latest_prices['Gold_IRR']
        
        self.logger.info("✓ Features computed successfully")
        
        # Cache features
        import json
        with open(LATEST_FEATURES_FILE, 'w') as f:
            json.dump(features, f, indent=2)
        
        return features
    
    def get_features_as_dataframe(self, features: Dict) -> pd.DataFrame:
        """
        Convert feature dict to DataFrame format expected by model
        
        Args:
            features: Dict of feature values
            
        Returns:
            Single-row DataFrame with correct column order
        """
        # Extract only the features used by model (not metadata)
        model_features = {k: v for k, v in features.items() 
                         if k in self.feature_columns}
        
        df = pd.DataFrame([model_features])
        
        # Ensure correct column order
        df = df[self.feature_columns]
        
        return df


# ==================== TESTING ====================

if __name__ == "__main__":
    import json
    
    engineer = LiveFeatureEngineer()
    
    # Test with sample latest prices
    test_prices = {
        'Gold_IRR': 11500000.0,
        'USD_IRR': 260000.0,
        'Ounce_USD': 1850.0,
        'Oil_USD': 68.5
    }
    
    features = engineer.compute_features_for_latest(test_prices)
    
    if features:
        print("✓ Feature engineering successful!")
        print(json.dumps(features, indent=2))
        
        # Convert to DataFrame
        df = engineer.get_features_as_dataframe(features)
        print("\nDataFrame shape:", df.shape)
        print(df.head())
    else:
        print("✗ Feature engineering failed")
