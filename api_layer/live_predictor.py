# api_layer/live_predictor.py (FIXED MODEL LOADING)

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict

# TensorFlow imports with proper error handling
try:
    import tensorflow as tf
    import keras
    print(f"TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    tf = None
    keras = None
from live_feature_engineering import LiveFeatureEngineer
from api_layer.api_config import (
    MODEL_PATH, SCALER_PATH, HISTORICAL_DATA_PATH,
    FEATURE_COLUMNS, LOOKBACK_DAYS, CACHE_DIR
)


class LivePredictor:
    """
    Real-time predictor using trained LSTM model
    """
    
    def __init__(self):
        self.feature_engineering = LiveFeatureEngineer()
        self.model = None
        self.scaler = None
        self.feature_columns = FEATURE_COLUMNS
        self.lookback = LOOKBACK_DAYS
        self.logger = self._setup_logger()
        
        # Load model and scaler
        self.load_model()
        self.load_scaler()
    
    def _setup_logger(self):
        logger = logging.getLogger('LivePredictor')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(CACHE_DIR.parent / 'logs' / 'predictions.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        if len(logger.handlers) < 2:
            logger.addHandler(console)
        
        return logger
    
    def load_model(self):
        """
        Load trained LSTM model with proper error handling
        """
        self.logger.info("Loading trained model...")
        
        if not MODEL_PATH.exists():
            self.logger.error(f"Model file not found: {MODEL_PATH}")
            return False
        
        try:
            # Try loading with compile=False (safer)
            self.model = keras.models.load_model(
                str(MODEL_PATH),
                compile=False
            )
            
            # Manually compile if needed
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            self.logger.info(f"✓ Model loaded: {MODEL_PATH}")
            self.logger.info(f"  Input shape: {self.model.input_shape}")
            self.logger.info(f"  Output shape: {self.model.output_shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.logger.info("Attempting alternative loading methods...")
            
            # Try alternative: Load weights only
            try:
                # Reconstruct model architecture (from your training code)
                self.model = self._build_model_architecture()
                self.model.load_weights(str(MODEL_PATH))
                self.logger.info("✓ Model weights loaded successfully")
                return True
            except Exception as e2:
                self.logger.error(f"Alternative loading also failed: {e2}")
                return False
    
    def _build_model_architecture(self):
        """
        Rebuild LSTM model architecture (must match training exactly)
        Based on your Phase 2 training code
        """
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        
        from config.settings import config as main_config
        n_features = len(self.feature_columns)
        
        model = Sequential([
            LSTM(main_config.model.LSTM_UNITS_1, return_sequences=True, input_shape=(self.lookback, n_features)),
            Dropout(main_config.model.DROPOUT_RATE),
            LSTM(main_config.model.LSTM_UNITS_2, return_sequences=False),
            Dropout(main_config.model.DROPOUT_RATE),
            Dense(main_config.model.DENSE_UNITS, activation='relu'),
            Dense(1)  # Predicting next-day log return
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def load_scaler(self):
        """Load fitted scaler"""
        if not SCALER_PATH.exists():
            self.logger.warning(f"Scaler not found: {SCALER_PATH}")
            self.logger.warning("Will use raw features (not recommended)")
            return False
        
        try:
            import joblib
            self.scaler = joblib.load(SCALER_PATH)
            self.logger.info(f"✓ Scaler loaded: {SCALER_PATH}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load scaler: {e}")
            return False
    
    def get_historical_features(self, days: int = None) -> Optional[pd.DataFrame]:
        """
        Load recent historical features for sequence building
        
        Args:
            days: Number of days to load (default: LOOKBACK_DAYS)
        
        Returns:
            DataFrame with features or None
        """
        if days is None:
            days = self.lookback
        
        self.logger.info(f"Fetching {days} days of historical data...")
        
        if not HISTORICAL_DATA_PATH.exists():
            self.logger.error(f"Historical data not found: {HISTORICAL_DATA_PATH}")
            return None
        
        try:
            df = pd.read_csv(HISTORICAL_DATA_PATH, parse_dates=['Date'])
            df = df.sort_values('Date')
            
            # Get last N days
            recent = df.tail(days).copy()
            
            # Ensure all feature columns exist
            missing = set(self.feature_columns) - set(recent.columns)
            if missing:
                self.logger.error(f"Missing features: {missing}")
                return None
            
            self.logger.info(f"✓ Loaded {len(recent)} historical records")
            return recent[['Date'] + self.feature_columns]
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            return None
    
    def prepare_sequence(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Prepare feature sequence for LSTM input
        
        Args:
            features_df: DataFrame with shape (lookback, n_features)
        
        Returns:
            Array with shape (1, lookback, n_features) or None
        """
        try:
            # Extract feature values
            X = features_df[self.feature_columns].values
            
            # Check shape
            if X.shape[0] != self.lookback:
                self.logger.error(
                    f"Wrong sequence length: expected {self.lookback}, got {X.shape[0]}"
                )
                return None
            
            # Scale features
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X  # Use raw features (not ideal)
            
            # Reshape for LSTM: (1, lookback, n_features)
            X_lstm = X_scaled.reshape(1, self.lookback, len(self.feature_columns))
            
            return X_lstm
            
        except Exception as e:
            self.logger.error(f"Error preparing sequence: {e}")
            return None
    
    def predict(self, features_df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """
        Make prediction on feature sequence
        
        Args:
            features_df: Recent features (must have exactly lookback rows)
        
        Returns:
            Dictionary with prediction results or None
        """
        if self.model is None:
            self.logger.error("No model loaded")
            return None
        
        # Prepare input sequence
        X_lstm = self.prepare_sequence(features_df)
        
        if X_lstm is None:
            return None
        
        try:
            # Make prediction
            pred_log_return = self.model.predict(X_lstm, verbose=0)[0, 0]
            
            # Convert to percentage
            pred_return_pct = pred_log_return * 100
            
            # Estimate confidence (you can improve this)
            confidence = self._estimate_confidence(pred_log_return)
            
            
            result = {
                'predicted_log_return': float(pred_log_return),
                'predicted_return_pct': float(pred_return_pct),
                'current_price': float(current_price),
                'confidence': float(confidence),
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(
                f"Prediction: {pred_return_pct:+.2f}% "
                f"(log: {pred_log_return:+.4f}) | "
                f"Confidence: {confidence:.1%}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None
    
    def _estimate_confidence(self, pred_log_return: float) -> float:
        """
        Estimate prediction confidence based on magnitude
        Simple heuristic - you can improve this with ensemble/variance
        """
        abs_return = abs(pred_log_return)
        
        if abs_return > 0.02:  # > 2% move
            return 0.85
        elif abs_return > 0.01:  # 1-2%
            return 0.75
        elif abs_return > 0.005:  # 0.5-1%
            return 0.65
        else:  # < 0.5%
            return 0.55
    
    def _build_live_feature_sequence(
        self,
        latest_prices: Dict[str, float],
        latest_features: Optional[Dict] = None
    ) -> Optional[pd.DataFrame]:
        """
        Build lookback sequence by combining historical features with the latest live feature row.
        """
        history_days = self.lookback - 1
        hist_df = self.get_historical_features(days=history_days)
        if hist_df is None or len(hist_df) != history_days:
            self.logger.error(
                f"Insufficient historical rows for live sequence. "
                f"Expected {history_days}, got {0 if hist_df is None else len(hist_df)}"
            )
            return None

        if latest_features is None:
            latest_features = self.feature_engineering.compute_features_for_latest(latest_prices)
        if latest_features is None:
            self.logger.error("Failed to compute latest live features")
            return None

        latest_row = {'Date': pd.Timestamp(datetime.now())}
        for col in self.feature_columns:
            latest_row[col] = latest_features.get(col, 0.0)

        live_df = pd.concat([hist_df[['Date'] + self.feature_columns], pd.DataFrame([latest_row])], ignore_index=True)
        return live_df

    def predict_from_latest_data(self, latest_prices, latest_features: Optional[Dict] = None) -> Optional[Dict]:
        """
        Full pipeline: load recent data and predict
        """
        self.logger.info("="*60)
        self.logger.info("Starting prediction pipeline...")
        
        if latest_prices is None:
            self.logger.error("latest_prices is required")
            return None

        features_df = self._build_live_feature_sequence(latest_prices, latest_features=latest_features)
        if features_df is None:
            self.logger.error("Failed to build live feature sequence")
            return None
        
        # Make prediction
        result = self.predict(features_df, current_price=float(latest_prices['Gold_IRR']))
        
        if result:
            # Cache the prediction
            cache_file = CACHE_DIR / 'latest_prediction.json'
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            self.logger.info(f"Cached prediction to {cache_file}")
        
        self.logger.info("="*60)
        return result


# ==================== TEST FUNCTION ====================

def test_predictor():
    """Test the predictor with historical data"""
    print("\n" + "="*60)
    print("TESTING LIVE PREDICTOR")
    print("="*60)
    
    predictor = LivePredictor()
    
    if predictor.model is None:
        print("\n✗ Model loading failed")
        print("\nTroubleshooting:")
        print("1. Check TensorFlow version:")
        print("   pip show tensorflow")
        print("2. Try reinstalling TensorFlow:")
        print("   pip install --upgrade tensorflow")
        print("3. Retrain model if version mismatch")
        return False
    
    print("\n✓ Model loaded successfully")
    
    # Test prediction
    latest_prices = {
        "Gold_IRR": 192347000.0,
        "USD_IRR": 1609350.0,
        "Ounce_USD": 4951.2001953125,
        "Oil_USD": 68.05000305175781,
        "timestamp": "2026-02-07T14:54:23.055527"
        }
    
    result = predictor.predict_from_latest_data(latest_prices)
    
    if result:
        print("\n✓ Prediction successful:")
        print(f"  Predicted Return: {result['predicted_return_pct']:+.2f}%")
        print(f"  Confidence: {result['confidence']:.1%}")
        if result['current_price']:
            print(f"  Current Price: {result['current_price']:,.0f} IRR")
        return True
    else:
        print("\n✗ Prediction failed")
        return False


if __name__ == "__main__":
    test_predictor()
