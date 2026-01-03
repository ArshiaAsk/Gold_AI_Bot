"""
FastAPI Prediction Service for Gold Price LSTM Model
"""
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from typing import Dict, List, Optional
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class GoldPricePredictor:
    """Production-ready predictor for gold prices"""

    def __init__(self, model_path: str, scaler_X_path: str, scaler_y_path: str):
        """  
        Initialize predictor with trained model and scalers
        
        Args:
            model_path: Path to trained Keras model
            scaler_X_path: Path to feature scaler
            scaler_y_path: Path to target scaler
        """ 
        self.model_path = model_path
        self.scaler_X_path = scaler_X_path
        self.scaler_y_path = scaler_y_path

        # Load model and scaler
        self._load_model()
        self._load_scaler()

        logger.info("GoldPricePredictor initialize successfully")

    
    def _load_model(self):
        """Load trained Keras model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"Model loaded form {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


    def _load_scaler(self):
        """Load fitted scalers"""
        try:
            self.scaler_X = joblib.load(self.scaler_X_path)
            self.scaler_y = joblib.load(self.scaler_y_path)
            logger.info(f"Scalers loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load scalers: {e}")
            raise


    def _validate_features(self, features: np.ndarray, expected_shape: tuple) -> bool:
        """
        Validate input features
        
        Args:
            features: Input feature array
            expected_shape: Expected shape (sequence_length, n_features)
            
        Returns:
            True if valid
        """
        if features.shape != expected_shape:
            raise ValueError(
                f"Invalid feature shape. Expected {expected_shape}, got {features.shape}"
            )
        return True
    

    def predict_log_return(self, features: np.ndarray) -> float:
        """
        Predict next day log return
        
        Args:
            features: Feature array of shape (sequence_length, n_features)
            
        Returns:
            Predicted log return
        """
        # Validate
        expected_shape = (30, 15)   # (sequence_length, n_features)
        self._validate_features(features, expected_shape)

        # Scale features
        features_2d = features.reshape(-1, features.shape[-1])
        features_scaled = self.scaler_X.transform(features_2d)
        features_scaled = features_scaled.reshape(1, *expected_shape)

        # predict
        log_return_scaled = self.model.predict(features_scaled, verbose=0)
        log_return = self.scaler_y.inverse_transform(log_return_scaled)[0, 0]

        return float(log_return)
    

    def predict_price(self, features: np.ndarray, current_price: float) -> Dict[str, float]:
        """
        Predict next day price
        
        Args:
            features: Feature array of shape (sequence_length, n_features)
            current_price: Current gold price
            
        Returns:
            Dictionary with prediction details
        """
        # Get log return prediction
        log_return = self.predict_log_return(features)

        # Caculate predicted price
        predicted_price = current_price * np.exp(log_return)
        price_change = predicted_price - current_price
        price_change_pct = (np.exp(log_return) - 1) * 100

        return {
            'current_pirce': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change': float(price_change),
            'price_change_percent': float(price_change_pct),
            'predicted_log_return': float(log_return),
            'prediction_timestamp': datetime.now().isoformat()
        }  
    

    def predict_with_confidence(
        self,
        features: np.ndarray,
        current_price: float,
        n_simulations: int = 100
    ) -> Dict [str, float]:
        """
        Predict with Monte Carlo confidence intervals
        
        Args:
            features: Feature array
            current_price: Current price
            n_simulations: Number of MC simulations
            
        Returns:
            Prediction with confidence intervals
        """
        predictions = []

        for _ in range(n_simulations):
            result = self.predict_price(features, current_price)
            predictions.append(result['predict_price'])

        predictions = np.ndarray(predictions)

        return {
            'current_price': float(current_price),
            'predicted_price': float(np.mean(predictions)),
            'confidence_interval_95': {
                'lower': float(np.percentile(predictions, 2.5)),
                'upper': float(np.percentile(predictions, 97.5))
            },
            'std_dev': float(np.std(predictions)),
            'prediction_timestamp': datetime.now().isoformat()
        }
    


class FeatureBuilder:
    """Build features from raw market data"""

    def __init__(self, sequence_length: int = 30):
        """
        Initialize feature builder
        
        Args:
            sequence_length: Number of time steps to use
        """
        self.sequence_length = sequence_length

    
    def build_features_from_history(self, history_df: pd.DataFrame) -> np.ndarray:
        """
        Build feature array from historical data
        
        Args:
            history_df: DataFrame with historical data (most recent last)
                       Must have at least sequence_length rows
            
        Returns:
            Feature array of shape (sequence_length, n_features)
        """
        if len(history_df) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} historical records, "
                f"got {len(history_df)}"
            )
        
        # Get last sequence_length rows
        recent_data = history_df.tail(self.sequence_length)

        # Expected feature columns (same as training)
        feature_columns = [
            'Gold_LogRet', 'USD_LogRet', 'Ounce_LogRet', 'Oil_LogRet',
            'SMA_7', 'RSI_14', 'MACD', 'MACD_Signal',
            'Bollinger_Upper', 'Bollinger_Lower',
            'Gold_LogRet_Lag_1', 'Gold_LogRet_Lag_2', 'Gold_LogRet_Lag_3',
            'USD_LogRet_Lag_1', 'USD_LogRet_Lag_2'
        ]

        # Extract features
        features = recent_data[feature_columns].values

        return features