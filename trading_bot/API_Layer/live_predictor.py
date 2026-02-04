"""
Live Prediction Module
Loads trained XGBoost model and generates real-time predictions
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

from api_config import MODEL_PATH, PIPELINE_PATH, FEATURE_SCALER_PATH
from live_feature_engineering import LiveFeatureEngineer


class LivePredictor:
    """
    Loads trained model and generates predictions on new data
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.model = None
        self.scaler = None
        self.pipeline = None
        self.feature_engineer = LiveFeatureEngineer()
        
        self.load_model()
    
    def _setup_logger(self):
        logger = logging.getLogger('LivePredictor')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('logs/predictions.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        
        return logger
    
    def load_model(self):
        """Load trained model and preprocessing objects"""
        self.logger.info("Loading trained model...")
        
        try:
            # Try loading full pipeline first (preferred)
            if PIPELINE_PATH.exists():
                with open(PIPELINE_PATH, 'rb') as f:
                    self.pipeline = pickle.load(f)
                self.logger.info(f"✓ Loaded pipeline from {PIPELINE_PATH}")
                return True
            
            # Otherwise load model and scaler separately
            if MODEL_PATH.exists():
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                self.logger.info(f"✓ Loaded model from {MODEL_PATH}")
            else:
                self.logger.error(f"Model file not found: {MODEL_PATH}")
                return False
            
            # Load scaler if exists
            if FEATURE_SCALER_PATH.exists():
                with open(FEATURE_SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.logger.info(f"✓ Loaded scaler from {FEATURE_SCALER_PATH}")
            else:
                self.logger.warning("No scaler found - using raw features")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features_df: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """
        Generate prediction using loaded model
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Tuple of (predicted_return, confidence) or None
        """
        if self.pipeline is None and self.model is None:
            self.logger.error("No model loaded")
            return None
        
        try:
            # Use pipeline if available
            if self.pipeline:
                prediction = self.pipeline.predict(features_df)[0]
            else:
                # Otherwise use model + scaler
                if self.scaler:
                    features_scaled = self.scaler.transform(features_df)
                else:
                    features_scaled = features_df.values
                
                prediction = self.model.predict(features_scaled)[0]
            
            # Calculate confidence (you can enhance this)
            # For now, use absolute value as proxy for confidence
            confidence = min(abs(prediction) / 0.05, 1.0)  # Scale to 0-1
            
            self.logger.info(f"Prediction: {prediction:.4f} (confidence: {confidence:.2f})")
            
            return prediction, confidence
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None
    
    def predict_from_latest_prices(self, 
                                   latest_prices: Dict[str, float]) -> Optional[Dict]:
        """
        Complete prediction pipeline: prices → features → prediction
        
        Args:
            latest_prices: Dict with Gold_IRR, USD_IRR, Ounce_USD, Oil_USD
            
        Returns:
            Dict with prediction results and metadata
        """
        self.logger.info("="*60)
        self.logger.info("Starting prediction pipeline...")
        
        # Step 1: Compute features
        features = self.feature_engineer.compute_features_for_latest(latest_prices)
        
        if features is None:
            self.logger.error("Feature computation failed")
            return None
        
        # Step 2: Convert to DataFrame
        features_df = self.feature_engineer.get_features_as_dataframe(features)
        
        # Step 3: Generate prediction
        result = self.predict(features_df)
        
        if result is None:
            return None
        
        predicted_return, confidence = result
        
        # Step 4: Calculate predicted price
        current_price = latest_prices['Gold_IRR']
        predicted_price = current_price * (1 + predicted_return)
        
        # Step 5: Compile results
        prediction_result = {
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'predicted_return': predicted_return,
            'predicted_return_pct': predicted_return * 100,
            'predicted_price': predicted_price,
            'price_change': predicted_price - current_price,
            'confidence': confidence,
            'features': {
                'RSI_14': features['RSI_14'],
                'MACD': features['MACD'],
                'SMA_7': features['SMA_7'],
                'SMA_30': features['SMA_30']
            }
        }
        
        self.logger.info(f"Current Price: {current_price:,.0f} IRR")
        self.logger.info(f"Predicted Return: {predicted_return*100:.2f}%")
        self.logger.info(f"Predicted Price: {predicted_price:,.0f} IRR")
        self.logger.info("="*60)
        
        return prediction_result


# ==================== TESTING ====================

if __name__ == "__main__":
    import json
    
    predictor = LivePredictor()
    
    # Test with sample prices
    test_prices = {
        'Gold_IRR': 11500000.0,
        'USD_IRR': 260000.0,
        'Ounce_USD': 1850.0,
        'Oil_USD': 68.5
    }
    
    result = predictor.predict_from_latest_prices(test_prices)
    
    if result:
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("="*60)
    else:
        print("Prediction failed")
