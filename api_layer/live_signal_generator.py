"""
Live Signal Generator
Converts ML predictions into actionable trading signals
Uses same logic as backtest signal generator
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from enum import Enum

from live_feature_engineering import LiveFeatureEngineer
from api_config import (
    BUY_THRESHOLD, SELL_THRESHOLD, 
    RSI_OVERSOLD, RSI_OVERBOUGHT,
    MIN_SIGNAL_CONFIDENCE
)


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class LiveSignalGenerator:
    """
    Generates trading signals from ML predictions and technical indicators
    """
    
    def __init__(self):
        self.feature_engineering = LiveFeatureEngineer()
        self.logger = self._setup_logger()
        self.signal_history = []
    
    def _setup_logger(self):
        logger = logging.getLogger('LiveSignalGenerator')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('logs/signals.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
        
        return logger
    
    def generate_signal(self, 
                       prediction_result: Dict,
                       ) -> Dict:
        """
        Generate trading signal from prediction
        
        Args:
            prediction_result: Output from LivePredictor
            
        Returns:
            Signal dictionary with action, reasoning, etc.
        """
        self.logger.info("Generating trading signal...")
        
        # Extract key values
        predicted_log_return = prediction_result['predicted_log_return']
        predicted_return_pct = prediction_result['predicted_return_pct']
        confidence = prediction_result['confidence']
        current_price = prediction_result['current_price']
        
        # Extract technical indicators
        # features = prediction_result['features']
        features = self.feature_engineering.get_cached_features() 
        rsi = features['RSI_14']
        macd = features['MACD']
        sma_7 = features['SMA_7']
        sma_30 = features['SMA_30']
        
        # Initialize signal
        signal = {
            'timestamp': datetime.now().isoformat(),
            'action': SignalType.HOLD.value,
            'predicted_log_return': predicted_log_return,
            'predicted_return_pct': predicted_return_pct * 100,
            'confidence': confidence,
            'current_price': current_price,
            'reasoning': [],
            'technical_indicators': {
                'RSI_14': rsi,
                'MACD': macd,
                'SMA_7': sma_7,
                'SMA_30': sma_30,
                'price_vs_sma7': ((current_price - sma_7) / sma_7) * 100,
                'price_vs_sma30': ((current_price - sma_30) / sma_30) * 100
            }
        }
        
        # Check confidence threshold
        if confidence < MIN_SIGNAL_CONFIDENCE:
            signal['reasoning'].append(f"Low confidence ({confidence:.2f} < {MIN_SIGNAL_CONFIDENCE})")
            self.logger.info(f"⚠️  HOLD - Low confidence signal")
            return signal
        
        # BUY Signal Logic
        if predicted_log_return >= BUY_THRESHOLD:
            buy_score = 0
            
            # Primary: Prediction is bullish
            signal['reasoning'].append(f"Predicted return {predicted_log_return*100:.2f}% > threshold {BUY_THRESHOLD*100:.2f}%")
            buy_score += 3
            
            # Supporting: RSI not overbought
            if rsi < RSI_OVERBOUGHT:
                signal['reasoning'].append(f"RSI {rsi:.1f} not overbought")
                buy_score += 2
            else:
                signal['reasoning'].append(f"⚠️  RSI {rsi:.1f} overbought - caution")
            
            # Supporting: Price above SMA7 (momentum)
            if current_price > sma_7:
                signal['reasoning'].append(f"Price above SMA7 (uptrend)")
                buy_score += 1
            
            # Supporting: MACD positive
            if macd > 0:
                signal['reasoning'].append(f"MACD {macd:,.0f} positive")
                buy_score += 1
            
            # Final decision
            if buy_score >= 4:  # Need at least 4 points to trigger BUY
                signal['action'] = SignalType.BUY.value
                signal['strength'] = buy_score / 7  # Normalize to 0-1
                self.logger.info(f"🟢 BUY Signal (strength: {signal['strength']:.2f})")
            else:
                signal['reasoning'].append(f"Insufficient confirmation (score: {buy_score}/7)")
                self.logger.info(f"⚠️  HOLD - Buy signal not confirmed")
        
        # SELL Signal Logic
        elif predicted_log_return <= SELL_THRESHOLD:
            sell_score = 0
            
            # Primary: Prediction is bearish
            signal['reasoning'].append(f"Predicted return {predicted_log_return*100:.2f}% < threshold {SELL_THRESHOLD*100:.2f}%")
            sell_score += 3
            
            # Supporting: RSI not oversold
            if rsi > RSI_OVERSOLD:
                signal['reasoning'].append(f"RSI {rsi:.1f} not oversold")
                sell_score += 2
            else:
                signal['reasoning'].append(f"⚠️  RSI {rsi:.1f} oversold - caution")
            
            # Supporting: Price below SMA7 (downtrend)
            if current_price < sma_7:
                signal['reasoning'].append(f"Price below SMA7 (downtrend)")
                sell_score += 1
            
            # Supporting: MACD negative
            if macd < 0:
                signal['reasoning'].append(f"MACD {macd:,.0f} negative")
                sell_score += 1
            
            # Final decision
            if sell_score >= 4:
                signal['action'] = SignalType.SELL.value
                signal['strength'] = sell_score / 7
                self.logger.info(f"🔴 SELL Signal (strength: {signal['strength']:.2f})")
            else:
                signal['reasoning'].append(f"Insufficient confirmation (score: {sell_score}/7)")
                self.logger.info(f"⚠️  HOLD - Sell signal not confirmed")
        
        # HOLD Signal (no strong prediction)
        else:
            signal['reasoning'].append(f"Predicted return {predicted_log_return*100:.2f}% in neutral zone")
            signal['strength'] = 0.0
            self.logger.info(f"⚪ HOLD Signal - Neutral prediction")
        
        # Add to history
        self.signal_history.append(signal)
        
        # Log detailed reasoning
        self.logger.info("Signal Reasoning:")
        for reason in signal['reasoning']:
            self.logger.info(f"  • {reason}")
        
        return signal
    
    def get_signal_summary(self, signal: Dict) -> str:
        """
        Generate human-readable signal summary
        
        Args:
            signal: Signal dictionary
            
        Returns:
            Formatted summary string
        """
        action = signal['action']
        pred_ret = signal['predicted_return_pct']
        conf = signal['confidence']
        price = signal['current_price']
        
        summary = f"""
╔══════════════════════════════════════════════╗
║         TRADING SIGNAL SUMMARY               ║
╚══════════════════════════════════════════════╝

📊 Signal: {action}
📈 Predicted Return: {pred_ret:+.2f}%
🎯 Confidence: {conf:.2%}
💰 Current Price: {price:,.0f} IRR

Technical Indicators:
  • RSI: {signal['technical_indicators']['RSI_14']:.1f}
  • MACD: {signal['technical_indicators']['MACD']:,.0f}
  • Price vs SMA7: {signal['technical_indicators']['price_vs_sma7']:+.2f}%
  • Price vs SMA30: {signal['technical_indicators']['price_vs_sma30']:+.2f}%

Reasoning:
"""
        for i, reason in enumerate(signal['reasoning'], 1):
            summary += f"  {i}. {reason}\n"
        
        if signal['action'] != SignalType.HOLD.value:
            summary += f"\n⚡ Action Strength: {signal['strength']:.0%}\n"
        
        summary += "\n" + "="*50
        
        return summary


# ==================== TESTING ====================

if __name__ == "__main__":
    generator = LiveSignalGenerator()
    
    # Test with sample prediction result
    test_prediction = {
        'timestamp': datetime.now().isoformat(),
        'current_price': 11500000.0,
        'predicted_return': 0.012,  # 1.2% bullish
        'predicted_return_pct': 1.2,
        'predicted_price': 11638000.0,
        'confidence': 0.75,
        'features': {
            'RSI_14': 55.0,
            'MACD': 50000.0,
            'SMA_7': 11400000.0,
            'SMA_30': 11300000.0
        }
    }
    
    signal = generator.generate_signal(test_prediction)
    
    print("\n" + generator.get_signal_summary(signal))
