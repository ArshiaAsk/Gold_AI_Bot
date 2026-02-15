"""
Signal Generator - Momentum Strategy
Converts predictions + technical indicators → BUY/SELL/HOLD signals
"""

import pandas as pd
import numpy as np
from typing import Literal, Dict, Any
from config import backtest_config as cfg

SignalType = Literal['BUY', 'SELL', 'HOLD', 'EXIT']


class MomentumSignalGenerator:
    """
    Generates trading signals based on:
    - Predicted returns
    - Technical indicators (RSI, MACD, SMA)
    - Confidence filters
    """
    
    def __init__(self):
        self.name = "Momentum Strategy"
        self.signal_history = []
        
    def generate_signal(
        self, 
        row: pd.Series,
        current_position: float = 0.0,
        entry_price: float = None
    ) -> Dict[str, Any]:
        """
        Generate trading signal for a single time step
        
        Args:
            row: Current market data row
            current_position: Current position size (0 if flat)
            entry_price: Entry price if currently in position
            
        Returns:
            Dictionary with signal and reasoning
        """
        
        # Extract features
        # Never use target labels as signals in backtests (prevents look-ahead leakage)
        predicted_return = row.get('Predicted_LogRet', 0.0)
        current_price = row['Gold_IRR']
        rsi = row.get('RSI_14', 50)
        sma_7 = row.get('SMA_7', current_price)
        macd = row.get('MACD', 0)
        macd_signal = row.get('MACD_Signal', 0)
        
        # Calculate unrealized P&L if in position
        unrealized_pnl_pct = 0.0
        if current_position > 0 and entry_price is not None:
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
        
        # Initialize signal
        signal = 'HOLD'
        confidence = 0.5
        reasons = []
        
        # ====================================================================
        # EXIT LOGIC (if currently in position)
        # ====================================================================
        
        if current_position > 0:
            # 1. Stop-loss hit
            if unrealized_pnl_pct <= -cfg.STOP_LOSS_PCT:
                return {
                    'signal': 'EXIT',
                    'confidence': 1.0,
                    'reasons': [f'Stop-loss hit: {unrealized_pnl_pct:.2%}'],
                    'predicted_return': predicted_return,
                    'rsi': rsi,
                    'price': current_price
                }
            
            # 2. Take-profit hit
            if unrealized_pnl_pct >= cfg.TAKE_PROFIT_PCT:
                return {
                    'signal': 'EXIT',
                    'confidence': 1.0,
                    'reasons': [f'Take-profit hit: {unrealized_pnl_pct:.2%}'],
                    'predicted_return': predicted_return,
                    'rsi': rsi,
                    'price': current_price
                }
            
            # 3. RSI overbought exit
            if rsi > cfg.RSI_EXIT:
                return {
                    'signal': 'EXIT',
                    'confidence': 0.8,
                    'reasons': [f'RSI overbought: {rsi:.1f}'],
                    'predicted_return': predicted_return,
                    'rsi': rsi,
                    'price': current_price
                }
            
            # 4. Strong negative prediction
            if predicted_return < cfg.SELL_THRESHOLD:
                return {
                    'signal': 'EXIT',
                    'confidence': 0.7,
                    'reasons': [f'Negative prediction: {predicted_return:.2%}'],
                    'predicted_return': predicted_return,
                    'rsi': rsi,
                    'price': current_price
                }
            
            # 5. Hold position if no exit conditions
            return {
                'signal': 'HOLD',
                'confidence': 0.6,
                'reasons': ['Holding position'],
                'predicted_return': predicted_return,
                'rsi': rsi,
                'price': current_price,
                'unrealized_pnl': unrealized_pnl_pct
            }
        
        # ====================================================================
        # ENTRY LOGIC (if flat / no position)
        # ====================================================================
        
        # BUY CONDITIONS
        buy_score = 0
        
        # 1. Strong positive prediction
        if predicted_return > cfg.BUY_THRESHOLD:
            buy_score += 3
            reasons.append(f'Strong prediction: {predicted_return:.2%}')
        elif predicted_return > cfg.HOLD_THRESHOLD_UPPER:
            buy_score += 1
            reasons.append(f'Moderate prediction: {predicted_return:.2%}')
        
        # 2. RSI not overbought
        if rsi < cfg.RSI_OVERBOUGHT:
            buy_score += 1
            if rsi < cfg.RSI_OVERSOLD:
                buy_score += 1
                reasons.append(f'RSI oversold: {rsi:.1f}')
        else:
            reasons.append(f'RSI too high: {rsi:.1f}')
            buy_score -= 2
        
        # 3. Price above SMA (trend confirmation)
        if cfg.USE_SMA_FILTER:
            if current_price > sma_7:
                buy_score += 1
                reasons.append('Price > SMA_7 (uptrend)')
            else:
                reasons.append('Price < SMA_7 (downtrend)')
                buy_score -= 1
        
        # 4. MACD bullish
        if cfg.USE_MACD_FILTER:
            if macd > macd_signal:
                buy_score += 1
                reasons.append('MACD bullish')
            else:
                reasons.append('MACD bearish')
        
        # Decide signal based on score
        if buy_score >= 5:
            signal = 'BUY'
            confidence = min(0.95, 0.6 + (buy_score - 5) * 0.1)
        elif buy_score >= 3:
            signal = 'BUY'
            confidence = 0.6
        else:
            signal = 'HOLD'
            confidence = 0.5
            if not reasons:
                reasons.append('Insufficient conviction')
        
        return {
            'signal': signal,
            'confidence': confidence,
            'reasons': reasons,
            'predicted_return': predicted_return,
            'rsi': rsi,
            'price': current_price,
            'buy_score': buy_score
        }
    
    def backtest_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals for entire dataset (used for analysis)
        
        Args:
            df: Historical data
            
        Returns:
            DataFrame with signals
        """
        signals = []
        
        for idx, row in df.iterrows():
            signal_info = self.generate_signal(row)
            signals.append({
                'Date': row['Date'],
                'Signal': signal_info['signal'],
                'Confidence': signal_info['confidence'],
                'Predicted_Return': signal_info['predicted_return'],
                'RSI': signal_info['rsi'],
                'Price': signal_info['price'],
                'Reasons': ' | '.join(signal_info['reasons'])
            })
        
        return pd.DataFrame(signals)


class MeanReversionSignalGenerator:
    """
    Alternative strategy: Mean Reversion
    Buy when oversold, sell when overbought
    (Not primary strategy but included for comparison)
    """
    
    def __init__(self):
        self.name = "Mean Reversion Strategy"
        
    def generate_signal(
        self, 
        row: pd.Series,
        current_position: float = 0.0,
        entry_price: float = None
    ) -> Dict[str, Any]:
        """Mean reversion signal logic"""
        
        current_price = row['Gold_IRR']
        rsi = row.get('RSI_14', 50)
        bb_upper = row.get('Bollinger_Upper', current_price * 1.02)
        bb_lower = row.get('Bollinger_Lower', current_price * 0.98)
        
        # Exit logic
        if current_position > 0 and entry_price is not None:
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
            
            # Stop-loss
            if unrealized_pnl_pct <= -cfg.STOP_LOSS_PCT:
                return {'signal': 'EXIT', 'confidence': 1.0, 'reasons': ['Stop-loss']}
            
            # Mean reversion exit: price reached upper band or RSI > 60
            if current_price > bb_upper or rsi > 60:
                return {'signal': 'EXIT', 'confidence': 0.7, 'reasons': ['Mean reversion exit']}
            
            return {'signal': 'HOLD', 'confidence': 0.6, 'reasons': ['Holding']}
        
        # Entry logic: buy when oversold
        if rsi < 35 and current_price < bb_lower:
            return {'signal': 'BUY', 'confidence': 0.8, 'reasons': ['Oversold + below BB']}
        
        return {'signal': 'HOLD', 'confidence': 0.5, 'reasons': ['Waiting for setup']}


def get_strategy(strategy_name: str = "momentum"):
    """Factory function to get strategy instance"""
    strategies = {
        'momentum': MomentumSignalGenerator,
        'mean_reversion': MeanReversionSignalGenerator
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    return strategy_class()


if __name__ == "__main__":
    # Test signal generator
    print("✅ Signal Generator Module Loaded")
    strategy = get_strategy("momentum")
    print(f"📊 Strategy: {strategy.name}")
