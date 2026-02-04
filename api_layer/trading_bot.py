"""
Live Trading Bot - Main Orchestrator
Coordinates all components: data fetching, prediction, signal generation
"""

import time
import json
import logging
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Optional

from api_config import (
    PRICE_UPDATE_INTERVAL, SIGNAL_GENERATION_INTERVAL,
    MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE,
    MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE,
    TRADING_DAYS, CACHE_DIR, LATEST_SIGNAL_FILE,
    ENABLE_NOTIFICATIONS, PAPER_TRADING_MODE
)

from data_fetcher import TGJUDataFetcher
from live_predictor import LivePredictor
from live_signal_generator import LiveSignalGenerator, SignalType


class TradingBot:
    """
    Main trading bot orchestrator
    Runs continuous loop: fetch → predict → signal → execute
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
        # Initialize components
        self.logger.info("Initializing Trading Bot...")
        self.data_fetcher = TGJUDataFetcher()
        self.predictor = LivePredictor()
        self.signal_generator = LiveSignalGenerator()
        
        # State
        self.is_running = False
        self.last_signal = None
        self.cycle_count = 0
        
        self.logger.info("✓ Trading Bot initialized")
    
    def _setup_logger(self):
        logger = logging.getLogger('TradingBot')
        logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler('logs/trading_bot.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
        
        return logger
    
    def is_market_hours(self) -> bool:
        """Check if current time is within market hours"""
        now = datetime.now()
        
        # Check day of week
        if now.weekday() not in TRADING_DAYS:
            return False
        
        # Check time
        market_open = dt_time(MARKET_OPEN_HOUR, MARKET_OPEN_MINUTE)
        market_close = dt_time(MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    def run_single_cycle(self) -> Optional[dict]:
        """
        Execute one complete trading cycle
        
        Returns:
            Signal dictionary or None
        """
        self.cycle_count += 1
        self.logger.info("="*70)
        self.logger.info(f"CYCLE #{self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*70)
        
        try:
            # Step 1: Fetch latest prices
            self.logger.info("📡 Step 1/3: Fetching latest prices...")
            latest_prices = self.data_fetcher.get_all_latest_prices()
            
            if latest_prices is None:
                self.logger.warning("⚠️  Failed to fetch prices - using cached data")
                latest_prices = self.data_fetcher.get_cached_prices()
                
                if latest_prices is None:
                    self.logger.error("✗ No cached data available - skipping cycle")
                    return None
            
            # Step 2: Generate prediction
            self.logger.info("🤖 Step 2/3: Generating ML prediction...")
            prediction_result = self.predictor.predict_from_latest_prices(latest_prices)
            
            if prediction_result is None:
                self.logger.error("✗ Prediction failed - skipping cycle")
                return None
            
            # Step 3: Generate trading signal
            self.logger.info("📊 Step 3/3: Generating trading signal...")
            signal = self.signal_generator.generate_signal(prediction_result)
            
            # Log signal summary
            summary = self.signal_generator.get_signal_summary(signal)
            self.logger.info(f"\n{summary}")
            
            # Save signal to file
            self._save_signal(signal)
            
            # Send notifications
            if ENABLE_NOTIFICATIONS:
                self._send_notification(signal)
            
            self.last_signal = signal
            
            return signal
            
        except Exception as e:
            self.logger.error(f"✗ Cycle failed with error: {e}", exc_info=True)
            return None
    
    def _save_signal(self, signal: dict):
        """Save signal to file for monitoring"""
        with open(LATEST_SIGNAL_FILE, 'w') as f:
            json.dump(signal, f, indent=2, ensure_ascii=False)
        
        # Also append to signal history
        history_file = CACHE_DIR / "signal_history.jsonl"
        with open(history_file, 'a') as f:
            f.write(json.dumps(signal, ensure_ascii=False) + '\n')
    
    def _send_notification(self, signal: dict):
        """Send notification about new signal"""
        if signal['action'] == SignalType.HOLD.value:
            return  # Don't spam with HOLD signals
        
        message = f"""
🔔 NEW TRADING SIGNAL
━━━━━━━━━━━━━━━━━━━━━━
Action: {signal['action']}
Predicted Return: {signal['predicted_return_pct']:+.2f}%
Confidence: {signal['confidence']:.0%}
Price: {signal['current_price']:,.0f} IRR
Time: {datetime.now().strftime('%H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━
"""
        
        self.logger.info(message)
        
        # TODO: Add email/telegram notifications here
    
    def run_continuous(self, 
                      check_market_hours: bool = True,
                      interval: int = SIGNAL_GENERATION_INTERVAL):
        """
        Run bot continuously with periodic signal generation
        
        Args:
            check_market_hours: Only run during market hours if True
            interval: Seconds between cycles
        """
        self.is_running = True
        self.logger.info("🚀 Starting continuous trading bot...")
        self.logger.info(f"Mode: {'PAPER TRADING' if PAPER_TRADING_MODE else '⚠️  LIVE TRADING'}")
        self.logger.info(f"Interval: {interval} seconds")
        self.logger.info(f"Market hours check: {'Enabled' if check_market_hours else 'Disabled'}")
        
        try:
            while self.is_running:
                # Check market hours
                if check_market_hours and not self.is_market_hours():
                    self.logger.info("⏸  Outside market hours - waiting...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Run trading cycle
                signal = self.run_single_cycle()
                
                # Wait until next cycle
                self.logger.info(f"⏳ Waiting {interval} seconds until next cycle...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("\n⏹  Bot stopped by user")
            self.is_running = False
        except Exception as e:
            self.logger.error(f"✗ Bot crashed: {e}", exc_info=True)
            self.is_running = False
    
    def run_once(self):
        """Run a single cycle (useful for testing)"""
        return self.run_single_cycle()
    
    def get_status(self) -> dict:
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'last_signal': self.last_signal,
            'market_hours': self.is_market_hours(),
            'timestamp': datetime.now().isoformat()
        }


# ==================== MAIN ENTRY POINT ====================

def main():
    """Main entry point for trading bot"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Gold Trading Bot')
    parser.add_argument('--mode', choices=['once', 'continuous'], default='once',
                       help='Run mode: once (single cycle) or continuous')
    parser.add_argument('--interval', type=int, default=300,
                       help='Interval between cycles in seconds (default: 300)')
    parser.add_argument('--no-market-check', action='store_true',
                       help='Disable market hours check')
    
    args = parser.parse_args()
    
    # Create bot
    bot = TradingBot()
    
    # Run based on mode
    if args.mode == 'once':
        print("\n" + "="*70)
        print("Running single trading cycle...")
        print("="*70 + "\n")
        signal = bot.run_once()
        
        if signal:
            print("\n✓ Cycle completed successfully")
            print(f"Signal: {signal['action']}")
        else:
            print("\n✗ Cycle failed")
    
    else:  # continuous
        print("\n" + "="*70)
        print("Starting continuous trading bot...")
        print("Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        bot.run_continuous(
            check_market_hours=not args.no_market_check,
            interval=args.interval
        )


if __name__ == "__main__":
    main()
