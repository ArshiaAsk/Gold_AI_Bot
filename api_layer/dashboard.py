"""
Simple Trading Bot Dashboard
Monitor bot performance and signals in real-time
"""

import json
import time
from datetime import datetime
from pathlib import Path
import os

try:
    from api_layer.api_config import LATEST_SIGNAL_FILE, CACHE_DIR
except ImportError:  # pragma: no cover - fallback for script-style execution
    from api_config import LATEST_SIGNAL_FILE, CACHE_DIR


def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name != 'nt' else 'cls')


def format_price(price: float) -> str:
    """Format price with thousands separator"""
    return f"{price:,.0f} IRR"


def format_percent(pct: float) -> str:
    """Format percentage"""
    return f"{pct:+.2f}%"


def get_signal_emoji(action: str) -> str:
    """Get emoji for signal type"""
    emojis = {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '⚪'
    }
    return emojis.get(action, '❓')


def load_latest_signal():
    """Load the latest trading signal"""
    if not LATEST_SIGNAL_FILE.exists():
        return None
    
    try:
        with open(LATEST_SIGNAL_FILE, 'r') as f:
            return json.load(f)
    except:
        return None


def load_signal_history(limit: int = 10):
    """Load recent signal history"""
    history_file = CACHE_DIR / "signal_history.jsonl"
    
    if not history_file.exists():
        return []
    
    signals = []
    with open(history_file, 'r') as f:
        lines = f.readlines()
        for line in lines[-limit:]:
            try:
                signals.append(json.loads(line))
            except:
                continue
    
    return signals


def display_dashboard():
    """Display real-time dashboard"""
    clear_screen()
    
    # Load data
    latest_signal = load_latest_signal()
    signal_history = load_signal_history(10)
    
    # Header
    print("╔" + "="*78 + "╗")
    print("║" + " "*25 + "GOLD TRADING BOT DASHBOARD" + " "*27 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    # Current time
    print(f"🕐 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Latest Signal
    if latest_signal:
        action = latest_signal['action']
        emoji = get_signal_emoji(action)
        
        print("┌" + "─"*78 + "┐")
        print("│" + " "*28 + "LATEST SIGNAL" + " "*37 + "│")
        print("├" + "─"*78 + "┤")
        
        signal_time = datetime.fromisoformat(latest_signal['timestamp'])
        time_ago = (datetime.now() - signal_time).total_seconds() / 60
        
        print(f"│ {emoji} Signal: {action:<20} │ Time: {signal_time.strftime('%H:%M:%S')} ({time_ago:.0f} min ago)" + " "*5 + "│")
        print(f"│ 💰 Price: {format_price(latest_signal['current_price']):<30} │" + " "*32 + "│")
        print(f"│ 📈 Predicted Return: {format_percent(latest_signal['predicted_return_pct']):<18} │" + " "*32 + "│")
        print(f"│ 🎯 Confidence: {latest_signal['confidence']:<21.0%} │" + " "*32 + "│")
        
        if action != 'HOLD':
            print(f"│ ⚡ Strength: {latest_signal.get('strength', 0):<23.0%} │" + " "*32 + "│")
        
        # Technical indicators
        tech = latest_signal.get('technical_indicators', {})
        if tech:
            print("│" + "─"*78 + "│")
            print("│ Technical Indicators:" + " "*57 + "│")
            print(f"│   RSI: {tech.get('RSI_14', 0):<10.1f} │ MACD: {tech.get('MACD', 0):<25,.0f} │" + " "*20 + "│")
            print(f"│   Price vs SMA7: {format_percent(tech.get('price_vs_sma7', 0)):<15} │" + " "*36 + "│")
            print(f"│   Price vs SMA30: {format_percent(tech.get('price_vs_sma30', 0)):<15} │" + " "*36 + "│")
        
        print("└" + "─"*78 + "┘")
    else:
        print("⚠️  No signals yet")
    
    print()
    
    # Signal History
    if signal_history:
        print("┌" + "─"*78 + "┐")
        print("│" + " "*29 + "SIGNAL HISTORY" + " "*35 + "│")
        print("├" + "─"*78 + "┤")
        print("│ Time     │ Action │ Return  │ Confidence │ Price        │" + " "*23 + "│")
        print("├" + "─"*78 + "┤")
        
        for signal in signal_history[-10:]:
            sig_time = datetime.fromisoformat(signal['timestamp'])
            time_str = sig_time.strftime('%H:%M')
            action = signal['action']
            emoji = get_signal_emoji(action)
            ret = signal['predicted_return_pct']
            conf = signal['confidence']
            price = signal['current_price']
            
            print(f"│ {time_str}   │ {emoji} {action:<4} │ {ret:+6.2f}% │ {conf:>5.0%}      │ {price:>12,.0f} │" + " "*23 + "│")
        
        print("└" + "─"*78 + "┘")
    
    print()
    
    # Statistics
    if signal_history:
        buy_count = sum(1 for s in signal_history if s['action'] == 'BUY')
        sell_count = sum(1 for s in signal_history if s['action'] == 'SELL')
        hold_count = sum(1 for s in signal_history if s['action'] == 'HOLD')
        
        print("📊 Statistics:")
        print(f"   Total Signals: {len(signal_history)}")
        print(f"   BUY: {buy_count} | SELL: {sell_count} | HOLD: {hold_count}")
    
    print()
    print("Press Ctrl+C to exit")


def run_dashboard(refresh_interval: int = 5):
    """Run dashboard with auto-refresh"""
    try:
        while True:
            display_dashboard()
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\n\nDashboard stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot Dashboard')
    parser.add_argument('--refresh', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    run_dashboard(refresh_interval=args.refresh)
