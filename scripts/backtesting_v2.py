import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import os

# --- Settings ---
TRANSACTION_FEE = 0.002  

def backtest_strategy_advanced(data_dir="processed_data_v2", model_path="models/gold_lstm_v2.h5"):
    print(f"---  Starting Advanced Backtest (Fee: {TRANSACTION_FEE*100}%) ---")
    
    # 1. Load Data
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    prices_test_base = np.load(f'{data_dir}/prices_test_base.npy').flatten()
    dates_test = np.load(f'{data_dir}/dates_test.npy', allow_pickle=True)
    
    scaler_y = joblib.load(f'{data_dir}/scaler_y.pkl')
    model = load_model(model_path)
    
    # 2. Predict
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    pred_log_ret_scaled = model.predict(X_test_reshaped, verbose=0)
    pred_log_ret = scaler_y.inverse_transform(pred_log_ret_scaled).flatten()
    actual_log_ret = scaler_y.inverse_transform(y_test).flatten()
    
    actual_prices_next_day = prices_test_base * np.exp(actual_log_ret)
    
    # 3. Simulation Variables
    initial_capital = 100_000_000
    
    # AI Portfolio
    ai_cash = initial_capital
    ai_gold_qty = 0
    ai_holding = False
    ai_portfolio_values = []
    
    # Trade Logging
    trade_events = {'date': [], 'price': [], 'type': [], 'color': []}
    trades_count = 0
    
    # Buy & Hold Portfolio
    bh_gold_qty = (initial_capital * (1 - TRANSACTION_FEE)) / prices_test_base[0] 
    bh_portfolio_values = []

    print("\n--- 🔄 Simulating... ---")
    
    for i in range(len(pred_log_ret)):
        current_price = prices_test_base[i]     
        next_price_real = actual_prices_next_day[i] 
        current_date = dates_test[i]
        
        # --- AI Logic ---
        signal_buy = pred_log_ret[i] > 0.0005
        
        if signal_buy:
            if not ai_holding:
                # BUY ACTION
                cost = ai_cash * TRANSACTION_FEE
                net_cash = ai_cash - cost
                ai_gold_qty = net_cash / current_price
                ai_cash = 0
                ai_holding = True
                
                trades_count += 1
                trade_events['date'].append(current_date)
                trade_events['price'].append(current_price)
                trade_events['type'].append('Buy')
                trade_events['color'].append('green')
            
            current_val = ai_gold_qty * next_price_real
            
        else:
            if ai_holding:
                # SELL ACTION
                gross_cash = ai_gold_qty * current_price
                cost = gross_cash * TRANSACTION_FEE
                ai_cash = gross_cash - cost
                ai_gold_qty = 0
                ai_holding = False
                
                trades_count += 1
                trade_events['date'].append(current_date)
                trade_events['price'].append(current_price)
                trade_events['type'].append('Sell')
                trade_events['color'].append('red')
            
            current_val = ai_cash
            
        ai_portfolio_values.append(current_val)
        
        # --- Buy & Hold Logic ---
        bh_val = bh_gold_qty * next_price_real
        bh_portfolio_values.append(bh_val)

    # 4. Results
    ai_final = ai_portfolio_values[-1]
    bh_final = bh_portfolio_values[-1]
    
    ai_roi = ((ai_final - initial_capital) / initial_capital) * 100
    bh_roi = ((bh_final - initial_capital) / initial_capital) * 100
    
    print(f"\n --- FINAL REPORT (With {TRANSACTION_FEE*100}% Fee) ---")
    print(f"Total Trades: {trades_count}")
    print(f"Buy & Hold ROI: {bh_roi:.2f}%  (End Val: {bh_final:,.0f})")
    print(f"AI Model ROI:   {ai_roi:.2f}%  (End Val: {ai_final:,.0f})")
    
    if ai_final > bh_final:
        diff = ai_final - bh_final
        print(f"✅ AI WON by {diff:,.0f} Toman")
    else:
        diff = bh_final - ai_final
        print(f"❌ AI LOST by {diff:,.0f} Toman (Fees ate the profit?)")

    # 5. Advanced Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})
    
    # Plot 1: Portfolio Value
    ax1.plot(dates_test, bh_portfolio_values, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
    ax1.plot(dates_test, ai_portfolio_values, label='AI Strategy', color='blue', linewidth=2)
    ax1.set_title(f'Portfolio Growth (Fee Included: {TRANSACTION_FEE*100}%)', fontsize=12)
    ax1.set_ylabel('Value (Toman)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Price Actions
    ax2.plot(dates_test, prices_test_base, label='Gold Price', color='black', alpha=0.5)
    
    buy_dates = [d for d, t in zip(trade_events['date'], trade_events['type']) if t == 'Buy']
    buy_prices = [p for p, t in zip(trade_events['price'], trade_events['type']) if t == 'Buy']
    
    sell_dates = [d for d, t in zip(trade_events['date'], trade_events['type']) if t == 'Sell']
    sell_prices = [p for p, t in zip(trade_events['price'], trade_events['type']) if t == 'Sell']
    
    ax2.scatter(buy_dates, buy_prices, marker='^', color='green', s=100, label='AI Buy', zorder=5)
    ax2.scatter(sell_dates, sell_prices, marker='v', color='red', s=100, label='AI Sell', zorder=5)
    
    ax2.set_title('AI Trade Execution (Green=Buy, Red=Sell)', fontsize=12)
    ax2.set_ylabel('Gold Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_result_v2_advanced.png')
    print("📈 Advanced chart saved as 'backtest_result_v2_advanced.png'")

if __name__ == "__main__":
    backtest_strategy_advanced()
