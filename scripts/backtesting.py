import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import os


def backtest_strategy(data_dir="processed_data_v2", model_path="models/gold_lstm_v2.h5"):
    print("--- Strating Backtesting Simulation ---")

    # 1. Load Data & Model
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')

    prices_test_base = np.load(f'{data_dir}/prices_test_base.npy').flatten()
    dates_test = np.load(f'{data_dir}/dates_test.npy', allow_pickle=True)

    scaler_y = joblib.load(f'{data_dir}/scaler_y.pkl')
    model = load_model(model_path)


    # 2. Generate Prediction (Signals)
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    pred_log_ret_scaled = model.predict(X_test_reshaped, verbose=0)

    pred_log_ret = scaler_y.inverse_transform(pred_log_ret_scaled).flatten()
    actual_log_ret = scaler_y.inverse_transform(y_test).flatten()

    actual_prices_next_day = prices_test_base * np.exp(actual_log_ret)


    # 3. Simulation Settings
    initial_capital = 100_000_000

    ai_capital = initial_capital
    ai_holding_gold = False
    ai_portfolio_history = []

    bh_capital = initial_capital
    bh_gold_amount = bh_capital / prices_test_base[0]
    bh_portfolio_history = []

    print("\n--- Simulation Trading day by day ---")


    # 4. Trading Loop
    for i in range(len(pred_log_ret)):
        current_price = prices_test_base[i]
        next_day_price_real = actual_prices_next_day[i]
        
      
        signal_buy = pred_log_ret[i] > 0.00 
        
        if signal_buy:
            if not ai_holding_gold:
                ai_gold_amount = ai_capital / current_price
                ai_holding_gold = True
                # print(f"Day {dates_test[i]}: BUY Signal 🟢")
            
            ai_capital = ai_gold_amount * next_day_price_real
            
        else: 
            if ai_holding_gold:
                ai_capital = ai_gold_amount * current_price 
                ai_holding_gold = False
                # print(f"Day {dates_test[i]}: SELL Signal 🔴")
            
            pass
            
        ai_portfolio_history.append(ai_capital)
        
        bh_value = bh_gold_amount * next_day_price_real
        bh_portfolio_history.append(bh_value)

    # 5. Analysis
    ai_final_value = ai_portfolio_history[-1]
    bh_final_value = bh_portfolio_history[-1]
    
    ai_roi = ((ai_final_value - initial_capital) / initial_capital) * 100
    bh_roi = ((bh_final_value - initial_capital) / initial_capital) * 100
    
    print(f"\n --- Results Summary ({len(dates_test)} Days) ---")
    print(f" Initial Capital: {initial_capital:,.0f} Toman")
    print("-" * 30)
    print(f" Buy & Hold Final Value: {bh_final_value:,.0f} Toman")
    print(f" Buy & Hold ROI: {bh_roi:.2f}%")
    print("-" * 30)
    print(f" AI Model Final Value:   {ai_final_value:,.0f} Toman")
    print(f" AI Model ROI:   {ai_roi:.2f}%")
    print("-" * 30)
    
    if ai_final_value > bh_final_value:
        print("✅ SUCCESS: AI outperformed the market!")
    else:
        print("❌ RESULT: Buy & Hold was better (Trend was too strong).")

    # 6. Plotting Wealth Curve
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, bh_portfolio_history, label=f'Buy & Hold ({bh_roi:.1f}%)', color='gray', linestyle='--')
    plt.plot(dates_test, ai_portfolio_history, label=f'AI Strategy ({ai_roi:.1f}%)', color='green', linewidth=2)
    
    plt.title('Profitability Analysis: AI Trader vs. Buy & Hold', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (Toman)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    
    plt.savefig('backtest_result.png')
    print("📈 Profit chart saved as 'backtest_result.png'")

if __name__ == "__main__":
    backtest_strategy()