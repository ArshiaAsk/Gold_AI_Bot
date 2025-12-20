import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    """
    Calculates Relative Strength Index (RSI).
    RSI > 70: Overbought (Potential drop).
    RSI < 30: Oversold (Potential rise).
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_advanced_features(input_path, output_path):
    print(f"--- Advanced Feature Engineering Started: Processing {input_path} ---")
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # CRITICAL: Sort by date ascending (Oldest first) for time-series calculations
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    print(f"Data sorted. Date Range: {df['Date'].min()} to {df['Date'].max()}")

    # ==========================================
    # 2. Technical Indicators (Market Psychology)
    # ==========================================
    print("Generating Technical Indicators...")
    
    # Simple Moving Averages (SMA) - Trend Identification
    df['SMA_7'] = df['Gold_Toman'].rolling(window=7).mean()   
    df['SMA_30'] = df['Gold_Toman'].rolling(window=30).mean() 
    
    # Exponential Moving Average (EMA) - More weight on recent prices
    df['EMA_12'] = df['Gold_Toman'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Gold_Toman'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence) - Momentum
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI (Relative Strength Index) - Overbought/Oversold
    df['RSI_14'] = calculate_rsi(df['Gold_Toman'], window=14)
    
    # Bollinger Bands (Volatility)
    # Upper/Lower bands: 2 standard deviations away from 20-day SMA
    sma_20 = df['Gold_Toman'].rolling(window=20).mean()
    std_20 = df['Gold_Toman'].rolling(window=20).std()
    df['Bollinger_Upper'] = sma_20 + (std_20 * 2)
    df['Bollinger_Lower'] = sma_20 - (std_20 * 2)

    # ==========================================
    # 3. Lag Features (Memory)
    # ==========================================
    print("Generating Lag Features...")
    
    # We want the model to know the price of: Yesterday (Lag_1), 2 days ago, 7 days ago.
    # IMPORTANT: We do this for Gold, USD, and Ounce (Global influence).
    
    features_to_lag = ['Gold_Toman', 'USD_Toman', 'Ounce_Toman']
    lags = [1, 2, 3, 7] # 1 day, 2 days, 3 days, 1 week
    
    for col in features_to_lag:
        for lag in lags:
            df[f'{col}_Lag_{lag}'] = df[col].shift(lag)

    # ==========================================
    # 4. Rate of Change (Percentage Growth)
    # ==========================================
    # How much did the price change compared to yesterday? (Daily Return)
    df['Gold_PCT_Change'] = df['Gold_Toman'].pct_change() * 100
    df['USD_PCT_Change'] = df['USD_Toman'].pct_change() * 100

    # ==========================================
    # 5. Target Creation (What to predict?)
    # ==========================================
    # We want to predict TOMORROW's Gold Price.
    # So we shift Gold_Toman backwards by 1 (Future into current row)
    df['Target_NextDay_Price'] = df['Gold_Toman'].shift(-1)
    
    # ==========================================
    # 6. Cleaning
    # ==========================================
    # Calculations (Rolling, Lags) create NaNs at the beginning.
    # The Shift(-1) creates a NaN at the very last row.
    initial_shape = df.shape
    df_clean = df.dropna()
    print(f"Dropped NaNs: {initial_shape[0] - df_clean.shape[0]} rows removed (due to window calculations).")
    
    # Save
    df_clean.to_csv(output_path, index=False)
    print(f"\n✅ ADVANCED FEATURES READY! Saved to: {output_path}")
    print("\nSample of new columns:\n", df_clean[['Date', 'Gold_Toman', 'SMA_7', 'RSI_14', 'Gold_Toman_Lag_1', 'Target_NextDay_Price']].tail())

if __name__ == "__main__":
    generate_advanced_features("datasets/final_features_toman.csv", "advanced_gold_features.csv")
