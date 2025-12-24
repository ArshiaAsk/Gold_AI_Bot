import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def process_features(input_path="cleaned_gold_dataset.csv", output_dir="processed_data"):
    print("--- ⚙️ Starting Advanced Feature Engineering (Log Returns Strategy) ---")
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Calculate Log Returns 
    cols_to_convert = ['Gold_Toman', 'USD_Toman', 'Ounce_Toman', 'Oil_Toman']
    
    for col in cols_to_convert:
        # ln(Price_t / Price_t-1)
        df[f'{col}_LogRet'] = np.log(df[col] / df[col].shift(1))
    
    df = df.dropna().reset_index(drop=True)

    # 3. Create Lag Features based on RETURNS 
    features_to_lag = [f'{c}_LogRet' for c in cols_to_convert]
    lags = [1, 2, 3, 5]  # تاخیرهای ۱ تا ۵ روزه
    
    cols = []
    for col in features_to_lag:
        for lag in lags:
            col_name = f'{col}_Lag_{lag}'
            df[col_name] = df[col].shift(lag)
            cols.append(col_name)
    
    # 4. Define Target 
    df['Target_NextDay_LogRet'] = df['Gold_Toman_LogRet'].shift(-1)
    
    df_final = df.dropna().reset_index(drop=True)
    
    print(f"Dataset Shape after processing: {df_final.shape}")
    
    # 5. Prepare X and y
    feature_cols = [c for c in df_final.columns if 'Lag' in c]
    X = df_final[feature_cols].values
    y = df_final['Target_NextDay_LogRet'].values.reshape(-1, 1)
    
    actual_prices = df_final['Gold_Toman'].values.reshape(-1, 1)
    dates = df_final['Date'].values
    
    # 6. Train/Test Split (Time-based)
    train_size = int(len(X) * 0.85) 
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    prices_test = actual_prices[train_size:] 
    dates_test = dates[train_size:]
    
    # 7. Scaling (Using StandardScaler for Returns)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # 8. Save Everything
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    np.save(f'{output_dir}/X_train.npy', X_train_scaled)
    np.save(f'{output_dir}/X_test.npy', X_test_scaled)
    np.save(f'{output_dir}/y_train.npy', y_train_scaled)
    np.save(f'{output_dir}/y_test.npy', y_test_scaled)
    
    np.save(f'{output_dir}/prices_test_base.npy', prices_test) 
    np.save(f'{output_dir}/dates_test.npy', dates_test)
    
    joblib.dump(scaler_X, f'{output_dir}/scaler_X.pkl')
    joblib.dump(scaler_y, f'{output_dir}/scaler_y.pkl')
    
    print("✅ Feature Engineering Complete. Data saved to 'processed_data/'")
    print(f"Features used: {len(feature_cols)}")

if __name__ == "__main__":
    process_features()
