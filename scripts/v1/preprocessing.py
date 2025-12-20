import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os



def preprocess_and_scale(input_path, output_dir='processed_data'):
    print(f"--- Starting Preprocessing & Scaling on {input_path}")

    # Creating folder for output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # 1. Load Data
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=True) 


    # 2. Select Feature (X) and Target (y)
    dates = df['Date'].values # Seprating Date Column

    y = df[['Target_NextDay_Price']].values

    X = df.drop(['Date', 'Target_NextDay_Price'], axis=1).values

    feature_names =  df.drop(['Date', 'Target_NextDay_Price'], axis=1).columns.tolist()

    print(f"Feature Count: {X.shape[1]}")
    print(f"Target Column: Target_NextDay_Price")


    # 3. Train/Test Split (Time-based split, NOT random)
    train_size = int(len(df) * 0.8)

    X_train_raw, X_test_raw = X[:train_size], X[train_size:]
    y_train_raw, y_test_raw = y[:train_size], y[train_size:]
    dates_train, dates_test = dates[:train_size], dates[train_size:]

    print(f"\nSplit Summary:")
    print(f"Train Samples: {len(X_train_raw)} (from {pd.to_datetime(dates_train[0]).date()} to {pd.to_datetime(dates_train[-1]).date()})")
    print(f"Test Samples: {len(X_test_raw)} (from {pd.to_datetime(dates_test[0]).date()} to {pd.to_datetime(dates_test[-1]).date()})")


    # 4. Scaling (Normalization to 0-1 range)
    print(f"\nScaling Data...")

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw)

    X_test_scaled = scaler_X.transform(X_test_raw)
    y_test_scaled = scaler_y.transform(y_test_raw)

    # 5. Save everything for Model
    print(f"\nSaving processed files...")

    np.save(f'{output_dir}/X_train.npy', X_train_scaled)
    np.save(f'{output_dir}/y_train.npy', y_train_scaled)
    np.save(f'{output_dir}/X_test.npy', X_test_scaled)
    np.save(f'{output_dir}/y_test.npy', y_test_scaled)
    np.save(f'{output_dir}/dates_test.npy', dates_test)

    joblib.dump(scaler_X, f'{output_dir}/scaler_X.pkl')
    joblib.dump(scaler_y, f'{output_dir}/scaler_y.pkl')

    joblib.dump(feature_names, f'{output_dir}/feature_names.pkl')

    print(f"✅ PREPROCESSING COMPLETE! All files saved in '{output_dir}/' folder.")

if __name__ == "__main__":
    preprocess_and_scale("datasets/advanced_gold_features.csv")