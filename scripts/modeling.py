import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt


np.random.seed(42)
tf.random.set_seed(42)


def build_and_train_model(data_dir="processed_data_v2", model_save_path="gold_lstm_v2.h5"):
    print("---  Loading Data (Log Returns Strategy) ---")
    
    # Load Data
    X_train = np.load(f'{data_dir}/X_train.npy')
    y_train = np.load(f'{data_dir}/y_train.npy')
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    # Reshape for LSTM
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    print(f"Input Shape: {X_train.shape}")
    
    # --- Architecture ---
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.3))
    
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1)) 
    
    optimizer = Adam(learning_rate=0.0005) 
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # --- Training ---
    print("\n---  Starting Training ---")
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    model.save(model_save_path)
    print(f"✅ Model saved to {model_save_path}")
    return model

def evaluate_and_plot(model, data_dir="processed_data_v2"):
    print("\n--- 📊 Reconstructing Prices & Plotting ---")
    
    # Load Data
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    
    # Load auxiliary files for reconstruction
    prices_test_base = np.load(f'{data_dir}/prices_test_base.npy')
    dates_test = np.load(f'{data_dir}/dates_test.npy', allow_pickle=True)
    scaler_y = joblib.load(f'{data_dir}/scaler_y.pkl')
    
    X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # 1. Predict Log Returns (Scaled)
    pred_log_ret_scaled = model.predict(X_test_reshaped)
    
    # 2. Inverse Scale (Get real Log Returns)
    pred_log_ret = scaler_y.inverse_transform(pred_log_ret_scaled)
    actual_log_ret = scaler_y.inverse_transform(y_test)
    
    # 3. Reconstruct Prices 
    # Price_Tomorrow = Price_Today * exp(Log_Return)
    
    pred_prices = prices_test_base * np.exp(pred_log_ret)
    
    actual_prices_next_day = prices_test_base * np.exp(actual_log_ret)

    # 4. Calculate RMSE on Real Prices
    rmse = np.sqrt(np.mean((pred_prices - actual_prices_next_day) ** 2))
    print(f"\n RMSE (Toman): {rmse:,.0f}")
    
    # 5. Plot
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, actual_prices_next_day, label='Actual Price', color='#1f77b4', linewidth=2)
    plt.plot(dates_test, pred_prices, label='AI Prediction (Returns Based)', color='#d62728', linewidth=1.5, linestyle='--')
    
    plt.title(f'Gold Price Prediction V2 (Log-Returns Strategy)\nRMSE: {rmse:,.0f} Toman', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price (Toman)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gcf().autofmt_xdate()
    
    plt.savefig('prediction_result_v2.png')
    print("📈 Chart saved as 'prediction_result_v2.png'")

if __name__ == "__main__":
    model = build_and_train_model()
    evaluate_and_plot(model)
