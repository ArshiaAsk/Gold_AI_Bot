import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import os


np.random.seed(42)
tf.random.set_seed(42)


def build_and_train_model(data_dir="processed_data", model_save_path="gold_lstm_model.h5"):
    print("--- Loading Data for Modeling ---")

    # 1. Load Data
    try:
        X_train = np.load(f'{data_dir}/X_train.npy')
        y_train = np.load(f'{data_dir}/y_train.npy')
        X_test = np.load(f'{data_dir}/X_test.npy')
        y_test = np.load(f'{data_dir}/y_test.npy')
    except FileNotFoundError:
        print(f"❌ Error: Data files not found in '{data_dir}'. Please run preprocessing first.")
        return None, None
    

    # 2. Reshaping for LSTM
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    print(f"Input Shape: {X_train.shape} (Samples, TimeSteps, Features)")


    # 3. Building the Architecture
    model = Sequential()

    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    model.add(LSTM(units=64, return_sequences=True)) 
    model.add(Dropout(0.2))

    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    
    # 4. Compile
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    print("\nModel Summary:")
    model.summary()


    # 5. Training Model
    print("\n--- Strating Training (This might take a moment) ---")

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    # Save Training Model
    model.save(model_save_path)
    print(f"\n Model trained and save to {model_save_path}")
    
    return model, history



def evaluate_model(model, data_dir="processed_data"):
    if model is None: return

    print(f"\n--- Evaluating Model Performance ---")
    
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    dates_test = np.load(f'{data_dir}/dates_test.npy', allow_pickle=True)
    scaler_y = joblib.load(f'{data_dir}/scaler_y.pkl')

    if len(X_test.shape) == 2:
        X_test_reshaped = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # 1. Prediction
    predicted_prices_scaled = model.predict(X_test_reshaped)

    # 2. Return Scaled Price to Real Price
    predicted_prices = scaler_y.inverse_transform(predicted_prices_scaled)
    actual_prices = scaler_y.inverse_transform(y_test)

    # 3. Caculating RMSE
    rmse = np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))
    print(f"\n Root Mean Squared Error (RMSE): {rmse:,.0f} Toman")
    
    # 4. Show Graph
    plt.figure(figsize=(14, 7))
    plt.plot(dates_test, actual_prices, color='blue', label='Actual Gold Price', alpha=0.6)
    plt.plot(dates_test, predicted_prices, color='red', label='Predicted Gold Price', alpha=0.8)

    plt.title('Gold Price Prediction: AI Model vs Reality', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Price (Toman)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    print(f" Chart saved as 'prediction_result.png'")
    plt.show()



if __name__ == "__main__":
    trained_model, trained_history = build_and_train_model()
    evaluate_model(trained_model)
