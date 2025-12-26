# Gold Price LSTM Prediction Model

Professional LSTM-based time series prediction model for Iran's gold price (IRR) using advanced feature engineering and deep learning.

## Project Overview

This project implements a robust LSTM neural network to predict gold prices in Iranian Rial (IRR) based on:
- Historical gold prices
- USD exchange rates
- Gold price in Ounce (USD)
- Oil prices
- Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- Lagged features

### Key Features

✅ **Clean Architecture**: Modular, maintainable, and scalable code structure  
✅ **Professional Logging**: Comprehensive logging system for tracking  
✅ **Automated Pipeline**: End-to-end training pipeline from data to evaluation  
✅ **Advanced Preprocessing**: StandardScaler normalization and time-series splitting  
✅ **Model Evaluation**: Multiple metrics (RMSE, MAE, R², MAPE) and visualizations  
✅ **Reproducibility**: Fixed random seeds for consistent results  

---

## Project Structure

Gold_Usd_Oil_IRR/

├── config/

│ └── settings.py # Configuration management

├── data/

│ ├── raw/ # Raw data files

│ └── processed/ # Processed features

│ └── advanced_gold_features.csv

├── src/

│ ├── config_settings.py # Updated config with dataclasses

│ ├── data_preprocessor.py # Data loading and preprocessing

│ ├── model_builder.py # LSTM model architecture

│ ├── model_evaluator.py # Evaluation and visualization

│ └── train_pipeline.py # Complete training pipeline

├── models/ # Saved models and scalers

│ ├── gold_lstm_v2.keras

│ ├── scaler_X.pkl

│ └── scaler_y.pkl

├── logs/ # Training logs and metrics

│ ├── training_YYYYMMDD_HHMMSS.log

│ ├── training_history.json

│ └── test_metrics.json

├── outputs/

│ └── plots/ # Generated visualizations

│ ├── training_history.png

│ ├── predictions.png

│ └── residuals.png

├── requirements.txt

└── README.md