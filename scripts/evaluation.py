import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(y_true, y_pred) -> tuple:
    """Evaluate numerical predictions using classic metrics"""
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_absolute_error(y_true, y_pred))
    percent_error = np.abs(y_pred - y_true) / y_true * 100
    
    results = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred,
        'Residual': y_pred - y_true,
        'PercentError': percent_error
    })
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'AvgPercentError': np.mean(percent_error)
    }
    
    return results, metrics