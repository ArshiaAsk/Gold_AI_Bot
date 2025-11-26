import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def train_linear_regression(df: pd.DataFrame, feature: list = None, target: str = 'Gold'):
    """Train a linear regression model on selected features and target"""
    
    if feature is None:
        feature = ['USD', 'Oil']
        
    X = df[feature]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False    
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'MSE': mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'Intercept': model.intercept_,
        'Coefficients': dict(zip(feature, model.coef_))
    }
    return model, metrics, y_test, y_pred

