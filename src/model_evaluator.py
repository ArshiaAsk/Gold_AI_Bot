"""
Model evaluation and visualization utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)



class ModelEvaluator:
    """Evaluate model predictions and visualize results"""

    def __init__(self, scaler_y):
        """
        Initialize evaluator

        Args:
            scaler_y: Fitted scaler for target variable
        """
        self.scaler_y = scaler_y

    
    def predict(self, model, X) -> np.ndarray:
        """
        Make predictions

        Args:
            model: Trained model
            X: Input features
        
        Returns:
            Predictions array            
        """
        predictions_scaled = model.predict(X, verbose=0)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        return predictions.flatten()
    

    def caculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Caculate evaluation metrics

        Args:
            y_true: True values
            y_pred: Prediction values

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

        logger.info(f"Metrics - RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.4f}")

        return metrics
    

    def reconstruct_prices(self,
                           base_prices: np.ndarray,
                           log_return: np.ndarray) -> np.ndarray:
        """
        Reconstruct prices from log returns
        
        Args:
            base_prices: Base prices (today's price)
            log_returns: Log returns for next day
            
        Returns:
            Reconstructed prices
        """
        if len(base_prices) != len(log_return):
            diff = len(base_prices) - len(log_return)
            base_prices = base_prices[diff:]
        
        return base_prices * np.exp(log_return)


    def evaluate_predictions(self,
                             model,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             base_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Evaluate predictions and reconstruct prices
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: True test targets (scaled)
            base_prices: Base prices for reconstruction
            
        Returns:
            actual_prices, predicted_prices, metrics
        """
        # Predict log returns
        pred_log_returns = self.predict(model, X_test) 
        actual_log_returns = self.scaler_y.inverse_transform(y_test).flatten()

        # Reconstruct prices
        predicted_prices = self.reconstruct_prices(base_prices, pred_log_returns)
        actual_prices = self.reconstruct_prices(base_prices, actual_log_returns)

        # Caculate metrics on prices
        metrics = self.caculate_metrics(actual_prices, predicted_prices)

        # Caculate metrics on log returns
        log_return_metrics = self.caculate_metrics(actual_log_returns, pred_log_returns)
        metrics['log_return_rmse'] = log_return_metrics['rmse']
        metrics['log_return_mae'] = log_return_metrics['mae']

        return actual_prices, predicted_prices, metrics
    

    
class Visualizer:
    """Create visualizations for model performance"""
    
    @staticmethod
    def plot_training_history(history: dict, save_path: str = None):
        """
        Plot training and validation loss
        
        Args:
            history: Training history dictionary
            save_path: Path to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE plot
        if 'mae' in history:
            ax2.plot(history['mae'], label='Training MAE', linewidth=2)
            ax2.plot(history['val_mae'], label='Validation MAE', linewidth=2)
            ax2.set_title('Model MAE Over Epochs', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_predictions(dates: np.ndarray,
                        actual_prices: np.ndarray,
                        predicted_prices: np.ndarray,
                        metrics: Dict[str, float],
                        save_path: str = None):
        """
        Plot actual vs predicted prices
        
        Args:
            dates: Date array
            actual_prices: Actual price values
            predicted_prices: Predicted price values
            metrics: Dictionary of evaluation metrics
            save_path: Path to save plot
        """
        if len(dates) != len(actual_prices):
            dates = dates[-len(actual_prices):]

        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot prices
        ax.plot(dates, actual_prices, 
                label='Actual Price', 
                color='#2E86AB', 
                linewidth=2.5, 
                alpha=0.8)
        ax.plot(dates, predicted_prices, 
                label='Predicted Price', 
                color='#A23B72', 
                linewidth=2, 
                linestyle='--', 
                alpha=0.8)
        
        # Add metrics text
        textstr = f"RMSE: {metrics['rmse']:,.0f} Toman\n"
        textstr += f"MAE: {metrics['mae']:,.0f} Toman\n"
        textstr += f"R²: {metrics['r2']:.4f}\n"
        textstr += f"MAPE: {metrics['mape']:.2f}%"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', bbox=props)
        
        ax.set_title('Gold Price Prediction - LSTM Model', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (Toman)', fontsize=12)
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Rotate date labels
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction plot saved to {save_path}")
        
        plt.close()
    
    @staticmethod
    def plot_residuals(actual: np.ndarray, 
                       predicted: np.ndarray,
                       save_path: str = None):
        """
        Plot residuals analysis
        
        Args:
            actual: Actual values
            predicted: Predicted values
            save_path: Path to save plot
        """
        residuals = actual - predicted
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals over time
        axes[0, 0].plot(residuals, color='#E63946', alpha=0.7)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[0, 1].hist(residuals, bins=50, color='#457B9D', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Residuals Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(actual, predicted, alpha=0.5, color='#2A9D8F')
        axes[1, 1].plot([actual.min(), actual.max()], 
                       [actual.min(), actual.max()], 
                       'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 1].set_title('Actual vs Predicted', fontweight='bold')
        axes[1, 1].set_xlabel('Actual Price')
        axes[1, 1].set_ylabel('Predicted Price')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {save_path}")
        
        plt.close()