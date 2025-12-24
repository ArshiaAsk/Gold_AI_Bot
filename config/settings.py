"""
Configuration settings for Gold Price LSTM Prediction
"""
import os
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PathConfig:
    """Path configurations"""
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_RAW_DIR: str = os.path.join(BASE_DIR, 'data', 'raw')
    DATA_PROCESSED_DIR: str = os.path.join(BASE_DIR, 'data', 'processed')
    MODELS_DIR: str = os.path.join(BASE_DIR, 'models')
    LOGS_DIR: str = os.path.join(BASE_DIR, 'logs')
    PLOTS_DIR: str = os.path.join(BASE_DIR, 'outputs', 'plots')
    
    def __post_init__(self):
        """Ensure all directories exist"""
        for dir_path in [self.DATA_RAW_DIR, self.DATA_PROCESSED_DIR, 
                         self.MODELS_DIR, self.LOGS_DIR, self.PLOTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    @property
    def raw_data_path(self) -> str:
        return os.path.join(self.DATA_RAW_DIR, 'final_gold_dataset.csv')
    
    @property
    def processed_data_path(self) -> str:
        return os.path.join(self.DATA_PROCESSED_DIR, 'advanced_gold_features.csv')
    
    @property
    def model_path(self) -> str:
        return os.path.join(self.MODELS_DIR, 'gold_lstm_v2.keras')
    
    @property
    def scaler_path(self) -> str:
        return os.path.join(self.MODELS_DIR, 'scaler.pkl')
    
    @property
    def training_history_path(self) -> str:
        return os.path.join(self.LOGS_DIR, 'training_history.json')


@dataclass
class DataConfig:
    """Data processing configurations"""
    TICKER_GOLD: str = "geram18"
    TICKER_USD: str = "price_dollar_rl"
    LOOKBACK_DAYS: int = 30
    TEST_SPLIT_RATIO: float = 0.15
    VAL_SPLIT_RATIO: float = 0.15
    RANDOM_STATE: int = 42
    
    # Feature columns
    FEATURE_COLUMNS: List[str] = None
    TARGET_COLUMN: str = 'Target_Next_LogRet'
    
    def __post_init__(self):
        if self.FEATURE_COLUMNS is None:
            self.FEATURE_COLUMNS = [
                'Gold_LogRet',
                'USD_LogRet', 
                'Ounce_LogRet',
                'Oil_LogRet',
                'SMA_7',
                'RSI_14',
                'MACD',
                'MACD_Signal',
                'Bollinger_Upper',
                'Bollinger_Lower',
                'Gold_LogRet_Lag_1',
                'Gold_LogRet_Lag_2',
                'Gold_LogRet_Lag_3',
                'USD_LogRet_Lag_1',
                'USD_LogRet_Lag_2',
            ]


@dataclass
class ModelConfig:
    """LSTM model architecture configurations"""
    # Architecture
    LSTM_UNITS_1: int = 64
    LSTM_UNITS_2: int = 32
    DENSE_UNITS: int = 16
    DROPOUT_RATE: float = 0.3
    
    # Training
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 100
    BATCH_SIZE: int = 16
    
    # Callbacks
    EARLY_STOPPING_PATIENCE: int = 15
    REDUCE_LR_PATIENCE: int = 5
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LR: float = 0.00001
    
    # Sequence settings
    SEQUENCE_LENGTH: int = 30  # LSTM input timesteps
    
    def get_optimizer_config(self) -> dict:
        """Return optimizer configuration"""
        return {'learning_rate': self.LEARNING_RATE}
    
    def get_early_stopping_config(self) -> dict:
        """Return early stopping configuration"""
        return {
            'monitor': 'val_loss',
            'patience': self.EARLY_STOPPING_PATIENCE,
            'restore_best_weights': True,
            'verbose': 1
        }
    
    def get_reduce_lr_config(self) -> dict:
        """Return ReduceLROnPlateau configuration"""
        return {
            'monitor': 'val_loss',
            'factor': self.REDUCE_LR_FACTOR,
            'patience': self.REDUCE_LR_PATIENCE,
            'min_lr': self.MIN_LR,
            'verbose': 1
        }


@dataclass
class TradingConfig:
    """Trading strategy configurations"""
    TRANSACTION_FEE: float = 0.002
    BUY_THRESHOLD: float = 0.0005
    SELL_THRESHOLD: float = -0.0005
    INITIAL_CAPITAL: float = 100_000_000


class Config:
    """Main configuration class"""
    def __init__(self):
        self.paths = PathConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.trading = TradingConfig()


# Global config instance
config = Config()
