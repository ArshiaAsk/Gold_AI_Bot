"""
LSTM Model Architecture for Gold Price Prediction
"""
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class LSTMModelBuilder:
    """Build and configure LSTM model for time series prediction"""

    def __init_(self, 
                lstm_units_1: int = 64,
                lstm_units_2: int = 32,
                dense_units: int = 16,
                dropout_rate: float = 0.3,
                learning_rate: float = 0.001,
                random_state: int = 42):
        """
        Initialize model builder
        
        Args:
            lstm_units_1: Units in first LSTM layer
            lstm_units_2: Units in second LSTM layer
            dense_units: Units in dense layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            random_state: Random seed
        """
        self.lstm_units_1 = lstm_units_1
        self.lstm_units_2 = lstm_units_2
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        # Set random seed for reproducibility
        tf.random.set_seed(random_state)


    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Build LSTM model architecture
        
        Args:
            input_shape: (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        logger.info(f"Building LSTM model with input shape: {input_shape}")

        model = Sequential([
            Input(shape=input_shape),

            # First LSTM layer with return sequences
            LSTM(units=self.lstm_units_1, return_sequences=True, name='lstm_1'),
            Dropout(self.dropout_rate, name='dropout_1'),

            # Second LSTM layer
            LSTM(units=self.lstm_units_2, return_sequences=False, name='lstm_2'),
            Dropout(self.dropout_rate, name='dropout_2'),

            # Dense layers
            Dense(units=self.dense_units, activation='relu', name='dense_1'),
            Dense(units=1, name='output') # Regression output
        ], name='Gold_LSTM_Model')

        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss = 'mean_squared_error',
            metrics=['mae', 'mse']
        )

        logger.info(f"Model compiled with {model.count_params():,} parameters")

        return model
    

    def get_callbacks(self,
                      early_stopping_config: dict,
                      reduce_lr_config: dict,
                      model_checkpoint_path: str = None) -> List:
        """
        Create training callbacks
        
        Args:
            early_stopping_config: EarlyStopping parameters
            reduce_lr_config: ReduceLROnPlateau parameters
            model_checkpoint_path: Path to save best model
            
        Returns:
            List of callbacks
        """
        callbacks = []

        # Early stopping
        early_stop = EarlyStopping(**early_stopping_config)
        callbacks.append(early_stop)

        # Reduce learning rate on plateau
        reduce_lr = ReduceLROnPlateau(**reduce_lr_config)
        callbacks.append(reduce_lr)

        # Model checkpoint (optional)
        if model_checkpoint_path:
            checkpoint = ModelCheckpoint(
                model_checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            callbacks.append(checkpoint)

        logger.info(f"Created {len(callbacks)} callbacks")
        return callbacks
    
    def print_model_summary(self, model: keras.Model):
        """Print model artitecture summary"""
        model.summary()