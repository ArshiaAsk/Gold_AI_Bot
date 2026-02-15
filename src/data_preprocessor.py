"""
Data preprocessing utilites for LSTM model
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import joblib 
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data preprocessing for LSTM training"""

    def __init__(self, feature_columns: list, target_column: str,
                 test_split: float = 0.15, val_split: float = 0.15,
                 random_state: int = 42):
        """
         Initialize preprocessor
        
        Args:
            feature_columns: List of feature column names
            target_column: Target column name
            test_split: Test set ratio
            val_split: Validation set ratio
            random_state: Random seed for reproducibility
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.test_split = test_split
        self.val_split = val_split
        self.random_state = random_state
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        np.random.seed(random_state)

    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load and validate dataset

        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded dataframe
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)

        # Validate required columns
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Missing target column: {self.target_column}")
        
        # Remove rows with NaN
        initial_rows = len(df)
        df = df.dropna(subset=self.feature_columns + [self.target_column])
        removed_rows = initial_rows - len(df)

        if removed_rows > 0:
            logger.warning(f"Reomved {removed_rows} rows with NaN values")

        logger.info(f"Loaded {len(df)} samples")
        return df
    

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets (time-series aware)
        
        Args:
            df: Input dataframe
            
        Returns:
            train_df, val_df, test_df
        """
        n_samples = len(df)
        test_size = int(n_samples * self.test_split)
        val_size = int(n_samples * self.val_split)
        train_size = n_samples - test_size - val_size
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size + val_size].copy()
        test_df = df.iloc[train_size + val_size:].copy()
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    

    def scale_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                       test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features and target using StandardScaler
        
        Args:
            train_df, val_df, test_df: DataFrames for each split
            
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test (all scaled)
        """
        # Fit scaler on training data only
        X_train = self.scaler_X.fit_transform(train_df[self.feature_columns])
        y_train = self.scaler_y.fit_transform(train_df[[self.target_column]])

        # Transform validation and test data
        X_val = self.scaler_X.transform(val_df[self.feature_columns])
        y_val = self.scaler_y.transform(val_df[[self.target_column]])
        
        X_test = self.scaler_X.transform(test_df[self.feature_columns])
        y_test = self.scaler_y.transform(test_df[[self.target_column]])

        logger.info(f"Features scaled - X Shape: {X_train.shape}, y shape: {y_train.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test
    

    def reshape_for_lstm(self, X: np.ndarray, sequence_length: int = 1) -> np.ndarray:
        """
         Reshape data for LSTM input: (samples, timesteps, features)
        
        Args:
            X: Input array of shape (samples, features)
            sequence_length: Number of timesteps
            
        Returns:
            Reshaped array of shape (samples, timesteps, features)
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if len(X) < sequence_length:
            return np.empty((0, sequence_length, X.shape[1]))

        X_3d = []
        for i in range(len(X) - sequence_length + 1):
            window = X[i: i + sequence_length]
            X_3d.append(window)

        return np.array(X_3d)

        # return X.reshape((X.shape[0], sequence_length, X.shape[1]))
    

    def prepare_data(self, filepath: str, sequence_length: int) -> Dict:
        """
        Complete data preparation pipeline
        
        Args:
            filepath: Path to CSV file
            sequence_length: LSTM sequence length
            
        Returns:
            Dictionary containing all prepared data
        """
        # Load data
        df = self.load_data(filepath)

        # Split data
        train_df, val_df, test_df = self.split_data(df)

        # Scale Feature
        X_train, y_train, X_val, y_val, X_test, y_test = self.scale_features(
            train_df, val_df, test_df
        )

        # Reshape for LSTM
        X_train = self.reshape_for_lstm(X_train, sequence_length)
        X_val = self.reshape_for_lstm(X_val, sequence_length)
        X_test = self.reshape_for_lstm(X_test, sequence_length)

        # Align target with the end of each rolling feature window
        y_train = y_train[sequence_length - 1:]
        y_val = y_val[sequence_length - 1:]
        y_test = y_test[sequence_length - 1:]

        # Store metadata for price reconstruction
        metadata = {
            'train_dates': train_df['Date'].values[sequence_length - 1:] if 'Date' in train_df.columns else None,
            'val_dates': val_df['Date'].values[sequence_length - 1:] if 'Date' in val_df.columns else None,
            'test_dates': test_df['Date'].values[sequence_length - 1:] if 'Date' in test_df.columns else None,
            'train_prices': train_df['Gold_IRR'].values[sequence_length - 1:] if 'Gold_IRR' in train_df.columns else None,
            'val_prices': val_df['Gold_IRR'].values[sequence_length - 1:] if 'Gold_IRR' in val_df.columns else None,
            'test_prices': test_df['Gold_IRR'].values[sequence_length - 1:] if 'Gold_IRR' in test_df.columns else None,
        }

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'metadata': metadata,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }


    def save_scalers(self, scaler_X_path: str, scaler_y_path: str):
        """Save fitted scalers"""
        joblib.dump(self.scaler_X, scaler_X_path)
        joblib.dump(self.scaler_y, scaler_y_path)
        logger.info(f"Scalers saved to {scaler_X_path} and {scaler_y_path}")


    def load_scalers(self, scaler_X_path: str, scaler_y_path: str):
        """Load fitted scalers"""
        self.scaler_X = joblib.load(scaler_X_path)
        self.scaler_y = joblib.load(scaler_y_path)
        logger.info(f"Scalers loaded from {scaler_X_path} and {scaler_y_path}")
