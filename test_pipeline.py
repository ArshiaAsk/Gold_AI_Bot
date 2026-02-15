"""
Unit Tests for Gold Price LSTM Pipeline

Run with: pytest test_pipeline.py -v
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.settings import config, DataConfig, ModelConfig
from src.data_preprocessor import DataPreprocessor
from src.model_builder import LSTMModelBuilder, ModelTrainer
from src.model_evaluator import ModelEvaluator


class TestDataPreprocessor(unittest.TestCase):
    """Test data preprocessing functionality"""
    
    def setUp(self):
        """Create sample data for testing"""
        self.feature_cols = ['feature1', 'feature2', 'feature3']
        self.target_col = 'target'
        
        # Create sample DataFrame
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=n_samples),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'target': np.random.randn(n_samples),
            'Gold_IRR': np.random.randint(10000000, 12000000, n_samples)
        })
        
        # Save to temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        self.df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        self.preprocessor = DataPreprocessor(
            feature_columns=self.feature_cols,
            target_column=self.target_col,
            test_split=0.2,
            val_split=0.2
        )
    
    def tearDown(self):
        """Clean up temporary file"""
        os.unlink(self.temp_file.name)
    
    def test_load_data(self):
        """Test data loading"""
        df = self.preprocessor.load_data(self.temp_file.name)
        self.assertEqual(len(df), 100)
        self.assertIn('feature1', df.columns)
    
    def test_split_data(self):
        """Test train/val/test split"""
        df = self.preprocessor.load_data(self.temp_file.name)
        train, val, test = self.preprocessor.split_data(df)
        
        # Check sizes
        self.assertEqual(len(train) + len(val) + len(test), len(df))
        self.assertGreater(len(train), len(val))
        self.assertGreater(len(train), len(test))
    
    def test_scale_features(self):
        """Test feature scaling"""
        df = self.preprocessor.load_data(self.temp_file.name)
        train, val, test = self.preprocessor.split_data(df)
        
        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocessor.scale_features(
            train, val, test
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[1], len(self.feature_cols))
        self.assertEqual(y_train.shape[1], 1)
        
        # Check scaling (mean should be close to 0, std close to 1)
        self.assertAlmostEqual(np.mean(X_train), 0, places=1)
        self.assertAlmostEqual(np.std(X_train), 1, places=1)
    
    def test_reshape_for_lstm(self):
        """Test LSTM reshaping"""
        X = np.random.randn(10, 5)  # 10 samples, 5 features
        X_reshaped = self.preprocessor.reshape_for_lstm(X, sequence_length=1)
        
        self.assertEqual(X_reshaped.shape, (10, 1, 5))
    
    def test_prepare_data_pipeline(self):
        """Test complete data preparation pipeline"""
        data = self.preprocessor.prepare_data(self.temp_file.name, sequence_length=1)
        
        # Check all required keys
        required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'metadata']
        for key in required_keys:
            self.assertIn(key, data)
        
        # Check shapes
        self.assertEqual(data['X_train'].ndim, 3)  # (samples, timesteps, features)
        self.assertEqual(data['y_train'].ndim, 2)  # (samples, 1)


class TestModelBuilder(unittest.TestCase):
    """Test model building functionality"""
    
    def setUp(self):
        """Initialize model builder"""
        self.builder = LSTMModelBuilder(
            lstm_units_1=32,
            lstm_units_2=16,
            dense_units=8,
            dropout_rate=0.2,
            learning_rate=0.001,
            random_state=42
        )
    
    def test_build_model(self):
        """Test model architecture"""
        model = self.builder.build_model(input_shape=(1, 10))
        
        # Check model exists
        self.assertIsNotNone(model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 1, 10))
        
        # Check output shape
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check trainable parameters
        self.assertGreater(model.count_params(), 0)
    
    def test_model_compilation(self):
        """Test model compilation"""
        model = self.builder.build_model(input_shape=(1, 5))
        
        # Check optimizer
        self.assertEqual(model.optimizer.__class__.__name__, 'Adam')
        
        # Check loss
        self.assertEqual(model.loss, 'mean_squared_error')
    
    def test_callbacks_creation(self):
        """Test callbacks creation"""
        callbacks = self.builder.get_callbacks(
            early_stopping_config={'monitor': 'val_loss', 'patience': 10},
            reduce_lr_config={'monitor': 'val_loss', 'factor': 0.5, 'patience': 5}
        )
        
        self.assertEqual(len(callbacks), 2)


class TestModelTrainer(unittest.TestCase):
    """Test model training functionality"""
    
    def setUp(self):
        """Create sample data and model"""
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.randn(50, 1, 5)
        self.y_train = np.random.randn(50, 1)
        self.X_val = np.random.randn(10, 1, 5)
        self.y_val = np.random.randn(10, 1)
        
        # Build model
        builder = LSTMModelBuilder(
            lstm_units_1=16,
            lstm_units_2=8,
            dense_units=8,
            dropout_rate=0.2,
            learning_rate=0.001,
            random_state=42
        )
        self.model = builder.build_model(input_shape=(1, 5))
        self.trainer = ModelTrainer(self.model)
    
    def test_training(self):
        """Test model training"""
        history = self.trainer.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=2,
            batch_size=16,
            verbose=0
        )
        
        # Check history exists
        self.assertIsNotNone(history)
        self.assertIn('loss', history.history)
        self.assertIn('val_loss', history.history)
    
    def test_evaluation(self):
        """Test model evaluation"""
        # Train briefly
        self.trainer.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=1,
            batch_size=16,
            verbose=0
        )
        
        # Evaluate
        metrics = self.trainer.evaluate(self.X_val, self.y_val, verbose=0)
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('mae', metrics)
        self.assertGreater(metrics['loss'], 0)


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation functionality"""
    
    def setUp(self):
        """Create sample data and evaluator"""
        from sklearn.preprocessing import StandardScaler
        
        # Create and fit a scaler
        self.scaler_y = StandardScaler()
        y_sample = np.random.randn(100, 1)
        self.scaler_y.fit(y_sample)
        
        self.evaluator = ModelEvaluator(self.scaler_y)
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = self.evaluator.calculate_metrics(y_true, y_pred)
        
        # Check all metrics exist
        required_metrics = ['rmse', 'mae', 'mse', 'r2', 'mape']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertGreater(metrics[metric], 0)
    
    def test_reconstruct_prices(self):
        """Test price reconstruction from log returns"""
        base_prices = np.array([10000000, 11000000, 12000000])
        log_returns = np.array([0.01, -0.02, 0.005])
        
        prices = self.evaluator.reconstruct_prices(base_prices, log_returns)
        
        # Check reconstruction
        self.assertEqual(len(prices), len(base_prices))
        expected_0 = base_prices[0] * np.exp(log_returns[0])
        self.assertAlmostEqual(prices[0], expected_0, places=2)


class TestConfiguration(unittest.TestCase):
    """Test configuration classes"""
    
    def test_data_config(self):
        """Test DataConfig"""
        data_config = DataConfig()
        
        self.assertEqual(data_config.LOOKBACK_DAYS, 30)
        self.assertEqual(data_config.RANDOM_STATE, 42)
        self.assertIsNotNone(data_config.FEATURE_COLUMNS)
    
    def test_model_config(self):
        """Test ModelConfig"""
        model_config = ModelConfig()
        
        self.assertEqual(model_config.LSTM_UNITS_1, 64)
        self.assertEqual(model_config.LEARNING_RATE, 0.001)
        self.assertGreater(model_config.EPOCHS, 0)
        
        # Test config methods
        optimizer_config = model_config.get_optimizer_config()
        self.assertIn('learning_rate', optimizer_config)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestModelBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
