"""
Complete Training Pipeline for Gold Price LSTM Model
"""
import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

# Import custom modules
from config.settings import Config
from data_preprocessor import DataPreprocessor
from model_builder import LSTMModelBuilder, ModelTrainer
from model_evaluator import ModelEvaluator, Visualizer


# Configure logging
def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)



class TrainingPipeline:
    """End-to-end training pipeline for LSTM model"""

    def __init__(self, config_obj):
        """
        Initialize training pipeline

        Args:
            config_obj: Configuration object
        """
        self.config = config_obj
        self.logger = setup_logging(self.config.paths.LOGS_DIR)

        self.preprocessor = None
        self.model_builder = None
        self.trainer = None
        self.evaluator = None
        
        self.data = None
        self.model = None
        self.history = None
        self.metrics = None

        self.logger.info("=" * 80)
        self.logger.info("Gold Price LSTM Training Pipeline Initialized")
        self.logger.info("=" * 80)


    def prepare_data(self):
        """Step 1: Prepare and preprocess data"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 1: Data Preparation")
        self.logger.info("=" * 80)

        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            feature_columns=self.config.data.FEATURE_COLUMNS,
            target_column=self.config.data.TARGET_COLUMN,
            test_split=self.config.data.TEST_SPLIT_RATIO,
            val_split=self.config.data.VAL_SPLIT_RATIO,
            random_state=self.config.data.RANDOM_STATE
        )

        # Prepare data
        self.data = self.preprocessor.prepare_data(
            filepath=self.config.paths.processed_data_path,
            sequence_length=self.config.model.SEQUENCE_LENGTH
        )

        # Save scaler
        scaler_X_path = self.config.paths.scaler_path.replace('.pkl', '_X.pkl')
        scaler_y_path = self.config.paths.scaler_path.replace('.pkl', '_y.pkl')
        self.preprocessor.save_scalers(scaler_X_path, scaler_y_path)

        self.logger.info(f"✓ Data preparation completed")
        self.logger.info(f"  - Train samples: {len(self.data['X_train'])}")
        self.logger.info(f"  - Validation samples: {len(self.data['X_val'])}")
        self.logger.info(f"  - Test samples: {len(self.data['X_test'])}")
        self.logger.info(f"  - Features: {self.data['X_train'].shape[2]}")


    def build_model(self):
        """Step 2: Build LSTM model"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 2: Model Building")
        self.logger.info("=" * 80)

        # Initialize model builder
        self.model_builder = LSTMModelBuilder(
            lstm_units_1 = self.config.model.LSTM_UNITS_1,
            lstm_units_2 = self.config.model.LSTM_UNITS_2,
            dense_units = self.config.model.DENSE_UNITS,
            dropout_rate = self.config.model.DROPOUT_RATE,
            learning_rate = self.config.model.LEARNING_RATE,
            random_state = self.config.data.RANDOM_STATE
        )

        # Build model
        input_shape = (self.data['X_train'].shape[1], self.data['X_train'].shape[2])
        self.model = self.model_builder.build_model(input_shape)

        # Print model summary
        self.logger.info("\nModel Architecture:")
        self.model_builder.print_model_summary(self.model)
        
        self.logger.info(f"✓ Model built successfully")


    def train_model(self):
        """Step 3: Train the model"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 3: Model Training")
        self.logger.info("=" * 80)
        
        # Initialize trainer
        self.trainer = ModelTrainer(self.model)

        # Prepare callbacks
        callbacks = self.model_builder.get_callbacks(
            early_stopping_config=self.config.model.get_early_stopping_config(),
            reduce_lr_config=self.config.model.get_reduce_lr_config(),
            model_checkpoint_path=self.config.paths.model_path
        )

        # Train model
        self.history = self.trainer.train(
            X_train=self.data['X_train'],
            y_train=self.data['y_train'],
            X_val=self.data['X_val'],
            y_val=self.data['y_val'],
            epochs=self.config.model.EPOCHS,
            batch_size=self.config.model.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.trainer.save_model(self.config.paths.model_path)

        # Save training history
        history_dict = self.trainer.get_training_history()
        history_path = self.config.paths.training_history_path
        with open(history_path, 'w') as f:
            # Convert numpy types to Python types for JSON serilization
            history_json = {k: [float(v) for v in vals] for k, vals in history_dict.items()}
            json.dump(history_path, f, indent=4)

        self.logger.info(f"✓ Training completed")
        self.logger.info(f"  - Total epochs: {len(history_dict['loss'])}")
        self.logger.info(f"  - Final train loss: {history_dict['loss'][-1]:.6f}")
        self.logger.info(f"  - Final val loss: {history_dict['val_loss'][-1]:.6f}")

    
    def evaluate_model(self):
        """Step 4: Evaluate model performance"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 4: Model Evaluation")
        self.logger.info("=" * 80)

        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.data['scaler_y'])

        # Evaluate on test set
        actual_prices, predicted_prices, self.metrics = self.evaluator.evaluate_predictions(
            model=self.model,
            X_test=self.data['X_test'],
            y_test=self.data['y_test'],
            base_prices=self.data['metadata']['test_prices']
        )

        # Print metrics
        self.logger.info("\nTest Set Metrics (Price Level):")
        self.logger.info(f"  - RMSE: {self.metrics['rmse']:,.2f} Toman")
        self.logger.info(f"  - MAE: {self.metrics['mae']:,.2f} Toman")
        self.logger.info(f"  - R² Score: {self.metrics['r2']:.4f}")
        self.logger.info(f"  - MAPE: {self.metrics['mape']:.2f}%")
        
        self.logger.info("\nTest Set Metrics (Log Return Level):")
        self.logger.info(f"  - RMSE: {self.metrics['log_return_rmse']:.6f}")
        self.logger.info(f"  - MAE: {self.metrics['log_return_mae']:.6f}")
        
        # Save metrics
        metrics_path = os.path.join(self.config.paths.LOGS_DIR, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)   

        self.logger.info(f"\n Evaluation completed")

        return actual_prices, predicted_prices
    

    def generate_visualization(self, actual_prices, predicted_prices):
        """Step 5: Generate visualizations"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STEP 5: Generating Visualizations")
        self.logger.info("=" * 80)

        # Training history plot
        history_plot_path = os.path.join(self.config.paths.PLOTS_DIR, 'tarining_history.png')
        Visualizer.plot_training_history(
            history=self.trainer.get_training_history(),
            save_path=history_plot_path
        )
        self.logger.info(f" Trining history plot saved")

        # Predictions plot
        predictions_plot_path = os.path.join(self.config.paths.PLOTS_DIR, 'predctions.png')
        Visualizer.plot_predictions(
            dates=self.data['metadata']['test_dates'],
            actual_prices=actual_prices,
            predicted_prices=predicted_prices,
            metrics=self.metrics,
            save_path=predictions_plot_path
        )
        self.logger.info(f" Predictions plot saved")

        # Residual plot
        residuals_plot_path = os.path.join(self.config.paths.PLOTS_DIR, 'residuals.png')
        Visualizer.plot_residuals(
            actual=actual_prices,
            predicted=predicted_prices,
            save_path=residuals_plot_path
        )
        self.logger.info(f"✓ Residuals plot saved")

    
    def run(self):
        """Execute complete trainig pipelines"""
        try:
            strat_time = datetime.now()
            self.logger.info(f"\nPipeline started at: {strat_time.strftime('%Y-%m-%d %H-%M-%S')}")

            # Step 1: Prepare data
            self.prepare_data()

            # Step 2: Build model
            self.build_model()

            # Step 3: Train model
            self.train_model()

            # Step 4: Evaluate model
            actual_prices, predicted_prices = self.evaluate_model()

            # Step 5: Generate visualizations
            self.generate_visualization(actual_prices, predicted_prices)

            # Caculate total time
            end_time = datetime.now()
            duration = end_time - strat_time

            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total duration: {duration}")
            self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("\nOutputs saved to:")
            self.logger.info(f"  - Model: {self.config.paths.model_path}")
            self.logger.info(f"  - Logs: {self.config.paths.LOGS_DIR}")
            self.logger.info(f"  - Plots: {self.config.paths.PLOTS_DIR}")

            return True
        
        except Exception as e:
            self.logger.error(f"\nPipeline failed with error: {str(e)}", exc_info=True)
            return False
        


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = TrainingPipeline(Config)

    # Run pipeline
    success = pipeline.run()

    if success:
        print(f"\n Training pipeline completed successfully!")
    else:
        print(f"\n Training pipeline failed. Check logs for details.")


if __name__ == "__main__":
    main()