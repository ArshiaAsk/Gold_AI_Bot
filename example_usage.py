"""
Example: How to use the Gold Price LSTM Training Pipeline

This script demonstrates the complete workflow from data preparation to model evaluation.
"""

import sys
import os

# Add src to path if running from root directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config_settings import config
from train_pipeline import TrainingPipeline


def main():
    """
    Main execution example
    """
    
    print("=" * 80)
    print("Gold Price LSTM Prediction - Training Example")
    print("=" * 80)
    print()
    
    # Display current configuration
    print("Current Configuration:")
    print(f"  Data Path: {config.paths.processed_data_path}")
    print(f"  Model Save Path: {config.paths.model_path}")
    print(f"  LSTM Units: [{config.model.LSTM_UNITS_1}, {config.model.LSTM_UNITS_2}]")
    print(f"  Learning Rate: {config.model.LEARNING_RATE}")
    print(f"  Epochs: {config.model.EPOCHS}")
    print(f"  Batch Size: {config.model.BATCH_SIZE}")
    print()
    
    # Ask for confirmation
    response = input("Start training with these settings? (y/n): ")
    
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("Starting Training Pipeline...")
    print("=" * 80 + "\n")
    
    # Initialize and run pipeline
    pipeline = TrainingPipeline(config)
    success = pipeline.run()
    
    # Display results
    print("\n" + "=" * 80)
    if success:
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\nYou can find the outputs at:")
        print(f"  📁 Model: {config.paths.model_path}")
        print(f"  📁 Logs: {config.paths.LOGS_DIR}")
        print(f"  📁 Plots: {config.paths.PLOTS_DIR}")
        print("\nNext steps:")
        print("  1. Review the training plots in outputs/plots/")
        print("  2. Check test metrics in logs/test_metrics.json")
        print("  3. Use the trained model for predictions")
    else:
        print("❌ TRAINING FAILED")
        print("=" * 80)
        print("\nPlease check the log files for error details.")
        print(f"  📁 Logs: {config.paths.LOGS_DIR}")


def quick_predict_example():
    """
    Example of making predictions with a trained model
    """
    import numpy as np
    import joblib
    from tensorflow import keras
    
    print("\n" + "=" * 80)
    print("Quick Prediction Example")
    print("=" * 80 + "\n")
    
    # Load trained model and scalers
    model = keras.models.load_model(config.paths.model_path)
    scaler_X = joblib.load(config.paths.scaler_path.replace('.pkl', '_X.pkl'))
    scaler_y = joblib.load(config.paths.scaler_path.replace('.pkl', '_y.pkl'))
    
    print("✓ Model and scalers loaded")
    
    # Example: Create dummy features for prediction
    # In real use, these would be actual market data
    n_features = len(config.data.FEATURE_COLUMNS)
    sample_features = np.random.randn(1, n_features)
    
    print(f"✓ Sample features shape: {sample_features.shape}")
    
    # Preprocess
    features_scaled = scaler_X.transform(sample_features)
    features_reshaped = features_scaled.reshape(1, 1, n_features)
    
    # Predict
    log_return_scaled = model.predict(features_reshaped, verbose=0)
    predicted_log_return = scaler_y.inverse_transform(log_return_scaled)[0, 0]
    
    print(f"\n📊 Predicted Log Return: {predicted_log_return:.6f}")
    
    # Convert to price change percentage
    price_change_pct = (np.exp(predicted_log_return) - 1) * 100
    print(f"📊 Predicted Price Change: {price_change_pct:+.2f}%")
    
    # Example base price
    current_price = 11_000_000  # Toman
    predicted_price = current_price * np.exp(predicted_log_return)
    
    print(f"\nIf current price is {current_price:,} Toman:")
    print(f"Predicted next day price: {predicted_price:,.0f} Toman")
    print(f"Change: {predicted_price - current_price:+,.0f} Toman")


if __name__ == "__main__":
    # Run training
    main()
    
    # Optionally, run prediction example after training
    # Uncomment the line below to test predictions
    # quick_predict_example()
