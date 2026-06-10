"""
Example MLOps Worker for Background Tasks
This demonstrates scheduled retraining, monitoring, and MLflow integration
"""

import time
import logging
import mlflow
import requests
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"
FASTAPI_URL = "http://localhost:8000"
RETRAIN_INTERVAL = 3600  # 1 hour

class MLOpsWorker:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.last_retrain = None
        
    def check_model_drift(self):
        """Check for model drift by analyzing recent predictions"""
        try:
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            
            if experiments:
                runs = client.search_runs(
                    experiment_ids=[experiments[0].experiment_id],
                    max_results=100
                )
                
                if len(runs) > 10:
                    logger.info(f"Analyzed {len(runs)} recent predictions")
                    return True  # Trigger retrain
            return False
        except Exception as e:
            logger.error(f"Drift check error: {e}")
            return False
    
    def retrain_model(self):
        """Simulate model retraining and log to MLflow"""
        try:
            logger.info("Starting model retraining...")
            
            with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Simulate training
                mlflow.log_param("training_data_size", 1000)
                mlflow.log_param("algorithm", "RandomForest")
                
                # Log metrics
                mlflow.log_metric("accuracy", 0.95)
                mlflow.log_metric("precision", 0.93)
                mlflow.log_metric("recall", 0.94)
                
                mlflow.set_tag("retrain_type", "scheduled")
                
            logger.info("Retraining complete")
            self.last_retrain = datetime.now()
            
            # Notify FastAPI to reload model
            try:
                requests.post(f"{FASTAPI_URL}/model/reload", timeout=5)
                logger.info("Notified FastAPI to reload model")
            except:
                logger.warning("Could not notify FastAPI")
                
        except Exception as e:
            logger.error(f"Retraining error: {e}")
    
    def run(self):
        """Main worker loop"""
        logger.info("MLOps Worker started")
        
        while True:
            try:
                # Check if retrain needed
                should_retrain = False
                
                if self.last_retrain is None:
                    should_retrain = True
                elif (datetime.now() - self.last_retrain).seconds > RETRAIN_INTERVAL:
                    should_retrain = True
                elif self.check_model_drift():
                    should_retrain = True
                
                if should_retrain:
                    self.retrain_model()
                
                # Heartbeat
                logger.info(f"Worker heartbeat: {datetime.now()}")
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    worker = MLOpsWorker()
    worker.run()
