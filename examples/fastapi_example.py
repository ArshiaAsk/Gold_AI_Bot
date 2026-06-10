"""
Example FastAPI Application with MLflow Integration
This demonstrates how to serve ML models and log predictions to MLflow
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import mlflow
import mlflow.pyfunc
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model API", version="1.0.0")

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "production-model"
model = None
model_version = None

class PredictionRequest(BaseModel):
    features: List[float]
    log_prediction: bool = True

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: Optional[str]
    timestamp: str

@app.on_event("startup")
async def load_model():
    global model, model_version
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        model_version = "1.0"
    except Exception as e:
        logger.warning(f"Using demo mode: {e}")

@app.get("/")
async def root():
    return {"service": "ML Model API", "status": "running"}

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "fastapi"})

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    try:
        if model is not None:
            input_array = np.array([request.features])
            prediction = float(model.predict(input_array)[0])
            confidence = 0.95
        else:
            prediction = float(np.mean(request.features))
            confidence = 0.75
        
        if request.log_prediction:
            background_tasks.add_task(log_to_mlflow, request.features, prediction)
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=model_version,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def log_to_mlflow(features, prediction):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run():
            mlflow.log_metric("prediction", prediction)
    except Exception as e:
        logger.error(f"MLflow error: {e}")
