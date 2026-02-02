"""
FastAPI Application for Gold Price Prediction
"""
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path

# Import our custom modules
from .predictor import GoldPricePredictor, FeatureBuilder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Pydantic Schemas ---
class PredictionRequest(BaseModel):
    """Request model for price prediction"""
    features: List[List[float]] = Field(
        ...,
        description="Feature array of shape (sequence_length, n_features)",
        min_items=30,
        max_items=30
    )
    current_price: float = Field(
        ...,
        description="Current gold price in Toman",
        gt=0
    )

    @validator('features')
    def validate_features(cls, v):
        if not all(len(row) == 15 for row in v):
            raise ValueError("Each feature row must have exactly 15 features")
        return v


class HistoricalDataRequest(BaseModel):
    """Request model using historical data"""
    historical_data: List[Dict[str, float]] = Field(
        ...,
        description="Historical market data (at least 30 days)",
        min_items=30
    )
    current_price: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    success: bool
    prediction: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: bool
    model_loaded: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    detail: Optional[str] = None


# --- Global State & Lifespan ---

# Dictionary to store loaded models and tools
model_artifacts = {
    "predictor": None,
    "feature_builder": None,
    "status": "startup"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    Runs before the app starts accepting requests (startup)
    and after it finishes (shutdown).
    """
    logger.info("🚀 Starting up... Loading models.")
    
    try:
        # Define paths (Relative to project root)
        # Assuming run from project root: python src/api/main.py
        base_path = Path(__file__).resolve().parent.parent.parent
        model_path = base_path / "models" / "gold_lstm_v2.keras"
        scaler_X_path = base_path / "models" / "scaler_X.pkl"
        scaler_y_path = base_path / "models" / "scaler_y.pkl"

        logger.info(f"Looking for model at: {model_path}")

        # Check if files exist to give better error messages
        if not model_path.exists():
            logger.warning(f"⚠️ Model file not found at {model_path}")
            # We don't raise here, so the server can still start for debugging
        else:
            # Load predictor
            predictor = GoldPricePredictor(
                model_path=str(model_path),
                scaler_X_path=str(scaler_X_path),
                scaler_y_path=str(scaler_y_path)
            )
            model_artifacts["predictor"] = predictor

            # Initialize feature builder
            feature_builder = FeatureBuilder(sequence_length=30)
            model_artifacts["feature_builder"] = feature_builder
            
            model_artifacts["status"] = "ready"
            logger.info("✅ Model and predictor loaded successfully")
            
    except Exception as e:
        logger.error(f"❌ Critical error during startup: {e}")
        model_artifacts["status"] = "failed"
        # We purposely do NOT raise e, to keep the /health endpoint alive

    yield  # Application runs here

    # Shutdown logic
    logger.info("🛑 Shutting down...")
    model_artifacts.clear()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Gold Price Prediction API",
    description="LSTM-based Gold Price Prediction Service for Iranian Market",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- Endpoints ---

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Gold Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    is_ready = model_artifacts["status"] == "ready"
    return HealthResponse(
        status=True, # The server itself is running
        model_loaded=is_ready,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """  
    Predict next day gold price using raw features
    """
    predictor = model_artifacts.get("predictor")
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check server logs."
        )
    
    try:
        # Convert to numpy array
        features = np.array(request.features)

        # Make prediction
        result = predictor.predict_price(features, request.current_price)

        return PredictionResponse(
            success=True,
            prediction=result,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction Failed: {str(e)}"
        )


@app.post("/predict-with-confidence", response_model=PredictionResponse)
async def predict_with_confidence(
    request: PredictionRequest,
    n_simulations: int 
):
    """
    Predict with Monte Carlo confidence intervals
    """
    predictor = model_artifacts.get("predictor")
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
    
    try:
        features = np.array(request.features)

        result = predictor.predict_with_confidence(
            features,
            request.current_price,
            n_simulations=n_simulations
        )

        return PredictionResponse(
            success=True,
            prediction=result,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-from-history", response_model=PredictionResponse)
async def predict_from_history(request: HistoricalDataRequest):
    """
    Predict using historical market data (auto feature extraction)
    """
    predictor = model_artifacts.get("predictor")
    feature_builder = model_artifacts.get("feature_builder")
    
    if predictor is None or feature_builder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or Feature Builder not loaded."
        )
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.historical_data)

        # Build features
        features = feature_builder.build_features_from_history(df)

        # Make Prediction
        result = predictor.predict_price(features, request.current_price)

        return PredictionResponse(
            success=True,
            prediction=result,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as ve:
        # Feature validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    predictor = model_artifacts.get("predictor")
    
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try: 
        model_summary = {
            "model_path": predictor.model_path,
            "input_shape": str(predictor.model.input_shape),
            "output_shape": str(predictor.model.output_shape),
            # Count params might be tricky if model isn't built fully, but load_model usually builds it
            "total_parameters": int(predictor.model.count_params()),
            "expected_features": {
                "sequence_length": 30,
                "n_features": 15,
                "features_name": [
                    'Gold_LogRet', 'USD_LogRet', 'Ounce_LogRet', 'Oil_LogRet',
                    'SMA_7', 'RSI_14', 'MACD', 'MACD_Signal',
                    'Bollinger_Upper', 'Bollinger_Lower',
                    'Gold_LogRet_Lag_1', 'Gold_LogRet_Lag_2', 'Gold_LogRet_Lag_3',
                    'USD_LogRet_Lag_1', 'USD_LogRet_Lag_2'
                ]
            }
        }

        return model_summary
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {e}"
        )


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"success": False, "error": "Invalid Input", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    # Use reload=True for development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
