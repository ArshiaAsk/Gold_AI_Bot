"""
FastAPI Application for Gold Price Prediction
"""
from fastapi import FastAPI, HTTPException, status, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import time
import shutil
import resource
from pathlib import Path

from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from config.logging import setup_logging
from src.mlops.api_integration import initialize_mlops
from src.mlops.background_tasks import UptimeTracker
from src.mlops.audit_store import AuditStore
from src.api.security import (
    AUTH_REQUIRED,
    RATE_LIMIT_ENABLED,
    rate_limiter,
    require_api_key,
    client_key,
)

try:
    # Package imports
    from src.api.predictor import GoldPricePredictor, FeatureBuilder
except ImportError:
    # Script-style import
    from predictor import GoldPricePredictor, FeatureBuilder

# Setup logging
setup_logging(base_log_dir="logs", level=os.getenv("LOG_LEVEL", "INFO"))
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
        arr = np.array(v, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("Features must contain only finite numeric values")
        if np.max(np.abs(arr)) > 1e9:
            raise ValueError("Features are out of expected numeric range")
        return v


class HistoricalDataRequest(BaseModel):
    """Request model using historical data"""
    historical_data: List[Dict[str, float]] = Field(
        ...,
        description="Historical market data (at least 30 days)",
        min_items=30
    )
    current_price: float = Field(..., gt=0)

    @validator("historical_data")
    def validate_historical_data(cls, value):
        if not value:
            raise ValueError("historical_data cannot be empty")
        for idx, row in enumerate(value):
            for key, item in row.items():
                try:
                    numeric = float(item)
                except Exception as exc:
                    raise ValueError(f"historical_data[{idx}]['{key}'] must be numeric") from exc
                if not np.isfinite(numeric):
                    raise ValueError(f"historical_data[{idx}]['{key}'] must be finite")
        return value


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
    checks: Dict[str, Any] = Field(default_factory=dict)


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

mlops_artifacts = {
    "integration": None,
    "uptime_tracker": UptimeTracker(),
    "audit_store": None,
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

        logger.info("Looking for model at: %s", model_path)

        # Check if files exist to give better error messages
        if not model_path.exists():
            logger.warning("⚠️ Model file not found at %s", model_path)
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

        integration = initialize_mlops()
        mlops_artifacts["integration"] = integration
        logger.info("✅ MLOps integration initialized")
        audit_store = AuditStore.from_env()
        mlops_artifacts["audit_store"] = audit_store
        logger.info("✅ Audit store initialized (healthy=%s)", audit_store.ping())
            
    except Exception as e:
        logger.error("❌ Critical error during startup: %s", e)
        model_artifacts["status"] = "failed"
        # We purposely do NOT raise e, to keep the /health endpoint alive

    yield  # Application runs here

    # Shutdown logic
    logger.info("🛑 Shutting down...")
    integration = mlops_artifacts.get("integration")
    if integration is not None:
        integration.stop()
    mlops_artifacts["integration"] = None
    mlops_artifacts["audit_store"] = None
    model_artifacts.clear()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Gold Price Prediction API",
    description="LSTM-based Gold Price Prediction Service for Iranian Market",
    version="1.0.0",
    lifespan=lifespan
)

def _cors_origins() -> List[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://localhost:8000")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["http://localhost:3000", "http://localhost:8000"]


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

PROTECTED_PATH_PREFIXES = ("/predict", "/mlops", "/model-info")
EXEMPT_PATHS = ("/", "/docs", "/openapi.json", "/health", "/metrics")


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not RATE_LIMIT_ENABLED:
        return await call_next(request)

    path = request.url.path
    if path in EXEMPT_PATHS:
        return await call_next(request)
    if not any(path.startswith(prefix) for prefix in PROTECTED_PATH_PREFIXES):
        return await call_next(request)

    key = client_key(request)
    if not rate_limiter.allow(key):
        retry_after = rate_limiter.retry_after_seconds(key)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded"},
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


# --- Endpoints ---

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Gold Price Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
        "mlops_health": "/mlops/health",
        "mlops_registry": "/mlops/model-registry",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    is_ready = model_artifacts["status"] == "ready"
    audit_store = mlops_artifacts.get("audit_store")
    checks = _health_checks(audit_store=audit_store, model_ready=is_ready)
    overall_ok = is_ready and checks["disk_ok"] and checks["memory_ok"] and checks["audit_store_ok"]
    return HealthResponse(
        status=overall_ok,
        model_loaded=is_ready,
        timestamp=datetime.now().isoformat(),
        checks=checks,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_price(
    request: PredictionRequest,
    http_request: Request,
    _: None = Depends(require_api_key),
):
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
        start = time.perf_counter()
        # Convert to numpy array
        features = np.array(request.features)

        # Make prediction
        result = predictor.predict_price(features, request.current_price)
        latency = time.perf_counter() - start

        integration = mlops_artifacts.get("integration")
        if integration is not None:
            integration.metrics_exporter.record_prediction(latency_seconds=latency, confidence=None)
        _log_prediction_audit(
            endpoint="/predict",
            request=http_request,
            result=result,
            latency_seconds=latency,
            success=True,
        )

        return PredictionResponse(
            success=True,
            prediction=result,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error("Prediction error: %s", e)
        integration = mlops_artifacts.get("integration")
        if integration is not None:
            integration.metrics_exporter.record_api_error()
        mlops_artifacts["uptime_tracker"].record_error()
        _log_prediction_audit(
            endpoint="/predict",
            request=http_request,
            result=None,
            latency_seconds=None,
            success=False,
            error_text=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction Failed: {str(e)}"
        )


@app.post("/predict-with-confidence", response_model=PredictionResponse)
async def predict_with_confidence(
    request: PredictionRequest,
    http_request: Request,
    n_simulations: int = Query(..., ge=10, le=10000),
    _: None = Depends(require_api_key),
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
        start = time.perf_counter()
        features = np.array(request.features)

        result = predictor.predict_with_confidence(
            features,
            request.current_price,
            n_simulations=n_simulations
        )
        latency = time.perf_counter() - start

        integration = mlops_artifacts.get("integration")
        if integration is not None:
            integration.metrics_exporter.record_prediction(latency_seconds=latency, confidence=None)
        _log_prediction_audit(
            endpoint="/predict-with-confidence",
            request=http_request,
            result=result,
            latency_seconds=latency,
            success=True,
        )

        return PredictionResponse(
            success=True,
            prediction=result,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error("Prediction error: %s", e)
        integration = mlops_artifacts.get("integration")
        if integration is not None:
            integration.metrics_exporter.record_api_error()
        mlops_artifacts["uptime_tracker"].record_error()
        _log_prediction_audit(
            endpoint="/predict-with-confidence",
            request=http_request,
            result=None,
            latency_seconds=None,
            success=False,
            error_text=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-from-history", response_model=PredictionResponse)
async def predict_from_history(
    request: HistoricalDataRequest,
    http_request: Request,
    _: None = Depends(require_api_key),
):
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
        start = time.perf_counter()
        # Convert to DataFrame
        df = pd.DataFrame(request.historical_data)

        # Build features
        features = feature_builder.build_features_from_history(df)

        # Make Prediction
        result = predictor.predict_price(features, request.current_price)
        latency = time.perf_counter() - start

        integration = mlops_artifacts.get("integration")
        if integration is not None:
            integration.metrics_exporter.record_prediction(latency_seconds=latency, confidence=None)
        _log_prediction_audit(
            endpoint="/predict-from-history",
            request=http_request,
            result=result,
            latency_seconds=latency,
            success=True,
        )

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
        logger.error("Prediction error: %s", e)
        integration = mlops_artifacts.get("integration")
        if integration is not None:
            integration.metrics_exporter.record_api_error()
        mlops_artifacts["uptime_tracker"].record_error()
        _log_prediction_audit(
            endpoint="/predict-from-history",
            request=http_request,
            result=None,
            latency_seconds=None,
            success=False,
            error_text=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    integration = mlops_artifacts.get("integration")
    if integration is not None:
        uptime = mlops_artifacts["uptime_tracker"].uptime_ratio()
        integration.metrics_exporter.set_api_uptime_ratio(uptime)
        payload = generate_latest(integration.metrics_exporter.registry)
    else:
        payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.get("/mlops/health")
async def mlops_health(_: None = Depends(require_api_key)):
    """MLOps runtime health endpoint."""
    integration = mlops_artifacts.get("integration")
    if integration is None:
        return {"status": "degraded", "reason": "MLOps integration is not initialized"}
    return integration.health()


@app.get("/mlops/model-registry")
async def mlops_model_registry(name: Optional[str] = None, _: None = Depends(require_api_key)):
    """List model versions in registry."""
    integration = mlops_artifacts.get("integration")
    if integration is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLOps integration is not initialized",
        )
    return {"models": integration.list_registry(name=name)}


@app.get("/model-info")
async def get_model_info(_: None = Depends(require_api_key)):
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

def _health_checks(audit_store: Optional[AuditStore], model_ready: bool) -> Dict[str, Any]:
    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / "models" / "gold_lstm_v2.keras"
    disk = shutil.disk_usage(str(base_path))
    disk_free_gb = round(disk.free / (1024**3), 2)
    disk_min_gb = float(os.getenv("HEALTH_DISK_MIN_FREE_GB", "1.0"))
    disk_ok = disk_free_gb >= disk_min_gb

    memory_max_mb = float(os.getenv("HEALTH_MEMORY_MAX_MB", "4096"))
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = round(rss_kb / 1024.0, 2)
    memory_ok = rss_mb <= memory_max_mb

    freshness_hours = None
    model_fresh = False
    max_model_age_hours = float(os.getenv("MODEL_MAX_AGE_HOURS", "720"))
    if model_path.exists():
        age_seconds = max(0.0, time.time() - model_path.stat().st_mtime)
        freshness_hours = round(age_seconds / 3600.0, 2)
        model_fresh = freshness_hours <= max_model_age_hours

    audit_ok = audit_store.ping() if audit_store else False

    return {
        "auth_required": AUTH_REQUIRED,
        "disk_free_gb": disk_free_gb,
        "disk_ok": disk_ok,
        "memory_rss_mb": rss_mb,
        "memory_ok": memory_ok,
        "model_freshness_hours": freshness_hours,
        "model_fresh": model_fresh if model_ready else False,
        "audit_store_ok": audit_ok,
    }


def _log_prediction_audit(
    *,
    endpoint: str,
    request: Request,
    result: Optional[Dict[str, Any]],
    latency_seconds: Optional[float],
    success: bool,
    error_text: Optional[str] = None,
) -> None:
    audit_store = mlops_artifacts.get("audit_store")
    if audit_store is None:
        return

    integration = mlops_artifacts.get("integration")
    model_version = None
    if integration is not None:
        production = integration.pipeline.registry.get_current_production_model()
        if production:
            model_version = production.get("version")

    predicted_price = None
    confidence_lower = None
    confidence_upper = None
    current_price = None
    if result:
        predicted_price = result.get("predicted_price")
        current_price = result.get("current_price")
        ci = result.get("confidence_interval_95")
        if isinstance(ci, dict):
            confidence_lower = ci.get("lower")
            confidence_upper = ci.get("upper")

    try:
        audit_store.log_prediction(
            endpoint=endpoint,
            client_ip=request.client.host if request.client else None,
            model_version=model_version,
            current_price=current_price,
            predicted_price=predicted_price,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            latency_ms=(latency_seconds * 1000.0) if latency_seconds is not None else None,
            success=success,
            error_text=error_text,
        )
    except Exception as exc:
        logger.warning("Failed to persist prediction audit log: %s", exc)


if __name__ == "__main__":
    import uvicorn
    # Use reload=True for development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
