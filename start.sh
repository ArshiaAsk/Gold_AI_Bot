#!/bin/bash

# ============================================================================
# Multi-Service ML Stack Startup Script for Hugging Face Spaces
# ============================================================================
# This script initializes the environment and starts all services via supervisord

set -e  # Exit on error

echo "=========================================="
echo "Starting ML Stack Initialization"
echo "=========================================="

# Environment variables
export APP_HOME=${APP_HOME:-/app}
export MLFLOW_HOME=${MLFLOW_HOME:-/mlflow}
export LOGS_HOME=${LOGS_HOME:-/logs}
export DATA_HOME=${DATA_HOME:-/data}
export MLFLOW_TRACKING_URI="http://localhost:5000"
export FASTAPI_URL="http://localhost:8000"

# Display environment info
echo "Application Home: ${APP_HOME}"
echo "MLflow Home: ${MLFLOW_HOME}"
echo "Data Home: ${DATA_HOME}"
echo "Logs Home: ${LOGS_HOME}"
echo "Python Version: $(python --version)"
echo "User ID: $(id -u)"
echo "=========================================="

# Verify directory permissions
echo "Verifying directory permissions..."
for dir in "${APP_HOME}" "${MLFLOW_HOME}" "${LOGS_HOME}" "${DATA_HOME}"; do
    if [ ! -w "$dir" ]; then
        echo "ERROR: Directory $dir is not writable!"
        exit 1
    fi
done
echo "All directories are writable ✓"

# Initialize MLflow database and artifact store
echo "Initializing MLflow backend..."
mkdir -p "${DATA_HOME}" "${MLFLOW_HOME}/artifacts"

# Check if MLflow DB exists, if not initialize it
if [ ! -f "${DATA_HOME}/mlflow.db" ]; then
    echo "Creating MLflow SQLite database..."
    touch "${DATA_HOME}/mlflow.db"
fi

# Create log directories
echo "Creating log directories..."
mkdir -p "${LOGS_HOME}/streamlit" "${LOGS_HOME}/fastapi" "${LOGS_HOME}/mlflow" "${LOGS_HOME}/worker"

# Pre-flight checks for critical files
echo "Running pre-flight checks..."

if [ ! -f "${APP_HOME}/streamlit_app.py" ]; then
    echo "WARNING: Streamlit streamlit_app.py not found. Creating minimal placeholder..."
    cat > "${APP_HOME}/streamlit_app.py" << 'EOF'
import streamlit as st
import requests
import os

st.set_page_config(page_title="ML Stack", layout="wide")

st.title("🚀 Multi-Service ML Stack")
st.write("Running on Hugging Face Spaces")

# Health checks
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("FastAPI Status")
    try:
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code == 200:
            st.success("✓ API Running")
        else:
            st.error("✗ API Error")
    except:
        st.warning("⏳ API Starting...")

with col2:
    st.subheader("MLflow Status")
    try:
        resp = requests.get("http://localhost:5000/health", timeout=2)
        st.success("✓ MLflow Running")
    except:
        st.warning("⏳ MLflow Starting...")

with col3:
    st.subheader("Environment")
    st.info(f"Python {os.sys.version.split()[0]}")

st.divider()
st.markdown("**Internal Services:**")
st.markdown("- FastAPI: `http://localhost:8000`")
st.markdown("- MLflow: `http://localhost:5000`")
EOF
fi

# Check if FastAPI main exists
if [ ! -f "${APP_HOME}/src/api/main.py" ]; then
    echo "WARNING: FastAPI main.py not found. Creating minimal placeholder..."
    mkdir -p "${APP_HOME}/src/api"
    cat > "${APP_HOME}/src/api/main.py" << 'EOF'
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="ML Model API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "ML Model API is running", "status": "healthy"}

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "fastapi"})

@app.post("/predict")
async def predict(data: dict):
    # Placeholder for model inference
    return {"prediction": "placeholder", "confidence": 0.95}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
fi

# Check if MLOps worker exists
if [ ! -f "${APP_HOME}/src/mlops/worker.py" ]; then
    echo "WARNING: MLOps worker.py not found. Creating minimal placeholder..."
    mkdir -p "${APP_HOME}/src/mlops"
    cat > "${APP_HOME}/src/mlops/worker.py" << 'EOF'
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("MLOps Worker started")
    while True:
        logger.info(f"Worker heartbeat: {datetime.now()}")
        # Placeholder for background tasks: retraining, monitoring, etc.
        time.sleep(60)

if __name__ == "__main__":
    main()
EOF
fi

# Check if Prometheus exporter exists
if [ ! -f "${APP_HOME}/src/monitoring/prometheus_exporter.py" ]; then
    echo "WARNING: Prometheus exporter not found. Creating minimal placeholder..."
    mkdir -p "${APP_HOME}/src/monitoring"
    cat > "${APP_HOME}/src/monitoring/prometheus_exporter.py" << 'EOF'
from prometheus_client import start_http_server, Gauge, Counter
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define metrics
model_predictions = Counter('model_predictions_total', 'Total model predictions')
model_latency = Gauge('model_latency_seconds', 'Model inference latency')

def main():
    start_http_server(9100)
    logger.info("Prometheus exporter started on port 9100")
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()
EOF
fi

# Create __init__.py files for Python modules
touch "${APP_HOME}/src/__init__.py"
touch "${APP_HOME}/src/api/__init__.py"
touch "${APP_HOME}/src/mlops/__init__.py"
touch "${APP_HOME}/src/monitoring/__init__.py"

echo "Pre-flight checks complete ✓"
echo "=========================================="

# Display startup message
cat << 'EOF'

  __  __ _        _____ _             _    
 |  \/  | |      / ____| |           | |   
 | \  / | |     | (___ | |_ __ _  ___| | __
 | |\/| | |      \___ \| __/ _` |/ __| |/ /
 | |  | | |____  ____) | || (_| | (__|   < 
 |_|  |_|______||_____/ \__\__,_|\___|_|\_\
                                            
Starting all services with supervisord...

EOF

echo "Service startup order:"
echo "  1. Streamlit UI (Port 7860) - PRIMARY"
echo "  2. FastAPI (Port 8000)"
echo "  3. MLflow (Port 5000)"
echo "  4. MLOps Worker (Background)"
echo "  5. Prometheus Exporter (Port 9100)"
echo "=========================================="

# Start supervisord (this will block and keep the container running)
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
EOF
