"""
Example Streamlit Application with Inter-Service Communication
This demonstrates how to communicate with FastAPI and MLflow from Streamlit
"""

import streamlit as st
import requests
import mlflow
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="ML Stack Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Service URLs (internal communication via localhost)
FASTAPI_URL = "http://localhost:8000"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


def check_service_health(service_name: str, url: str) -> bool:
    """Check if a service is healthy"""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_fastapi_prediction(features: list) -> dict:
    """Get prediction from FastAPI service"""
    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json={"features": features},
            timeout=5
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_mlflow_experiments():
    """Fetch experiments from MLflow"""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        return experiments
    except Exception as e:
        st.error(f"MLflow connection error: {e}")
        return []


def get_mlflow_runs(experiment_id: str, max_results: int = 10):
    """Fetch recent runs from an experiment"""
    try:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        return runs
    except Exception as e:
        st.error(f"Error fetching runs: {e}")
        return []


# Main UI
st.title("🚀 Multi-Service ML Stack Dashboard")
st.markdown("Demonstrating inter-service communication in Hugging Face Spaces")

# Service Health Status
st.header("📊 Service Health Status")
col1, col2, col3 = st.columns(3)

with col1:
    fastapi_healthy = check_service_health("FastAPI", FASTAPI_URL)
    if fastapi_healthy:
        st.success("✅ FastAPI: Running")
    else:
        st.error("❌ FastAPI: Offline")

with col2:
    mlflow_healthy = check_service_health("MLflow", MLFLOW_TRACKING_URI)
    if mlflow_healthy:
        st.success("✅ MLflow: Running")
    else:
        st.error("❌ MLflow: Offline")

with col3:
    st.info(f"🕒 {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# Prediction Interface
st.header("🎯 Model Prediction (via FastAPI)")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Input Features")
    
    # Example feature inputs
    feature1 = st.slider("Feature 1", 0.0, 10.0, 5.0)
    feature2 = st.slider("Feature 2", 0.0, 10.0, 5.0)
    feature3 = st.slider("Feature 3", 0.0, 10.0, 5.0)
    
    if st.button("🚀 Get Prediction", type="primary"):
        with st.spinner("Calling FastAPI..."):
            result = get_fastapi_prediction([feature1, feature2, feature3])
            
            if "error" in result:
                st.error(f"Prediction failed: {result['error']}")
            else:
                st.session_state.prediction_history.append({
                    "timestamp": datetime.now(),
                    "features": [feature1, feature2, feature3],
                    "prediction": result.get("prediction", "N/A"),
                    "confidence": result.get("confidence", 0.0)
                })
                
                st.success(f"✅ Prediction: {result.get('prediction', 'N/A')}")
                st.metric("Confidence", f"{result.get('confidence', 0) * 100:.1f}%")

with col2:
    st.subheader("Recent Predictions")
    if st.session_state.prediction_history:
        for pred in st.session_state.prediction_history[-5:]:
            st.text(f"{pred['timestamp'].strftime('%H:%M:%S')}: {pred['prediction']}")
    else:
        st.info("No predictions yet")

st.divider()

# MLflow Integration
st.header("📈 MLflow Experiments & Models")

tab1, tab2 = st.tabs(["Experiments", "Model Registry"])

with tab1:
    st.subheader("Recent Experiments")
    
    experiments = get_mlflow_experiments()
    
    if experiments:
        exp_data = []
        for exp in experiments:
            exp_data.append({
                "Name": exp.name,
                "Experiment ID": exp.experiment_id,
                "Lifecycle": exp.lifecycle_stage
            })
        
        st.dataframe(pd.DataFrame(exp_data), use_container_width=True)
        
        # Show runs for selected experiment
        selected_exp = st.selectbox(
            "Select Experiment",
            options=[exp.experiment_id for exp in experiments],
            format_func=lambda x: next(exp.name for exp in experiments if exp.experiment_id == x)
        )
        
        if selected_exp:
            runs = get_mlflow_runs(selected_exp)
            
            if runs:
                st.subheader(f"Recent Runs ({len(runs)})")
                
                run_data = []
                for run in runs:
                    run_data.append({
                        "Run ID": run.info.run_id[:8],
                        "Start Time": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
                        "Status": run.info.status,
                        "Metrics": ", ".join([f"{k}={v:.4f}" for k, v in run.data.metrics.items()][:3])
                    })
                
                st.dataframe(pd.DataFrame(run_data), use_container_width=True)
    else:
        st.info("No experiments found. Start logging experiments to MLflow!")

with tab2:
    st.subheader("Registered Models")
    
    try:
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()
        
        if models:
            model_data = []
            for model in models:
                model_data.append({
                    "Name": model.name,
                    "Latest Version": model.latest_versions[0].version if model.latest_versions else "N/A",
                    "Stage": model.latest_versions[0].current_stage if model.latest_versions else "N/A"
                })
            
            st.dataframe(pd.DataFrame(model_data), use_container_width=True)
        else:
            st.info("No registered models found.")
    except Exception as e:
        st.warning(f"Model registry not available: {e}")

st.divider()

# System Information
with st.expander("🔧 System Information"):
    st.json({
        "FastAPI URL": FASTAPI_URL,
        "MLflow Tracking URI": MLFLOW_TRACKING_URI,
        "Prediction History Count": len(st.session_state.prediction_history),
        "Current Time": datetime.now().isoformat()
    })

# Auto-refresh option
st.sidebar.header("Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh health status")
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)
    st.sidebar.info(f"Refreshing every {refresh_interval}s")
    import time
    time.sleep(refresh_interval)
    st.rerun()
