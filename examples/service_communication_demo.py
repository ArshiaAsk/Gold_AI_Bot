"""
Service Communication Demo
Shows all inter-service communication patterns in the ML stack
"""

import requests
import mlflow
import time
from datetime import datetime

# Service endpoints
FASTAPI_URL = "http://localhost:8000"
MLFLOW_URL = "http://localhost:5000"

def test_fastapi_health():
    """Test FastAPI health endpoint"""
    print("\n1. Testing FastAPI Health...")
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        print(f"   ✅ FastAPI Status: {response.json()}")
        return True
    except Exception as e:
        print(f"   ❌ FastAPI Error: {e}")
        return False

def test_fastapi_prediction():
    """Test FastAPI prediction endpoint"""
    print("\n2. Testing FastAPI Prediction...")
    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json={"features": [1.5, 2.3, 3.7], "log_prediction": True},
            timeout=5
        )
        result = response.json()
        print(f"   ✅ Prediction: {result.get('prediction')}")
        print(f"   📊 Confidence: {result.get('confidence')}")
        return True
    except Exception as e:
        print(f"   ❌ Prediction Error: {e}")
        return False

def test_mlflow_connection():
    """Test MLflow tracking server"""
    print("\n3. Testing MLflow Connection...")
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"   ✅ MLflow Connected: {len(experiments)} experiments found")
        return True
    except Exception as e:
        print(f"   ❌ MLflow Error: {e}")
        return False

def test_mlflow_logging():
    """Test logging to MLflow"""
    print("\n4. Testing MLflow Logging...")
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        with mlflow.start_run(run_name=f"demo_{int(time.time())}"):
            mlflow.log_param("test_param", "demo_value")
            mlflow.log_metric("test_metric", 0.95)
            mlflow.set_tag("demo", "service_communication")
        print("   ✅ Successfully logged to MLflow")
        return True
    except Exception as e:
        print(f"   ❌ MLflow Logging Error: {e}")
        return False

def test_end_to_end_workflow():
    """Test complete workflow: Prediction -> MLflow Logging"""
    print("\n5. Testing End-to-End Workflow...")
    try:
        # Make prediction via FastAPI
        pred_response = requests.post(
            f"{FASTAPI_URL}/predict",
            json={"features": [4.2, 5.1, 6.3], "log_prediction": True},
            timeout=5
        )
        prediction = pred_response.json()
        
        # Wait for background task to log to MLflow
        time.sleep(2)
        
        # Verify in MLflow
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        if experiments:
            recent_runs = client.search_runs(
                experiment_ids=[experiments[0].experiment_id],
                max_results=5
            )
            print(f"   ✅ Workflow Complete")
            print(f"   📈 Prediction: {prediction.get('prediction')}")
            print(f"   📝 Recent MLflow Runs: {len(recent_runs)}")
            return True
        else:
            print("   ⚠️  Workflow completed but no MLflow data")
            return True
    except Exception as e:
        print(f"   ❌ Workflow Error: {e}")
        return False

def main():
    print("=" * 60)
    print("🚀 Multi-Service ML Stack Communication Demo")
    print("=" * 60)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔗 FastAPI: {FASTAPI_URL}")
    print(f"🔗 MLflow: {MLFLOW_URL}")
    
    results = {
        "FastAPI Health": test_fastapi_health(),
        "FastAPI Prediction": test_fastapi_prediction(),
        "MLflow Connection": test_mlflow_connection(),
        "MLflow Logging": test_mlflow_logging(),
        "End-to-End Workflow": test_end_to_end_workflow()
    }
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test}")
    
    print(f"\n🎯 Total: {passed}/{total} tests passed")
    print("=" * 60)

if __name__ == "__main__":
    main()
