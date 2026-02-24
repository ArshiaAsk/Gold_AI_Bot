# MLOps Configuration

## Environment Variables

### API and Logging
- `LOG_LEVEL` default: `INFO`
- `API_HOST` default: `0.0.0.0`
- `API_PORT` default: `8000`

### Model Paths
- `MODEL_PATH` default: `models/gold_lstm_v2.keras`
- `SCALER_X_PATH` default: `models/scaler_X.pkl`
- `SCALER_Y_PATH` default: `models/scaler_y.pkl`
- `MODEL_REGISTRY_PATH` default: `models/registry`

### Scheduler
- `MLOPS_SCHEDULER_ENABLED` default: `true`
- `MLOPS_SCHEDULER_POLL_SECONDS` default: `60`
- `MLOPS_RETRAIN_DAY` default: `sunday`
- `MLOPS_RETRAIN_TIME` default: `02:00`
- `MLOPS_DRIFT_TIME` default: `06:00`

### Drift and Validation
- `DRIFT_P_VALUE_THRESHOLD` default: `0.05`
- `DRIFT_MEAN_SHIFT_THRESHOLD` default: `0.2`
- `VALIDATION_RMSE_MAX` default: `120000`
- `VALIDATION_MAE_MAX` default: `90000`
- `VALIDATION_R2_MIN` default: `0.70`
- `VALIDATION_MAPE_MAX` default: `25.0`

### Monitoring
- `PROMETHEUS_ENABLED` default: `true`
- `PROMETHEUS_PORT` default: `8001`
- `PROMETHEUS_WORKER_PORT` default: `8001`

### MLflow
- `MLFLOW_TRACKING_URI` default: empty
- `MLFLOW_EXPERIMENT_NAME` default: `gold_price_retraining`

## Tuning Guidance

- Increase `MLOPS_SCHEDULER_POLL_SECONDS` if host is resource-constrained.
- Tighten drift thresholds only after observing false positives for at least 1 week.
- Tighten validation thresholds only if baseline model quality remains stable.
