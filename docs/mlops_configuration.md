# MLOps Configuration

## Environment Variables

### API and Logging
- `LOG_LEVEL` default: `INFO`
- `API_HOST` default: `0.0.0.0`
- `API_PORT` default: `8000`
- `CORS_ALLOW_ORIGINS` default: `http://localhost:3000,http://localhost:8000`

### API Security
- `API_KEY_REQUIRED` default: `true`
- `API_KEY` default: empty (must be set in production)
- `API_KEY_HEADER_NAME` default: `X-API-Key`
- `RATE_LIMIT_ENABLED` default: `true`
- `RATE_LIMIT_REQUESTS` default: `120`
- `RATE_LIMIT_WINDOW_SECONDS` default: `60`

### Model Paths
- `MODEL_PATH` default: `models/gold_lstm_v2.keras`
- `SCALER_X_PATH` default: `models/scaler_X.pkl`
- `SCALER_Y_PATH` default: `models/scaler_y.pkl`
- `MODEL_REGISTRY_PATH` default: `models/registry`
- `MODEL_MAX_AGE_HOURS` default: `720`

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
- `PREDICTION_SLA_SECONDS` default: `1.0`
- `HEALTH_DISK_MIN_FREE_GB` default: `1.0`
- `HEALTH_MEMORY_MAX_MB` default: `4096`

### Persistence
- `AUDIT_DB_PATH` default: `data/mlops_audit.db`

### Alert Channels (Optional)
- `SLACK_WEBHOOK_URL` default: empty
- `TELEGRAM_BOT_TOKEN` default: empty
- `TELEGRAM_CHAT_ID` default: empty
- `ALERT_SMTP_HOST` default: empty
- `ALERT_SMTP_PORT` default: `587`
- `ALERT_SMTP_USER` default: empty
- `ALERT_SMTP_PASSWORD` default: empty
- `ALERT_EMAIL_FROM` default: empty
- `ALERT_EMAIL_TO` default: empty

### MLflow
- `MLFLOW_TRACKING_URI` default: empty
- `MLFLOW_EXPERIMENT_NAME` default: `gold_price_retraining`

## Tuning Guidance

- Increase `MLOPS_SCHEDULER_POLL_SECONDS` if host is resource-constrained.
- Tighten drift thresholds only after observing false positives for at least 1 week.
- Tighten validation thresholds only if baseline model quality remains stable.
