# MLOps Troubleshooting

## 1. API starts but MLOps endpoints show degraded

Symptoms:
- `/mlops/health` returns degraded status

Actions:
1. Check API logs: `docker-compose logs -f api`
2. Confirm `config/` is present in image and import errors are absent
3. Validate registry path permissions (`models/registry`)

## 2. Worker is running but no retraining executes

Symptoms:
- `mlops-worker` healthy
- No weekly retraining logs

Actions:
1. Check scheduler thread status via `/mlops/health`
2. Verify container time zone / system clock
3. Trigger manual run in shell:

```python
from config.settings import Config
from src.mlops.retraining_pipeline import AutoRetrainingPipeline
p = AutoRetrainingPipeline(Config())
p.trigger_retraining()
```

## 3. Prometheus has no data for worker

Actions:
1. Ensure worker port is exposed (`8001:8001`)
2. Check worker metrics endpoint: `curl http://localhost:8001/metrics`
3. Open Prometheus targets page: `http://localhost:9090/targets`

## 4. Grafana dashboard empty

Actions:
1. Ensure datasource is provisioned (`Prometheus`)
2. Check query in Explore mode
3. Verify metric names exist in Prometheus

## 5. Model rollback needed

```python
from src.mlops.model_registry import ModelRegistry
r = ModelRegistry()
print(r.list_all("gold_lstm"))
r.rollback(target_version=1, name="gold_lstm")
```

## 6. MLflow errors

`ExperimentTracker` degrades gracefully. If tracking is required:
1. Set `MLFLOW_TRACKING_URI`
2. Ensure backend store is reachable
3. Restart API and worker
