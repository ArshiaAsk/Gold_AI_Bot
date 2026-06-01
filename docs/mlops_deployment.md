# MLOps Deployment Guide

## Services

Phase 4 + Phase 5 deployment runs 5 services:
- `api` (`:8000`) FastAPI prediction service with MLOps endpoints
- `mlops-worker` (`:8001`) scheduler + retraining worker with metrics server
- `mlflow` (`:5000`) experiment tracking UI and backend
- `prometheus` (`:9090`) metrics storage/scraping
- `grafana` (`:3000`) monitoring dashboards

## Prerequisites

- Docker + Docker Compose
- Project files available locally (`models/`, `data/`, `logs/`)

## Start

```bash
docker-compose up --build -d
```

## Verify

```bash
curl http://localhost:8000/health
curl -H "X-API-Key: $API_KEY" http://localhost:8000/mlops/health
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics
curl http://localhost:9090/-/healthy
curl http://localhost:5000/
```

Phase 5 endpoints (API key required):
- `GET /mlops/ab-test` — canary routing status
- `POST /mlops/ab-test/promote` — promote canary to production
- `POST /mlops/ab-test/disable` — disable canary
- `GET /mlops/experiments` — recent MLflow runs
- `GET /mlops/model-card` — generate governance model card

Phase 5 environment variables:
- `MLFLOW_TRACKING_URI` (default in compose: `http://mlflow:5000`)
- `AB_TESTING_ENABLED` (`true`/`false`)
- `AB_CANARY_TRAFFIC_PERCENT` (0–100)
- `PROMOTE_VIA_CANARY` (`true` registers improved models as canary first)
- `ONNX_EXPORT_ENABLED` (`false` in compose; enable for ONNX artifact export)

If `API_KEY_REQUIRED=true`, use header `X-API-Key` for:
- `/predict`
- `/predict-with-confidence`
- `/predict-from-history`
- `/mlops/health`
- `/mlops/model-registry`
- `/model-info`

Grafana:
- URL: `http://localhost:3000`
- User: `admin`
- Password: `admin`

## Stop

```bash
docker-compose down
```

## Logs

```bash
docker-compose logs -f api
docker-compose logs -f mlops-worker
docker-compose logs -f prometheus
```

## Manual Rollback

Rollback is model-registry based.
1. Open `models/registry/metadata.json`
2. Find target version for model `gold_lstm`
3. Run rollback command via Python shell:

```python
from src.mlops.model_registry import ModelRegistry
registry = ModelRegistry()
registry.rollback(target_version=2, name="gold_lstm")
```
