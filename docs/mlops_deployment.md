# MLOps Deployment Guide

## Services

Phase 4 deployment runs 4 services:
- `api` (`:8000`) FastAPI prediction service with MLOps endpoints
- `mlops-worker` (`:8001`) scheduler + retraining worker with metrics server
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
curl http://localhost:8000/mlops/health
curl http://localhost:8000/metrics
curl http://localhost:8001/metrics
curl http://localhost:9090/-/healthy
```

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
