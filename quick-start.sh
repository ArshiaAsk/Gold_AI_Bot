#!/bin/bash
# Quick Start Script for Multi-Service ML Stack

set -e

echo "=========================================="
echo "🚀 ML Stack Quick Start"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✅ Docker found: $(docker --version)"

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Docker Compose not found"
    exit 1
fi

echo "✅ Compose found: $($COMPOSE_CMD version)"

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data mlflow logs examples

# Check for required files
echo ""
echo "Checking required files..."
required_files=("Dockerfile" "supervisord.conf" "start.sh" ".streamlit/config.toml")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

echo "✅ All required files present"

# Build and start services
echo ""
echo "=========================================="
echo "Building Docker image (this may take a few minutes)..."
echo "=========================================="
$COMPOSE_CMD build

echo ""
echo "=========================================="
echo "Starting services..."
echo "=========================================="
$COMPOSE_CMD up -d

echo ""
echo "⏳ Waiting for services to start (30 seconds)..."
sleep 30

# Check service health
echo ""
echo "=========================================="
echo "Checking service health..."
echo "=========================================="

check_service() {
    local name=$1
    local url=$2
    
    if curl -sf "$url" > /dev/null 2>&1; then
        echo "✅ $name is running"
        return 0
    else
        echo "⚠️  $name is not responding yet"
        return 1
    fi
}

check_service "Streamlit" "http://localhost:7860/_stcore/health"
check_service "FastAPI" "http://localhost:8000/health"
check_service "MLflow" "http://localhost:5000/health"

echo ""
echo "=========================================="
echo "🎉 ML Stack is running!"
echo "=========================================="
echo ""
echo "📱 Access your services:"
echo "   Streamlit UI:  http://localhost:7860"
echo "   FastAPI Docs:  http://localhost:8000/docs"
echo "   MLflow UI:     http://localhost:5000"
echo "   Prometheus:    http://localhost:9100"
echo ""
echo "📋 Useful commands:"
echo "   View logs:     $COMPOSE_CMD logs -f"
echo "   Stop services: $COMPOSE_CMD down"
echo "   Restart:       $COMPOSE_CMD restart"
echo "   Shell access:  docker exec -it \$(docker ps -qf 'name=ml-stack') /bin/bash"
echo ""
echo "🧪 Run tests:"
echo "   python examples/service_communication_demo.py"
echo ""
echo "=========================================="
