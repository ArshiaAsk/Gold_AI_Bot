#!/bin/bash
# Quick Start Guide for Docker Services
# Gold Price Prediction System

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Gold Price Predictor - Docker Quick Start"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${BLUE}✓ Docker installed: $(docker --version)${NC}"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo -e "${YELLOW}🎯 AVAILABLE OPTIONS${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

show_menu() {
    echo "1) 🏃 Run Full Stack Locally (Streamlit + API + MLOps + Monitoring)"
    echo "2) 🎨 Run Streamlit Only (Fast, lightweight)"
    echo "3) 🐳 Build Docker Images (for deployment)"
    echo "4) 📤 Push to Docker Hub (requires login)"
    echo "5) 🏥 Check Service Health"
    echo "6) 📋 View Logs"
    echo "7) 🛑 Stop All Services"
    echo "8) 📚 Show Documentation"
    echo "9) ❌ Exit"
    echo ""
}

# Function: Full Stack
run_full_stack() {
    echo -e "${BLUE}Starting Full Stack...${NC}"
    echo ""
    docker-compose -f docker-compose.enhanced.yml up --build
    echo ""
    echo -e "${GREEN}✓ Services started!${NC}"
    echo ""
    echo "Access services at:"
    echo "  • Streamlit UI:  http://localhost:8501"
    echo "  • API:           http://localhost:8000"
    echo "  • MLflow:        http://localhost:5000"
    echo "  • Prometheus:    http://localhost:9090"
    echo "  • Grafana:       http://localhost:3000"
    echo ""
}

# Function: Streamlit Only
run_streamlit_only() {
    echo -e "${BLUE}Starting Streamlit Only...${NC}"
    echo ""
    
    # Check if requirements installed
    if ! python -c "import streamlit" 2>/dev/null; then
        echo "Installing dependencies..."
        pip install -r requirements-hf.txt -q
    fi
    
    streamlit run streamlit_app.py
}

# Function: Build Images
build_images() {
    echo -e "${BLUE}Building Docker Images...${NC}"
    echo ""
    
    chmod +x build_and_push.sh
    ./build_and_push.sh build
    
    echo ""
    echo -e "${GREEN}✓ Build complete!${NC}"
}

# Function: Push to Registry
push_registry() {
    echo -e "${BLUE}Pushing to Docker Hub...${NC}"
    echo ""
    
    read -p "Enter Docker Hub username: " DOCKER_USERNAME
    export DOCKER_USERNAME
    
    chmod +x build_and_push.sh
    ./build_and_push.sh push
    
    echo ""
    echo -e "${GREEN}✓ Push complete!${NC}"
}

# Function: Health Check
check_health() {
    echo -e "${BLUE}Checking Service Health...${NC}"
    echo ""
    
    services=("Streamlit:8501" "API:8000" "MLflow:5000" "Prometheus:9090")
    
    for service in "${services[@]}"; do
        name="${service%:*}"
        port="${service#*:}"
        
        if curl -sf "http://localhost:$port" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name (port $port) is healthy${NC}"
        else
            echo -e "${YELLOW}⚠ $name (port $port) not responding${NC}"
        fi
    done
    
    echo ""
}

# Function: View Logs
view_logs() {
    echo -e "${BLUE}Available services:${NC}"
    echo "1) Streamlit"
    echo "2) API"
    echo "3) MLOps Worker"
    echo "4) All"
    echo ""
    read -p "Select service (1-4): " choice
    
    case $choice in
        1) docker-compose -f docker-compose.enhanced.yml logs -f streamlit ;;
        2) docker-compose -f docker-compose.enhanced.yml logs -f api ;;
        3) docker-compose -f docker-compose.enhanced.yml logs -f mlops-worker ;;
        4) docker-compose -f docker-compose.enhanced.yml logs -f ;;
    esac
}

# Function: Stop Services
stop_services() {
    echo -e "${YELLOW}Stopping all services...${NC}"
    docker-compose -f docker-compose.enhanced.yml down
    echo -e "${GREEN}✓ Services stopped${NC}"
}

# Function: Show Documentation
show_docs() {
    echo ""
    echo -e "${BLUE}📚 Documentation Available:${NC}"
    echo ""
    echo "1) DOCKER_DEPLOYMENT_GUIDE.md    - Complete Docker guide"
    echo "2) SYSTEM_ARCHITECTURE.md        - System design & flow"
    echo "3) DEPLOYMENT_HF.md              - Streamlit HF deployment"
    echo "4) MLOps_Complete_Journey.md     - Project phases"
    echo "5) README.md                     - Main documentation"
    echo ""
    read -p "Open file (1-5, or 'q' to cancel): " choice
    
    case $choice in
        1) cat DOCKER_DEPLOYMENT_GUIDE.md | less ;;
        2) cat SYSTEM_ARCHITECTURE.md | less ;;
        3) cat DEPLOYMENT_HF.md | less ;;
        4) cat MLOps_Complete_Journey.md | less ;;
        5) cat README.md | less ;;
        q) return ;;
    esac
}

# Main Loop
while true; do
    clear
    echo "════════════════════════════════════════════════════════════════"
    echo -e "${YELLOW}🏆 Gold Price Prediction System - Docker Quick Start${NC}"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    show_menu
    read -p "Select option (1-9): " choice
    
    case $choice in
        1) run_full_stack ;;
        2) run_streamlit_only ;;
        3) build_images ;;
        4) push_registry ;;
        5) check_health ;;
        6) view_logs ;;
        7) stop_services ;;
        8) show_docs ;;
        9) echo "Goodbye! 👋"; exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done
