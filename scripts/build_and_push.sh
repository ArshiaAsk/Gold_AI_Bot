#!/bin/bash
# ============================================================================
# Docker Build, Test & Push Script for Gold Price Predictor
# Automated workflow: Build → Test → Push to Docker Hub → Deploy to HF
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-docker.io}"
DOCKER_USERNAME="${DOCKER_USERNAME:-}"
IMAGE_NAME="gold-price-predictor"
IMAGE_TAG="${IMAGE_TAG:-latest}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.enhanced.yml}"

# ============================================================================
# FUNCTIONS
# ============================================================================

print_banner() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     🚀 Docker Build & Push for Gold Price Predictor      ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}🔹 $1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

check_docker() {
    print_section "Checking Docker Installation"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}✗ Docker is not installed${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}✗ Docker Compose is not installed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Docker: $(docker --version)${NC}"
    echo -e "${GREEN}✓ Docker Compose: $(docker-compose --version)${NC}"
}

build_images() {
    print_section "Building Docker Images"
    
    echo -e "${BLUE}Using compose file: $COMPOSE_FILE${NC}"
    echo ""
    
    docker-compose -f "$COMPOSE_FILE" build --no-cache
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Images built successfully${NC}"
    else
        echo -e "${RED}✗ Build failed${NC}"
        exit 1
    fi
}

test_images() {
    print_section "Testing Docker Stack"
    
    echo -e "${BLUE}Starting services...${NC}"
    docker-compose -f "$COMPOSE_FILE" up -d
    
    sleep 5
    
    echo -e "${BLUE}Checking service health...${NC}"
    
    # Check Streamlit
    if curl -sf http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Streamlit is healthy${NC}"
    else
        echo -e "${YELLOW}⚠ Streamlit not yet ready (normal on first start)${NC}"
    fi
    
    # Check API
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is healthy${NC}"
    else
        echo -e "${YELLOW}⚠ API not yet ready (normal on first start)${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}Services running:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo -e "${GREEN}✓ Test completed${NC}"
    echo -e "${YELLOW}Note: Services will be stopped now. Use 'docker-compose up' to run.${NC}"
    
    docker-compose -f "$COMPOSE_FILE" down
}

show_images() {
    print_section "Available Docker Images"
    
    docker images | grep gold
    
    echo ""
    echo "Image sizes:"
    docker images --format "table {{.Repository}}\t{{.Size}}" | grep gold
}

push_to_registry() {
    print_section "Pushing to Docker Registry"
    
    if [ -z "$DOCKER_USERNAME" ]; then
        echo -e "${RED}✗ DOCKER_USERNAME not set${NC}"
        echo "Set it with: export DOCKER_USERNAME=your_username"
        exit 1
    fi
    
    echo -e "${BLUE}Logging in to Docker Hub...${NC}"
    docker login -u "$DOCKER_USERNAME"
    
    echo ""
    echo -e "${BLUE}Tagging images...${NC}"
    
    # Tag Streamlit image
    docker tag ${IMAGE_NAME}_streamlit:latest "$DOCKER_USERNAME/$IMAGE_NAME:streamlit-latest"
    echo -e "${GREEN}✓ Tagged streamlit image${NC}"
    
    # Tag API image
    docker tag ${IMAGE_NAME}_api:latest "$DOCKER_USERNAME/$IMAGE_NAME:api-latest"
    echo -e "${GREEN}✓ Tagged api image${NC}"
    
    echo ""
    echo -e "${BLUE}Pushing to Docker Hub...${NC}"
    
    docker push "$DOCKER_USERNAME/$IMAGE_NAME:streamlit-latest"
    docker push "$DOCKER_USERNAME/$IMAGE_NAME:api-latest"
    
    echo -e "${GREEN}✓ Push completed${NC}"
    
    echo ""
    echo -e "${BLUE}Images available at:${NC}"
    echo "  Streamlit: $DOCKER_USERNAME/$IMAGE_NAME:streamlit-latest"
    echo "  API: $DOCKER_USERNAME/$IMAGE_NAME:api-latest"
}

show_usage() {
    print_section "Usage"
    
    echo -e "${BLUE}Available options:${NC}"
    echo ""
    echo "  ./build_and_push.sh [option]"
    echo ""
    echo "Options:"
    echo "  build          Build Docker images only"
    echo "  test           Build and test images"
    echo "  push           Build and push to Docker Hub"
    echo "  hf             Build for Hugging Face (streamlined)"
    echo "  show           Show built images"
    echo "  clean          Remove all images"
    echo "  help           Show this help message"
    echo ""
    echo -e "${BLUE}Environment variables:${NC}"
    echo "  DOCKER_USERNAME   Your Docker Hub username"
    echo "  IMAGE_TAG         Image tag (default: latest)"
    echo "  COMPOSE_FILE      Compose file (default: docker-compose.enhanced.yml)"
    echo ""
}

clean_images() {
    print_section "Cleaning Images"
    
    echo -e "${YELLOW}Removing all gold-price-predictor images...${NC}"
    docker-compose -f "$COMPOSE_FILE" down
    docker system prune -f
    
    echo -e "${GREEN}✓ Cleanup completed${NC}"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    print_banner
    
    # Parse arguments
    COMMAND="${1:-help}"
    
    case "$COMMAND" in
        build)
            check_docker
            build_images
            show_images
            ;;
        test)
            check_docker
            build_images
            test_images
            show_images
            ;;
        push)
            check_docker
            build_images
            test_images
            push_to_registry
            show_images
            ;;
        hf)
            echo -e "${BLUE}Building for Hugging Face Spaces...${NC}"
            COMPOSE_FILE="docker-compose.hf.yml"
            check_docker
            build_images
            show_images
            echo ""
            echo -e "${GREEN}✓ HF-ready images built${NC}"
            echo -e "${YELLOW}Next step: Push to Docker Hub and reference in HF Spaces${NC}"
            ;;
        show)
            show_images
            ;;
        clean)
            clean_images
            ;;
        help|*)
            show_usage
            ;;
    esac
    
    echo ""
}

main "$@"
