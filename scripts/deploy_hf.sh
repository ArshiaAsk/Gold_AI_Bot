#!/bin/bash
# Deployment script for Hugging Face Spaces
# Phase 5 Gold Price Prediction System

set -e

echo "🚀 Starting deployment to Hugging Face Spaces..."
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Verify requirements
echo -e "${BLUE}Step 1: Verifying environment...${NC}"
command -v git >/dev/null 2>&1 || { echo "❌ git is required but not installed."; exit 1; }
command -v pip >/dev/null 2>&1 || { echo "❌ pip is required but not installed."; exit 1; }
echo -e "${GREEN}✓ Environment verified${NC}"
echo ""

# Step 2: Install dependencies
echo -e "${BLUE}Step 2: Installing dependencies...${NC}"
pip install -r requirements-hf.txt --quiet
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 3: Test local deployment
echo -e "${BLUE}Step 3: Testing Streamlit app locally...${NC}"
echo -e "${YELLOW}⚠ To test locally, run: streamlit run streamlit_app.py${NC}"
echo ""

# Step 4: Deployment instructions
echo -e "${BLUE}Step 4: Deployment Instructions${NC}"
echo ""
echo "To deploy to Hugging Face Spaces:"
echo ""
echo "1. Create a Hugging Face account at https://huggingface.co"
echo ""
echo "2. Create a new Space:"
echo "   - Go to https://huggingface.co/spaces"
echo "   - Click 'Create new Space'"
echo "   - Name: gold-price-predictor"
echo "   - License: MIT"
echo "   - SDK: Streamlit"
echo "   - Visibility: Public"
echo ""
echo "3. Clone the Space locally (optional):"
echo "   git clone https://huggingface.co/spaces/[YOUR-USERNAME]/gold-price-predictor"
echo "   cd gold-price-predictor"
echo ""
echo "4. Copy files to Space:"
echo "   - streamlit_app.py"
echo "   - requirements-hf.txt → requirements.txt"
echo "   - .streamlit/config.toml"
echo ""
echo "5. Push to Hugging Face:"
echo "   git add ."
echo "   git commit -m 'Initial commit: Phase 5 MLOps deployment'"
echo "   git push"
echo ""
echo -e "${GREEN}✓ Your app will automatically build and deploy!${NC}"
echo ""

# Step 5: Success message
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎉 Deployment setup complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Next steps:"
echo "1. Test locally: streamlit run streamlit_app.py"
echo "2. Push to Hugging Face Spaces"
echo "3. Share your app: https://huggingface.co/spaces/[YOUR-USERNAME]/gold-price-predictor"
echo ""
