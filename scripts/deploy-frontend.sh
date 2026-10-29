#!/usr/bin/env bash
set -euo pipefail

# Deployment script for the retail-inventory frontend
# Builds the frontend and deploys to Vercel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Frontend Deployment Script"
echo "=========================================="
echo ""

# Check if frontend directory exists
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "ERROR: Frontend directory not found at $FRONTEND_DIR"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "ERROR: npm is not installed or not in PATH"
    exit 1
fi

cd "$FRONTEND_DIR"

# Check if node_modules exists, if not install dependencies
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "ERROR: npm install failed"
        exit 1
    fi
fi

echo "Building frontend..."
npm run build

if [ $? -ne 0 ]; then
    echo "ERROR: Frontend build failed"
    exit 1
fi

echo ""
echo "Frontend build complete!"
echo ""

# Check if Vercel CLI is installed
if command -v vercel &> /dev/null; then
    echo "Deploying to Vercel (production)..."
    vercel --prod
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Vercel deployment failed"
        exit 1
    fi
    
    echo ""
    echo "=========================================="
    echo "Frontend deployed successfully!"
    echo "=========================================="
else
    echo "Vercel CLI not found. Build artifacts are in the 'dist/' directory."
    echo ""
    echo "To deploy manually:"
    echo "  1. Install Vercel CLI: npm i -g vercel"
    echo "  2. Login: vercel login"
    echo "  3. Deploy: cd frontend && vercel --prod"
    echo ""
    echo "Or deploy the 'dist/' folder to any static hosting service."
    echo ""
fi
