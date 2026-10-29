#!/usr/bin/env bash
set -euo pipefail

# Deployment script for the retail-inventory backend API
# Builds and tags the Docker image for deployment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Backend Deployment Script"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

# Check if Dockerfile.api exists
if [ ! -f "Dockerfile.api" ]; then
    echo "ERROR: Dockerfile.api not found in project root"
    exit 1
fi

# Generate version tag
VERSION_TAG="$(date +%Y%m%d)"

echo "Building Docker image: retail-inventory-backend:latest"
docker build -t retail-inventory-backend:latest -f Dockerfile.api .

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi

echo ""
echo "Tagging image with version: retail-inventory-backend:${VERSION_TAG}"
docker tag retail-inventory-backend:latest "retail-inventory-backend:${VERSION_TAG}"

if [ $? -ne 0 ]; then
    echo "ERROR: Docker tag failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Backend deployment build complete!"
echo "=========================================="
echo "Images built:"
echo "  - retail-inventory-backend:latest"
echo "  - retail-inventory-backend:${VERSION_TAG}"
echo ""
echo "Next steps:"
echo "  - Push to registry: docker push retail-inventory-backend:latest"
echo "  - Or run locally: docker run -p 8000:8000 retail-inventory-backend:latest"
echo ""
