# Deployment script for the retail-inventory backend API (Windows PowerShell)
# Builds and tags the Docker image for deployment

param(
    [string]$ImageName = "retail-inventory-backend"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir

Set-Location $projectRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Backend Deployment Script (PowerShell)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: Docker is not installed or not in PATH"
    exit 1
}

# Check if Dockerfile.api exists
if (-not (Test-Path "Dockerfile.api")) {
    Write-Error "ERROR: Dockerfile.api not found in project root"
    exit 1
}

# Generate version tag
$versionTag = Get-Date -Format "yyyyMMdd"

Write-Host "Building Docker image: ${ImageName}:latest" -ForegroundColor Yellow
docker build -t "${ImageName}:latest" -f Dockerfile.api .

if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Docker build failed"
    exit 1
}

Write-Host ""
Write-Host "Tagging image with version: ${ImageName}:${versionTag}" -ForegroundColor Yellow
docker tag "${ImageName}:latest" "${ImageName}:${versionTag}"

if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Docker tag failed"
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Backend deployment build complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Images built:" -ForegroundColor Green
Write-Host "  - ${ImageName}:latest" -ForegroundColor Green
Write-Host "  - ${ImageName}:${versionTag}" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  - Push to registry: docker push ${ImageName}:latest" -ForegroundColor White
Write-Host "  - Or run locally: docker run -p 8000:8000 ${ImageName}:latest" -ForegroundColor White
Write-Host ""
