# Deployment script for the retail-inventory frontend (Windows PowerShell)
# Builds the frontend and deploys to Vercel

param(
    [string]$FrontendDir = "frontend"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
$frontendPath = Join-Path $projectRoot $FrontendDir

Set-Location $projectRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Frontend Deployment Script (PowerShell)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if frontend directory exists
if (-not (Test-Path $frontendPath)) {
    Write-Error "ERROR: Frontend directory not found at $frontendPath"
    exit 1
}

# Check if Node.js is installed
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: Node.js is not installed or not in PATH"
    exit 1
}

# Check if npm is installed
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: npm is not installed or not in PATH"
    exit 1
}

Set-Location $frontendPath

# Check if node_modules exists, if not install dependencies
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    npm install
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: npm install failed"
        exit 1
    }
}

Write-Host "Building frontend..." -ForegroundColor Yellow
npm run build

if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Frontend build failed"
    exit 1
}

Write-Host ""
Write-Host "Frontend build complete!" -ForegroundColor Green
Write-Host ""

# Check if Vercel CLI is installed
if (Get-Command vercel -ErrorAction SilentlyContinue) {
    Write-Host "Deploying to Vercel (production)..." -ForegroundColor Yellow
    vercel --prod
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "ERROR: Vercel deployment failed"
        exit 1
    }
    
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "Frontend deployed successfully!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
} else {
    Write-Host "Vercel CLI not found. Build artifacts are in the 'dist/' directory." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To deploy manually:" -ForegroundColor White
    Write-Host "  1. Install Vercel CLI: npm i -g vercel" -ForegroundColor White
    Write-Host "  2. Login: vercel login" -ForegroundColor White
    Write-Host "  3. Deploy: cd frontend && vercel --prod" -ForegroundColor White
    Write-Host ""
    Write-Host "Or deploy the 'dist/' folder to any static hosting service." -ForegroundColor White
    Write-Host ""
}
