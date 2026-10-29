param(
  [string]$HostAddress = "127.0.0.1",
  [int]$Port = 8000
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$required = @(
  "checkpoints/tft/tft_latest.pt",
  "data/processed/tft_reference.parquet",
  "data/processed/tft_runtime_config.yaml",
  "data/processed/tft_fingerprint.json",
  "data/processed/cl_metrics_latest.json"
)

$missing = @()
foreach ($path in $required) {
  if (-not (Test-Path $path)) {
    $missing += $path
  }
}

if ($missing.Count -gt 0) {
  Write-Error "Missing required pretrained artifacts:`n - $($missing -join "`n - ")"
  exit 1
}

if (-not (Test-Path ".\.venv39\Scripts\python.exe")) {
  Write-Error "Virtual env not found at .venv39"
  exit 1
}

Write-Host "Starting backend..."
Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$projectRoot'; .\.venv39\Scripts\python -m uvicorn src.api.main:app --host $HostAddress --port $Port"
)

Write-Host "Waiting for backend health..."
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
  & ".\.venv39\Scripts\python" "scripts/check_backend.py" --base-url "http://${HostAddress}:$Port" --timeout 2 | Out-Null
  if ($LASTEXITCODE -eq 0) {
    $ok = $true
    break
  }
  Start-Sleep -Seconds 1
}

if (-not $ok) {
  Write-Error "Backend did not become healthy on http://${HostAddress}:$Port"
  exit 2
}

Write-Host "Starting frontend..."
Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$projectRoot\frontend'; if (-not (Test-Path node_modules)) { npm install }; npm run dev"
)

Write-Host ""
Write-Host "Ready (pretrained mode)."
Write-Host "Frontend: http://localhost:5173"
Write-Host "Backend:  http://${HostAddress}:$Port/health"
