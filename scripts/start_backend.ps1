param(
  [string]$HostAddress = "127.0.0.1",
  [int]$Port = 8000
)

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path ".\.venv39\Scripts\python.exe")) {
  Write-Error "Virtual env not found at .venv39. Create/install it first."
  exit 1
}

$env:PYTHONPATH = "."
Write-Host "Starting backend on http://${HostAddress}:$Port"
& ".\.venv39\Scripts\python.exe" -m uvicorn src.api.main:app --host $HostAddress --port $Port
