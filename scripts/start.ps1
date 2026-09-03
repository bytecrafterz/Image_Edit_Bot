# Photo Robot - arranque.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$venvPython = Join-Path $backend ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Falta el entorno virtual. Ejecuta primero:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File `"$(Join-Path $PSScriptRoot 'setup.ps1')`""
    exit 1
}

# Auto-reload is on by default: saving a .py restarts the server.
# Pass --no-reload for a production run, e.g. start.ps1 --no-reload
& $venvPython (Join-Path $backend "run.py") @args
