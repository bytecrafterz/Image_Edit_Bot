# Photo Robot - instalacion en Windows.
# Idempotente: se puede volver a ejecutar sin romper nada.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$venv = Join-Path $backend ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"

function Write-Section($text) {
    Write-Host ""
    Write-Host ("=" * 64)
    Write-Host "  $text"
    Write-Host ("=" * 64)
}

Write-Section "1. Buscando Python 3.12"

$candidates = @(
    "C:\Program Files\Python312\python.exe",
    "C:\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
)
$python = $null
foreach ($candidate in $candidates) {
    if (Test-Path $candidate) { $python = $candidate; break }
}
if (-not $python) {
    try {
        $found = (Get-Command py -ErrorAction Stop).Source
        if ($found) { $python = "py -3.12" }
    } catch { }
}
if (-not $python) {
    try {
        $cmd = (Get-Command python -ErrorAction Stop).Source
        $version = & $cmd --version 2>&1
        if ($version -match "3\.1[12]") { $python = $cmd }
    } catch { }
}
if (-not $python) {
    Write-Host ""
    Write-Host "  No se ha encontrado Python 3.12." -ForegroundColor Red
    Write-Host "  Descargalo aqui e instalalo marcando 'Add python.exe to PATH':"
    Write-Host "  https://www.python.org/downloads/release/python-31210/"
    exit 1
}
Write-Host "  Python encontrado: $python"

Write-Section "2. Entorno virtual"

if (Test-Path $venvPython) {
    Write-Host "  Ya existe en $venv"
} else {
    Write-Host "  Creando en $venv ..."
    & $python -m venv "$venv"
    if (-not (Test-Path $venvPython)) {
        Write-Host "  No se pudo crear el entorno virtual." -ForegroundColor Red
        exit 1
    }
}

Write-Section "3. Dependencias"

& $venvPython -m pip install --upgrade pip setuptools wheel --quiet
Write-Host "  Instalando (puede tardar varios minutos la primera vez)..."
& $venvPython -m pip install -r (Join-Path $backend "requirements.txt") --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Fallo la instalacion de dependencias." -ForegroundColor Red
    exit 1
}

Write-Section "4. Comprobacion"

$check = @'
import cv2, mediapipe, numpy, fastapi, scipy, skimage, PIL
print("  numpy      ", numpy.__version__)
print("  opencv     ", cv2.__version__)
print("  mediapipe  ", mediapipe.__version__)
print("  fastapi    ", fastapi.__version__)
print("  scipy      ", scipy.__version__)
print("  pillow     ", PIL.__version__)
'@
& $venvPython -c $check
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Alguna libreria no se instalo bien." -ForegroundColor Red
    exit 1
}

Write-Section "5. Carpetas de datos"

foreach ($name in @("uploads", "outputs", "previews", "profiles", "cache", "logs", "scenes")) {
    $dir = Join-Path (Join-Path $root "data") $name
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
}
Write-Host "  Listas en $(Join-Path $root 'data')"

Write-Section "Instalacion terminada"
Write-Host ""
Write-Host "  Para arrancar el sistema:"
Write-Host ""
Write-Host "      powershell -ExecutionPolicy Bypass -File `"$(Join-Path $PSScriptRoot 'start.ps1')`""
Write-Host ""
Write-Host "  Se abrira en http://localhost:8080"
Write-Host "  La primera cuenta que crees sera la administradora."
Write-Host ""
