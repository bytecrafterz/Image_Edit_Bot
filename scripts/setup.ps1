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

Write-Section "4. Runtime de Visual C++ (lo necesita MediaPipe)"

# MediaPipe carga DLLs compiladas con Visual C++ 2015-2022 y necesita su
# runtime actual (msvcp140.dll 14.3x y vcruntime140_1.dll).  Windows Server
# trae de fabrica uno de 2016 (14.00.24215) sin vcruntime140_1.dll, y con el
# la importacion falla con "DLL initialization routine failed" (2026-09-29).
$vcDll = Join-Path $env:WINDIR "System32\vcruntime140_1.dll"
$msvcp = Join-Path $env:WINDIR "System32\msvcp140.dll"
$vcOk = $false
if ((Test-Path $vcDll) -and (Test-Path $msvcp)) {
    $v = (Get-Item $msvcp).VersionInfo
    $vcOk = ($v.FileMajorPart -gt 14) -or ($v.FileMajorPart -eq 14 -and $v.FileMinorPart -ge 30)
}
if ($vcOk) {
    Write-Host "  Instalado: msvcp140.dll $((Get-Item $msvcp).VersionInfo.FileVersion)"
} else {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
               ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    $redist = Join-Path $env:TEMP "vc_redist.x64.exe"
    Write-Host "  Falta o es antiguo. Descargando el instalador oficial de Microsoft..."
    Invoke-WebRequest -UseBasicParsing "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $redist
    $sig = Get-AuthenticodeSignature $redist
    if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch "Microsoft Corporation") {
        Write-Host "  El instalador descargado no tiene una firma valida de Microsoft; no se ejecuta." -ForegroundColor Red
        exit 1
    }
    if ($isAdmin) {
        $p = Start-Process -FilePath $redist -ArgumentList "/install","/quiet","/norestart" -Wait -PassThru
        Write-Host "  Instalador terminado (codigo $($p.ExitCode); 0 o 3010 es correcto)."
    } else {
        Write-Host ""
        Write-Host "  Hace falta instalarlo como administrador. Abre PowerShell con" -ForegroundColor Yellow
        Write-Host "  'Ejecutar como administrador' y ejecuta:" -ForegroundColor Yellow
        Write-Host "      & `"$redist`" /install /quiet /norestart"
        Write-Host "  Despues vuelve a ejecutar este script."
        exit 1
    }
}

Write-Section "5. Comprobacion"

# The check runs from a file, not from ``python -c``: Windows PowerShell 5.1
# strips the double quotes inside an argument handed to a native program, so
# the inline version failed with "SyntaxError: '(' was never closed" on a
# perfectly good install (2026-09-29).
$check = @'
import cv2, mediapipe, numpy, fastapi, scipy, skimage, PIL
print("  numpy      ", numpy.__version__)
print("  opencv     ", cv2.__version__)
print("  mediapipe  ", mediapipe.__version__)
print("  fastapi    ", fastapi.__version__)
print("  scipy      ", scipy.__version__)
print("  pillow     ", PIL.__version__)
'@
$checkFile = Join-Path $env:TEMP "photorobot_setup_check.py"
Set-Content -Path $checkFile -Value $check -Encoding ascii
& $venvPython $checkFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Alguna libreria no se instalo bien." -ForegroundColor Red
    exit 1
}

Write-Section "6. Modelos de rostro y cuerpo"

& $venvPython (Join-Path $PSScriptRoot "fetch_face_model.py")
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Faltan modelos: revisa la conexion y vuelve a ejecutar." -ForegroundColor Red
    exit 1
}

Write-Section "7. Carpetas de datos"

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
