$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$apiDir = Join-Path $root "api"

if (-not (Test-Path -LiteralPath (Join-Path $apiDir "app\main.py"))) {
  throw "API app not found. Expected api\app\main.py"
}

Set-Location $apiDir

if (-not (Test-Path -LiteralPath ".venv")) {
  python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
