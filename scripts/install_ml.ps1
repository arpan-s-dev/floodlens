$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$mlDir = Join-Path $root "ml"

if (-not (Test-Path -LiteralPath (Join-Path $mlDir "requirements.txt"))) {
  throw "ML requirements not found. Expected ml\requirements.txt"
}

Set-Location $mlDir

if (-not (Test-Path -LiteralPath ".venv")) {
  python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "ML environment ready: $mlDir\.venv"
