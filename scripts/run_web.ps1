$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$webDir = Join-Path $root "web"

if (-not (Test-Path -LiteralPath (Join-Path $webDir "package.json"))) {
  throw "Web app not found. Expected web\package.json"
}

Set-Location $webDir

if (-not (Test-Path -LiteralPath "node_modules")) {
  npm install
}

npm run dev
