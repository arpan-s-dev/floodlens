$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Space Disaster Mapper smoke check"
Write-Host "Root: $root"

$required = @(
  "README.md",
  ".gitignore",
  "PROJECT_STATUS.md",
  "sample_data\README.md"
)

foreach ($path in $required) {
  if (-not (Test-Path -LiteralPath $path)) {
    throw "Missing required file: $path"
  }
}

if (Test-Path -LiteralPath "api\app\main.py") {
  python -m py_compile "api\app\main.py"
}

if (Test-Path -LiteralPath "ml\space_mapper") {
  $pyFiles = Get-ChildItem -LiteralPath "ml\space_mapper" -Filter "*.py" -Recurse
  foreach ($file in $pyFiles) {
    python -m py_compile $file.FullName
  }
}

if (Test-Path -LiteralPath "web\package.json") {
  Write-Host "web\package.json found"
}

Write-Host "Smoke check completed"
