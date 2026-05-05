# Vendors everything the Gradio demo needs into demo/ so it can be pushed to a
# self-contained Hugging Face Space. Run from the project root.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$demo = Join-Path $root "demo"

# 1. Inference code
$srcPkg = Join-Path $root "ml\space_mapper"
$dstPkg = Join-Path $demo "space_mapper"
if (Test-Path $dstPkg) { Remove-Item $dstPkg -Recurse -Force }
Copy-Item $srcPkg $dstPkg -Recurse
# drop caches
Get-ChildItem $dstPkg -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# 2. Trained checkpoint (the diverse 10-country model is the good one)
$modelDir = Join-Path $demo "model"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
Copy-Item (Join-Path $root "ml\artifacts\tiny_unet_diverse.pt") $modelDir -Force

# 3. A few example chips (varied countries + water levels look best in the demo)
$exDir = Join-Path $demo "examples"
New-Item -ItemType Directory -Force -Path $exDir | Out-Null
$examples = @("Spain_5923267", "Mekong_52610", "Nigeria_529525", "Ghana_264787")
foreach ($id in $examples) {
    $src = Join-Path $root "data\raw\sen1floods11\images\$id.tif"
    if (Test-Path $src) { Copy-Item $src $exDir -Force }
}

Write-Host "Packaged demo/ for a Hugging Face Space:" -ForegroundColor Green
Write-Host "  demo/space_mapper/   (inference code)"
Write-Host "  demo/model/          (checkpoint)"
Write-Host "  demo/examples/       (sample chips)"
Write-Host "Next: create a Space, then push the contents of demo/ to it."
