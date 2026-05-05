# Downloads a small subset of the Sen1Floods11 hand-labeled dataset over plain HTTPS.
# Source: public Google Cloud bucket gs://sen1floods11 (no gsutil / no login needed).
# Saves paired optical images + water masks with MATCHING filenames so the
# manifest tool can pair them by stem.
#
# Usage (from the project root):
#   .\scripts\download_sen1floods11_subset.ps1
#   .\scripts\download_sen1floods11_subset.ps1 -Count 50    # grab more pairs

param(
    [int]$Count = 80
)

$ErrorActionPreference = "Stop"

# --- Where things go -------------------------------------------------------
$Root      = Split-Path -Parent $PSScriptRoot          # project root
$OutDir    = Join-Path $Root "data\raw\sen1floods11"
$ImageDir  = Join-Path $OutDir "images"
$MaskDir   = Join-Path $OutDir "masks"
New-Item -ItemType Directory -Force -Path $ImageDir | Out-Null
New-Item -ItemType Directory -Force -Path $MaskDir  | Out-Null

# --- Remote layout ---------------------------------------------------------
$Bucket   = "https://storage.googleapis.com/sen1floods11/v1.1"
$SplitCsv = "$Bucket/splits/flood_handlabeled/flood_train_data.csv"
$S2Dir    = "$Bucket/data/flood_events/HandLabeled/S2Hand"
$LabelDir = "$Bucket/data/flood_events/HandLabeled/LabelHand"

Write-Host "Fetching file list from split CSV..." -ForegroundColor Cyan
$csv = (Invoke-WebRequest -Uri $SplitCsv -UseBasicParsing).Content
$allRows = $csv -split "`n" | Where-Object { $_.Trim() -ne "" }

# --- Pick a DIVERSE subset: round-robin across countries -------------------
# Group rows by country (the text before the first underscore, e.g. "Ghana").
$byCountry = @{}
foreach ($row in $allRows) {
    $country = (($row -split "_")[0]).Trim()
    if (-not $byCountry.ContainsKey($country)) { $byCountry[$country] = New-Object System.Collections.ArrayList }
    [void]$byCountry[$country].Add($row)
}
Write-Host ("Countries available: {0}" -f ($byCountry.Keys -join ", ")) -ForegroundColor DarkGray

# Take one from each country in turn until we reach $Count.
$lines = New-Object System.Collections.ArrayList
$idx = 0
while ($lines.Count -lt $Count) {
    $added = $false
    foreach ($country in ($byCountry.Keys | Sort-Object)) {
        if ($idx -lt $byCountry[$country].Count) {
            [void]$lines.Add($byCountry[$country][$idx])
            $added = $true
            if ($lines.Count -ge $Count) { break }
        }
    }
    if (-not $added) { break }   # exhausted all countries
    $idx++
}

Write-Host "Downloading $($lines.Count) diverse image/mask pairs to $OutDir" -ForegroundColor Cyan

$ok = 0
foreach ($line in $lines) {
    # Each line looks like: Ghana_103272_S1Hand.tif,Ghana_103272_LabelHand.tif
    $firstCol = ($line -split ",")[0].Trim()
    # Base id = filename without the "_S1Hand.tif" suffix, e.g. "Ghana_103272"
    $baseId = $firstCol -replace "_S1Hand\.tif$", ""

    $s2Url    = "$S2Dir/${baseId}_S2Hand.tif"
    $labelUrl = "$LabelDir/${baseId}_LabelHand.tif"

    # Renamed so image + mask share the SAME stem for pairing.
    $imgOut  = Join-Path $ImageDir "$baseId.tif"
    $maskOut = Join-Path $MaskDir  "$baseId.tif"

    try {
        Invoke-WebRequest -Uri $s2Url    -OutFile $imgOut  -UseBasicParsing
        Invoke-WebRequest -Uri $labelUrl -OutFile $maskOut -UseBasicParsing
        $ok++
        Write-Host ("  [{0}/{1}] {2}" -f $ok, $lines.Count, $baseId)
    }
    catch {
        Write-Warning "  Skipped $baseId : $($_.Exception.Message)"
        if (Test-Path $imgOut)  { Remove-Item $imgOut  -Force }
        if (Test-Path $maskOut) { Remove-Item $maskOut -Force }
    }
}

Write-Host ""
Write-Host "Done. Downloaded $ok pairs." -ForegroundColor Green
Write-Host "Images: $ImageDir"
Write-Host "Masks:  $MaskDir"
Write-Host ""
Write-Host "Provenance: Sen1Floods11 (Bonafilia et al., CVPR 2020 Workshops)." -ForegroundColor DarkGray
Write-Host "Source bucket: gs://sen1floods11/v1.1  (public)" -ForegroundColor DarkGray
