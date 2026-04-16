# Space Disaster Mapper ML slice

This folder contains a small, reproducible Python package for a portfolio-ready satellite disaster segmentation MVP. It is inspired by Prithvi-style foundation model workflows, Sen1Floods11 flood mapping, and ImpactMesh-style disaster intelligence, but it does **not** download datasets or claim trained benchmark metrics.

## What is implemented

- Local image/mask manifest creation for binary segmentation datasets.
- PNG/JPEG/BMP/WebP image support with Pillow.
- GeoTIFF reading when `rasterio` is installed; `.tif/.tiff` inputs fail with a clear install message otherwise.
- Binary segmentation metrics: IoU, Dice/F1, precision, recall, and confusion counts.
- Optional lightweight PyTorch U-Net baseline for CPU-friendly experiments on small subsets.
- Deterministic heuristic prediction for demos without trained weights. This is a **heuristic baseline**, not model performance.
- CLI entry points: `prepare_sample_manifest`, `train_baseline`, `evaluate_predictions`, `predict_sample`.

## Setup

From `Z:\Projects\Space Project`:

```powershell
python -m venv ml\.venv
ml\.venv\Scripts\python -m pip install --upgrade pip
ml\.venv\Scripts\python -m pip install -e ml
```

Optional dependencies:

```powershell
# For PyTorch baseline training/inference, install the CPU build recommended by pytorch.org.
ml\.venv\Scripts\python -m pip install torch

# For GeoTIFF/Sentinel-style rasters.
ml\.venv\Scripts\python -m pip install rasterio
```

## Dataset layout

Use small local subsets while iterating:

```text
data\sample\images\tile_001.png
data\sample\images\tile_002.png
data\sample\masks\tile_001.png
data\sample\masks\tile_002.png
```

Image and mask files are paired by filename stem.

Recommended public datasets to explore manually:

- **Sen1Floods11**: Sentinel-1 flood segmentation labels for flood mapping experiments.
- **ImpactMesh / disaster response datasets**: useful inspiration for multi-source disaster intelligence products.
- **Prithvi / HLS examples**: useful future direction for foundation-model embeddings and multispectral transfer learning.

Download datasets outside this package and point the commands at local folders. Do not commit large imagery.

## Commands

Create a manifest:

```powershell
prepare_sample_manifest --images data\sample\images --masks data\sample\masks --output ml\artifacts\manifest.csv
```

Train the tiny U-Net on a small subset:

```powershell
train_baseline --manifest ml\artifacts\manifest.csv --output ml\artifacts\unet_baseline.pt --epochs 2 --resize 128 --batch-size 2 --max-samples 32
```

Create a deterministic heuristic prediction without model weights:

```powershell
predict_sample --image data\sample\images\tile_001.png --output ml\artifacts\predictions\tile_001.png --disaster flood
```

Use a trained checkpoint instead:

```powershell
predict_sample --image data\sample\images\tile_001.png --output ml\artifacts\predictions\tile_001.png --checkpoint ml\artifacts\unet_baseline.pt
```

Evaluate predictions against masks:

```powershell
evaluate_predictions --pred-dir ml\artifacts\predictions --mask-dir data\sample\masks --output ml\artifacts\metrics.json
```

Run the smoke test:

```powershell
python ml\smoke_test.py
```

## Future work

- Add real Sen1Floods11/ImpactMesh dataset adapters and download checksums.
- Add geospatial tiling/stitching and CRS-aware output metadata.
- Add Prithvi embedding extraction or fine-tuning adapters.
- Add experiment tracking and robust train/validation/test protocols.
- Add cloud-optimized GeoTIFF outputs and API/web integration.
