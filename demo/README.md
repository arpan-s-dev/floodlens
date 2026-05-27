---
title: FloodLens
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Space Disaster Mapper — Gradio demo

Flood-water segmentation on Sentinel-2 imagery, trained on a subset of
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Bonafilia et al., CVPR 2020
Workshops). A Prithvi-inspired **implementation & evaluation** project — not a production
flood model.

Upload a 13-band Sentinel-2 `.tif` chip (or pick an example) and the app predicts a
water mask with a small CPU U-Net, plus a heuristic (NDWI-like) baseline for comparison.

## Deploying this as a Hugging Face Space

This folder is self-contained once you vendor two things next to `app.py`:

1. `space_mapper/` — copy of `../ml/space_mapper/` (the inference code)
2. `model/tiny_unet_sen1floods11.pt` — copy of `../ml/artifacts/tiny_unet_sen1floods11.pt`
3. `examples/*.tif` — a few sample chips (optional but nice)

The repo has a helper: `scripts/package_space.ps1` copies these in for you.

Running locally (from the repo root): `ml/.venv/Scripts/python.exe demo/app.py`
— locally it falls back to the repo copies, so no vendoring needed to test.
