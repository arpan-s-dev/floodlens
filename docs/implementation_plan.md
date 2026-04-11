# Space Disaster Mapper - Implementation Plan

This plan turns the research direction into a resume-worthy ML engineering project. The project should be built in phases so there is always a working demo, even before full datasets and GPU training are available.

## Tier 1 - MVP

**Goal:** working local demo with sample images, heuristic/baseline masks, API, and web UI.

Estimated time: 2-3 weeks.

### Milestone 1.1 - Project scaffold

- Create root README, `.gitignore`, sample-data policy, and smoke-check scripts.
- Split project into `ml\`, `api\`, `web\`, and `docs\`.
- Keep large datasets, checkpoints, generated masks, and runtime outputs out of git.

Status: started.

### Milestone 1.2 - ML core

- Implement dataset utilities for local image/mask pairs.
- Support normal images immediately and GeoTIFF when `rasterio` is installed.
- Implement binary segmentation metrics:
  - IoU
  - Dice/F1
  - precision
  - recall
  - confusion counts
- Implement a lightweight PyTorch U-Net baseline if PyTorch is available.
- Implement a deterministic heuristic fallback so the demo works without trained weights.
- Add CLI entry points:
  - create sample manifest
  - predict sample
  - evaluate predictions
  - train baseline

### Milestone 1.3 - API

- FastAPI service with:
  - `GET /health`
  - `GET /samples`
  - `POST /predict`
- Accept known sample IDs and uploaded images.
- Return prediction metadata, pixel stats, and a mask image/base64 payload.
- Use local fallback prediction if the ML package is unavailable.

### Milestone 1.4 - Web demo

- React + TypeScript frontend.
- List samples from the API.
- Allow image upload.
- Call `/predict`.
- Display source image, predicted mask, and metrics.
- Fall back to polished mock samples when API is offline.

### MVP deliverable

A recruiter can run:

```powershell
cd "Z:\Projects\Space Project"
.\scripts\smoke_check.ps1
.\scripts\run_api.ps1
.\scripts\run_web.ps1
```

Then open the Vite URL and see a working satellite disaster-mask demo.

## Tier 2 - Strong resume version

**Goal:** real dataset evaluation with honest metrics.

Estimated time: 4-6 weeks.

### Milestone 2.1 - Sen1Floods11 ingestion

- Download Sen1Floods11 outside git.
- Build a manifest from image/mask files.
- Validate shapes, missing masks, class balance, and image statistics.
- Create a tiny reproducible subset for smoke tests.

### Milestone 2.2 - Baseline training

- Train a lightweight U-Net or DeepLab-style baseline on a small subset first.
- Log IoU, Dice/F1, precision, recall, and loss curves.
- Save configs, seed, dataset split, and checkpoint path.
- Produce qualitative prediction panels for the README.

### Milestone 2.3 - Foundation-model path

- Reproduce or run inference with official Prithvi flood/burn-scar resources where feasible.
- If GPU limits block full fine-tuning, clearly frame this as:
  - official Prithvi model/reference
  - local U-Net baseline
  - local inference/evaluation harness
- Do not claim unsupported Prithvi metrics unless reproduced locally.

### Milestone 2.4 - Ablations

Recommended ablations:

| Ablation | Values |
| --- | --- |
| Model | heuristic, U-Net, Prithvi/reference |
| Input type | RGB/optical, SAR if available, optical+SAR if available |
| Dataset size | tiny subset, 25%, 100% |
| Resolution | 256, 512 |
| Threshold/fallback | fixed threshold, adaptive threshold |

### Milestone 2.5 - Results table

Create `results\main_table.md`:

```markdown
| Model | Dataset | IoU | Dice/F1 | Precision | Recall | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Heuristic baseline | Tiny demo subset | TBD | TBD | TBD | TBD | No training |
| U-Net baseline | Sen1Floods11 subset | TBD | TBD | TBD | TBD | Local training |
| Prithvi reference/inference | Sen1Floods11 | TBD | TBD | TBD | TBD | Cite exact source or local run |
```

Only fill in numbers after running the experiment.

## Tier 3 - Polished portfolio version

**Goal:** public GitHub + screenshots + live/demo-ready app.

Estimated time: 6-8+ weeks.

### Milestone 3.1 - Better geospatial support

- Preserve GeoTIFF CRS/transform metadata when possible.
- Save masks as GeoTIFF and PNG preview.
- Add RGB composite rendering for multispectral inputs.
- Add overlay alpha controls in the frontend.

### Milestone 3.2 - ImpactMesh subset

- Download a manageable ImpactMesh subset.
- Compare flood vs fire samples.
- Add modality ablations:
  - Sentinel-2 optical
  - Sentinel-1 SAR
  - optical + SAR
  - optical + SAR + DEM

### Milestone 3.3 - Demo deployment

- Containerize API if needed.
- Deploy web frontend to Vercel/Netlify or serve locally for demos.
- Optionally build a Hugging Face Space or Gradio companion demo for model inference.
- Add screenshots/GIFs to the README.

### Milestone 3.4 - Technical report

Write a short report:

1. Motivation
2. Related work
3. Dataset
4. Method
5. Experiments
6. Results
7. Limitations
8. Future work

Keep the language honest: this project implements and extends existing research; it does not invent a new foundation model.

## Resource estimate

| Stage | Compute | Storage | Notes |
| --- | --- | --- | --- |
| MVP | CPU okay | <1 GB | Synthetic/tiny images only |
| Baseline training | Local GPU preferred | 10-20 GB | Sen1Floods11 subset/full |
| Prithvi/ImpactMesh | GPU strongly preferred | 50+ GB | Use subsets first |
| Portfolio demo | CPU okay for small inference | small runtime outputs | Use optimized checkpoints or heuristic fallback |

## Resume framing

After MVP:

> Built a FastAPI + React satellite disaster-mapping demo that generates and visualizes flood/wildfire segmentation masks from uploaded imagery.

After real evaluation:

> Implemented a paper-backed geospatial ML pipeline for flood/wildfire segmentation using PyTorch and satellite imagery; benchmarked heuristic and U-Net baselines with IoU, Dice/F1, precision, and recall on Sen1Floods11/ImpactMesh-style data.

After full polish:

> Extended a Prithvi-inspired satellite disaster-mapping workflow with reproducible training/evaluation, modality ablations, and a deployed FastAPI + React mask-visualization demo for flood and wildfire imagery.
