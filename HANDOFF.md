# FloodLens — Project Handoff / State Analysis

_Written 2026-07-19 to hand this project off to a fresh IDE session (Antigravity).
It records what exists, what's deployed, what was built where, and what's left._

---

## 1. What FloodLens is

Flood-water **semantic segmentation** on Sentinel-2 satellite imagery. Given a 13-band
Sentinel-2 GeoTIFF chip, it predicts a per-pixel water mask. It's a **portfolio /
implementation-and-evaluation project** — inspired by NASA/IBM Prithvi + Sen1Floods11 —
not a new model or a production service.

- **Local folder:** `Z:\Projects\Space Project`
- **GitHub (empty, not pushed yet):** https://github.com/arpan-s-dev/floodlens
- **Live demo (DEPLOYED):** https://huggingface.co/spaces/arpanjeet/floodlens
  — static HF Space, model runs fully in-browser (ONNX + WebAssembly).

## 2. Current results (real, measured)

Trained on a 102-chip, 10-country subset of Sen1Floods11 (72 train / 15 val / 15 test).

| Method | IoU | Dice/F1 | Precision | Recall |
|---|---|---|---|---|
| NDWI heuristic baseline | 0.108 | 0.195 | 0.110 | 0.858 |
| **Tiny U-Net (weighted BCE + Dice)** | **0.678** | **0.808** | **0.699** | **0.958** |

**The build story (all real, all measured):**
- v1 — 30 Ghana chips, plain BCE → **IoU 0.000** (majority-class collapse; ~1.8% water)
- v2 — same data, weighted BCE + soft Dice (pos_weight≈91) → **IoU 0.193**
- v3 — 102 chips / 10 countries, same loss → **IoU 0.678**

## 3. Who built what

**Claude Code session (this repo, earlier):** data download scripts, ML env, dataset
inspection, manifests, the three training runs, `losses.py` + training flags, evaluation
pipeline, comparison figures, the **Gradio** demo (`demo/app.py`), the root `README.md`
with Mermaid diagrams, and `docs/ml_math_explained.md`.

**Cursor session (most recent):** ONNX export + parity check (`scripts/export_onnx.py`),
the **static browser demo** (`demo_static/` — plain HTML/JS, geotiff.js, onnxruntime-web,
KaTeX, Mermaid), **deployed the live HF static Space**, reframed the README as "AI Flood
Area Detection," and added the "download a sample + re-upload" UX to the static site.
It also drafted a backdated-commit plan and a portfolio-pipeline analysis — **neither was
executed** (see §7).

## 4. File map (what matters)

```
Space Project/
├── ml/space_mapper/          ML core (Python package)
│   ├── io.py                 GeoTIFF/mask reading  (int16 → /32767 normalization)
│   ├── datasets.py           manifests, splits, resize, PyTorch Dataset
│   ├── models/unet.py        Tiny U-Net (~0.5 MB, base_channels=16)
│   ├── losses.py             soft Dice + weighted-BCE + compute_pos_weight()
│   ├── metrics.py            IoU, Dice/F1, precision, recall
│   ├── inference.py          heuristic_predict() + predict_with_checkpoint()
│   └── cli/                  prepare_sample_manifest, train_baseline,
│                             predict_sample, evaluate_predictions
├── ml/artifacts/             tiny_unet_diverse.pt (BEST), tiny_unet_sen1floods11.pt (old)
├── ml/.venv/                 Python 3.12 env (torch CPU, rasterio, onnx, gradio)
├── api/                      FastAPI backend (/health /samples /predict)
├── web/                      React + Vite front end (full-stack variant)
├── demo/app.py               Gradio demo (LOCAL only; HF paywalls Gradio Spaces now)
├── demo_static/              *** THE LIVE SPACE SOURCE ***
│   ├── index.html            page + explainer + Mermaid + KaTeX
│   ├── app.js                in-browser inference (mirrors Python pipeline exactly)
│   ├── model/tiny_unet_diverse.onnx   (468 KB, verified 100% parity with PyTorch)
│   ├── examples/*.tif        4 sample chips (Spain, Mekong, 2× Ghana)
│   └── README.md             HF Space config header (sdk: static)
├── scripts/                  download, train pipeline, make_comparison, export_onnx,
│                             package_space, run_api/web, smoke_check
├── docs/                     research_summary, implementation_plan, ml_math_explained,
│                             assets/comparison_diverse.png
└── data/                     LOCAL ONLY, gitignored: raw chips, manifests, predictions, results
```

**Stray file to deal with:** `tech-stack.tsx` at repo root (and `Z:\Projects\Space Project\ml`
also has `smoke_test.py`). `tech-stack.tsx` looks orphaned — review before committing.

## 5. How to run things

```powershell
# ML env is already built at ml\.venv (Python 3.12, torch CPU, rasterio, onnx, gradio)
$py = "ml\.venv\Scripts\python.exe"

# retrain best model
& $py -m space_mapper.cli.train_baseline --manifest ..\data\manifests\sen1floods11_diverse.csv `
  --output artifacts\tiny_unet_diverse.pt --epochs 30 --batch-size 8 --loss bce_dice --pos-weight auto
# (run from ml\ ; or set PYTHONPATH=ml and use full paths)

# evaluate + comparison figure (from repo root, with PYTHONPATH=%CD%\ml)
& $py scripts\run_pipeline.py --split test --manifest data\manifests\sen1floods11_diverse.csv `
  --checkpoint ml\artifacts\tiny_unet_diverse.pt --tag _diverse
& $py scripts\make_comparison.py --manifest data\manifests\sen1floods11_diverse.csv `
  --out data\results\comparison_diverse.png

# re-export ONNX after retraining (also runs a parity check)
& $py scripts\export_onnx.py

# Gradio demo (local)              # static demo (what's deployed)
& $py demo\app.py                  # open demo_static\index.html via any static server
```

**Redeploy the live Space** (needs an HF write token; the previously pasted one should be
rotated):
```python
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="demo_static", repo_id="arpanjeet/floodlens", repo_type="space")
```

## 6. Key technical facts to preserve

- **Normalization:** int16 bands ÷ 32767 → [0,1]. The static JS `app.js` replicates this
  exactly, plus area-average downscale to 128 and bilinear upscale of probabilities. Keep
  Python and JS in lockstep or parity breaks.
- **Loss matters more than architecture here:** plain BCE collapses on ~2% positives;
  `bce_dice` + `pos_weight auto` is what unlocked detection. Don't "simplify" it back.
- **Masks:** Sen1Floods11 labels are {-1 no-data, 0 not-water, 1 water}; pipeline treats
  only `==1` as positive.
- **ONNX:** opset 17, `dynamo=False`, input `[1,13,128,128]` named `input`, output `logits`.
  Parity verified (max|logit diff| tiny, mask agreement ~100%).

## 7. What's LEFT to do

1. **Git:** repo is **not initialized locally** and GitHub remote is **empty**. Needs:
   fix `.gitignore` to allow `demo_static/examples/*.tif` + `demo_static/model/*.onnx`
   (keep `data/`, `.venv/`, `*.pt` ignored) → init → commit with **honest, real dates**
   (author: Arpanjeet Singh, no AI co-author trailers) → push.
   > Note: a backdated April–June 2026 history was proposed but is **not** recommended and
   > was not done — repo/Space were created in July 2026, so backdated commits would be
   > internally inconsistent and misrepresent the timeline. Commit with real dates.
2. **Portfolio:** add FloodLens to `C:\Users\ranji\my_portfolio` (projects grid +
   architecture entry) — not done yet.
3. **Token hygiene:** rotate the HF write token.
4. **Roadmap (optional):** Sentinel-1 (radar, sees through cloud), GPU-scale training
   (RTX 5050), threshold calibration to raise precision (currently 0.70), Prithvi
   reference comparison, ImpactMesh multi-hazard.

## 8. Honest limitations (keep these visible)

Small subset (102 chips), optical-only (cloud hides water), 15-chip test set (metrics
indicative not definitive), precision 0.70 (some over-painting). Framing everywhere:
implementation & evaluation project, **not** a new foundation model or production service.
