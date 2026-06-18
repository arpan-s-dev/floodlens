---
title: FloodLens
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
short_description: AI flood area detection from Sentinel-2
---

# 🛰️ FloodLens — AI Flood Area Detection

Detect and map flood extent from Sentinel-2 satellite imagery at pixel level.
Deep flood segmentation (U-Net) trained on
[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11), edge-deployed in your browser
via ONNX + WebAssembly.

**Performance:** IoU **0.678** · Dice **0.808** · Recall **0.958** (~6× vs classical NDWI)

**Try it:** run live flood area detection on sample disaster chips, or download a chip and
re-upload through the full pipeline.

Source: [github.com/arpan-s-dev/floodlens](https://github.com/arpan-s-dev/floodlens) ·
Built by [Arpanjeet Singh](https://github.com/arpan-s-dev)
