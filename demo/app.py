"""FloodLens — flood-water segmentation demo (Terra Nova design language).

Loads the trained Tiny U-Net and predicts a water mask from a Sentinel-2 GeoTIFF.
Runs locally (`python demo/app.py`) and on a free Hugging Face CPU Space.

UI: light editorial single-page experience — Space Grotesk type, green-700 primary,
pill buttons, large-radius white cards — with a full technical explainer below the demo
(problem, data, models, equations, architecture diagrams, results). Light mode is
forced (Gradio's automatic dark mode fights the palette), and diagrams render with
mermaid.js loaded in <head>.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Make space_mapper importable: prefer a vendored copy, else the repo's ml/.
for candidate in (HERE, REPO / "ml"):
    if (candidate / "space_mapper").is_dir():
        sys.path.insert(0, str(candidate))
        break

from space_mapper.inference import heuristic_predict, predict_with_checkpoint  # noqa: E402
from space_mapper.io import read_image  # noqa: E402

# Checkpoint: prefer the diverse (best) model, vendored first, then repo copies.
CKPT = None
for candidate in (HERE / "model" / "tiny_unet_diverse.pt",
                  HERE / "model" / "tiny_unet_sen1floods11.pt",
                  REPO / "ml" / "artifacts" / "tiny_unet_diverse.pt",
                  REPO / "ml" / "artifacts" / "tiny_unet_sen1floods11.pt"):
    if candidate.exists():
        CKPT = candidate
        break

EXAMPLES_DIR = HERE / "examples"
EXAMPLE_FILES = sorted(str(p) for p in EXAMPLES_DIR.glob("*.tif")) if EXAMPLES_DIR.is_dir() else []

WATER_RGB = (14, 165, 233)  # sky-500 — reads as water on any land cover


def _stretch(band: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(band, 2), np.percentile(band, 98)
    b = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    return (b * 255).astype(np.uint8)


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Best-effort true-color preview from a multi-band or RGB image."""
    if image.ndim == 2:
        g = _stretch(image)
        return np.dstack([g, g, g])
    c = image.shape[-1]
    if c >= 4:  # Sentinel-2: B4=idx3 red, B3=idx2 green, B2=idx1 blue
        return np.dstack([_stretch(image[..., 3]), _stretch(image[..., 2]), _stretch(image[..., 1])])
    if c == 3:
        return np.dstack([_stretch(image[..., i]) for i in range(3)])
    g = _stretch(image[..., 0])
    return np.dstack([g, g, g])


def _overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    tint = np.array(WATER_RGB, dtype=np.float32)
    out[m] = 0.3 * out[m] + 0.7 * tint
    return out.astype(np.uint8)


def predict(file_path: str, method: str):
    if not file_path:
        return None, None, (
            "<div class='fl-stats fl-card'><div class='stat'><span class='stat-label'>STATUS</span>"
            "<span class='stat-value'>Upload a Sentinel-2 GeoTIFF (.tif) or pick an example.</span></div></div>"
        )
    image = read_image(file_path, normalize=True)
    rgb = _to_rgb(image)

    if method == "Tiny U-Net" and CKPT is not None:
        result = predict_with_checkpoint(image, CKPT)
        label = "Tiny U-Net · trained checkpoint"
    else:
        result = heuristic_predict(image, disaster="flood")
        label = "Heuristic baseline · NDWI-like"

    mask = result.mask
    water_pct = 100.0 * float(mask.mean())
    info = (
        "<div class='fl-stats fl-card'>"
        f"<div class='stat'><span class='stat-label'>METHOD</span><span class='stat-value'>{label}</span></div>"
        f"<div class='stat'><span class='stat-label'>WATER COVERAGE</span><span class='stat-value green'>{water_pct:.1f}%</span></div>"
        f"<div class='stat'><span class='stat-label'>RESOLUTION</span><span class='stat-value'>{mask.shape[1]}×{mask.shape[0]} px</span></div>"
        "</div>"
    )
    return rgb, _overlay(rgb, mask), info


# ---------------------------------------------------------------------------
# <head>: force light mode + load mermaid for the architecture diagrams
# ---------------------------------------------------------------------------

HEAD = """
<script>
  // FloodLens is a light design; Gradio's automatic dark mode breaks the palette.
  (function () {
    const forceLight = () => {
      document.documentElement.classList.remove('dark');
      if (document.body) document.body.classList.remove('dark');
    };
    forceLight();
    new MutationObserver(forceLight)
      .observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    document.addEventListener('DOMContentLoaded', () => {
      forceLight();
      if (document.body) {
        new MutationObserver(forceLight)
          .observe(document.body, { attributes: true, attributeFilter: ['class'] });
      }
    });
  })();
</script>
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  mermaid.initialize({
    startOnLoad: false,
    theme: "neutral",
    themeVariables: {
      fontFamily: "Space Grotesk, sans-serif",
      primaryColor: "#f0fdf4",
      primaryBorderColor: "#15803d",
      primaryTextColor: "#111827",
      lineColor: "#15803d",
      secondaryColor: "#ffffff",
      tertiaryColor: "#f9fafb",
      noteBkgColor: "#f0fdf4",
      noteBorderColor: "#15803d",
      actorBorder: "#15803d",
      actorBkg: "#ffffff",
      signalColor: "#374151",
      signalTextColor: "#374151"
    }
  });
  // Render each diagram individually with a unique id (bulk mermaid.run collides
  // and stacks every SVG into the first card). Idempotent per node, so it survives
  // Gradio re-mounts: unprocessed nodes get picked up on the next tick.
  let seq = 0;
  const renderAll = async () => {
    const nodes = Array.from(document.querySelectorAll("pre.mermaid:not(.fl-done)"));
    for (const node of nodes) {
      if (!node.dataset.code) node.dataset.code = node.textContent;
      try {
        const { svg } = await mermaid.render("flmmd" + (seq++), node.dataset.code.trim());
        node.innerHTML = svg;
        node.classList.add("fl-done");
      } catch (e) { console.error("mermaid:", e); }
    }
  };
  let ticks = 0;
  const tick = async () => { await renderAll(); if (++ticks < 40) setTimeout(tick, 700); };
  setTimeout(tick, 800);
</script>
"""

# ---------------------------------------------------------------------------
# Terra Nova design system, rebuilt for Gradio — light mode, forced
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Geist+Mono:wght@400;600&display=swap');

:root, .dark {
  --fl-bg: #f4f5f6;
  --fl-card: #ffffff;
  --fl-fg: #111827;
  --fl-body: #374151;
  --fl-muted: #6b7280;
  --fl-border: #e5e7eb;
  --fl-green: #15803d;
  --fl-green-dark: #166534;
  --fl-green-soft: #f0fdf4;
  --fl-radius: 16px;

  /* Pin the Gradio theme variables that dark mode would otherwise flip. */
  --body-background-fill: #f4f5f6 !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #f9fafb !important;
  --block-background-fill: #ffffff !important;
  --body-text-color: #111827 !important;
  --block-label-text-color: #6b7280 !important;
  --block-title-text-color: #111827 !important;
  --border-color-primary: #e5e7eb !important;
  --input-background-fill: #ffffff !important;
}

html, body { background: var(--fl-bg) !important; }

.gradio-container {
  background: var(--fl-bg) !important;
  font-family: 'Space Grotesk', ui-sans-serif, system-ui !important;
  color: var(--fl-fg) !important;
  max-width: 1180px !important;
  margin: 0 auto !important;
}

/* ---------- hero ---------- */
.fl-hero {
  border-radius: 28px;
  padding: 56px 48px 52px;
  background:
    radial-gradient(1100px 480px at 85% -10%, rgba(56,189,248,0.30), transparent 60%),
    radial-gradient(900px 420px at 8% 112%, rgba(34,197,94,0.22), transparent 55%),
    linear-gradient(160deg, #0b2530 0%, #0e3542 48%, #114237 100%);
  color: #ffffff;
  margin-bottom: 18px;
}
.fl-hero .pill-logo {
  display: inline-flex; align-items: center; gap: 8px;
  background: #ffffff; color: #111827;
  border-radius: 999px; padding: 8px 18px;
  font-weight: 600; font-size: 14px; letter-spacing: -0.01em;
}
.fl-hero .pill-logo .green { color: #15803d; }
.fl-hero h1 {
  font-size: clamp(34px, 5vw, 56px);
  font-weight: 700; letter-spacing: -0.03em; line-height: 1.05;
  margin: 26px 0 14px; color: #ffffff;
}
.fl-hero p.tag { color: rgba(255,255,255,0.85); font-size: 17px; max-width: 640px; margin: 0 0 26px; }
.fl-hero .metric-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.fl-hero .metric-pills span {
  border: 1px solid rgba(255,255,255,0.30); border-radius: 999px;
  padding: 7px 16px; font-size: 13px; color: rgba(255,255,255,0.95);
  font-family: 'Geist Mono', monospace;
  background: rgba(255,255,255,0.08);
}

/* ---------- cards / blocks ---------- */
.gradio-container .block, .gradio-container .form {
  background: var(--fl-card) !important;
  border: 1px solid var(--fl-border) !important;
  border-radius: var(--fl-radius) !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.gradio-container button.primary {
  background: var(--fl-green) !important;
  color: #ffffff !important;
  border-radius: 999px !important;
  border: none !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 15px !important;
  padding: 12px 26px !important;
}
.gradio-container button.primary:hover { background: var(--fl-green-dark) !important; }
.gradio-container label span, .gradio-container .label-wrap span {
  font-family: 'Geist Mono', monospace !important;
  text-transform: uppercase; letter-spacing: 0.08em;
  font-size: 11px !important; color: var(--fl-muted) !important;
}

/* Radio pills: unselected = quiet outline, selected = solid green with white text */
.gradio-container label:has(input[type="radio"]) {
  border-radius: 999px !important;
  border: 1px solid var(--fl-border) !important;
  background: #ffffff !important;
}
.gradio-container label.selected:has(input[type="radio"]),
.gradio-container label:has(input[type="radio"]:checked) {
  background: var(--fl-green) !important;
  border-color: var(--fl-green) !important;
}
.gradio-container label:has(input[type="radio"]:checked) span {
  color: #ffffff !important;
}

/* ---------- stats strip ---------- */
.fl-card {
  background: var(--fl-card); border: 1px solid var(--fl-border);
  border-radius: var(--fl-radius); padding: 18px 22px;
}
.fl-stats { display: flex; gap: 36px; flex-wrap: wrap; }
.fl-stats .stat { display: flex; flex-direction: column; gap: 3px; }
.fl-stats .stat-label {
  font-family: 'Geist Mono', monospace; font-size: 10px;
  letter-spacing: 0.12em; color: var(--fl-muted);
}
.fl-stats .stat-value { font-size: 16px; font-weight: 600; color: var(--fl-fg); }
.fl-stats .stat-value.green { color: var(--fl-green); font-family: 'Geist Mono', monospace; }

/* ---------- explainer sections ---------- */
.fl-section { padding: 8px 6px 2px; }
.fl-section, .fl-section * { color: var(--fl-body); }
.fl-kicker, .fl-section .fl-kicker {
  font-family: 'Geist Mono', monospace; font-size: 12px; font-weight: 600;
  letter-spacing: 0.14em; color: var(--fl-green); text-transform: uppercase;
  margin: 30px 0 6px; display: block;
}
.fl-section h2, .fl-section h2 * {
  font-size: 28px; font-weight: 700; letter-spacing: -0.02em;
  margin: 0 0 12px; color: var(--fl-fg);
}
.fl-section h3 { font-size: 19px; font-weight: 600; margin: 18px 0 8px; color: var(--fl-fg); }
.fl-section strong { color: var(--fl-fg); }
.fl-section a { color: var(--fl-green); }
.fl-section p, .fl-section li { font-size: 15.5px; line-height: 1.65; }
.fl-section table { width: 100%; border-collapse: collapse; margin: 12px 0; background: var(--fl-card); border-radius: 12px; }
.fl-section th, .fl-section td {
  border-bottom: 1px solid var(--fl-border); padding: 9px 12px;
  text-align: left; font-size: 14.5px;
}
.fl-section th {
  font-family: 'Geist Mono', monospace; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.08em; color: var(--fl-muted);
}
.fl-section code {
  font-family: 'Geist Mono', monospace; font-size: 13.5px;
  background: var(--fl-green-soft); color: var(--fl-green-dark);
  border-radius: 6px; padding: 2px 6px;
}

/* ---------- mermaid diagram cards ---------- */
.fl-diagram {
  background: var(--fl-card); border: 1px solid var(--fl-border);
  border-radius: var(--fl-radius); padding: 22px;
  margin: 14px 0; overflow-x: auto;
}
.fl-diagram .mermaid { display: flex; justify-content: center; background: transparent; }
.fl-diagram-caption {
  font-family: 'Geist Mono', monospace; font-size: 11px; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--fl-muted); margin: 10px 4px 0; text-align: center;
}

.fl-footer {
  border-top: 1px solid var(--fl-border); margin-top: 34px; padding: 22px 6px 10px;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
  color: var(--fl-muted); font-size: 14px;
}
.fl-footer a { color: var(--fl-green); text-decoration: none; font-weight: 600; }
footer { display: none !important; }
"""

HERO = """
<div class="fl-hero">
  <span class="pill-logo">🛰️ <span><span class="green">Flood</span>Lens</span></span>
  <h1>See floods from space.</h1>
  <p class="tag">Pixel-level flood-water mapping from Sentinel-2 satellite imagery.
  A compact U-Net — trained end-to-end on the Sen1Floods11 benchmark — traces water
  through coastlines, river valleys, and cloudy scenes.</p>
  <div class="metric-pills">
    <span>IoU 0.68</span><span>Dice 0.81</span><span>Recall 0.96</span>
    <span>~0.5 MB model</span><span>10 countries</span>
  </div>
</div>
"""

LATEX = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

EXPLAINER_1 = r"""
<span class="fl-kicker">01 · The problem</span>
## Floods are the most frequent natural disaster — and clouds hide them

When a river bursts its banks, responders need one map above all: **which pixels are under
water, right now.** Satellites see the whole planet every few days, but turning a raw
13-band Sentinel-2 image into a water map is a **semantic segmentation** problem: a
512×512 chip means **262,144 individual water / not-water decisions.** Hand-drawn rules
break down across geographies — a formula tuned for a clear Spanish coastline fails over a
cloudy Ghanaian floodplain. FloodLens learns the mapping from data instead.

<span class="fl-kicker">02 · The data</span>
## Sen1Floods11

We train and evaluate on [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)
(Bonafilia et al., CVPR 2020 Workshops): hand-labeled 512×512 chips from **11 real flood
events**, where every pixel is labeled water / not-water by analysts. Our subset:
**102 chips across 10 countries** (Ghana, India, Mekong, Nigeria, Pakistan, Paraguay,
Somalia, Spain, Sri-Lanka, USA), split 72 train / 15 val / 15 test, ~9.9% water pixels.
Each input is a 13-band Sentinel-2 image — visible light plus infrared bands invisible
to the eye.

<span class="fl-kicker">03 · Model 1</span>
## The physics baseline — NDWI (no learning)

Water absorbs near-infrared light but reflects green. The classic Normalized Difference
Water Index exploits that:

$$\text{NDWI} = \frac{G - \text{NIR}}{G + \text{NIR}}$$

A pixel is called water when NDWI exceeds a cutoff. No parameters are learned — it's our
*floor to beat*, and it fails in a telling way: clouds and bright soil fool it into painting
water everywhere (precision 0.11).

<span class="fl-kicker">04 · Model 2</span>
## The learned model — Tiny U-Net

A **U-Net** is a convolutional encoder–decoder. The encoder repeatedly convolves and
downsamples to learn *what* is in the image (edges → textures → "water-like region");
the decoder upsamples back to full resolution to say *where* it is; **skip connections**
carry fine detail across so shoreline boundaries stay sharp. Ours is deliberately tiny —
**~0.5 MB, trainable on a laptop CPU in minutes** — and outputs one logit $z$ per pixel,
squashed to a probability:

$$p = \sigma(z) = \frac{1}{1 + e^{-z}} \qquad \text{water if } p \ge 0.5$$
"""

UNET_DIAGRAM = """
<div class="fl-diagram">
<pre class="mermaid">
flowchart LR
    IN["Input<br/>13 × 128 × 128"] --> E1["Conv ×2 + ReLU<br/>16 ch"]
    E1 --> P1["MaxPool ↓"]
    P1 --> E2["Conv ×2 + ReLU<br/>32 ch"]
    E2 --> P2["MaxPool ↓"]
    P2 --> BN["Bottleneck<br/>Conv ×2 · 64 ch"]
    BN --> U2["Upsample ↑"]
    U2 --> D2["Conv ×2<br/>32 ch"]
    D2 --> U1["Upsample ↑"]
    U1 --> D1["Conv ×2<br/>16 ch"]
    D1 --> OUT["1×1 Conv → logit<br/>sigmoid → P(water)"]
    E2 -. skip connection .-> D2
    E1 -. skip connection .-> D1
</pre>
<div class="fl-diagram-caption">Tiny U-Net — encoder · bottleneck · decoder, with skip connections</div>
</div>
"""
# Inside <pre>, the browser would parse <br/> into a real element and corrupt the
# mermaid source text — escape so mermaid receives the literal token.
UNET_DIAGRAM = UNET_DIAGRAM.replace("<br/>", "&lt;br/&gt;")

EXPLAINER_2 = r"""
<span class="fl-kicker">05 · The key equation</span>
## The loss function — and the trap hiding in it

Training minimizes **binary cross-entropy** between prediction $p$ and truth
$y \in \{0,1\}$:

$$\mathcal{L}_{\text{BCE}} = -\big[\,y \log p + (1-y)\log(1-p)\,\big]$$

Its gradient w.r.t. the logit is simply $\partial\mathcal{L}/\partial z = p - y$. Averaged
over an image that is ~98% "not water," that gradient is dominated by the majority class —
so plain BCE is minimized by predicting **no water anywhere**. Our first model did exactly
that: **IoU 0.000.** The fix is to make rare water pixels count. We weight the positive
class by $w_+ \approx N_{\text{neg}}/N_{\text{pos}} \approx 91$ and add a **soft Dice loss**
that optimizes overlap directly and ignores true negatives:

$$\mathcal{L} = \underbrace{-\big[\,w_+\, y \log p + (1-y)\log(1-p)\,\big]}_{\text{weighted BCE}}
\;+\; \underbrace{1 - \frac{2\sum_i p_i y_i + \epsilon}{\sum_i p_i + \sum_i y_i + \epsilon}}_{\text{soft Dice}}$$

Optimized with **AdamW** (momentum + adaptive step size + weight decay), 30 epochs.

<span class="fl-kicker">06 · Measuring honestly</span>
## Metrics that can't be fooled

Accuracy is misleading here — predicting "never water" scores 98% accuracy and finds zero
floods. We report overlap-based metrics built from true/false positives and negatives:

$$\text{IoU} = \frac{TP}{TP + FP + FN} \qquad
\text{Dice/F1} = \frac{2\,TP}{2\,TP + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP} \qquad
\text{Recall} = \frac{TP}{TP + FN}$$

<span class="fl-kicker">07 · Results</span>
## Held-out test set (15 chips)

| Method | IoU ↑ | Dice/F1 ↑ | Precision | Recall |
|---|---|---|---|---|
| Heuristic (NDWI-like) | 0.108 | 0.195 | 0.110 | 0.858 |
| **Tiny U-Net (weighted BCE + Dice)** | **0.678** | **0.808** | **0.699** | **0.958** |

And the experiment trail that earned it — every step measured:

| Version | Data | Loss | IoU | Lesson |
|---|---|---|---|---|
| v1 | 30 chips, one event | plain BCE | 0.000 | Class imbalance collapses plain BCE |
| v2 | 30 chips, one event | weighted BCE + Dice | 0.193 | Imbalance-aware loss stops the collapse |
| v3 | 102 chips, 10 countries | weighted BCE + Dice | **0.678** | Diverse data does the rest |
"""

SYSTEM_HEADER = """
<span class="fl-kicker">08 · System design</span>

## How FloodLens is built
"""

SYSTEM_DIAGRAM = """
<div class="fl-diagram">
<pre class="mermaid">
flowchart LR
    DL["Sen1Floods11<br/>download (public GCS)"] --> MF["Manifest<br/>72 / 15 / 15 split"]
    MF --> TR["Train Tiny U-Net<br/>weighted BCE + Dice · AdamW"]
    TR --> EV["Evaluate<br/>IoU · Dice · P · R"]
    TR --> CK[("Checkpoint<br/>~0.5 MB")]
</pre>
<div class="fl-diagram-caption">Offline · the training pipeline produces one small checkpoint</div>
</div>

<div class="fl-diagram">
<pre class="mermaid">
flowchart LR
    CK[("Checkpoint<br/>~0.5 MB")] --> UN
    UP["GeoTIFF<br/>upload"] --> IO["Read + normalize<br/>13 bands"]
    IO --> CH{"Model<br/>choice"}
    CH -->|"Tiny U-Net"| UN["U-Net<br/>forward pass"]
    CH -->|"Baseline"| ND["NDWI<br/>threshold"]
    UN --> TH["sigmoid<br/>≥ 0.5"]
    ND --> TH
    TH --> OV["Blue overlay<br/>+ water %"]
</pre>
<div class="fl-diagram-caption">Online · this app loads that checkpoint and serves predictions</div>
</div>

<div class="fl-diagram">
<pre class="mermaid">
sequenceDiagram
    actor U as User
    participant G as Gradio UI
    participant I as inference.py
    participant M as Tiny U-Net
    U->>G: upload Sentinel-2 .tif (13 × 512 × 512)
    G->>I: read_image() → normalize to [0, 1]
    I->>M: tensor 1 × 13 × 128 × 128
    M-->>I: logits → sigmoid → P(water) per pixel
    I-->>G: threshold ≥ 0.5 → mask, resized to 512 × 512
    G-->>U: true-color preview + blue overlay + stats
</pre>
<div class="fl-diagram-caption">One prediction, end to end</div>
</div>
"""
SYSTEM_DIAGRAM = SYSTEM_DIAGRAM.replace("<br/>", "&lt;br/&gt;")

EXPLAINER_3 = r"""
<span class="fl-kicker">09 · Limits & roadmap</span>
## What this is not

FloodLens is an **implementation-and-evaluation portfolio project** — not a production
flood service and not a new foundation model. Honest limits: a small subset (102 chips),
optical-only inputs (heavy cloud can hide water), a 15-chip test set (metrics are
indicative, not definitive), and precision 0.70 (some over-painting). Roadmap: Sentinel-1
radar inputs that see through cloud, GPU-scale training, threshold calibration, and a
NASA/IBM **Prithvi** reference comparison.
"""

FOOTER = """
<div class="fl-footer">
  <span><b><span style="color:#15803d">Flood</span>Lens</b> · flood-water segmentation on Sentinel-2</span>
  <span>
    <a href="https://github.com/arpan-s-dev/floodlens">GitHub</a> ·
    <a href="https://github.com/cloudtostreet/Sen1Floods11">Sen1Floods11 dataset</a> ·
    Built by <a href="https://github.com/arpan-s-dev">Arpanjeet Singh</a>
  </span>
</div>
"""

try:
    import gradio as gr

    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Space Grotesk"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Geist Mono"), "monospace"],
    )

    with gr.Blocks(title="FloodLens — flood mapping from space", theme=theme,
                   css=CSS, head=HEAD) as demo:
        gr.HTML(HERO)
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.File(label="Sentinel-2 GeoTIFF", type="filepath",
                              file_types=[".tif", ".tiff"])
                method = gr.Radio(["Tiny U-Net", "Heuristic baseline"], value="Tiny U-Net",
                                  label="Model")
                btn = gr.Button("Predict water mask ↗", variant="primary")
                if EXAMPLE_FILES:
                    gr.Examples(examples=[[f] for f in EXAMPLE_FILES], inputs=inp,
                                label="Example chips")
            with gr.Column(scale=2):
                with gr.Row():
                    out_rgb = gr.Image(label="Satellite · true color")
                    out_overlay = gr.Image(label="Predicted water · blue")
                out_info = gr.HTML(
                    "<div class='fl-stats fl-card'><div class='stat'>"
                    "<span class='stat-label'>STATUS</span>"
                    "<span class='stat-value'>Awaiting input — upload a Sentinel-2 chip and hit predict.</span>"
                    "</div></div>"
                )
        btn.click(predict, inputs=[inp, method], outputs=[out_rgb, out_overlay, out_info])

        with gr.Column(elem_classes=["fl-section"]):
            gr.Markdown(EXPLAINER_1, latex_delimiters=LATEX)
            gr.HTML(UNET_DIAGRAM)
            gr.Markdown(EXPLAINER_2, latex_delimiters=LATEX)
            gr.Markdown(SYSTEM_HEADER)
            gr.HTML(SYSTEM_DIAGRAM)
            gr.Markdown(EXPLAINER_3, latex_delimiters=LATEX)
        gr.HTML(FOOTER)

    if __name__ == "__main__":
        demo.launch()
except ImportError:
    if __name__ == "__main__":
        print("gradio not installed. `pip install gradio` to run the UI.")
