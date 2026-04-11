# Results — Sen1Floods11 flood-water segmentation

**Dataset:** Sen1Floods11 hand-labeled chips, Sentinel-2 optical (13-band), 512×512.
**Source:** Bonafilia et al., *Sen1Floods11*, CVPR 2020 Workshops. Public bucket `gs://sen1floods11/v1.1`.
**Model:** Tiny U-Net (CPU-friendly, ~0.5 MB), trained locally.

## Headline result (diverse 10-country subset)

**102 chips** across 10 countries (Ghana, India, Mekong, Nigeria, Pakistan, Paraguay,
Somalia, Spain, Sri-Lanka, USA). Split **72 train / 15 val / 15 test** (seed 13).
Water prevalence ~9.9%.

| Method | IoU ↑ | Dice/F1 ↑ | Precision | Recall |
|---|---|---|---|---|
| Heuristic (NDWI-like) | 0.108 | 0.195 | 0.110 | 0.858 |
| **Tiny U-Net (BCE+Dice, weighted)** | **0.678** | **0.808** | **0.699** | **0.958** |

The U-Net beats the heuristic baseline by ~6× on IoU. See `comparison_diverse.png`
(columns: RGB · Truth · Heuristic · Tiny U-Net) — the model tracks true water across a
sea/lagoon, a barely-flooded valley, and a cloud-covered chip with no water (it correctly
predicts almost no water there, so it did not just learn "always predict water").

## How we got here (the experiment trail)

| Version | Data | Loss | IoU | What it shows |
|---|---|---|---|---|
| v1 | 30 chips, all Ghana | plain BCE | 0.000 | **Collapsed** — class imbalance (~1.8% water) → predicts no water |
| v2 | 30 chips, all Ghana | weighted BCE + Dice | 0.193 | Imbalance-aware loss stops the collapse |
| **v3** | **102 chips, 10 countries** | weighted BCE + Dice | **0.678** | Diverse, less-imbalanced data → strong result |

**Two changes drove the improvement, each measured:**
1. **Loss:** plain BCE → weighted BCE + soft Dice (`--loss bce_dice --pos-weight auto`).
   Fixed the majority-class collapse (IoU 0.00 → 0.19 on identical data).
2. **Data:** single-event cloudy Ghana → 10-country diverse subset (water 1.8% → 9.9%).
   Gave the model varied, learnable water (IoU 0.19 → 0.68).

## Honest limitations

- Still a **small subset** (102 chips) and **optical-only** (cloud hides water in some scenes).
- Test set is 15 chips — metrics have real variance; treat as indicative, not definitive.
- Precision 0.70 means some over-painting remains.

## Next experiments (planned)

- [ ] Scale up (more chips) + train on GPU (RTX 5050).
- [ ] Sentinel-1 (radar) inputs, which see through cloud — the dataset's primary task.
- [ ] Threshold tuning / test-time calibration to lift precision.
- [ ] Prithvi foundation-model reference comparison.
