"""Render a multi-sample comparison montage: RGB | Truth | Heuristic | U-Net.

    ml/.venv/Scripts/python.exe scripts/make_comparison.py \
        --manifest data/manifests/sen1floods11_diverse.csv \
        --out data/results/comparison_diverse.png --n 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from space_mapper.datasets import read_manifest  # noqa: E402


def stretch(b):
    lo, hi = np.percentile(b, 2), np.percentile(b, 98)
    return (np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def load_pred(name, sample_id):
    p = ROOT / f"data/predictions/{name}/{sample_id}.png"
    return (np.asarray(Image.open(p).convert("L")) > 127).astype(np.uint8)


def colorize(mask, rgb):
    out = rgb.copy()
    out[mask == 1] = [0, 255, 255]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/sen1floods11_diverse.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/results/comparison_diverse.png")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    recs = read_manifest(ROOT / args.manifest, split=args.split)

    def water_frac(rec):
        with rasterio.open(rec.mask_path) as s:
            return (s.read(1) == 1).mean()

    # spread of examples: wettest, median, driest-with-some-water
    ranked = sorted(recs, key=water_frac, reverse=True)
    picks = [ranked[0], ranked[len(ranked) // 2], ranked[-1]][: args.n]

    H = W = 512
    gap = 8
    rows = []
    for rec in picks:
        with rasterio.open(rec.image_path) as s:
            img = s.read()
        rgb = np.dstack([stretch(img[3]), stretch(img[2]), stretch(img[1])])
        with rasterio.open(rec.mask_path) as s:
            truth = (s.read(1) == 1).astype(np.uint8)
        heur = load_pred("heuristic", rec.sample_id)
        unet = load_pred("unet_tiny", rec.sample_id)
        panels = [rgb, colorize(truth, rgb), colorize(heur, rgb), colorize(unet, rgb)]
        row = np.full((H, W * 4 + gap * 3, 3), 255, np.uint8)
        for i, p in enumerate(panels):
            row[:, i * (W + gap): i * (W + gap) + W] = p
        rows.append(row)
        print(f"{rec.sample_id}: {water_frac(rec)*100:.1f}% water")

    montage = np.full((H * len(rows) + gap * (len(rows) - 1), rows[0].shape[1], 3), 255, np.uint8)
    for i, row in enumerate(rows):
        montage[i * (H + gap): i * (H + gap) + H] = row

    out = ROOT / args.out
    Image.fromarray(montage).save(out)
    print(f"columns: RGB | Truth | Heuristic | Tiny U-Net")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
