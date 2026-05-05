"""End-to-end: predict heuristic + U-Net over a manifest split, then evaluate both.

Run from the ml/.venv python so space_mapper + torch are importable:
    ml/.venv/Scripts/python.exe scripts/run_pipeline.py --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_mapper.datasets import read_manifest
from space_mapper.inference import predict_file
from space_mapper.io import read_mask
from space_mapper.metrics import ConfusionCounts, confusion_counts, metrics_from_counts

ROOT = Path(__file__).resolve().parents[1]
PRED_ROOT = ROOT / "data/predictions"
RESULTS = ROOT / "data/results"


def evaluate(records, pred_dir: Path, threshold: float = 0.5) -> dict:
    totals = ConfusionCounts(0, 0, 0, 0)
    for rec in records:
        pred_path = pred_dir / f"{rec.sample_id}.png"
        pred = read_mask(pred_path, positive_threshold=threshold)
        truth = read_mask(rec.mask_path)
        c = confusion_counts(truth, pred, threshold=threshold)
        totals = ConfusionCounts(
            totals.true_positive + c.true_positive,
            totals.false_positive + c.false_positive,
            totals.true_negative + c.true_negative,
            totals.false_negative + c.false_negative,
        )
    return {"num_samples": len(records), **metrics_from_counts(totals)}


def run_method(records, name: str, *, checkpoint: Path | None) -> dict:
    pred_dir = PRED_ROOT / name
    pred_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        predict_file(
            rec.image_path,
            pred_dir / f"{rec.sample_id}.png",
            checkpoint_path=str(checkpoint) if checkpoint else None,
            disaster="flood",
        )
    metrics = evaluate(records, pred_dir)
    print(f"\n[{name}] {json.dumps(metrics, indent=2)}")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", default="data/manifests/sen1floods11_tiny.csv")
    ap.add_argument("--checkpoint", default="ml/artifacts/tiny_unet_sen1floods11.pt")
    ap.add_argument("--tag", default="", help="Suffix for the output metrics filename.")
    args = ap.parse_args()

    manifest = ROOT / args.manifest
    ckpt = ROOT / args.checkpoint
    records = read_manifest(manifest, split=args.split)
    if not records:
        raise SystemExit(f"No records for split={args.split}")
    print(f"Evaluating {len(records)} '{args.split}' samples")

    results = {"split": args.split, "methods": {}}
    results["methods"]["heuristic"] = run_method(records, "heuristic", checkpoint=None)
    if ckpt.exists():
        results["methods"]["tiny_unet"] = run_method(records, "unet_tiny", checkpoint=ckpt)
    else:
        print(f"\n(no checkpoint at {ckpt}; skipping U-Net)")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"metrics_{args.split}{args.tag}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
