from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_mapper.datasets import discover_pairs
from space_mapper.io import read_mask
from space_mapper.metrics import ConfusionCounts, confusion_counts, metrics_from_counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate binary segmentation predictions against masks.")
    parser.add_argument("--pred-dir", required=True, help="Directory containing predicted masks.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing ground-truth masks with matching stems.")
    parser.add_argument("--output", help="Optional JSON metrics output path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for prediction masks if they are probabilistic.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = discover_pairs(args.pred_dir, args.mask_dir)
    if not pairs:
        raise SystemExit("No prediction/mask pairs found. Filenames must share stems.")

    totals = ConfusionCounts(0, 0, 0, 0)
    for pair in pairs:
        pred = read_mask(pair.image_path, positive_threshold=args.threshold)
        truth = read_mask(pair.mask_path)
        counts = confusion_counts(truth, pred, threshold=args.threshold)
        totals = ConfusionCounts(
            totals.true_positive + counts.true_positive,
            totals.false_positive + counts.false_positive,
            totals.true_negative + counts.true_negative,
            totals.false_negative + counts.false_negative,
        )

    result = {"num_samples": len(pairs), **metrics_from_counts(totals)}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
