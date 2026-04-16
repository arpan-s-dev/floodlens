from __future__ import annotations

import argparse
from collections import Counter

from space_mapper.datasets import make_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a sample image/mask manifest for segmentation.")
    parser.add_argument("--images", required=True, help="Directory containing input images.")
    parser.add_argument("--masks", required=True, help="Directory containing binary masks with matching filename stems.")
    parser.add_argument("--output", required=True, help="Output CSV manifest path.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = make_manifest(
        args.images,
        args.masks,
        args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    counts = Counter(record.split for record in records)
    print(f"Wrote {len(records)} paired samples to {args.output}: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
