from __future__ import annotations

import argparse
import json
from pathlib import Path

from space_mapper.inference import predict_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one sample mask with a checkpoint or deterministic heuristic baseline.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output PNG mask path.")
    parser.add_argument("--checkpoint", help="Optional TinyUNet checkpoint. If omitted, uses a heuristic baseline.")
    parser.add_argument("--disaster", choices=["flood", "wildfire", "generic"], default="flood")
    parser.add_argument("--resize", type=int, help="Resize side length for checkpoint inference.")
    parser.add_argument("--threshold", type=float, help="Prediction threshold or heuristic cutoff.")
    parser.add_argument("--device", default="cpu", help="Torch device for checkpoint inference.")
    parser.add_argument("--metadata-output", help="Optional JSON metadata path. Defaults to output path with .json suffix.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = predict_file(
            args.image,
            args.output,
            checkpoint_path=args.checkpoint,
            disaster=args.disaster,
            resize=args.resize,
            threshold=args.threshold,
            device=args.device,
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    metadata = {
        "image": args.image,
        "output": args.output,
        "method": result.method,
        "details": result.details,
        "note": "Heuristic outputs are deterministic demo baselines, not validated model performance."
        if result.method == "heuristic_baseline"
        else "Checkpoint output depends on provided weights.",
    }
    metadata_path = Path(args.metadata_output) if args.metadata_output else Path(args.output).with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
