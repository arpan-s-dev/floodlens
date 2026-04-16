from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:
    raise SystemExit(f"Smoke test requires base requirements. Install with `pip install -e ml`. Missing: {exc}")

from space_mapper.datasets import make_manifest
from space_mapper.inference import heuristic_predict
from space_mapper.io import read_image, save_mask_png
from space_mapper.metrics import binary_metrics


def main() -> int:
    work = ROOT / ".smoke_artifacts"
    if work.exists():
        shutil.rmtree(work)
    images = work / "images"
    masks = work / "masks"
    preds = work / "preds"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    preds.mkdir(parents=True)

    for idx in range(2):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 1] = 120
        image[8:24, 8:24, :] = 25 + idx
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        Image.fromarray(image, mode="RGB").save(images / f"tile_{idx}.png")
        Image.fromarray(mask, mode="L").save(masks / f"tile_{idx}.png")

    records = make_manifest(images, masks, work / "manifest.csv", train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    assert len(records) == 2

    metrics = []
    for record in records:
        prediction = heuristic_predict(read_image(record.image_path), disaster="flood").mask
        save_mask_png(prediction, preds / f"{record.sample_id}.png")
        truth = np.asarray(Image.open(record.mask_path)) > 0
        metrics.append(binary_metrics(truth, prediction))

    shutil.rmtree(work)
    print(f"Smoke test passed for {len(records)} samples. Example metrics keys: {sorted(metrics[0])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
