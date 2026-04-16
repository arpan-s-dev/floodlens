"""Space Disaster Mapper ML utilities."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = ["__version__", "predict_image", "segment_image", "predict"]


def _to_numpy(image: Any):
    import numpy as np

    return np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)


def predict_image(image: Any):
    """Return a PIL mask for API integration using the deterministic heuristic baseline."""

    from PIL import Image

    from .inference import heuristic_predict

    result = heuristic_predict(_to_numpy(image), disaster="flood")
    return Image.fromarray((result.mask > 0).astype("uint8") * 255, mode="L")


def segment_image(image: Any):
    return predict_image(image)


def predict(image: Any):
    return predict_image(image)
