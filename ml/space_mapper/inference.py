from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .datasets import resize_array
from .io import read_image, save_mask_png


@dataclass(frozen=True)
class PredictionResult:
    mask: np.ndarray
    method: str
    details: str


def _normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image).astype(np.float32)
    if image.size == 0:
        return image
    max_value = float(np.nanmax(image))
    if max_value > 1.5:
        image = image / max_value
    return np.nan_to_num(image, copy=False)


def heuristic_predict(image: np.ndarray, *, disaster: str = "flood", threshold: float | None = None) -> PredictionResult:
    """Create a deterministic heuristic mask for demos without trained weights."""

    image = _normalize(image)
    if image.ndim == 2:
        gray = image
        channels = 1
    else:
        channels = image.shape[-1]
        gray = image.mean(axis=-1)

    disaster = disaster.lower()
    if disaster == "flood" and channels >= 4:
        green = image[..., 1]
        nir = image[..., 3]
        score = (green - nir) / (green + nir + 1e-6)
        cutoff = 0.0 if threshold is None else threshold
        mask = score > cutoff
        details = "NDWI-like green/NIR heuristic for multispectral flood/water detection."
    elif disaster == "flood":
        cutoff = 0.35 if threshold is None else threshold
        mask = gray < cutoff
        details = "Dark-water normalized intensity fallback because no NIR channel was available."
    elif disaster == "wildfire" and channels >= 3:
        red = image[..., 0]
        green = image[..., 1]
        blue = image[..., 2]
        score = red - 0.5 * (green + blue)
        cutoff = float(score.mean() + 0.5 * score.std()) if threshold is None else threshold
        mask = score > cutoff
        details = "Red-dominance heuristic for smoke/fire scar demos; not a validated burn model."
    else:
        cutoff = float(gray.mean()) if threshold is None else threshold
        mask = gray > cutoff
        details = "Generic normalized threshold fallback."

    return PredictionResult(mask=mask.astype(np.uint8), method="heuristic_baseline", details=details)


def _adapt_channels(image: np.ndarray, expected_channels: int) -> np.ndarray:
    if image.ndim == 2:
        image = image[..., None]
    channels = image.shape[-1]
    if channels == expected_channels:
        return image
    if channels > expected_channels:
        return image[..., :expected_channels]
    padding = np.zeros((*image.shape[:2], expected_channels - channels), dtype=image.dtype)
    return np.concatenate([image, padding], axis=-1)


def predict_with_checkpoint(
    image: np.ndarray,
    checkpoint_path: str | Path,
    *,
    resize: int | None = None,
    threshold: float = 0.5,
    device: str = "cpu",
) -> PredictionResult:
    """Run the TinyUNet baseline from a saved checkpoint."""

    from .models import TinyUNet, require_torch

    torch = require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    in_channels = int(checkpoint.get("in_channels", 3))
    base_channels = int(checkpoint.get("base_channels", 16))
    model = TinyUNet(in_channels=in_channels, out_channels=1, base_channels=base_channels).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    original_hw = image.shape[:2]
    image = _adapt_channels(_normalize(image), in_channels)
    if resize is None:
        resize = checkpoint.get("resize")
    if resize:
        image = resize_array(image, int(resize), is_mask=False)
    tensor = torch.from_numpy(np.moveaxis(image, -1, 0)[None, ...].astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    if probs.shape != original_hw:
        probs = resize_array(probs, original_hw, is_mask=False)
    mask = (probs >= threshold).astype(np.uint8)
    return PredictionResult(mask=mask, method="tiny_unet_checkpoint", details=f"Checkpoint inference from {checkpoint_path}")


def predict_file(
    image_path: str | Path,
    output_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    disaster: str = "flood",
    resize: int | None = None,
    threshold: float | None = None,
    device: str = "cpu",
) -> PredictionResult:
    image = read_image(image_path, normalize=True)
    if checkpoint_path:
        result = predict_with_checkpoint(image, checkpoint_path, resize=resize, threshold=0.5 if threshold is None else threshold, device=device)
    else:
        result = heuristic_predict(image, disaster=disaster, threshold=threshold)
    save_mask_png(result.mask, output_path)
    return result
