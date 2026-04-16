from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
GEOTIFF_SUFFIXES = {".tif", ".tiff"}


class SpaceMapperError(Exception):
    """Base exception for Space Disaster Mapper ML utilities."""


class GeoTiffSupportError(ImportError):
    """Raised when GeoTIFF input is requested without rasterio installed."""


class ImageReadError(RuntimeError):
    """Raised when an image cannot be read into an array."""


@dataclass(frozen=True)
class RasterData:
    array: np.ndarray
    profile: dict[str, Any] | None = None


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Pillow is required for common image formats. Install with `pip install Pillow`.") from exc
    return Image


def _normalize_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype == np.bool_:
        return array.astype(np.float32)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        return array.astype(np.float32) / float(info.max)
    array = array.astype(np.float32)
    max_value = float(np.nanmax(array)) if array.size else 1.0
    if max_value > 1.5:
        array = array / max_value
    return np.nan_to_num(array, copy=False)


def read_geotiff(path: str | Path, *, normalize: bool = True) -> RasterData:
    """Read a GeoTIFF with rasterio, returning HxW or HxWxC arrays."""

    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise GeoTiffSupportError(
            "GeoTIFF support requires rasterio. Install it with `pip install rasterio`, "
            "or convert the sample to PNG/JPEG for the lightweight local workflow."
        ) from exc

    path = Path(path)
    with rasterio.open(path) as src:
        array = src.read()
        profile = dict(src.profile)

    if array.shape[0] == 1:
        array = array[0]
    else:
        array = np.moveaxis(array, 0, -1)
    if normalize:
        array = _normalize_array(array)
    return RasterData(array=array, profile=profile)


def read_image(path: str | Path, *, normalize: bool = True) -> np.ndarray:
    """Read a supported image into a NumPy array.

    TIFF files are treated as GeoTIFF-capable rasters and require rasterio so
    geospatial and multispectral data is handled intentionally.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ImageReadError(f"Unsupported image suffix `{suffix}` for {path}")
    if suffix in GEOTIFF_SUFFIXES:
        return read_geotiff(path, normalize=normalize).array

    Image = _require_pillow()
    try:
        with Image.open(path) as img:
            array = np.asarray(img.convert("RGB") if img.mode in {"P", "1"} else img).copy()
    except Exception as exc:  # pragma: no cover - PIL-specific details
        raise ImageReadError(f"Failed to read image {path}: {exc}") from exc

    if normalize:
        array = _normalize_array(array)
    return array


def read_mask(path: str | Path, *, positive_threshold: float | None = None) -> np.ndarray:
    """Read a binary segmentation mask as uint8 values in {0, 1}."""

    array = read_image(path, normalize=False)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.bool_:
        return array.astype(np.uint8)

    numeric = array.astype(np.float32)
    if positive_threshold is None:
        positive_threshold = 0.5 if numeric.size and float(np.nanmax(numeric)) <= 1.0 else 0.0
    return (numeric > positive_threshold).astype(np.uint8)


def save_mask_png(mask: np.ndarray, path: str | Path) -> None:
    """Save a binary mask as an 8-bit PNG with values 0 and 255."""

    Image = _require_pillow()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
    Image.fromarray(mask_uint8, mode="L").save(path)


def list_image_files(root: str | Path) -> list[Path]:
    """Return supported image files recursively, sorted for reproducibility."""

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
