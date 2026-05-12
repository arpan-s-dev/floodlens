import base64
import importlib
import io
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageStat, UnidentifiedImageError

API_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIR = API_ROOT / "runtime" / "predictions"
MAX_BASE64_BYTES = 256 * 1024
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@dataclass
class PredictionResult:
    provider: str
    mask: Image.Image
    stats: Dict[str, Any]
    metadata: Dict[str, Any]


class MLSpaceMapperAdapter:
    """Optional adapter for a future ml.space_mapper package.

    The API must run without ml/ being present. This adapter imports lazily and
    falls back safely when the package or an expected callable is unavailable.
    """

    provider_name = "ml.space_mapper"

    def __init__(self) -> None:
        self._module: Optional[Any] = None
        self._load_error: Optional[str] = None
        errors: list[str] = []
        for module_name in ("space_mapper", "ml.space_mapper"):
            try:
                self._module = importlib.import_module(module_name)
                self.provider_name = module_name
                return
            except Exception as exc:  # pragma: no cover - depends on optional package
                errors.append(f"{module_name}: {exc}")
        self._load_error = "; ".join(errors)

    @property
    def available(self) -> bool:
        return self._module is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def predict(self, image: Image.Image) -> Optional[PredictionResult]:
        if self._module is None:
            return None

        for name in ("predict_image", "segment_image", "predict"):
            callable_obj = getattr(self._module, name, None)
            if callable(callable_obj):
                raw_result = callable_obj(image)
                mask = self._coerce_mask(raw_result)
                if mask is None:
                    continue
                return PredictionResult(
                    provider=self.provider_name,
                    mask=mask,
                    stats=_mask_stats(mask),
                    metadata={"adapter_callable": name},
                )
        return None

    def _coerce_mask(self, raw_result: Any) -> Optional[Image.Image]:
        if isinstance(raw_result, Image.Image):
            return raw_result.convert("L")
        if isinstance(raw_result, dict):
            mask = raw_result.get("mask") or raw_result.get("segmentation_mask")
            if isinstance(mask, Image.Image):
                return mask.convert("L")
        return None


class FallbackHeuristicPredictor:
    provider_name = "fallback-heuristic"

    def predict(self, image: Image.Image) -> PredictionResult:
        rgb = image.convert("RGB")
        gray = rgb.convert("L")
        stat = ImageStat.Stat(gray)
        mean = stat.mean[0]
        stddev = stat.stddev[0]
        threshold = int(max(80, min(235, mean + (0.45 * stddev))))
        mask = gray.point(lambda value: 255 if value >= threshold else 0, mode="L")
        stats = _mask_stats(mask)
        stats["threshold"] = threshold
        return PredictionResult(
            provider=self.provider_name,
            mask=mask,
            stats=stats,
            metadata={
                "heuristic": "grayscale threshold for unusually bright flood/smoke/urban pixels",
                "threshold": threshold,
            },
        )


def active_predictor_name() -> str:
    adapter = MLSpaceMapperAdapter()
    return adapter.provider_name if adapter.available else FallbackHeuristicPredictor.provider_name


def predict_image(image: Image.Image) -> PredictionResult:
    adapter = MLSpaceMapperAdapter()
    if adapter.available:
        try:
            result = adapter.predict(image)
            if result is not None:
                return result
        except Exception as exc:  # pragma: no cover - defensive for future package
            fallback = FallbackHeuristicPredictor().predict(image)
            fallback.metadata["ml_adapter_error"] = str(exc)
            return fallback

    fallback = FallbackHeuristicPredictor().predict(image)
    if adapter.load_error:
        fallback.metadata["ml_adapter_status"] = "unavailable"
    return fallback


def load_image_from_bytes(content: bytes) -> Image.Image:
    if not content:
        raise ValueError("Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file exceeds the 10 MB demo limit.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a supported image.") from exc


def persist_mask(mask: Image.Image, prediction_id: str) -> Dict[str, Any]:
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTION_DIR / f"{prediction_id}_mask.png"
    mask.convert("L").save(output_path, format="PNG")

    raw = output_path.read_bytes()
    include_base64 = len(raw) <= MAX_BASE64_BYTES
    return {
        "path": str(output_path.relative_to(API_ROOT)),
        "base64": base64.b64encode(raw).decode("ascii") if include_base64 else None,
        "base64_included": include_base64,
    }


def new_prediction_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid.uuid4().hex[:10]}"


def _mask_stats(mask: Image.Image) -> Dict[str, Any]:
    binary = mask.convert("L")
    pixels = list(binary.getdata())
    total = len(pixels)
    mask_pixels = sum(1 for value in pixels if value > 0)
    return {
        "mask_pixels": mask_pixels,
        "total_pixels": total,
        "mask_ratio": round(mask_pixels / total, 6) if total else 0.0,
    }
