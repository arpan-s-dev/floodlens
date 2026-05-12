from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    service: str = Field(..., example="space-disaster-mapper-api")
    predictor: str = Field(..., example="fallback-heuristic")


class SampleMetadata(BaseModel):
    id: str
    title: str
    description: str
    width: int
    height: int
    modality: str = "synthetic-rgb"
    bands: List[str] = Field(default_factory=lambda: ["R", "G", "B"])
    tags: List[str] = Field(default_factory=list)


class PredictionStats(BaseModel):
    mask_pixels: int
    total_pixels: int
    mask_ratio: float
    threshold: Optional[int] = None


class PredictionResponse(BaseModel):
    prediction_id: str
    provider: str
    source_type: str
    sample_id: Optional[str] = None
    filename: Optional[str] = None
    width: int
    height: int
    created_at: str
    mask_output_path: str
    mask_base64: Optional[str] = None
    mask_base64_included: bool
    stats: PredictionStats
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    detail: str
