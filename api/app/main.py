from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from .predictor import active_predictor_name, load_image_from_bytes, new_prediction_id, persist_mask, predict_image
from .samples import build_sample_image, has_sample, list_samples
from .schemas import ErrorResponse, HealthResponse, PredictionResponse, PredictionStats, SampleMetadata

app = FastAPI(
    title="Space Disaster Mapper API",
    version="0.1.0",
    description="Demo API for satellite disaster segmentation predictions.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    payload = ErrorResponse(error="request_error", detail=detail)
    return JSONResponse(status_code=exc.status_code, content=_model_to_dict(payload))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="space-disaster-mapper-api", predictor=active_predictor_name())


@app.get("/samples", response_model=List[SampleMetadata])
def samples() -> List[SampleMetadata]:
    return list_samples()


@app.post("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def predict(request: Request) -> PredictionResponse:
    parsed = await _parse_prediction_request(request)
    sample_id = parsed.get("sample_id")
    filename = parsed.get("filename")
    upload_bytes = parsed.get("upload_bytes")

    if sample_id and upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either sample_id or an uploaded image, not both.",
        )
    if not sample_id and not upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide a known sample_id or upload an image file in the file/image form field.",
        )

    source_type = "sample" if sample_id else "upload"
    if sample_id:
        if not has_sample(sample_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown sample_id '{sample_id}'. Call GET /samples for valid ids.",
            )
        image = build_sample_image(sample_id)
    else:
        try:
            image = load_image_from_bytes(upload_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    prediction_id = new_prediction_id()
    result = predict_image(image)
    persisted = persist_mask(result.mask, prediction_id)

    stats = PredictionStats(**result.stats)
    return PredictionResponse(
        prediction_id=prediction_id,
        provider=result.provider,
        source_type=source_type,
        sample_id=sample_id,
        filename=filename,
        width=image.width,
        height=image.height,
        created_at=datetime.now(timezone.utc).isoformat(),
        mask_output_path=persisted["path"],
        mask_base64=persisted["base64"],
        mask_base64_included=persisted["base64_included"],
        stats=stats,
        metadata=result.metadata,
    )


async def _parse_prediction_request(request: Request) -> Dict[str, Any]:
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body.") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="JSON body must be an object.")
        sample_id = payload.get("sample_id")
        if sample_id is not None and not isinstance(sample_id, str):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="sample_id must be a string.")
        return {"sample_id": sample_id.strip() if sample_id else None}

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        form = await request.form()
        sample_id_value = form.get("sample_id")
        sample_id = str(sample_id_value).strip() if sample_id_value else None
        upload = form.get("file") or form.get("image")
        upload_bytes: Optional[bytes] = None
        filename: Optional[str] = None
        if isinstance(upload, StarletteUploadFile):
            filename = upload.filename
            upload_bytes = await upload.read()
        return {"sample_id": sample_id, "upload_bytes": upload_bytes, "filename": filename}

    sample_id = request.query_params.get("sample_id")
    if sample_id:
        return {"sample_id": sample_id.strip()}

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Use application/json with sample_id, multipart/form-data with file/image, or ?sample_id=.",
    )
