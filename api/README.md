# Space Disaster Mapper API

FastAPI backend for a demo satellite segmentation workflow. It works without the future `ml/` package by using a local image heuristic. If `ml.space_mapper` is later available on `PYTHONPATH`, the adapter in `app/predictor.py` can use it.

## Run on Windows

```powershell
cd "Z:\Projects\Space Project\api"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health` - service status and active predictor.
- `GET /samples` - built-in synthetic sample metadata.
- `POST /predict` - create a demo mask from either JSON/form `sample_id` or an uploaded image file.

JSON sample request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"sample_id":"coastal-flood-demo"}'
```

Multipart upload request:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -F "file=@C:\path\to\satellite-image.png"
```

Prediction masks are written under `runtime\predictions\`, which is ignored by git.
