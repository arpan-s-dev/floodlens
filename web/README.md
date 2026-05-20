# Space Disaster Mapper Web

Lightweight React + TypeScript portfolio frontend for the Space Disaster Mapper demo.

## Run

```powershell
npm install
npm run dev
```

For a production bundle:

```powershell
npm run build
npm run preview
```

The app expects the API at `http://localhost:8000` by default and calls:

- `GET /samples`
- `POST /predict`

Override the API URL with:

```powershell
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

If the API is not running, the UI falls back to mock samples and mock prediction output so screenshots still look complete.
