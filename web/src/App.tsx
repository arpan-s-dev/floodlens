import { useEffect, useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import './App.css';

type Sample = {
  id: string;
  name: string;
  description: string;
  hazard: string;
  location: string;
  date: string;
  imageUrl?: string;
};

type Prediction = {
  label: string;
  confidence: number;
  summary: string;
  imageUrl?: string;
  maskUrl?: string;
  metrics: Record<string, string | number>;
  fallback?: boolean;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

const imagePlaceholder = (title: string, subtitle: string, accent = '#8ce9ff') => {
  const svg = `
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 420">
    <defs>
      <linearGradient id="sky" x1="0" x2="1" y1="0" y2="1">
        <stop offset="0" stop-color="#0b1220"/>
        <stop offset="0.55" stop-color="#18243b"/>
        <stop offset="1" stop-color="#05070d"/>
      </linearGradient>
      <linearGradient id="scan" x1="0" x2="1">
        <stop offset="0" stop-color="${accent}" stop-opacity="0.72"/>
        <stop offset="1" stop-color="#b7a0ff" stop-opacity="0.68"/>
      </linearGradient>
    </defs>
    <rect width="720" height="420" fill="url(#sky)"/>
    <path d="M0 272 C110 212 180 308 276 248 S462 176 720 226 V420 H0Z" fill="#17243a"/>
    <path d="M0 332 C120 294 180 344 306 304 S510 260 720 310 V420 H0Z" fill="#243753"/>
    <g opacity="0.75">
      <circle cx="102" cy="78" r="2" fill="#fff"/>
      <circle cx="628" cy="92" r="2" fill="#fff"/>
      <circle cx="470" cy="54" r="1.6" fill="#fff"/>
      <circle cx="315" cy="118" r="1.4" fill="#fff"/>
    </g>
    <rect x="80" y="78" width="560" height="264" rx="22" fill="none" stroke="url(#scan)" stroke-width="3" stroke-dasharray="10 12"/>
    <rect x="116" y="114" width="180" height="18" rx="9" fill="${accent}" opacity="0.28"/>
    <rect x="116" y="146" width="320" height="14" rx="7" fill="#dbeafe" opacity="0.18"/>
    <text x="116" y="246" fill="#f8fbff" font-family="Arial, sans-serif" font-size="34" font-weight="700">${title}</text>
    <text x="116" y="286" fill="#b8c7dc" font-family="Arial, sans-serif" font-size="18">${subtitle}</text>
  </svg>`;

  return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
};

const mockSamples: Sample[] = [
  {
    id: 'coastal-flooding',
    name: 'Coastal Flooding',
    description: 'Sentinel-style scene showing inundation risk near dense coastal infrastructure.',
    hazard: 'Flood',
    location: 'Bay Delta',
    date: '2026-06-12',
    imageUrl: imagePlaceholder('Coastal Scene', 'Demo satellite sample', '#67e8f9'),
  },
  {
    id: 'wildfire-burn-scar',
    name: 'Wildfire Burn Scar',
    description: 'Multispectral crop of vegetation loss and possible fire perimeter expansion.',
    hazard: 'Wildfire',
    location: 'Sierra Foothills',
    date: '2026-05-21',
    imageUrl: imagePlaceholder('Burn Scar', 'Demo satellite sample', '#fb7185'),
  },
  {
    id: 'cyclone-damage',
    name: 'Cyclone Damage',
    description: 'Before/after inspired footprint for rapid disaster response triage.',
    hazard: 'Storm',
    location: 'Island Corridor',
    date: '2026-04-03',
    imageUrl: imagePlaceholder('Storm Impact', 'Demo satellite sample', '#c4b5fd'),
  },
];

const mockMask = imagePlaceholder('Predicted Mask', 'Highlighted affected regions', '#facc15');

const architectureCards = [
  {
    title: 'Data ingest',
    copy: 'Upload an image or pick a curated sample, then send it to the prediction service.',
  },
  {
    title: 'Model API',
    copy: 'The frontend expects /samples and /predict endpoints and works with JSON or multipart uploads.',
  },
  {
    title: 'Decision view',
    copy: 'Results are summarized as confidence, affected area, response priority, and visual masks.',
  },
];

const toNumber = (value: unknown, fallback = 0) => {
  const numeric = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

const toMetricValue = (value: unknown, fallback: string): string | number =>
  typeof value === 'string' || typeof value === 'number' ? value : fallback;

const asObject = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' ? (value as Record<string, unknown>) : {};

const formatRatio = (value: unknown) => {
  const numeric = toNumber(value, Number.NaN);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : '—';
};

const normalizeSample = (raw: unknown, index: number): Sample | null => {
  if (typeof raw === 'string') {
    return {
      id: raw,
      name: raw.replace(/[-_]/g, ' '),
      description: 'Sample provided by the API.',
      hazard: 'Satellite scene',
      location: 'API catalog',
      date: 'Live',
    };
  }

  if (!raw || typeof raw !== 'object') {
    return null;
  }

  const item = raw as Record<string, unknown>;
  const id = String(item.id ?? item.sample_id ?? item.slug ?? `sample-${index + 1}`);
  const name = String(item.name ?? item.title ?? id.replace(/[-_]/g, ' '));

  return {
    id,
    name,
    description: String(item.description ?? item.summary ?? 'Satellite sample from the API catalog.'),
    hazard: String(item.hazard ?? item.type ?? item.disaster_type ?? 'Disaster'),
    location: String(item.location ?? item.region ?? 'Unknown region'),
    date: String(item.date ?? item.captured_at ?? 'Recent'),
    imageUrl: String(item.imageUrl ?? item.image_url ?? item.thumbnailUrl ?? item.thumbnail_url ?? item.url ?? ''),
  };
};

const normalizePrediction = (payload: unknown): Prediction => {
  const envelope =
    payload && typeof payload === 'object' && 'prediction' in payload
      ? (payload as Record<string, unknown>).prediction
      : payload && typeof payload === 'object' && 'result' in payload
        ? (payload as Record<string, unknown>).result
        : payload;

  const data = envelope && typeof envelope === 'object' ? (envelope as Record<string, unknown>) : {};
  const stats = asObject(data.stats);
  const confidenceSource = data.confidence ?? data.score ?? data.probability ?? stats.mask_ratio;
  const confidence = Math.min(1, Math.max(0, toNumber(confidenceSource, 0.0)));
  const metricsPayload = asObject(data.metrics);
  const apiMetrics = Object.entries(metricsPayload).reduce<Record<string, string | number>>((metrics, [key, value]) => {
    if (typeof value === 'string' || typeof value === 'number') {
      metrics[key] = value;
    }
    return metrics;
  }, {});
  const statsMetrics =
    Object.keys(stats).length > 0
      ? {
          'Mask ratio': formatRatio(stats.mask_ratio),
          'Mask pixels': toMetricValue(stats.mask_pixels, '—'),
          'Total pixels': toMetricValue(stats.total_pixels, '—'),
          ...(stats.threshold !== undefined ? { Threshold: toMetricValue(stats.threshold, '—') } : {}),
        }
      : {};
  const metricsSource =
    Object.keys(apiMetrics).length > 0
      ? apiMetrics
      : Object.keys(statsMetrics).length > 0
        ? statsMetrics
      : {
          'Affected area': toMetricValue(data.damage_area_km2, '18.4 km²'),
          'Pixels flagged': toMetricValue(data.pixels_flagged, '42%'),
          Priority: toMetricValue(data.priority, 'High'),
        };
  const maskBase64 = typeof data.mask_base64 === 'string' ? `data:image/png;base64,${data.mask_base64}` : '';
  const maskUrl = String(data.maskUrl ?? data.mask_url ?? data.overlayUrl ?? data.overlay_url ?? maskBase64);
  const provider = data.provider ? ` via ${String(data.provider)}` : '';

  return {
    label: String(data.label ?? data.class ?? data.disaster_type ?? `Segmentation mask${provider}`),
    confidence,
    summary: String(
      data.summary ??
        data.message ??
        'The API returned a binary mask and pixel statistics for the selected scene.',
    ),
    imageUrl: String(data.imageUrl ?? data.image_url ?? data.source_image_url ?? ''),
    maskUrl,
    metrics: metricsSource,
  };
};

const resolveAssetUrl = (url?: string) => {
  if (!url) {
    return '';
  }

  if (url.startsWith('http') || url.startsWith('data:') || url.startsWith('blob:')) {
    return url;
  }

  return `${API_BASE}${url.startsWith('/') ? url : `/${url}`}`;
};

const makeFallbackPrediction = (sample?: Sample, hasUpload = false): Prediction => ({
  label: hasUpload ? 'Uploaded scene analysis' : `${sample?.hazard ?? 'Disaster'} impact detected`,
  confidence: 0.84,
  summary:
    'Mock result shown because the API is unavailable. Start the backend to replace this card with live predictions.',
  maskUrl: mockMask,
  metrics: {
    'Affected area': '18.4 km²',
    'Pixels flagged': '42%',
    Priority: sample?.hazard === 'Wildfire' ? 'Critical' : 'High',
  },
  fallback: true,
});

function App() {
  const [samples, setSamples] = useState<Sample[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadPreview, setUploadPreview] = useState('');
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [notice, setNotice] = useState('Connecting to the API sample catalog...');

  useEffect(() => {
    const loadSamples = async () => {
      try {
        const response = await fetch(`${API_BASE}/samples`);
        if (!response.ok) {
          throw new Error(`Samples request failed: ${response.status}`);
        }

        const payload = await response.json();
        const payloadObject =
          payload && typeof payload === 'object' ? (payload as Record<string, unknown>) : {};
        const list = Array.isArray(payload)
          ? payload
          : Array.isArray(payloadObject.samples)
            ? payloadObject.samples
            : Array.isArray(payloadObject.data)
              ? payloadObject.data
              : [];
        const normalized = list
          .map((item: unknown, index: number) => normalizeSample(item, index))
          .filter((item: Sample | null): item is Sample => Boolean(item));
        const nextSamples = normalized.length > 0 ? normalized : mockSamples;

        setSamples(nextSamples);
        setSelectedSampleId((current) => current || nextSamples[0]?.id || '');
        setNotice(
          normalized.length > 0
            ? `Loaded ${normalized.length} sample${normalized.length === 1 ? '' : 's'} from ${API_BASE}.`
            : 'API returned no samples, so demo placeholders are shown.',
        );
      } catch {
        setSamples(mockSamples);
        setSelectedSampleId((current) => current || mockSamples[0].id);
        setNotice('API is not running yet. Using polished mock samples for the portfolio demo.');
      } finally {
        setSamplesLoading(false);
      }
    };

    void loadSamples();
  }, []);

  useEffect(() => {
    if (!uploadedFile) {
      setUploadPreview('');
      return;
    }

    const objectUrl = URL.createObjectURL(uploadedFile);
    setUploadPreview(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [uploadedFile]);

  const selectedSample = useMemo(
    () => samples.find((sample) => sample.id === selectedSampleId),
    [samples, selectedSampleId],
  );

  const sourceImage =
    uploadPreview ||
    resolveAssetUrl(prediction?.imageUrl) ||
    resolveAssetUrl(selectedSample?.imageUrl) ||
    imagePlaceholder('Satellite Input', 'Select a sample or upload an image', '#8ce9ff');
  const maskImage = resolveAssetUrl(prediction?.maskUrl) || mockMask;
  const canPredict = Boolean(uploadedFile || selectedSampleId);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    setUploadedFile(nextFile);
    if (nextFile) {
      setPrediction(null);
      setNotice(`Ready to analyze ${nextFile.name}.`);
    }
  };

  const runPrediction = async () => {
    if (!canPredict) {
      return;
    }

    setPredicting(true);
    setNotice('Sending scene to /predict...');

    try {
      const requestInit: RequestInit = {
        method: 'POST',
      };

      if (uploadedFile) {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        requestInit.body = formData;
      } else {
        requestInit.headers = { 'Content-Type': 'application/json' };
        requestInit.body = JSON.stringify({ sample_id: selectedSampleId });
      }

      const response = await fetch(`${API_BASE}/predict`, requestInit);
      if (!response.ok) {
        throw new Error(`Prediction request failed: ${response.status}`);
      }

      const payload = await response.json();
      setPrediction(normalizePrediction(payload));
      setNotice('Live prediction loaded from the API.');
    } catch {
      setPrediction(makeFallbackPrediction(selectedSample, Boolean(uploadedFile)));
      setNotice('Prediction API unavailable. Displaying a mock result so the demo remains screenshot-ready.');
    } finally {
      setPredicting(false);
    }
  };

  return (
    <main className="app-shell">
      <div className="content">
        <section className="hero">
          <div className="hero-card">
            <p className="eyebrow">Orbital AI disaster intelligence</p>
            <h1>
              Space Disaster <span className="gradient-text">Mapper</span>
            </h1>
            <p className="hero-copy">
              A lightweight portfolio frontend for mapping disasters from satellite imagery. Select a
              sample, upload a scene, and visualize the predicted impact mask returned by the API.
            </p>
            <div className="hero-actions">
              <button className="primary-button" onClick={runPrediction} disabled={!canPredict || predicting}>
                {predicting ? 'Analyzing...' : 'Run prediction'}
              </button>
              <a className="secondary-button" href="#prediction-workspace">
                Explore demo
              </a>
            </div>
          </div>

          <aside className="hero-card status-card">
            <div>
              <div className="orbital-badge" aria-hidden="true" />
              <span className="status-label">API base URL</span>
              <p className="api-url">{API_BASE}</p>
            </div>
            <p className="notice">{notice}</p>
          </aside>
        </section>

        <section className="grid metrics-grid" aria-label="Project metrics">
          <div className="metric-card">
            <span className="metric-value">3</span>
            <span className="metric-label">API endpoints expected</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">3+</span>
            <span className="metric-label">Demo disaster samples</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">0</span>
            <span className="metric-label">Heavy UI libraries</span>
          </div>
          <div className="metric-card">
            <span className="metric-value">Live</span>
            <span className="metric-label">Fetch-based integration</span>
          </div>
        </section>

        <section className="grid architecture-grid" aria-label="Architecture overview">
          {architectureCards.map((card) => (
            <article className="architecture-card" key={card.title}>
              <h3>{card.title}</h3>
              <p>{card.copy}</p>
            </article>
          ))}
        </section>

        <section className="workspace" id="prediction-workspace">
          <aside className="panel">
            <div className="section-heading">
              <div>
                <h2>Choose input</h2>
                <p className="muted">Use API samples or upload an image for a multipart prediction.</p>
              </div>
              <span className="pill">{samplesLoading ? 'Loading' : `${samples.length} samples`}</span>
            </div>

            <label className="field">
              API sample
              <select
                className="select"
                value={selectedSampleId}
                onChange={(event) => {
                  setSelectedSampleId(event.target.value);
                  setPrediction(null);
                }}
              >
                {samples.map((sample) => (
                  <option key={sample.id} value={sample.id}>
                    {sample.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="field">
              Upload image
              <input className="file-input" type="file" accept="image/*" onChange={handleFileChange} />
            </label>

            <button className="primary-button" onClick={runPrediction} disabled={!canPredict || predicting}>
              {predicting ? 'Analyzing scene...' : 'Call /predict'}
            </button>

            <div className="sample-list" aria-label="Available samples">
              {samples.map((sample) => (
                <button
                  className={`sample-card ${sample.id === selectedSampleId ? 'active' : ''}`}
                  key={sample.id}
                  onClick={() => {
                    setSelectedSampleId(sample.id);
                    setPrediction(null);
                  }}
                  type="button"
                >
                  <h3>{sample.name}</h3>
                  <p className="muted">{sample.description}</p>
                  <div className="sample-meta">
                    <span>{sample.hazard}</span>
                    <span>{sample.location}</span>
                    <span>{sample.date}</span>
                  </div>
                </button>
              ))}
            </div>
          </aside>

          <section className="panel">
            <div className="section-heading">
              <div>
                <h2>Prediction panel</h2>
                <p className="muted">Image, mask, and response-ready model summary.</p>
              </div>
              <span className="pill">{prediction?.fallback ? 'Mock' : prediction ? 'Live' : 'Ready'}</span>
            </div>

            <div className="prediction-grid">
              <figure className="image-card">
                <img src={sourceImage} alt="Selected satellite input" />
                <figcaption className="image-caption">Source image</figcaption>
              </figure>
              <figure className="image-card">
                <img src={maskImage} alt="Predicted disaster mask" />
                <figcaption className="image-caption">Mask / overlay</figcaption>
              </figure>

              <article className="result-card">
                <div className="result-header">
                  <div>
                    <h3 className="result-title">{prediction?.label ?? 'Awaiting prediction'}</h3>
                    <p className="muted">
                      {prediction?.summary ??
                        'Run a prediction to view disaster classification, confidence, and response metrics.'}
                    </p>
                  </div>
                  <div className="confidence">
                    <span>Confidence</span>
                    <strong>{Math.round((prediction?.confidence ?? 0) * 100)}%</strong>
                  </div>
                </div>

                <div className="result-metrics">
                  {Object.entries(
                    prediction?.metrics ?? {
                      'Affected area': '—',
                      'Pixels flagged': '—',
                      Priority: '—',
                    },
                  ).map(([label, value]) => (
                    <div className="result-metric" key={label}>
                      <span>{label}</span>
                      <strong>{value}</strong>
                    </div>
                  ))}
                </div>
              </article>
            </div>
          </section>
        </section>
      </div>
    </main>
  );
}

export default App;
