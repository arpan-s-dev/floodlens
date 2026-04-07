# Project Status

## Active work split

| Slice | Owner | Status |
| --- | --- | --- |
| Root scaffold | Main agent | Done |
| ML pipeline | `ml-pipeline` agent | Done |
| API service | `api-service` agent | Done |
| Web demo | `web-demo` agent | Done |
| Research docs | `research-docs` agent | Done |
| Integration | Main agent | Done |

## Definition of done for MVP

- ML slice exposes dataset loading, metrics, baseline inference, and documented commands.
- API slice starts with FastAPI and serves `/health`, `/samples`, and `/predict`.
- Web slice runs locally and can call the API or show graceful placeholders.
- Docs explain the research basis, what is original, and how to frame the project honestly.
- Smoke checks pass without downloading large datasets.
