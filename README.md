# Remote-Sensing AI Assistant Backend

FastAPI backend for a dual-audience remote-sensing chat assistant.
It pairs a swapable LLM provider with native tool-calling against STAC
(Sentinel-2 L2A via Element 84 Earth Search) and an external commercial
aggregator (`Maxar`, `Planet`, `AxelGlobe` via `api.geohub.vn`).

The frontend consumes the backend over HTTP and SSE; the backend is
stateless per request and carries conversation history in the request body.

## Key Features

- `POST /api/chat`: streaming chat endpoint using Server-Sent Events
- Tool-enabled LLM workflow with native tool calls for:
  - geocoding (`geocode_location`)
  - satellite imagery search (`search_satellite_imagery`)
  - UI map actions (`clear_roi`, `clear_results`, `focus_location`)
- Unified satellite search API:
  - `provider=sentinel` → STAC Sentinel-2 L2A
  - `provider=maxar|planet|axelglobe` → commercial aggregator
- Dual-audience prompts for `expert` and `beginner` modes
- Strict separation of geometry from LLM context: model sees summaries,
  frontend receives full geometry for map display

## Architecture

```
HTTP → app/routers/*  →  app/services/*  →  external APIs
                     ↑
              app/dependencies.py (singleton services)
              app/config.py        (.env settings)
              app/models/schemas.py (shared Pydantic contract)
              app/prompts/system_prompts.py (expert / beginner)
```

## Setup

1. Create or update Python environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and set your provider and API keys.

```powershell
copy .env.example .env
```

3. Configure `LLM_PROVIDER` and any provider-specific credentials in `.env`.

## Run

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Health check

```powershell
curl http://localhost:8000/healthz
```

### OpenAPI docs

Open in the browser:

```
http://localhost:8000/docs
```

## Important Notes

- There is no test suite, linter, or formatter configured in this repository.
- The active LLM provider is selected by `LLM_PROVIDER` in `.env`.
- If the configured LLM SDK is missing, the app may fail at startup.
- This service is designed for a frontend at `http://localhost:5173` by default,
  but CORS is configurable in settings.

## Endpoints

- `POST /api/chat` — chat streaming endpoint
- `GET /healthz` — service health and settings summary
- `GET /api/search-satellite` — direct satellite scene search

## Notes on spatial filters

- `bbox` format: `[min_lon, min_lat, max_lon, max_lat]`
- `UIAction.params.center` format: `[lat, lon]`
- `SceneSearchArgs.bbox` uses `List[float]` for LLM compatibility

## Useful files

- `app/main.py` — FastAPI application entrypoint
- `app/routers/chat.py` — chat streaming router
- `app/routers/satellite.py` — unified satellite search router
- `app/services/llm_service.py` — tool-enabled LLM orchestration
- `app/services/stac_service.py` — STAC search wrapper
- `app/services/external_provider_service.py` — commercial provider aggregator
- `app/models/schemas.py` — Pydantic request/response models
- `app/prompts/system_prompts.py` — system prompts for expert/beginner modes

## License

No license is included in this repository.
