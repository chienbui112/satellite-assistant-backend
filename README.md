# Remote-Sensing AI Assistant Backend

FastAPI backend for a dual-audience remote-sensing chat assistant.
It pairs a swappable LLM provider with native tool calling against STAC
(Sentinel-2 L2A via Element 84 Earth Search) and a commercial aggregator
(`Maxar`, `Planet`, `AxelGlobe` via `api.geohub.vn`).

The backend is stateless per request and keeps conversation history in the
chat request payload rather than server memory.

## Key Features

- `POST /api/chat`: streaming chat endpoint using Server-Sent Events
- Tool-enabled LLM workflow for:
  - geocoding (`geocode_location`)
  - satellite imagery search (`get_sentinel_scenes` / `search_satellite_imagery`)
  - UI actions (`clear_roi`, `clear_results`, `focus_location`)
- Unified satellite search API:
  - `provider=sentinel` → STAC Sentinel-2 L2A
  - `provider=maxar|planet|axelglobe` → commercial aggregator
- Direct STAC search endpoint for frontend bypass: `GET /api/scenes/search`
- Dual-audience prompts for `expert` and `beginner` modes
- Geometry is never exposed to the LLM; the model sees summaries only

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

1. Create or activate the Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and update credentials.

```powershell
copy .env.example .env
```

3. Set `LLM_PROVIDER` and provider-specific env vars in `.env`.

- `LLM_PROVIDER=ollama` uses local Ollama.
- `LLM_PROVIDER=google` uses Gemini via `GOOGLE_API_KEY`.
- `LLM_PROVIDER=openai` uses `OPENAI_API_KEY`.
- `LLM_PROVIDER=anthropic` uses `ANTHROPIC_API_KEY`.
- `LLM_PROVIDER=groq` uses `GROQ_API_KEY`.

For Gemini, `GOOGLE_MODEL` should normally be set to `gemini-2.5-flash`.

## Run

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Health check

```powershell
curl http://localhost:8000/healthz
```

## OpenAPI docs

Open in the browser:

```
http://localhost:8000/docs
```

## API Endpoints

- `POST /api/chat` — streaming chat powered by the LLM tool loop
- `POST /api/clear-history` — reset token metrics for a new conversation
- `GET /api/search-satellite` — direct satellite search frontend endpoint
- `GET /api/scenes/search` — direct STAC scene search endpoint
- `GET /healthz` — service health and settings summary

## Notes

- There is no test suite, linter, or formatter configured.
- The backend is stateless; history and context travel in the request body.
- The `search-satellite` endpoint is direct frontend ↔ backend and bypasses the LLM.
- `SceneSearchArgs.bbox` uses `List[float]` because LLM tool schema validators reject tuple-style JSON schema.
- `CHAT_HISTORY_WINDOW_SIZE` controls how many past user-anchored turns are sent to the LLM.
- `EXTERNAL_PROVIDER_URL=` empty forces mock mode for external commercial search.

## CORS

By default the app allows origins from `http://localhost:5173` plus a few common local hosts.
Adjust `CORS_ORIGINS` in `.env` as needed.

## Useful files

- `app/main.py` — FastAPI application entrypoint
- `app/config.py` — environment-based settings and provider validation
- `app/routers/chat.py` — SSE chat router
- `app/routers/history.py` — clear-history endpoint
- `app/routers/satellite.py` — unified satellite search router
- `app/routers/scenes.py` — direct STAC search router
- `app/services/llm_service.py` — tool-enabled LLM orchestration
- `app/services/stac_service.py` — STAC search wrapper
- `app/services/external_provider_service.py` — commercial aggregator wrapper
- `app/models/schemas.py` — Pydantic models and validation rules
- `app/prompts/system_prompts.py` — expert/beginner prompt text

## README bằng tiếng Việt

### Tổng quan

Dịch vụ backend FastAPI cho trợ lý chat địa không gian, sử dụng LLM và tìm kiếm ảnh vệ tinh.
Nó hoạt động không trạng thái trên mỗi yêu cầu; lịch sử hội thoại được gửi theo payload thay vì lưu trên máy chủ.

### Tính năng chính

- `POST /api/chat`: hội thoại streaming với Server-Sent Events
- Hỗ trợ tìm kiếm ảnh vệ tinh và định vị địa lý qua tool call
- `provider=sentinel` dùng STAC Sentinel-2 L2A
- `provider=maxar|planet|axelglobe` dùng bộ tổng hợp thương mại
- `GET /api/scenes/search` cho tìm kiếm STAC trực tiếp
- Prompt dành cho hai chế độ `expert` và `beginner`

### Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Chỉnh `LLM_PROVIDER` và các khóa API phù hợp trong `.env`.

### Khởi chạy

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Kiểm tra sức khỏe

```powershell
curl http://localhost:8000/healthz
```

### Chú ý

- Không có bộ test, linter hoặc formatter mặc định.
- Khi dùng Gemini, đặt `LLM_PROVIDER=google` và `GOOGLE_API_KEY`.
- `CHAT_HISTORY_WINDOW_SIZE` điều khiển số lượt trò chuyện gửi cho LLM.
- `search-satellite` là endpoint trực tiếp, không đi qua LLM.
