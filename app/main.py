"""
FastAPI entrypoint for the Remote-Sensing AI Assistant backend.

Run locally:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import chat, history, satellite, scenes

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)

app = FastAPI(
    title="Remote-Sensing AI Assistant",
    description=(
        "Dual-audience chat backend (expert + beginner) for Sentinel-2 imagery, "
        "powered by a local Ollama LLM with STAC function-calling."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(history.router)
app.include_router(satellite.router)
app.include_router(scenes.router)


@app.get("/healthz", tags=["health"])
async def healthz():
    window = settings.chat_history_window_size
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider.value,
        "model": settings.active_model,
        "stac_url": settings.stac_api_url,
        "collection": settings.stac_collection,
        "chat_history_window_size": window,
        "history_mode": "unlimited" if window is None else f"last {window} turns",
        "tokenizer_method": settings.tokenizer_method.value,
        "max_tokens": settings.max_tokens,
        "warning_threshold": settings.warning_threshold,
    }
