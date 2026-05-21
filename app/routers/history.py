"""
Conversation-history housekeeping endpoints.

The backend itself is stateless (history travels in the chat request payload),
so /api/clear-history is mostly a confirmation: it lets the frontend get an
authoritative "zero" token-metrics shape back, and gives us a single place to
hang server-side housekeeping later (e.g. session logs, billing).
"""

from fastapi import APIRouter, Depends

from app.config import Settings
from app.dependencies import get_settings, get_tokenizer
from app.models.schemas import ClearHistoryResponse, TokenMetrics
from app.services.tokenizer_service import Tokenizer

router = APIRouter(prefix="/api", tags=["history"])


@router.post("/clear-history", response_model=ClearHistoryResponse)
async def clear_history(
    settings: Settings = Depends(get_settings),
    tokenizer: Tokenizer = Depends(get_tokenizer),
) -> ClearHistoryResponse:
    return ClearHistoryResponse(
        ok=True,
        token_metrics=TokenMetrics(
            current_tokens=0,
            max_tokens=settings.max_tokens,
            warning_threshold=settings.warning_threshold,
            method=tokenizer.method,
        ),
    )
