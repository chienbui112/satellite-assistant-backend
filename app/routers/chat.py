"""
Streaming chat endpoint.

`POST /api/chat` returns a Server-Sent Events stream (`text/event-stream`).
The body is a normal JSON `ChatRequest`. The frontend reads the stream via
fetch + ReadableStream (EventSource is GET-only, doesn't fit our POST body).

Event types yielded by the underlying LLMService.stream_chat():
  - chat_start
  - chat_message            (initial confirmation, then final summary)
  - tool_call_trace
  - parameters_extracted    (only for search_satellite_imagery)
  - provider_update         (one per provider as the fan-out completes)
  - ui_action
  - token_metrics
  - updated_history
  - done                    (always emitted last)
  - error                   (emitted in place of `done` on a 5xx-equivalent)

SSE frame format:
    event: <type>
    data: <one-line JSON>
    <blank line>
"""

import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.dependencies import get_llm_service
from app.models.schemas import ChatRequest
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def _sse(event_type: str, data) -> bytes:
    """Format one SSE frame. `data` is JSON-serialised onto one line."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    llm: LLMService = Depends(get_llm_service),
) -> StreamingResponse:
    async def event_source():
        try:
            async for ev in llm.stream_chat(
                message=payload.message,
                mode=payload.mode,
                history=payload.history,
                bbox=payload.bbox,
                geometry=payload.geometry,
            ):
                # Bail early if the client closed the connection — both for
                # politeness (don't keep four upstream STAC calls running)
                # and to surface cancellation up to the create_task siblings
                # in _fan_out_providers.
                if await request.is_disconnected():
                    logger.info("client disconnected mid-stream")
                    return
                yield _sse(ev["event"], ev["data"])
        except Exception as e:  # noqa: BLE001
            logger.exception("stream_chat failed")
            yield _sse("error", {"detail": f"LLM pipeline failed: {e}"})
            yield _sse("done", {})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            # Discourage proxies (nginx, CDNs) from buffering the stream.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
