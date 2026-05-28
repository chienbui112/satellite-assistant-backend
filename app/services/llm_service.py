"""
LLM orchestration for the dual-audience assistant.

Flow per chat turn:
  1. Build a message list = [system(mode), ...history, user(message)].
  2. If the request carries a bbox drawn on the map, inject it as a system
     note BEFORE the user message so the model treats it as authoritative
     ROI context (and won't re-ask for coordinates).
  3. Ask the model with `bind_tools([search_satellite_imagery])`. Qwen-2.5 and
     Llama-3.1+ Instruct both support native tool calling via Ollama.
  4. If the model emits a tool call, execute it against the STAC service,
     append a ToolMessage with the JSON result, and ask the model once more
     to produce the user-facing natural-language reply.
  5. Return reply + the structured scene list so the frontend can draw it.

We deliberately keep this as a hand-rolled tiny loop instead of LangGraph;
it's two steps and easier to reason about.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from app.config import Settings
from app.models.schemas import (
    BBox,
    ChatMessage,
    ChatRole,
    ClearResultsArgs,
    ClearROIArgs,
    COMMERCIAL_PROVIDERS,
    FocusLocationArgs,
    GeocodeArgs,
    GeoJSONGeometry,
    Provider,
    Scene,
    SceneSearchArgs,
    SceneSearchArgsLLM,
    SearchPagination,
    SearchParams,
    SearchResultsPayload,
    SortOrder,
    TokenMetrics,
    ToolCallTrace,
    UIAction,
    UIActionParams,
    UICommand,
    UserMode,
    scene_to_lite,
)
from app.prompts.system_prompts import get_system_prompt
from app.services.external_provider_service import (
    ExternalProviderError,
    ExternalProviderService,
)
from app.services.geocode_service import GeocodeError, GeocodeService
from app.services.llm_factory import build_chat_model
from app.services.stac_service import STACService, STACSearchError
from app.services.tokenizer_service import Tokenizer

# Per-turn cap on the LLM tool-dispatch loop.
MAX_TOOL_ROUNDS = 5

# The four providers a single search_satellite_imagery call fans out across.
ALL_PROVIDERS: List[Provider] = [
    Provider.SENTINEL,
    Provider.MAXAR,
    Provider.PLANET,
    Provider.AXELGLOBE,
]

# Bilingual fallback when the LLM produces no text alongside its search tool
# call. Stable + short so the frontend can show "we're searching" instantly.
INITIAL_CONFIRMATION = (
    "Đang tìm kiếm ảnh trên cả 4 hệ thống vệ tinh "
    "(Sentinel-2, Maxar, Planet, AxelGlobe)… / "
    "Searching all four providers in parallel…"
)

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self,
        settings: Settings,
        stac: STACService,
        geocode: GeocodeService,
        tokenizer: Tokenizer,
        external: ExternalProviderService,
    ):
        self._settings = settings
        self._stac = stac
        self._geocode = geocode
        self._tokenizer = tokenizer
        self._external = external
        self._llm = build_chat_model(settings)
        # All tools are declared via @tool decorators and bound here. Bodies
        # are never invoked directly — we dispatch by name in the chat loop.
        self._tools = self._build_tools()
        self._llm_with_tools = self._llm.bind_tools(self._tools)

    # ---------- tool factories ----------

    def _build_tools(self):
        @tool("geocode_location", args_schema=GeocodeArgs)
        def geocode_location(location_name):
            """Resolve a place name into a bbox + center point.

            Call this FIRST when the user mentions a city/region/country by
            name (e.g. "Hanoi", "Hà Nội", "Tokyo", "Da Nang") and you do not
            already have a bbox for it from the map context. Returns:
              {name, center: [lat, lon],
               bbox: [min_lon, min_lat, max_lon, max_lat],
               has_polygon: true | false}.

            On success the backend automatically:
              - paints the administrative polygon (when has_polygon is true,
                otherwise the bbox rectangle) on the user's map via a
                SET_SEARCH_AREA ui_action,
              - remembers the polygon server-side for the upcoming search.

            Just pass the returned `bbox` to `search_satellite_imagery`. Do
            NOT try to construct or pass a polygon — you don't have the
            coordinates and inventing them will break the tool call.
            """
            raise RuntimeError("tool body should not be invoked directly")

        @tool("search_satellite_imagery", args_schema=SceneSearchArgsLLM)
        def search_satellite_imagery(
            bbox,
            datetime_from=None,
            datetime_to=None,
            max_cloud_cover=None,
            limit=10,
            image_type=None,
            gsd=None,
        ):
            """Search satellite imagery across ALL FOUR providers in parallel.

            One call here triggers a parallel fan-out: Sentinel-2 (free ESA),
            Maxar (premium sub-meter), Planet (high cadence), AxelGlobe
            (high cadence). The backend streams per-provider results to the
            frontend as they arrive, and gives you back a single aggregated
            text summary covering all four.

            Use this whenever the user asks to find, list, browse, or look at
            satellite imagery for a region/time. Always pass a bbox in
            [min_lon, min_lat, max_lon, max_lat]. Dates are ISO-8601
            (YYYY-MM-DD). max_cloud_cover is a percent (0-100). limit is the
            per-provider page size (default 10).

            NOTE: If geocode_location was called earlier this turn (or the
            user drew a polygon), the backend AUTOMATICALLY uses the precise
            polygon as the spatial filter — you only pass the bbox.
            """
            raise RuntimeError("tool body should not be invoked directly")

        @tool("clear_roi", args_schema=ClearROIArgs)
        def clear_roi(reason):
            """Clear the drawn bounding box / ROI from the map.

            Call when the user asks to remove, clear, delete the drawn area
            or ROI. Vietnamese examples: "xóa vùng vẽ", "bỏ vùng chọn",
            "xóa hộp". English: "clear the box", "remove the ROI".
            """
            raise RuntimeError("tool body should not be invoked directly")

        @tool("clear_results", args_schema=ClearResultsArgs)
        def clear_results(reason):
            """Clear the current satellite scene list and footprints.

            Call when the user asks to wipe results, hide scenes, or start
            fresh. Vietnamese examples: "xóa kết quả tìm kiếm", "ẩn ảnh",
            "xóa hết ảnh". English: "clear results", "remove footprints".
            """
            raise RuntimeError("tool body should not be invoked directly")

        @tool("focus_location", args_schema=FocusLocationArgs)
        def focus_location(location_name, center, zoom=10):
            """Pan and zoom the map to a named geographic location.

            Call when the user asks to fly / focus / pan / go to a place.
            YOU must supply approximate [lat, lon] from your own knowledge.
            Vietnamese examples: "bay tới Hồ Chí Minh", "focus vào Hà Nội",
            "đi đến Đà Nẵng". English: "fly to Tokyo", "focus on Singapore".
            Use zoom=10 for cities, zoom=6 for countries.
            """
            raise RuntimeError("tool body should not be invoked directly")

        return [
            geocode_location,
            search_satellite_imagery,
            clear_roi,
            clear_results,
            focus_location,
        ]

    # ---------- public entry: streaming ----------

    async def stream_chat(
        self,
        message: str,
        mode: UserMode,
        history: List[ChatMessage],
        bbox: BBox | None,
        geometry: GeoJSONGeometry | None = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run one chat turn and yield typed events as work completes.

        Event types yielded (in roughly this order):
          - "chat_start"           : work has begun (data: {}).
          - "chat_message"         : assistant text — initial confirmation,
                                     and again at the end with the final
                                     consolidated summary.
          - "tool_call_trace"      : one per tool invocation (debug aid).
          - "ui_action"            : one per UI command from clear_*/focus_*.
          - "provider_update"      : ONE per provider as the parallel fan-out
                                     completes (Sentinel + 3 commercial).
          - "token_metrics"        : final token-bar snapshot.
          - "updated_history"      : full conversation history for the
                                     frontend to replace its local copy.
          - "done"                 : terminator (data: {}). Always sent.
        """
        yield _event("chat_start", {})

        # Derive an envelope bbox from the polygon if the caller only gave us
        # a geometry — the LLM tool requires a bbox arg, and MAP CONTEXT text
        # is bbox-shaped. The polygon itself is the source of truth for the
        # actual spatial filter and is auto-injected at search dispatch time.
        effective_bbox = bbox
        if geometry is not None and effective_bbox is None:
            effective_bbox = _envelope_of_geometry(geometry)

        messages: List[BaseMessage] = [SystemMessage(content=get_system_prompt(mode))]
        if effective_bbox is not None:
            if geometry is not None:
                messages.append(
                    SystemMessage(
                        content=(
                            "MAP CONTEXT: the user has drawn a polygon AOI on the map. "
                            f"Its bounding envelope is bbox={list(effective_bbox)} "
                            "(EPSG:4326). Pass this bbox to search_satellite_imagery — "
                            "the backend already has the full polygon and will use it "
                            "as the precise spatial filter (more accurate than the envelope). "
                            "Do NOT call geocode_location for any place name; the AOI is set."
                        )
                    )
                )
            else:
                messages.append(
                    SystemMessage(
                        content=(
                            "MAP CONTEXT: the user has drawn a bounding box on the map. "
                            f"Use this ROI for any spatial search: bbox={list(effective_bbox)} "
                            "(EPSG:4326, [min_lon, min_lat, max_lon, max_lat])."
                        )
                    )
                )

        window = self._settings.chat_history_window_size
        trimmed = _trim_history(history, window)
        if len(trimmed) < len(history):
            logger.info(
                "history trimmed: %d -> %d messages (window=%s turns)",
                len(history), len(trimmed), window,
            )
        messages.extend(_history_to_lc(trimmed))
        messages.append(HumanMessage(content=message))

        ai_msg: AIMessage = await self._llm_with_tools.ainvoke(messages)
        messages.append(ai_msg)

        # Per-turn state shared across tool calls — keeps the geocoded
        # polygon out of the LLM context (it's huge) while letting search
        # auto-inject it server-side. Reset every turn.
        #
        # Seed `pending_geometry` with the user-drawn polygon (if any) so
        # search_satellite_imagery auto-injects it without the LLM having
        # to know the coordinates. A subsequent geocode_location in the
        # same turn would override this — that matches user intent (naming
        # a place AFTER drawing means "search there instead").
        turn_state: Dict[str, Any] = {
            "pending_geometry": geometry.model_dump() if geometry is not None else None,
        }

        rounds = 0
        while ai_msg.tool_calls and rounds < MAX_TOOL_ROUNDS:
            rounds += 1
            for call in ai_msg.tool_calls:
                async for ev in self._dispatch_tool_call(call, messages, turn_state):
                    yield ev

            ai_msg = await self._llm_with_tools.ainvoke(messages)
            messages.append(ai_msg)

        if rounds >= MAX_TOOL_ROUNDS and ai_msg.tool_calls:
            logger.warning(
                "Reached MAX_TOOL_ROUNDS=%d without a final reply", MAX_TOOL_ROUNDS
            )

        # Flush any deferred SET_SEARCH_AREA that wasn't consumed by a search.
        # geocode_location buffers its ui_action in turn_state instead of
        # yielding immediately, so multi-place turns ("So sánh Hà Nội với
        # TP.HCM") coalesce into ONE map update, even when the LLM spreads
        # the geocode calls across separate tool-rounds.
        pending = turn_state.get("pending_search_area")
        if pending is not None:
            yield _event("ui_action", pending)
            turn_state["pending_search_area"] = None

        # Final assistant reply — consolidated summary if a search ran.
        final_text = _text_of(ai_msg)
        if final_text:
            yield _event("chat_message", {
                "role": "assistant",
                "content": final_text,
                "stage": "final",
            })

        token_metrics = self._compute_token_metrics(messages)
        yield _event("token_metrics", token_metrics.model_dump())

        updated_history = _lc_to_history(messages[1 + (1 if effective_bbox else 0):])
        yield _event(
            "updated_history",
            [m.model_dump() for m in updated_history],
        )
        yield _event("done", {})

    # ---------- tool dispatch (event-emitting) ----------

    async def _dispatch_tool_call(
        self,
        call: Dict[str, Any],
        messages: List[BaseMessage],
        turn_state: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        name = call.get("name")
        args = call.get("args") or {}
        tool_call_id = call.get("id") or name

        yield _event("tool_call_trace", {"name": name, "arguments": args})

        if name == "geocode_location":
            geocode_result = await self._execute_geocode(args)
            # Per prompt 19: paint the resolved area on the user's map BEFORE
            # the satellite search starts. The frontend's SET_SEARCH_AREA
            # handler clears any user-drawn rectangle and flies the map to
            # fit the polygon/bbox. If geocoding errored, geocode_result has
            # `error` and we skip — no map mutation on failure.
            #
            # Dedupe across the whole turn: instead of yielding immediately,
            # stash the ui_action in turn_state. It gets flushed either right
            # before the next search runs (so the user sees the polygon
            # before results stream in) or at the very end of the turn (if
            # no search follows). This collapses multi-place turns
            # ("So sánh Hà Nội với TP.HCM") into ONE map update — last
            # geocode wins — even when the LLM spreads the geocode calls
            # across separate tool-rounds.
            if isinstance(geocode_result, dict) and "error" not in geocode_result:
                turn_state["pending_search_area"] = {
                    "command": UICommand.SET_SEARCH_AREA.value,
                    "params": {
                        "location_name": geocode_result.get("name"),
                        "center": geocode_result.get("center"),
                        "bbox": geocode_result.get("bbox"),
                        "geometry": geocode_result.get("geometry"),
                    },
                }
                # Stash the polygon for the upcoming search call.
                # The LLM never sees it (the full geometry can be 1000s of
                # tokens for an admin boundary like Hà Nội — that alone
                # bursts Groq's 6000 TPM free-tier ceiling).
                turn_state["pending_geometry"] = geocode_result.get("geometry")

            # LLM-facing payload — strip the heavy geometry; just tell the
            # model that one is available so it doesn't try to invent
            # coords or skip the search.
            result_payload = {
                **{k: v for k, v in (geocode_result or {}).items() if k != "geometry"},
                "has_polygon": bool(
                    isinstance(geocode_result, dict)
                    and geocode_result.get("geometry")
                ),
            }

        elif name == "search_satellite_imagery":
            # Flush any deferred SET_SEARCH_AREA from prior geocode calls in
            # this turn before results start streaming. The user sees the
            # polygon appear → then the "Searching..." confirmation → then
            # provider_update rows. If the LLM emitted multiple geocodes
            # earlier (same round or across rounds), only the LAST one's
            # area is flushed here — no flicker A → B.
            pending = turn_state.get("pending_search_area")
            if pending is not None:
                yield _event("ui_action", pending)
                turn_state["pending_search_area"] = None

            # An immediate confirmation message into the chat so the user
            # sees instant feedback while the four providers spin up.
            yield _event("chat_message", {
                "role": "assistant",
                "content": INITIAL_CONFIRMATION,
                "stage": "confirmation",
            })

            # ALWAYS discard any `geometry` the LLM emitted — it never sees
            # the actual polygon (stripped from geocode_location's result),
            # so any value it sends is either omitted (good) or hallucinated
            # (bad: smaller models like Llama 3.3 70B fabricate degenerate
            # polygons that break Groq's tool-call parser). Then re-inject
            # the real polygon from turn_state when we have one.
            args = {k: v for k, v in args.items() if k != "geometry"}
            if turn_state.get("pending_geometry"):
                args["geometry"] = turn_state["pending_geometry"]

            # Fan out: each provider_update event streams as soon as its
            # provider finishes (in arrival order, not declaration order).
            aggregate_lines: List[str] = []
            try:
                parsed = SceneSearchArgs.model_validate(args)
            except Exception as e:
                logger.warning("Invalid search args from LLM: %s", e)
                result_payload = {"error": f"invalid arguments: {e}"}
            else:
                yield _event("parameters_extracted", {
                    "bbox": list(parsed.bbox),
                    "date_start": parsed.datetime_from,
                    "date_end": parsed.datetime_to,
                    "max_cloud": parsed.max_cloud_cover,
                    "limit": parsed.limit,
                    # New in prompt 17: surface what the LLM detected so the
                    # UI can pre-select the Image Type tab + GSD dropdown.
                    "image_type": parsed.image_type or "optical",
                    "gsd": parsed.gsd or "Very-high",
                    # New in prompt 19: the LLM may pass an admin polygon
                    # received from geocode_location. The frontend already
                    # painted it via SET_SEARCH_AREA; including it here
                    # keeps `search_params` round-trippable for Load More.
                    "geometry": parsed.geometry.model_dump() if parsed.geometry else None,
                })

                async for ev, line in self._fan_out_providers(parsed):
                    yield ev
                    if line:
                        aggregate_lines.append(line)

                result_payload = {
                    "summary": (
                        "Aggregate of parallel search across four providers:\n"
                        + "\n".join(aggregate_lines)
                        + "\n(Detailed scene rows have been streamed to the "
                        "frontend; do NOT enumerate them in the reply.)"
                    ),
                }

        elif name in {"clear_roi", "clear_results", "focus_location"}:
            result_payload, action = self._execute_ui_action(name, args)
            if action is not None:
                yield _event("ui_action", action.model_dump())

        else:
            result_payload = {"error": f"unknown tool: {name}"}

        messages.append(
            ToolMessage(
                content=json.dumps(result_payload, default=str),
                tool_call_id=tool_call_id,
            )
        )

    async def _fan_out_providers(
        self,
        parsed: SceneSearchArgs,
    ) -> AsyncIterator[Tuple[Dict[str, Any], str]]:
        """Kick off all four searches concurrently; yield events + one
        labelled summary line per provider as each one completes.

        The line is what the LLM sees in the aggregate tool result for its
        final consolidated reply. Pattern:
          "Sentinel-2: total=12 page=10 cloud_range=3-18%"
        """
        bbox_tuple: Tuple[float, float, float, float] = tuple(parsed.bbox)  # type: ignore[assignment]
        common = {
            "date_start": parsed.datetime_from,
            "date_end":   parsed.datetime_to,
            "max_cloud":  parsed.max_cloud_cover,
            "limit":      parsed.limit,
        }

        queue: asyncio.Queue = asyncio.Queue()
        tasks = [
            asyncio.create_task(
                self._run_one_provider(p, bbox_tuple, parsed, queue),
                name=f"search:{p.value}",
            )
            for p in ALL_PROVIDERS
        ]

        completed = 0
        try:
            while completed < len(ALL_PROVIDERS):
                update = await queue.get()
                completed += 1
                provider_name: str = update["provider"]

                if "error" in update:
                    yield (
                        _event("provider_update", {
                            "provider": provider_name,
                            "error": update["error"],
                            "search_params": common | {"bbox": list(bbox_tuple)},
                        }),
                        f"{provider_name}: ERROR ({update['error'][:80]})",
                    )
                    continue

                payload: SearchResultsPayload = update["payload"]
                # Build the LLM-facing one-line summary now that we know the
                # numbers — no geometry, no scene IDs.
                clouds = [r.cloud_cover for r in payload.results if r.cloud_cover is not None]
                cloud_blurb = ""
                if clouds:
                    cloud_blurb = f" cloud_range={min(clouds):.0f}-{max(clouds):.0f}%"
                line = (
                    f"{provider_name}: total={payload.pagination.total_records} "
                    f"page={len(payload.results)}{cloud_blurb}"
                )

                yield (
                    _event("provider_update", {
                        "provider": provider_name,
                        "results":     [r.model_dump() for r in payload.results],
                        "geometries":  [g.model_dump() for g in payload.geometries],
                        "pagination":  payload.pagination.model_dump(),
                        "search_params": payload.search_params.model_dump() if payload.search_params else None,
                        "total_records": payload.pagination.total_records,
                    }),
                    line,
                )
        finally:
            # Cancel any still-running tasks if the consumer aborted.
            for t in tasks:
                if not t.done():
                    t.cancel()

    async def _run_one_provider(
        self,
        provider: Provider,
        bbox: Tuple[float, float, float, float],
        parsed: SceneSearchArgs,
        queue: asyncio.Queue,
    ) -> None:
        """Run one provider's search; always pushes one item to the queue."""
        try:
            geometry_dict = (
                parsed.geometry.model_dump() if parsed.geometry else None
            )
            if provider == Provider.SENTINEL:
                scenes, matched = await self._stac.search_scenes(
                    bbox=bbox,
                    datetime_from=parsed.datetime_from,
                    datetime_to=parsed.datetime_to,
                    max_cloud_cover=parsed.max_cloud_cover,
                    limit=parsed.limit,
                    page=1,
                    sort_by=SortOrder.NEWEST,
                    geometry=geometry_dict,
                )
            elif provider in COMMERCIAL_PROVIDERS:
                cloud_max = float(parsed.max_cloud_cover) if parsed.max_cloud_cover is not None else 100.0
                effective_gsd = parsed.gsd or "Very-high"
                if provider == Provider.AXELGLOBE:
                    effective_gsd = None  # AxelGlobe doesn't accept GSD filtering
                scenes, matched = await self._external.search(
                    provider=provider,
                    bbox=bbox,
                    date_start=parsed.datetime_from,
                    date_end=parsed.datetime_to,
                    cloud_range=(0.0, cloud_max),
                    page=1,
                    limit=parsed.limit,
                    image_type=parsed.image_type or "optical",
                    gsd=effective_gsd,
                    geometry=geometry_dict,
                )
            else:
                raise RuntimeError(f"unsupported provider {provider}")

            payload = SearchResultsPayload(
                results=[scene_to_lite(s) for s in scenes],
                geometries=[s.geometry for s in scenes],
                pagination=SearchPagination(
                    total_records=matched,
                    current_page=1,
                    limit=parsed.limit,
                ),
                search_params=SearchParams(
                    bbox=list(bbox),
                    # Echo the polygon actually used (user-drawn or geocoded)
                    # so the frontend's Load More re-issues the SAME spatial
                    # filter — not the bbox envelope, which would widen the
                    # search and return extra scenes on later pages.
                    geometry=parsed.geometry,
                    date_start=parsed.datetime_from,
                    date_end=parsed.datetime_to,
                    max_cloud=parsed.max_cloud_cover,
                    sort_by=SortOrder.NEWEST,
                    image_type=parsed.image_type or "optical",
                    # Echo back NO gsd for AxelGlobe so the frontend's stored
                    # state matches the disabled-dropdown UX.
                    gsd=(parsed.gsd or "Very-high") if provider != Provider.AXELGLOBE else None,
                ),
            )
            await queue.put({"provider": provider.value, "payload": payload})
        except (STACSearchError, ExternalProviderError) as e:
            await queue.put({"provider": provider.value, "error": str(e)})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error in %s search", provider.value)
            await queue.put({"provider": provider.value, "error": repr(e)})

    def _compute_token_metrics(self, messages: List[BaseMessage]) -> TokenMetrics:
        """Estimate the token surface of *this* turn for the UI progress bar.

        We count the messages we actually sent to the LLM (system prompts +
        trimmed history + current user message + tool/assistant messages from
        the multi-round loop). Tool *schemas* (bound via bind_tools) are
        additional fixed overhead the provider injects; we approximate that
        as +800 tokens for our current 5 tools so the bar reflects what the
        provider counts toward its TPM limit.
        """
        FIXED_TOOL_SCHEMA_OVERHEAD = 800
        total = self._tokenizer.count_many(_text_of(m) for m in messages)
        total += FIXED_TOOL_SCHEMA_OVERHEAD
        return TokenMetrics(
            current_tokens=total,
            max_tokens=self._settings.max_tokens,
            warning_threshold=self._settings.warning_threshold,
            method=self._tokenizer.method,
        )

    # ---------- tool dispatch ----------

    async def _execute_geocode(self, args: dict):
        try:
            parsed = GeocodeArgs.model_validate(args)
        except Exception as e:
            logger.warning("Invalid geocode args from LLM: %s", e)
            return {"error": f"invalid arguments: {e}"}

        try:
            result = await self._geocode.geocode(parsed.location_name)
        except GeocodeError as e:
            return {"error": str(e)}

        # Already in the shape the model needs to consume next turn.
        return result

    def _execute_ui_action(self, name: str, args: dict):
        """Validate UI tool args, return (tool_result_payload, UIAction|None)."""
        try:
            if name == "clear_roi":
                ClearROIArgs.model_validate(args)
                action = UIAction(
                    command=UICommand.CLEAR_ROI,
                    params=UIActionParams(reason=args.get("reason")),
                )
            elif name == "clear_results":
                ClearResultsArgs.model_validate(args)
                action = UIAction(
                    command=UICommand.CLEAR_RESULTS,
                    params=UIActionParams(reason=args.get("reason")),
                )
            elif name == "focus_location":
                parsed = FocusLocationArgs.model_validate(args)
                action = UIAction(
                    command=UICommand.FOCUS_LOCATION,
                    params=UIActionParams(
                        center=parsed.center,
                        location_name=parsed.location_name,
                        zoom=parsed.zoom,
                    ),
                )
            else:
                return {"error": f"unhandled ui tool: {name}"}, None
        except Exception as e:
            logger.warning("Invalid UI tool args for %s: %s", name, e)
            return {"error": f"invalid arguments: {e}"}, None

        return {"ok": True, "command": action.command.value}, action


# ---------- streaming helper ----------

def _event(event_type: str, data: Any) -> Dict[str, Any]:
    """Wrap a (type, data) pair into the dict that the route turns into an
    SSE frame. Centralising the shape keeps the router thin and lets future
    consumers (e.g. a websocket route) reuse the same emitter."""
    return {"event": event_type, "data": data}


# ---------- token-saving helpers ----------

def _format_scene_summary(
    scenes: List[Scene],
    matched: Optional[int] = None,
    page: int = 1,
    limit: Optional[int] = None,
) -> dict:
    """Build a tight text summary of STAC results for the LLM.

    The LLM does NOT need scene IDs, geometry, or bboxes — those go to the
    frontend via the response payload.

    CRITICAL labelling: the LLM has historically confused "items on this
    page" with "total matched". We now hand it three explicitly-labelled
    numbers (the exact labels the system prompt teaches it to quote back):

      - Total matched in database
      - Displaying on current page
      - Page number

    Plus a structured copy of the same fields so a future LLM that prefers
    JSON keys can use them too.
    """
    n_page = len(scenes)
    matched_val = matched if matched is not None else n_page
    page_limit = limit if (limit and limit > 0) else (n_page or 1)
    # ceil(matched / limit)
    total_pages = max(1, (matched_val + page_limit - 1) // page_limit) if matched_val else 1

    header = (
        f"Total matched in database: {matched_val}\n"
        f"Displaying on current page: {n_page}\n"
        f"Page number: {page} of {total_pages}\n"
    )

    if n_page == 0:
        body = (
            "No scenes match this query. Suggest the user widen the cloud "
            "cover threshold, extend the date range, or pick a different area."
        )
        return {
            "total_matched_in_database": matched_val,
            "displaying_on_current_page": 0,
            "page_number": f"{page} of {total_pages}",
            "summary": header + body,
        }

    pairs = []
    clouds: List[float] = []
    for s in scenes:
        d = s.datetime.date().isoformat()
        if s.cloud_cover is None:
            pairs.append(f"{d} (cloud n/a)")
        else:
            pairs.append(f"{d} ({s.cloud_cover:.0f}%)")
            clouds.append(s.cloud_cover)

    stats_line = ""
    if clouds:
        clouds_sorted = sorted(clouds)
        median = clouds_sorted[len(clouds_sorted) // 2]
        stats_line = (
            f"Cloud range on this page: {min(clouds):.0f}–{max(clouds):.0f}%, "
            f"median {median:.0f}%.\n"
        )

    pagination_hint = ""
    if matched_val > n_page:
        pagination_hint = (
            "Remaining matches can be paged through in the right-rail panel — "
            "the LLM does NOT need to fetch them.\n"
        )
    elif matched_val == n_page:
        pagination_hint = "All matches are on this page; pagination is not needed.\n"

    body = (
        f"Scene dates and cloud cover: {', '.join(pairs)}.\n"
        f"{stats_line}"
        f"{pagination_hint}"
        "Full scene metadata (IDs, footprints, thumbnails) is delivered to "
        "the frontend separately."
    )

    return {
        "total_matched_in_database": matched_val,
        "displaying_on_current_page": n_page,
        "page_number": f"{page} of {total_pages}",
        "summary": header + body,
    }


def _envelope_of_geometry(geom: GeoJSONGeometry) -> Tuple[float, float, float, float]:
    """Return [min_lon, min_lat, max_lon, max_lat] for a Polygon/MultiPolygon.

    Used only for the LLM-facing MAP CONTEXT text — we never want the LLM
    seeing raw polygon coords (admin boundaries can be 1000s of vertices,
    bursting the TPM ceiling). The backend keeps the actual polygon and
    routes it to the search call server-side.
    """
    coords: Any = geom.coordinates
    # Walk down nested lists until we reach a [lon, lat] pair.
    while coords and isinstance(coords[0], list) and (
        not coords[0] or isinstance(coords[0][0], list)
    ):
        coords = coords[0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))


def _trim_history(
    history: List[ChatMessage],
    k_turns: Optional[int],
) -> List[ChatMessage]:
    """Keep only the last k user-anchored turns of past history.

    Semantics, driven by `CHAT_HISTORY_WINDOW_SIZE` env var:
      - k_turns is None  → unlimited; pass the full history through.
      - k_turns is 0     → drop all past turns.
      - k_turns > 0      → keep the last k turns (each turn = one user
                           message + everything that follows it until the
                           next user message).

    Anchoring the trim at user messages is important: tool messages must
    follow their matching tool_call assistant message, otherwise OpenAI /
    Groq / Anthropic 400-reject the request. Cutting at user boundaries
    keeps every assistant/tool pair intact.

    Why a plain list slice instead of LangChain's
    ConversationBufferWindowMemory(k=...): our backend is stateless per
    request — history arrives in the API payload — so a memory object
    would just wrap a list, badly. The user explicitly chose this
    formulation in prompt5.md.
    """
    if k_turns is None:
        return history
    if k_turns <= 0 or not history:
        return []
    user_idx = [i for i, m in enumerate(history) if m.role == ChatRole.USER]
    if len(user_idx) <= k_turns:
        return history
    return history[user_idx[-k_turns]:]


# ---------- history conversion ----------

def _history_to_lc(history: List[ChatMessage]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in history:
        if m.role == ChatRole.USER:
            out.append(HumanMessage(content=m.content))
        elif m.role == ChatRole.ASSISTANT:
            out.append(AIMessage(content=m.content, tool_calls=m.tool_calls or []))
        elif m.role == ChatRole.SYSTEM:
            out.append(SystemMessage(content=m.content))
        elif m.role == ChatRole.TOOL and m.tool_call_id:
            out.append(ToolMessage(content=m.content, tool_call_id=m.tool_call_id))
    return out


def _lc_to_history(messages: List[BaseMessage]) -> List[ChatMessage]:
    out: List[ChatMessage] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(ChatMessage(role=ChatRole.USER, content=_text_of(m)))
        elif isinstance(m, AIMessage):
            out.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=_text_of(m),
                    tool_calls=getattr(m, "tool_calls", None) or None,
                )
            )
        elif isinstance(m, ToolMessage):
            out.append(
                ChatMessage(
                    role=ChatRole.TOOL,
                    content=_text_of(m),
                    tool_call_id=m.tool_call_id,
                )
            )
        elif isinstance(m, SystemMessage):
            out.append(ChatMessage(role=ChatRole.SYSTEM, content=_text_of(m)))
    return out


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # Some providers return content as a list of blocks.
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)
