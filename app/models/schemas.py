from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ---------- Enums ----------

class UserMode(str, Enum):
    EXPERT = "expert"
    BEGINNER = "beginner"


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class SortOrder(str, Enum):
    NEWEST = "newest"            # datetime desc
    LOWEST_CLOUD = "lowest_cloud" # eo:cloud_cover asc


class Provider(str, Enum):
    """
    Search backends the unified /api/search-satellite endpoint routes to:
      - SENTINEL : free Element 84 STAC catalogue (Sentinel-2-L2A).
      - MAXAR    : commercial sub-meter optical (WorldView constellation).
      - PLANET   : commercial high-cadence optical (PlanetScope / SkySat).
      - AXELGLOBE: commercial high-cadence optical (Axelspace GRUS).
    The three commercial providers all share one external aggregator API
    (api.geohub.vn); only the `provider_code` differs per request.
    """
    SENTINEL = "sentinel"
    MAXAR = "maxar"
    PLANET = "planet"
    AXELGLOBE = "axelglobe"


# Which commercial providers route through the external aggregator.
COMMERCIAL_PROVIDERS = {Provider.MAXAR, Provider.PLANET, Provider.AXELGLOBE}


# ---------- Geometry ----------

BBox = Tuple[float, float, float, float]  # [min_lon, min_lat, max_lon, max_lat]


class GeoJSONGeometry(BaseModel):
    """Minimal GeoJSON geometry (Polygon / MultiPolygon)."""
    type: str
    coordinates: List[Any]


# ---------- Chat ----------

class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    # Optional tool-call metadata (for assistant turns that invoked a tool)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    mode: UserMode = UserMode.BEGINNER
    history: List[ChatMessage] = Field(default_factory=list)
    # ROI drawn on the map; if present, it is injected into the model context
    bbox: Optional[BBox] = None

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: Optional[BBox]) -> Optional[BBox]:
        if v is None:
            return v
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError("longitude out of range")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError("latitude out of range")
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat] with min < max")
        return v


# ---------- Scene metadata returned to the frontend ----------

class SceneAsset(BaseModel):
    href: str
    type: Optional[str] = None
    title: Optional[str] = None
    roles: List[str] = Field(default_factory=list)


class Scene(BaseModel):
    """A normalised scene from any provider (Sentinel STAC or commercial).

    Used internally; the search endpoints serialise it into a split form
    (SceneLite + parallel geometry list) so the frontend can render the
    list without paying for GeoJSON, and so the LLM never sees the geometry.
    """
    id: str
    provider: "Provider" = Field(default=Provider.SENTINEL)
    collection: str
    datetime: datetime
    cloud_cover: Optional[float] = None
    bbox: BBox
    geometry: GeoJSONGeometry
    thumbnail_url: Optional[str] = None
    # Higher-res image suitable for the preview modal (RGB composite or
    # full-resolution browse image). For Sentinel this is the `visual` asset;
    # for AxelGlobe `tci`; for Maxar `browse`; for Planet `ortho_visual`.
    preview_url: Optional[str] = None
    # Names of the downloadable raster bands (asset keys with role="data").
    # The frontend looks up href via `assets[band_name]` for the actual URL.
    bands: List[str] = Field(default_factory=list)
    # Full raw asset dictionary keyed by asset name → SceneAsset.
    assets: Dict[str, SceneAsset] = Field(default_factory=dict)
    platform: Optional[str] = None
    instrument: Optional[str] = None


class SceneLite(BaseModel):
    """Scene minus geometry — the row format for the right-rail list."""
    id: str
    provider: "Provider" = Field(default=Provider.SENTINEL)
    collection: str
    datetime: datetime
    cloud_cover: Optional[float] = None
    bbox: BBox
    thumbnail_url: Optional[str] = None
    preview_url: Optional[str] = None
    bands: List[str] = Field(default_factory=list)
    assets: Dict[str, SceneAsset] = Field(default_factory=dict)
    platform: Optional[str] = None
    instrument: Optional[str] = None


# ---------- Paginated search payload ----------

class SearchPagination(BaseModel):
    total_records: int
    current_page: int = Field(..., ge=1)
    limit: int = Field(..., ge=1)


class SearchParams(BaseModel):
    """The query the LLM extracted from natural language. The frontend stores
    this after the first turn so it can re-issue paginated requests against
    /api/search-satellite without round-tripping through the LLM."""
    bbox: List[float] = Field(..., min_length=4, max_length=4)
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    max_cloud: Optional[float] = None
    sort_by: SortOrder = SortOrder.NEWEST
    # New in prompt 17: explicit filter-panel fields.
    # image_type: "optical" | "thermal" | "sar". Drives the external
    # aggregator's `image_type` body field; Sentinel ignores it.
    image_type: Optional[str] = "optical"
    # gsd: "Very-high" | "High" | "Medium" | "Low". Maps to the external
    # aggregator's `gsd` body field; ignored for Sentinel and AxelGlobe.
    gsd: Optional[str] = "Very-high"


class SearchResultsPayload(BaseModel):
    """Standardised search response — same shape from both /api/chat
    (embedded under `search_results`) and /api/search-satellite (top level).
    `results[i]` and `geometries[i]` are index-aligned."""
    results: List[SceneLite]
    geometries: List[GeoJSONGeometry]
    pagination: SearchPagination
    # `search_params` is None when the client supplied them (i.e. on direct
    # GET to /api/search-satellite); populated when the LLM extracted them.
    search_params: Optional[SearchParams] = None


def scene_to_lite(scene: Scene) -> SceneLite:
    return SceneLite(
        id=scene.id,
        provider=scene.provider,
        collection=scene.collection,
        datetime=scene.datetime,
        cloud_cover=scene.cloud_cover,
        bbox=scene.bbox,
        thumbnail_url=scene.thumbnail_url,
        preview_url=scene.preview_url,
        bands=scene.bands,
        assets=scene.assets,
        platform=scene.platform,
        instrument=scene.instrument,
    )


# ---------- STAC tool arguments ----------

class SceneSearchArgs(BaseModel):
    """Arguments the LLM emits when it calls get_sentinel_scenes.

    NOTE: `bbox` is typed `List[float]` (not the BBox tuple alias) because
    Gemini's tool-schema validator rejects the `prefixItems` JSON-Schema that
    Pydantic emits for tuples. A uniform `items` array schema with min/max
    length is accepted by OpenAI, Anthropic, Ollama, and Gemini alike.
    """
    bbox: List[float] = Field(
        ...,
        description="[min_lon, min_lat, max_lon, max_lat] in EPSG:4326",
        min_length=4,
        max_length=4,
    )
    datetime_from: Optional[str] = Field(
        None, description="ISO-8601 start, e.g. 2026-05-01"
    )
    datetime_to: Optional[str] = Field(
        None, description="ISO-8601 end, e.g. 2026-05-31"
    )
    max_cloud_cover: Optional[float] = Field(
        None, ge=0, le=100, description="Max cloud cover percent"
    )
    # Items per page. The frontend's Load More walks through the remaining
    # pages via /api/search-satellite. Default 10 matches the right-rail's
    # tab page size across all providers.
    limit: int = Field(10, ge=1, le=50)
    # Optional — the LLM may extract these from user intent ("SAR over X",
    # "Tìm ảnh nhiệt..."). The UI's filter panel uses them to pre-select
    # the right tab/dropdown.
    image_type: Optional[str] = Field(
        None,
        description="optical | thermal | sar (default 'optical')",
    )
    gsd: Optional[str] = Field(
        None,
        description="Ground Sample Distance bucket: Very-high | High | Medium | Low",
    )
    # New in prompt 19: when geocode_location returns a polygon, the LLM
    # should pass the GeoJSON geometry here. STAC then searches with
    # `intersects` (more precise than the bbox envelope), and the external
    # aggregator gets the actual administrative polygon in its `aois` field.
    # bbox still required as a fallback for the polygon's envelope.
    geometry: Optional[GeoJSONGeometry] = Field(
        None,
        description=(
            "Optional GeoJSON Polygon/MultiPolygon returned by geocode_location. "
            "When present, used as the spatial filter; bbox becomes a fallback."
        ),
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 elements")
        min_lon, min_lat, max_lon, max_lat = v
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("bbox must have min < max for both axes")
        return v


# ---------- UI control actions ----------

class UICommand(str, Enum):
    CLEAR_ROI = "CLEAR_ROI"
    CLEAR_RESULTS = "CLEAR_RESULTS"
    FOCUS_LOCATION = "FOCUS_LOCATION"
    # New in prompt 19: emitted right after `geocode_location` so the
    # resolved polygon paints onto the map BEFORE provider results arrive.
    # Both the polygon (preferred) and the bbox (fallback) ride along so
    # the frontend can render whichever it has.
    SET_SEARCH_AREA = "SET_SEARCH_AREA"


class UIActionParams(BaseModel):
    """Params for a UI action. Most fields are command-specific; we keep
    them all in one model so the response schema stays flat for the
    frontend to consume."""
    # FOCUS_LOCATION + SET_SEARCH_AREA
    center: Optional[List[float]] = Field(
        None,
        description="[lat, lon] in EPSG:4326 (Leaflet's order, NOT bbox order)",
        min_length=2,
        max_length=2,
    )
    location_name: Optional[str] = None
    zoom: Optional[int] = Field(None, ge=1, le=20)
    # SET_SEARCH_AREA — both bbox (always) and geometry (when Nominatim
    # returned a polygon for this place).
    bbox: Optional[List[float]] = Field(
        None,
        description="[min_lon, min_lat, max_lon, max_lat] EPSG:4326",
        min_length=4,
        max_length=4,
    )
    geometry: Optional[GeoJSONGeometry] = Field(
        None,
        description="Polygon | MultiPolygon for the resolved administrative area",
    )
    # Shared
    reason: Optional[str] = None


class UIAction(BaseModel):
    command: UICommand
    params: UIActionParams = Field(default_factory=UIActionParams)


# ---------- Tool argument schemas for UI actions ----------

class ClearROIArgs(BaseModel):
    reason: str = Field(
        ..., description="Why the ROI is being cleared (1 short sentence)"
    )


class ClearResultsArgs(BaseModel):
    reason: str = Field(
        ..., description="Why the results are being cleared (1 short sentence)"
    )


class GeocodeArgs(BaseModel):
    """Args the LLM emits when it calls geocode_location."""
    location_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Place name in any language, e.g. 'Hanoi', 'Hà Nội', 'Tokyo'.",
    )


class FocusLocationArgs(BaseModel):
    location_name: str = Field(
        ..., description="Human-readable place name, e.g. 'Hanoi'"
    )
    center: List[float] = Field(
        ...,
        description="Approximate [lat, lon] for the place, EPSG:4326",
        min_length=2,
        max_length=2,
    )
    zoom: int = Field(
        10,
        ge=1,
        le=20,
        description="Leaflet zoom level; ~10 for a city, ~6 for a country",
    )

    @field_validator("center")
    @classmethod
    def validate_center(cls, v: List[float]) -> List[float]:
        lat, lon = v
        if not (-90 <= lat <= 90):
            raise ValueError("latitude out of range")
        if not (-180 <= lon <= 180):
            raise ValueError("longitude out of range")
        return v


# ---------- Chat response ----------

class ToolCallTrace(BaseModel):
    name: str
    arguments: Dict[str, Any]


class TokenMetrics(BaseModel):
    current_tokens: int
    max_tokens: int
    warning_threshold: int
    method: str  # which tokenizer produced the count (for debug/display)


class ChatResponse(BaseModel):
    reply: str
    mode: UserMode
    tool_calls: List[ToolCallTrace] = Field(default_factory=list)
    # First page of the search the LLM triggered this turn (None if no search).
    # The frontend stores `search_results.search_params` and re-issues paged
    # requests against /api/search-satellite without involving the LLM.
    search_results: Optional[SearchResultsPayload] = None
    # UI actions the LLM wants the frontend to execute this turn.
    ui_actions: List[UIAction] = Field(default_factory=list)
    # Token-usage snapshot for the UI progress bar.
    token_metrics: Optional[TokenMetrics] = None
    # Echoed back so the frontend can keep its conversation state in sync.
    updated_history: List[ChatMessage] = Field(default_factory=list)


class ClearHistoryResponse(BaseModel):
    ok: bool = True
    token_metrics: TokenMetrics


# ---------- Direct STAC search endpoint ----------

class SceneSearchResponse(BaseModel):
    count: int
    scenes: List[Scene]
