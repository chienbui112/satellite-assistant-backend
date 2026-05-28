"""
Unified satellite-search router.

ONE endpoint — `/api/search-satellite` — dispatches to the right data source
based on the `provider` query parameter:

  - provider=sentinel  → free Element 84 STAC (Sentinel-2-L2A).
  - provider=capella   → external commercial aggregator (SAR).
  - provider=planet    → external commercial aggregator (optical).
  - provider=axelglobe → external commercial aggregator (optical, high cadence).

The frontend never has to know which back-end fulfilled the request; the
unified `SearchResultsPayload` shape (`results`, `geometries`, `pagination`,
`search_params`) is identical regardless of provider.

LLM-context discipline: no GeoJSON ever passes through the LLM context.
This endpoint is direct frontend ↔ backend, no LLM involvement.
"""

import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import Settings
from app.dependencies import (
    get_external_provider_service,
    get_settings,
    get_stac_service,
)
from app.models.schemas import (
    BBox,
    COMMERCIAL_PROVIDERS,
    Provider,
    SearchPagination,
    SearchParams,
    SearchResultsPayload,
    SortOrder,
    scene_to_lite,
)
from app.services.external_provider_service import (
    ExternalProviderError,
    ExternalProviderService,
)
from app.services.geometry_utils import simplify_geojson_geometry
from app.services.stac_service import STACSearchError, STACService

router = APIRouter(prefix="/api", tags=["satellite"])

# Defensive cap on STAC pagination depth (see stac_service notes).
MAX_PAGE = 100


def _parse_geometry(geometry: Optional[str]) -> Optional[Dict[str, Any]]:
    """Decode a URL-encoded GeoJSON Polygon/MultiPolygon from the query string.

    GET requests can't carry a body, so the frontend sends the geometry via
    a `geometry=<encodeURIComponent(JSON.stringify(g))>` query param. We
    validate shape strictly here so providers downstream can trust it.
    """
    if not geometry:
        return None
    try:
        decoded = json.loads(geometry)
    except (ValueError, TypeError) as e:
        # Include a truncated snippet of the offending value so the frontend
        # team can see whether they sent `[object Object]`, an unquoted JS
        # object literal, etc. Common bug: passing the geometry through
        # URLSearchParams without JSON.stringify first.
        snippet = geometry[:120] + ("…" if len(geometry) > 120 else "")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "geometry must be a URL-encoded JSON Polygon/MultiPolygon "
                f"(got {snippet!r}; JSON parse error: {e})"
            ),
        )
    if not isinstance(decoded, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"geometry must be a GeoJSON object (got {type(decoded).__name__})",
        )
    gtype = decoded.get("type")
    if gtype not in ("Polygon", "MultiPolygon"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"geometry.type must be Polygon or MultiPolygon, got {gtype!r}",
        )
    if not isinstance(decoded.get("coordinates"), list):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="geometry.coordinates missing or not a list",
        )
    return decoded


def _parse_bbox(bbox: str) -> BBox:
    try:
        parts = [float(p.strip()) for p in bbox.split(",")]
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bbox must be 4 comma-separated floats: min_lon,min_lat,max_lon,max_lat",
        )
    if len(parts) != 4:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bbox must have exactly 4 elements",
        )
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bbox must have min < max for both axes",
        )
    return (min_lon, min_lat, max_lon, max_lat)  # type: ignore[return-value]


@router.get("/search-satellite", response_model=SearchResultsPayload)
async def search_satellite(
    provider: Provider = Query(
        Provider.SENTINEL,
        description="Which catalogue to query: sentinel | capella | planet | axelglobe",
    ),
    bbox: str = Query(
        ..., description="Comma-separated: min_lon,min_lat,max_lon,max_lat",
    ),
    geometry: Optional[str] = Query(
        None,
        description=(
            "Optional URL-encoded GeoJSON Polygon/MultiPolygon. "
            "When present, overrides bbox as the spatial filter."
        ),
    ),
    date_start: Optional[str] = Query(None, description="ISO-8601 date YYYY-MM-DD"),
    date_end: Optional[str] = Query(None, description="ISO-8601 date YYYY-MM-DD"),
    max_cloud: Optional[float] = Query(None, ge=0, le=100),
    sort_by: SortOrder = Query(
        SortOrder.NEWEST,
        description="'newest' (datetime desc) or 'lowest_cloud' (cloud asc)",
    ),
    image_type: Optional[str] = Query(
        "optical", description="optical | thermal | sar",
    ),
    gsd: Optional[str] = Query(
        "Very-high", description="Very-high | High | Medium | Low",
    ),
    page: int = Query(1, ge=1, le=MAX_PAGE),
    # Spec: default 10 across all providers (Sentinel changes 5→10 too).
    limit: int = Query(10, ge=1, le=50),
    stac: STACService = Depends(get_stac_service),
    external: ExternalProviderService = Depends(get_external_provider_service),
    settings: Settings = Depends(get_settings),
) -> SearchResultsPayload:
    bbox_tuple = _parse_bbox(bbox)
    geometry_obj = _parse_geometry(geometry)
    # Defensive simplify — geocode already simplifies at source, but the
    # frontend might pass in a user-drawn polygon (rare) or a polygon from
    # an unrelated source. Cheap no-op when already small.
    if geometry_obj is not None:
        geometry_obj = simplify_geojson_geometry(
            geometry_obj, settings.geometry_simplify_tolerance,
        )

    if provider == Provider.SENTINEL:
        try:
            scenes, matched = await stac.search_scenes(
                bbox=bbox_tuple,
                datetime_from=date_start,
                datetime_to=date_end,
                max_cloud_cover=max_cloud,
                limit=limit,
                page=page,
                sort_by=sort_by,
                geometry=geometry_obj,
            )
        except STACSearchError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
            ) from e

    elif provider in COMMERCIAL_PROVIDERS:
        # The aggregator's `cloud_cover` is a [min, max] range; we map the
        # caller's single `max_cloud` to the upper bound, with 0 as floor.
        cloud_range = (0.0, float(max_cloud) if max_cloud is not None else 100.0)
        # AxelGlobe doesn't accept GSD filtering — strip it for that provider
        # so the request body matches what their API expects. The frontend
        # also disables the dropdown when this tab is active, but this is
        # the defensive backend half of that contract.
        effective_gsd = gsd if provider != Provider.AXELGLOBE else None
        try:
            scenes, matched = await external.search(
                provider=provider,
                bbox=bbox_tuple,
                date_start=date_start,
                date_end=date_end,
                cloud_range=cloud_range,
                page=page,
                limit=limit,
                image_type=image_type or "optical",
                gsd=effective_gsd,
                geometry=geometry_obj,
            )
        except ExternalProviderError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
            ) from e

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported provider: {provider}",
        )

    return SearchResultsPayload(
        results=[scene_to_lite(s) for s in scenes],
        geometries=[s.geometry for s in scenes],
        pagination=SearchPagination(
            total_records=matched,
            current_page=page,
            limit=limit,
        ),
        search_params=SearchParams(
            bbox=list(bbox_tuple),
            date_start=date_start,
            date_end=date_end,
            max_cloud=max_cloud,
            sort_by=sort_by,
            image_type=image_type,
            gsd=gsd if provider != Provider.AXELGLOBE else None,
        ),
    )
