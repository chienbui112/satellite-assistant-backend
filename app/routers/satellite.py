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

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import (
    get_external_provider_service,
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
from app.services.stac_service import STACSearchError, STACService

router = APIRouter(prefix="/api", tags=["satellite"])

# Defensive cap on STAC pagination depth (see stac_service notes).
MAX_PAGE = 100


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
    date_start: Optional[str] = Query(None, description="ISO-8601 date YYYY-MM-DD"),
    date_end: Optional[str] = Query(None, description="ISO-8601 date YYYY-MM-DD"),
    max_cloud: Optional[float] = Query(None, ge=0, le=100),
    sort_by: SortOrder = Query(
        SortOrder.NEWEST,
        description="'newest' (datetime desc) or 'lowest_cloud' (cloud asc)",
    ),
    page: int = Query(1, ge=1, le=MAX_PAGE),
    # Spec: default 10 across all providers (Sentinel changes 5→10 too).
    limit: int = Query(10, ge=1, le=50),
    stac: STACService = Depends(get_stac_service),
    external: ExternalProviderService = Depends(get_external_provider_service),
) -> SearchResultsPayload:
    bbox_tuple = _parse_bbox(bbox)

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
            )
        except STACSearchError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
            ) from e

    elif provider in COMMERCIAL_PROVIDERS:
        # The aggregator's `cloud_cover` is a [min, max] range; we map the
        # caller's single `max_cloud` to the upper bound, with 0 as floor.
        cloud_range = (0.0, float(max_cloud) if max_cloud is not None else 100.0)
        try:
            scenes, matched = await external.search(
                provider=provider,
                bbox=bbox_tuple,
                date_start=date_start,
                date_end=date_end,
                cloud_range=cloud_range,
                page=page,
                limit=limit,
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
        ),
    )
