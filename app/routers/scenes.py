"""
Direct STAC search endpoint - useful for the frontend to call the same
function the LLM would call (e.g. when the expert user wants to bypass chat
and search directly from the map UI).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.dependencies import get_stac_service
from app.models.schemas import BBox, SceneSearchResponse
from app.services.stac_service import STACSearchError, STACService

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


@router.get("/search", response_model=SceneSearchResponse)
async def search_scenes(
    min_lon: float = Query(..., ge=-180, le=180),
    min_lat: float = Query(..., ge=-90, le=90),
    max_lon: float = Query(..., ge=-180, le=180),
    max_lat: float = Query(..., ge=-90, le=90),
    datetime_from: Optional[str] = Query(None, description="ISO-8601 date"),
    datetime_to: Optional[str] = Query(None, description="ISO-8601 date"),
    max_cloud_cover: Optional[float] = Query(None, ge=0, le=100),
    limit: int = Query(10, ge=1, le=50),
    stac: STACService = Depends(get_stac_service),
) -> SceneSearchResponse:
    if min_lon >= max_lon or min_lat >= max_lat:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bbox must have min < max for both axes",
        )

    bbox: BBox = (min_lon, min_lat, max_lon, max_lat)

    try:
        scenes, _matched = await stac.search_scenes(
            bbox=bbox,
            datetime_from=datetime_from,
            datetime_to=datetime_to,
            max_cloud_cover=max_cloud_cover,
            limit=limit,
        )
    except STACSearchError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
        ) from e

    return SceneSearchResponse(count=len(scenes), scenes=scenes)
