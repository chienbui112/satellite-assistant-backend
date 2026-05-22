"""
Thin async-friendly wrapper around `pystac-client` for Element 84 Earth Search.

`pystac-client` is synchronous (it uses `requests` under the hood). We push
its calls to a worker thread via `asyncio.to_thread` so they don't block the
FastAPI event loop, and shape the output into our `Scene` schema so the
frontend gets a stable contract regardless of upstream STAC quirks.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Tuple

from pystac_client import Client
from pystac_client.exceptions import APIError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.models.schemas import (
    BBox,
    GeoJSONGeometry,
    Provider,
    Scene,
    SceneAsset,
    SortOrder,
)


_SORT_FIELDS = {
    SortOrder.NEWEST:       [{"field": "properties.datetime",       "direction": "desc"}],
    SortOrder.LOWEST_CLOUD: [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
}

logger = logging.getLogger(__name__)


# Asset keys we surface explicitly; everything else still goes in `assets`.
_THUMBNAIL_KEYS = ("thumbnail", "preview")
_VISUAL_KEYS = ("visual", "rendered_preview")


class STACSearchError(RuntimeError):
    pass


class STACService:
    def __init__(self, settings: Settings):
        self._settings = settings
        # pystac-client opens an HTTP session lazily; cheap to keep one client.
        self._client: Optional[Client] = None

    def _client_sync(self) -> Client:
        if self._client is None:
            self._client = Client.open(self._settings.stac_api_url)
        return self._client

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(APIError),
    )
    def _search_sync(
        self,
        bbox: BBox,
        datetime_range: Optional[str],
        max_cloud_cover: Optional[float],
        page: int,
        limit: int,
        sort_by: SortOrder,
        geometry: Optional[dict] = None,
    ) -> Tuple[List[Scene], int]:
        """Fetch one page from STAC + the total-matched count.

        Spatial filter rules:
          - `geometry` (Polygon / MultiPolygon, EPSG:4326): preferred when
            present — sent as `intersects` per the STAC spec. More precise
            than the bbox envelope for non-rectangular admin areas.
          - `bbox`: used as a fallback when no polygon is available.

        pystac-client doesn't expose direct offset paging; we fetch up to
        page*limit items (the client follows `next` rel links transparently)
        and slice the last `limit` off the end. Linear in page number, which
        is fine for the realistic case (users rarely scroll past page ~20).
        """
        client = self._client_sync()
        query = {}
        if max_cloud_cover is not None:
            query["eo:cloud_cover"] = {"lt": max_cloud_cover}

        max_items = page * limit
        search_kwargs = {
            "collections": [self._settings.stac_collection],
            "datetime": datetime_range,
            "query": query or None,
            "limit": limit,
            "max_items": max_items,
            "sortby": _SORT_FIELDS[sort_by],
        }
        if geometry:
            # `intersects` and `bbox` are mutually exclusive in STAC; honour
            # the polygon and skip the envelope.
            search_kwargs["intersects"] = geometry
        else:
            search_kwargs["bbox"] = list(bbox)
        search = client.search(**search_kwargs)

        # `matched()` returns the total from the STAC `context` extension —
        # Earth Search v1 includes it. Returns None if the server doesn't,
        # in which case we fall back to len(items_seen) which under-reports
        # for deep paginations but at least gives a non-lying total_pages=1.
        items = list(search.items())
        scenes = [self._item_to_scene(item) for item in items]
        matched = search.matched()
        if matched is None:
            matched = len(scenes) + ((page - 1) * limit)

        start = (page - 1) * limit
        page_scenes = scenes[start:start + limit]
        return page_scenes, matched

    @staticmethod
    def _item_to_scene(item) -> Scene:
        props = item.properties or {}
        assets = item.assets or {}

        thumbnail_url = next(
            (assets[k].href for k in _THUMBNAIL_KEYS if k in assets), None
        )
        visual_url = next(
            (assets[k].href for k in _VISUAL_KEYS if k in assets), None
        )

        # Geometry can be a shapely-style dict or already a mapping.
        geom = item.geometry
        if not isinstance(geom, dict):
            geom = dict(geom)

        bbox = tuple(item.bbox) if item.bbox else _bbox_from_geometry(geom)

        dt = item.datetime or _parse_iso(props.get("datetime"))
        if dt is None:
            raise STACSearchError(f"STAC item {item.id} has no datetime")

        normalised_assets: dict[str, SceneAsset] = {}
        bands: list[str] = []
        for k, a in assets.items():
            roles = list(a.roles) if a.roles else []
            normalised_assets[k] = SceneAsset(
                href=a.href, type=a.media_type, title=a.title, roles=roles,
            )
            # "data" role marks the asset as a raw raster band — Sentinel
            # marks red/green/blue/nir/swir/etc with role="data".
            # We skip the `-jp2` alternates: Earth Search ships every band
            # twice (COG .tif and JP2K) and the JP2 variant duplicates the
            # entry in the Download menu without adding new content.
            if "data" in roles and not k.endswith("-jp2"):
                bands.append(k)

        return Scene(
            id=item.id,
            provider=Provider.SENTINEL,
            collection=item.collection_id or "",
            datetime=dt,
            cloud_cover=props.get("eo:cloud_cover"),
            bbox=bbox,  # type: ignore[arg-type]
            geometry=GeoJSONGeometry(type=geom["type"], coordinates=geom["coordinates"]),
            thumbnail_url=thumbnail_url,
            preview_url=visual_url,  # full-res RGB composite from STAC
            bands=bands,
            assets=normalised_assets,
            platform=props.get("platform"),
            instrument=props.get("instruments", [None])[0] if props.get("instruments") else None,
        )

    async def search_scenes(
        self,
        bbox: BBox,
        datetime_from: Optional[str] = None,
        datetime_to: Optional[str] = None,
        max_cloud_cover: Optional[float] = None,
        limit: Optional[int] = None,
        page: int = 1,
        sort_by: SortOrder = SortOrder.NEWEST,
        geometry: Optional[dict] = None,
    ) -> Tuple[List[Scene], int]:
        """Returns (scenes_for_this_page, total_matched).

        If `geometry` is provided, it's used as the STAC `intersects` filter
        (more precise than the bbox envelope). `bbox` is still required as
        a fallback for the polygon's bounding box.
        """
        if page < 1:
            page = 1
        limit = min(
            limit or self._settings.stac_default_limit,
            self._settings.stac_max_limit,
        )
        datetime_range = _format_datetime_range(datetime_from, datetime_to)

        try:
            return await asyncio.to_thread(
                self._search_sync,
                bbox, datetime_range, max_cloud_cover, page, limit, sort_by,
                geometry,
            )
        except APIError as e:
            logger.exception("STAC API error")
            raise STACSearchError(f"STAC API error: {e}") from e


# ---------- helpers ----------

def _format_datetime_range(
    start: Optional[str], end: Optional[str]
) -> Optional[str]:
    if not start and not end:
        return None
    # STAC accepts "start/end", "start/..", "../end", or a single instant.
    s = start or ".."
    e = end or ".."
    if s == ".." and e == "..":
        return None
    return f"{s}/{e}"


def _parse_iso(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _bbox_from_geometry(geom: dict) -> Tuple[float, float, float, float]:
    coords = geom["coordinates"]
    # Flatten until we reach numeric pairs.
    while coords and isinstance(coords[0], list) and (
        not coords[0] or isinstance(coords[0][0], list)
    ):
        coords = coords[0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (min(lons), min(lats), max(lons), max(lats))
