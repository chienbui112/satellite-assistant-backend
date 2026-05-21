"""
Client for the geohub.vn external imagery aggregator, which fronts:
  - Maxar    (WorldView; sub-meter optical)
  - Planet   (PlanetScope/SkySat; high-cadence optical)
  - AxelGlobe (Axelspace GRUS; high-cadence optical)

POST shape (per spec):
  URL: https://api.geohub.vn/v1/vmrs/api/v1/search?provider_code={p}
  Body: {
    "gsd": "Very-high",
    "date_from": <epoch ms>, "date_to": <epoch ms>,
    "cloud_cover": [0, 100],
    "aois": [{ "type": "Polygon", "coordinates": [[NW, NE, SE, SW, NW]] }],
    "page_number": <int>, "page_size": <int>,
    "provider_code": "maxar" | "planet" | "axelglobe",
    "image_type": "optical"
  }

Response parser is tuned to the three response variants captured in the spec
sample:
  - AxelGlobe: `properties["eo:cloud_cover"]["parsedValue"]`,  `tci` asset is preview,
                band0..band5 marked with `roles=["data"]`.
  - Maxar:     `properties["eo:cloud_cover"]` as a number,    `browse` asset is preview,
                no `data`-roled assets in the listing (ordering required).
  - Planet:    `properties["eo:cloud_cover"]` as a number,    no preview-roled asset;
                `thumbnail` is JPEG.

The actual upstream response is a LIST of items (not the single-item example
in the spec). We accept several common envelope shapes.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import Settings
from app.models.schemas import (
    BBox,
    COMMERCIAL_PROVIDERS,
    GeoJSONGeometry,
    Provider,
    Scene,
    SceneAsset,
)

logger = logging.getLogger(__name__)


class ExternalProviderError(RuntimeError):
    pass


class ExternalProviderService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "vegastar-rs-assistant/0.1",
            }
            key = self._settings.external_provider_api_key
            if key:
                headers["Authorization"] = f"Bearer {key}"
            cookie = self._settings.external_provider_cookie
            if cookie:
                headers["Cookie"] = cookie
            self._client = httpx.AsyncClient(timeout=20.0, headers=headers)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---------------- public ----------------

    async def search(
        self,
        provider: Provider,
        bbox: BBox,
        date_start: Optional[str],
        date_end: Optional[str],
        cloud_range: Tuple[float, float],
        page: int,
        limit: int,
    ) -> Tuple[List[Scene], int]:
        """Returns (scenes, total_matched). Empty result if no URL configured."""
        if provider not in COMMERCIAL_PROVIDERS:
            raise ExternalProviderError(
                f"{provider} is not routed through the external aggregator"
            )

        base_url = (self._settings.external_provider_url or "").strip()
        if not base_url:
            logger.info(
                "EXTERNAL_PROVIDER_URL unset — returning mock empty page for %s",
                provider.value,
            )
            return [], 0

        payload = _build_external_payload(
            provider=provider,
            bbox=bbox,
            date_start=date_start,
            date_end=date_end,
            cloud_range=cloud_range,
            page=page,
            limit=limit,
            fallback_page_size=self._settings.external_provider_page_size,
        )
        # Per spec: provider_code is supplied as BOTH a query param and a
        # body field. Some aggregators inspect one or the other depending on
        # the route; sending both is harmless and matches the contract.
        url = f"{base_url}?provider_code={provider.value}"

        try:
            resp = await self._http().post(url, json=payload)
        except httpx.HTTPError as e:
            raise ExternalProviderError(f"network error talking to {base_url}: {e}") from e

        if resp.status_code >= 400:
            raise ExternalProviderError(
                f"upstream {resp.status_code}: {resp.text[:240]}"
            )

        try:
            data = resp.json()
        except ValueError as e:
            raise ExternalProviderError(
                f"upstream returned non-JSON: {resp.text[:240]}"
            ) from e

        return _parse_external_response(data, provider=provider)


# ---------------- payload construction ----------------


def _bbox_to_polygon_ring(bbox: BBox) -> List[List[List[float]]]:
    """[min_lon, min_lat, max_lon, max_lat] → closed GeoJSON ring (NW→NE→SE→SW→NW)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return [[
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat],
        [min_lon, max_lat],
    ]]


def _to_epoch_ms(date_str: Optional[str], end_of_day: bool = False) -> Optional[int]:
    if not date_str:
        return None
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ExternalProviderError(f"invalid date {date_str!r}: {e}") from e
    if end_of_day:
        d = d.replace(hour=23, minute=59, second=59, microsecond=999_000)
    return int(d.timestamp() * 1000)


def _build_external_payload(
    *,
    provider: Provider,
    bbox: BBox,
    date_start: Optional[str],
    date_end: Optional[str],
    cloud_range: Tuple[float, float],
    page: int,
    limit: int,
    fallback_page_size: int,
) -> Dict[str, Any]:
    """Build the JSON body the geohub aggregator expects."""
    return {
        "gsd": "Very-high",
        "date_from": _to_epoch_ms(date_start, end_of_day=False),
        "date_to":   _to_epoch_ms(date_end,   end_of_day=True),
        "cloud_cover": [float(cloud_range[0]), float(cloud_range[1])],
        "aois": [{
            "type": "Polygon",
            "coordinates": _bbox_to_polygon_ring(bbox),
        }],
        "page_number": page,
        "page_size": limit or fallback_page_size,
        "provider_code": provider.value,
        "image_type": "optical",  # all three commercial vendors are optical
    }


# ---------------- response normalisation ----------------


def _pick_int(*candidates: Any) -> Optional[int]:
    """Return the first candidate that converts to a valid int.

    Important: we DO NOT use a Python `or` chain here because `0` is a
    legitimate total count and would be silently skipped by `or`. We need a
    real "is this defined?" check.
    """
    for c in candidates:
        if c is None:
            continue
        try:
            return int(c)
        except (TypeError, ValueError):
            continue
    return None


def _parse_external_response(
    data: Any,
    *,
    provider: Provider,
) -> Tuple[List[Scene], int]:
    """Normalise the aggregator's response to (scenes, total_matched).

    Accepts both list-style and envelope-style responses:
      - A bare list of items.
      - { items / results / features / data: [...], total / total_records / count: N }
    """
    if isinstance(data, list):
        items_raw = data
        total_candidates: List[Any] = []
    elif isinstance(data, dict):
        items_raw = (
            data.get("items")
            or data.get("results")
            or data.get("features")
            or data.get("data")
            or []
        )
        total_candidates = [
            data.get("total"),
            data.get("total_records"),
            data.get("count"),
            data.get("total_count"),
        ]
    else:
        return [], 0

    # Server-reported total, falling back to the page length only if the
    # upstream didn't provide one. (Falsy-chain would mistreat total=0.)
    total = _pick_int(*total_candidates)
    if total is None:
        total = len(items_raw)

    scenes: List[Scene] = []
    skip_reasons: Dict[str, int] = {}
    first_skip_sample: Optional[Dict[str, Any]] = None
    for raw in items_raw:
        try:
            scenes.append(_item_to_scene(raw, provider=provider))
        except Exception as e:
            reason = type(e).__name__ + ": " + str(e)[:80]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            # Capture a redacted snapshot of the FIRST skipped item to help
            # diagnose unexpected schemas. Only top-level keys (and a peek
            # at `properties` keys) so we don't dump 1000s of coordinates.
            if first_skip_sample is None and isinstance(raw, dict):
                p = raw.get("properties") or {}
                first_skip_sample = {
                    "id": raw.get("id"),
                    "raw.datetime": raw.get("datetime"),
                    "raw.published": raw.get("published"),
                    "raw.updated": raw.get("updated"),
                    "props.datetime": p.get("datetime") if isinstance(p, dict) else None,
                    "props.acquired": p.get("acquired") if isinstance(p, dict) else None,
                    "props.acquisition_time": p.get("acquisition_time") if isinstance(p, dict) else None,
                    "props.collect_time_start": p.get("collect_time_start") if isinstance(p, dict) else None,
                    "props.published": p.get("published") if isinstance(p, dict) else None,
                    "props.updated": p.get("updated") if isinstance(p, dict) else None,
                }

    if skip_reasons:
        breakdown = ", ".join(f"{k} ×{v}" for k, v in skip_reasons.items())
        logger.warning(
            "parsed %d of %d items from %s (skipped %d — %s) first_skip=%r",
            len(scenes), len(items_raw), provider.value,
            sum(skip_reasons.values()), breakdown, first_skip_sample,
        )
    else:
        logger.info(
            "parsed %d of %d items from %s (total_matched=%d)",
            len(scenes), len(items_raw), provider.value, total,
        )

    return scenes, total


def _item_to_scene(raw: Dict[str, Any], *, provider: Provider) -> Scene:
    """Map one raw external item to our normalised Scene."""
    scene_id = raw.get("id") or raw.get("scene_id") or raw.get("uuid")
    if not scene_id:
        raise ValueError("missing id")

    props = raw.get("properties") or {}

    # Planet preview-stage items routinely have `datetime: null` everywhere
    # except `published` / `updated`. Trying a wider list keeps them in the
    # result set (better to render an item with publish time than to drop
    # it silently and confuse the counter).
    dt = _parse_datetime_any(
        raw.get("datetime")
        or props.get("datetime")
        or props.get("acquired")
        or props.get("acquisition_time")
        or props.get("collect_time_start")
        or props.get("published")
        or props.get("updated")
        or raw.get("published")
        or raw.get("updated")
    )
    if dt is None:
        raise ValueError("missing datetime")

    cloud = _parse_cloud_cover(props.get("eo:cloud_cover"))
    if cloud is None:
        cloud = _parse_cloud_cover(raw.get("cloud_cover"))

    # Determine bbox first — we may need it as a geometry fallback below.
    raw_bbox = raw.get("bbox")
    geom_raw = raw.get("geometry")

    geometry: GeoJSONGeometry
    if isinstance(geom_raw, dict) and "coordinates" in geom_raw:
        geometry = GeoJSONGeometry(
            type=geom_raw.get("type", "Polygon"),
            coordinates=geom_raw["coordinates"],
        )
        bbox = raw_bbox or _bbox_from_geometry(geom_raw)
    elif raw_bbox and len(raw_bbox) == 4:
        # Upstream omitted geometry but gave us a bbox — synthesise a
        # rectangular polygon so the item isn't silently dropped. This
        # commonly happens with Maxar's catalogue rows that index data
        # whose footprint isn't yet ingested. Map renders the rectangle,
        # which is honest about precision.
        bbox = tuple(raw_bbox)  # type: ignore[assignment]
        geometry = GeoJSONGeometry(
            type="Polygon",
            coordinates=_bbox_to_polygon_ring(bbox),  # type: ignore[arg-type]
        )
    else:
        raise ValueError("missing geometry and bbox")

    assets_raw = raw.get("assets") or {}
    normalised_assets: Dict[str, SceneAsset] = {}
    bands: List[str] = []
    for key, a in assets_raw.items():
        if not isinstance(a, dict):
            continue
        href = a.get("href")
        if not isinstance(href, str):
            continue
        roles = list(a.get("roles") or [])
        normalised_assets[key] = SceneAsset(
            href=href,
            type=a.get("type"),
            title=a.get("title"),
            roles=roles,
        )
        if "data" in roles:
            bands.append(key)

    thumbnail_url = _asset_href(normalised_assets, ("thumbnail",))
    preview_url = _asset_href(
        normalised_assets,
        # Try by name first, then by role.
        ("tci", "browse", "ortho_visual", "rendered_preview", "visual"),
    ) or _asset_href_by_role(normalised_assets, "visual")

    return Scene(
        id=str(scene_id),
        provider=provider,
        collection=str(raw.get("collection") or provider.value),
        datetime=dt,
        cloud_cover=cloud,
        bbox=tuple(bbox),  # type: ignore[arg-type]
        geometry=geometry,
        thumbnail_url=thumbnail_url,
        preview_url=preview_url,
        bands=bands,
        assets=normalised_assets,
        platform=props.get("platform") or raw.get("platform"),
        instrument=_first_instrument(props.get("instruments")) or props.get("instrument"),
    )


def _parse_cloud_cover(v: Any) -> Optional[float]:
    """AxelGlobe sends `{source:"95.0", parsedValue:95}`; Maxar/Planet send a number."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        pv = v.get("parsedValue", v.get("source"))
        try:
            return float(pv) if pv is not None else None
        except (TypeError, ValueError):
            return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# Match the fractional-seconds group of an ISO-8601 timestamp, but only
# when followed by a timezone indicator or end-of-string — otherwise we'd
# falsely match decimal numbers elsewhere in the string.
_FRAC_RE = re.compile(r"\.(\d+)(?=([+-]\d{2}:?\d{2}|Z|$))")


def _normalise_iso_fractional(s: str) -> str:
    """Make ISO-8601 strings parseable by Python 3.10's fromisoformat.

    3.10 only accepts EXACTLY 3 or 6 fractional digits. Planet's preview
    feed emits 5 digits (e.g. "2026-05-16T03:36:37.59235Z") which would be
    silently rejected, dropping the entire item. We pad/truncate to 6.
    """
    def fix(m: "re.Match[str]") -> str:
        d = m.group(1)
        if len(d) == 3 or len(d) == 6:
            return m.group(0)
        d6 = d[:6] if len(d) > 6 else d.ljust(6, "0")
        return "." + d6
    return _FRAC_RE.sub(fix, s, count=1)


def _parse_datetime_any(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        secs = value / 1000.0 if value > 1e11 else float(value)
        return datetime.fromtimestamp(secs, tz=timezone.utc)
    if isinstance(value, str):
        s = value.replace("Z", "+00:00")
        s = _normalise_iso_fractional(s)
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None
    return None


def _bbox_from_geometry(geom: Dict[str, Any]) -> List[float]:
    coords = geom.get("coordinates")
    while coords and isinstance(coords[0], list) and (
        not coords[0] or isinstance(coords[0][0], list)
    ):
        coords = coords[0]
    if not coords:
        raise ValueError("empty geometry coordinates")
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def _asset_href(assets: Dict[str, SceneAsset], keys: Tuple[str, ...]) -> Optional[str]:
    for k in keys:
        a = assets.get(k)
        if a and a.href:
            return a.href
    return None


def _asset_href_by_role(assets: Dict[str, SceneAsset], role: str) -> Optional[str]:
    for a in assets.values():
        if role in (a.roles or []):
            return a.href
    return None


def _first_instrument(value: Any) -> Optional[str]:
    if isinstance(value, list) and value:
        return str(value[0])
    return None
