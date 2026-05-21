"""
Geocoding via OpenStreetMap Nominatim.

Nominatim's free public endpoint is ToS-restricted:
  - One request per second per IP (sequentially, not bursts).
  - HTTP User-Agent must identify the application + a contact.
  - Heavy users should self-host or use a paid mirror.

For chat-driven place lookups we hit it rarely; we still:
  - send a descriptive User-Agent,
  - cache resolved names so "Hanoi" twice in a session is one network call.

Nominatim's `boundingbox` is `[south, north, west, east]` (strings, lat-major).
Our internal `bbox` is `[min_lon, min_lat, max_lon, max_lat]` — we convert.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import httpx

from app.config import Settings

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = (
    "vegastar-rs-assistant/0.1 "
    "(remote-sensing chatbot; contact vanchienvs@vegastar.com.vn)"
)


class GeocodeError(RuntimeError):
    pass


class GeocodeService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=10.0,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Language": "en,vi;q=0.8",
            },
        )

    async def geocode(self, location_name: str) -> dict:
        """Resolve a place name to bbox + center. Cached in-process."""
        name = (location_name or "").strip()
        if not name:
            raise GeocodeError("location_name is empty")

        cached = _cache_get(name.lower())
        if cached is not None:
            return cached

        resp = await self._client.get(
            NOMINATIM_URL,
            params={
                "q": name,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 0,
            },
        )
        if resp.status_code != 200:
            raise GeocodeError(
                f"Nominatim HTTP {resp.status_code}: {resp.text[:120]}"
            )

        results = resp.json()
        if not results:
            raise GeocodeError(f"no match for: {location_name!r}")

        result = _normalise(results[0])
        _cache_put(name.lower(), result)
        return result

    async def aclose(self) -> None:
        await self._client.aclose()


def _normalise(raw: dict) -> dict:
    try:
        # Nominatim: lat/lon are strings; boundingbox is [south, north, west, east]
        lat = float(raw["lat"])
        lon = float(raw["lon"])
        s, n, w, e = (float(x) for x in raw["boundingbox"])
    except (KeyError, ValueError, TypeError) as exc:
        raise GeocodeError(f"unexpected Nominatim payload: {raw!r}") from exc

    return {
        "name": raw.get("display_name") or raw.get("name") or "",
        "center": [lat, lon],
        "bbox": [w, s, e, n],  # to [min_lon, min_lat, max_lon, max_lat]
        "type": raw.get("type"),
        "osm_id": raw.get("osm_id"),
    }


# Tiny module-level cache. functools.lru_cache wants hashable args; we keep a
# dict so async callers can populate it without going through lru_cache's
# sync-only decorator.
_CACHE: dict[str, dict] = {}
_CACHE_MAX = 256


def _cache_get(key: str) -> Optional[dict]:
    return _CACHE.get(key)


def _cache_put(key: str, value: dict) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        # Drop one arbitrary entry — we don't need true LRU semantics here.
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = value
