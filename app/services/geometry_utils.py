"""
Geometry helpers — Douglas-Peucker simplification via shapely.

Why we simplify:
  - Nominatim admin boundaries (Hà Nội, Việt Nam, ...) can have 5000+ vertices.
  - URL query strings have practical limits (~2-8 KB depending on proxy).
  - SSE payloads sent to the frontend balloon when raw polygons ride through.
  - Spatial filters at the satellite-scene scale don't need sub-meter
    precision — 0.001° (≈110 m at the equator) is well below Sentinel-2's
    10 m pixel and AxelGlobe's 2.5 m, so safe for footprint filtering.

shapely.simplify(preserve_topology=True) keeps the polygon valid (no
self-intersections), which matters because STAC's `intersects` and the
aggregator's `aois` both reject invalid geometries.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from shapely.geometry import mapping, shape
    _SHAPELY_AVAILABLE = True
except ImportError:
    _SHAPELY_AVAILABLE = False

logger = logging.getLogger(__name__)


def simplify_geojson_geometry(
    geom: Optional[Dict[str, Any]],
    tolerance: float,
) -> Optional[Dict[str, Any]]:
    """Run Douglas-Peucker on a GeoJSON Polygon/MultiPolygon dict.

    No-op (returns input unchanged) when:
      - geom is None / not a dict / not Polygon|MultiPolygon
      - tolerance <= 0
      - shapely missing, or simplification raises (logged, original returned)
      - simplified geometry is empty (would lose the AOI entirely)
    """
    if not isinstance(geom, dict) or tolerance <= 0:
        return geom
    if geom.get("type") not in ("Polygon", "MultiPolygon"):
        return geom
    if not _SHAPELY_AVAILABLE:
        logger.warning("shapely not available — geometry not simplified")
        return geom

    try:
        sh = shape(geom)
        simplified = sh.simplify(tolerance, preserve_topology=True)
        if simplified.is_empty:
            return geom
        result = mapping(simplified)
    except Exception as e:  # noqa: BLE001
        logger.warning("simplify failed (%s); using original geometry", e)
        return geom

    # shapely's mapping() emits nested tuples; pydantic + json prefer lists.
    coords = _tuples_to_lists(result["coordinates"])
    _log_savings(geom, {"type": result["type"], "coordinates": coords})
    return {"type": result["type"], "coordinates": coords}


def _tuples_to_lists(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)):
        return [_tuples_to_lists(x) for x in obj]
    return obj


def _vertex_count(geom: Dict[str, Any]) -> int:
    """Total vertex count across all rings of a Polygon/MultiPolygon."""
    coords = geom.get("coordinates")

    def walk(x: Any) -> int:
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float)):
            return 1  # leaf [lon, lat] (or [lon, lat, alt])
        if isinstance(x, (list, tuple)):
            return sum(walk(c) for c in x)
        return 0

    return walk(coords)


def _log_savings(original: Dict[str, Any], simplified: Dict[str, Any]) -> None:
    before = _vertex_count(original)
    after = _vertex_count(simplified)
    if before and after < before:
        logger.info(
            "geometry simplified: %d → %d vertices (%.0f%% reduction)",
            before, after, 100.0 * (1.0 - after / before),
        )
