"""Quick SSE check for prompt 19's polygon path.

Asks the chat to search over Hà Nội; expects to see:
  1. tool_call_trace: geocode_location
  2. ui_action: SET_SEARCH_AREA  ← with a Polygon/MultiPolygon
  3. tool_call_trace: search_satellite_imagery
  4. parameters_extracted with geometry passed through
  5. 4× provider_update
  6. done
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import httpx  # noqa: E402

BODY = {
    "message": "Tìm ảnh Sentinel-2 trên Hà Nội tháng 5 này mây dưới 100%",
    "mode": "expert",
    "history": [],
    "bbox": None,
}

saw = {"SET_SEARCH_AREA": 0, "geocode_geometry_type": None,
       "provider_update": 0, "geometry_in_params_extracted": False}

with httpx.stream("POST", "http://localhost:8000/api/chat",
                  json=BODY, timeout=240.0) as r:
    event_type = "message"
    data_line = ""
    for line in r.iter_lines():
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_line += line[5:].strip()
        elif line == "":
            if not data_line:
                continue
            try:
                data = json.loads(data_line)
            except ValueError:
                data = data_line
            if event_type == "ui_action" and isinstance(data, dict) and \
                    data.get("command") == "SET_SEARCH_AREA":
                saw["SET_SEARCH_AREA"] += 1
                params = data.get("params") or {}
                geom = params.get("geometry")
                gtype = geom.get("type") if isinstance(geom, dict) else None
                saw["geocode_geometry_type"] = gtype
                name = (params.get("location_name") or "")[:60]
                print(f"  SET_SEARCH_AREA: name={name!r} "
                      f"geom.type={gtype} bbox={params.get('bbox')}")
            elif event_type == "provider_update":
                saw["provider_update"] += 1
                print(f"  provider_update: {data.get('provider'):10s} "
                      f"results={len(data.get('results') or [])} "
                      f"total={data.get('total_records')}")
            elif event_type == "parameters_extracted":
                geom = data.get("geometry")
                saw["geometry_in_params_extracted"] = bool(geom)
                gtype = geom.get("type") if isinstance(geom, dict) else None
                print(f"  parameters_extracted: bbox={data.get('bbox')} "
                      f"geom_passed={gtype or 'no'}")
            elif event_type == "tool_call_trace":
                print(f"  tool_call: {data.get('name')}")
            elif event_type == "error":
                print(f"  ERROR: {data}")
            elif event_type == "done":
                print("  done")
            data_line = ""
            event_type = "message"

print()
print("summary:", saw)
