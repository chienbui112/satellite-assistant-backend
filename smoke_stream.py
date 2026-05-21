"""Quick SSE smoke test for /api/chat. Prints each event's type + a digest.

Usage:
    python smoke_stream.py
"""
import io
import json
import sys
import time

# Force UTF-8 so Vietnamese diacritics in chat_message events don't crash
# the print on Windows (default cp1252 can't encode them).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import httpx  # noqa: E402

URL = "http://localhost:8000/api/chat"
BODY = {
    "message": "Tìm ảnh khu vực Đà Nẵng tháng 5 này ít mây",
    "mode": "expert",
    "history": [],
    "bbox": None,
}

def main():
    t0 = time.monotonic()
    received = {"provider_update": 0}
    print(f"POST {URL}")
    with httpx.stream("POST", URL, json=BODY, timeout=120.0) as r:
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.read()[:200]}")
            sys.exit(1)
        event_type = "message"
        data_line = ""
        for line in r.iter_lines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_line += line[5:].strip()
            elif line == "":
                # frame end
                if not data_line:
                    continue
                elapsed = time.monotonic() - t0
                try:
                    data = json.loads(data_line)
                except Exception:
                    data = data_line
                digest = _digest(event_type, data)
                print(f"  +{elapsed:6.2f}s  [{event_type:18s}] {digest}")
                received[event_type] = received.get(event_type, 0) + 1
                data_line = ""
                event_type = "message"

    print()
    print("== summary ==")
    for k, v in received.items():
        print(f"  {k}: {v}")

def _digest(event_type, data):
    if event_type == "provider_update":
        if isinstance(data, dict) and data.get("error"):
            return f"{data.get('provider')}: ERROR {data['error'][:80]}"
        return (
            f"{data.get('provider'):10s}"
            f" results={len(data.get('results') or [])}"
            f" total={data.get('total_records')}"
        )
    if event_type == "chat_message":
        return f"stage={data.get('stage','?')} content={(data.get('content') or '')[:80]!r}"
    if event_type == "parameters_extracted":
        return f"bbox={data.get('bbox')} dates={data.get('date_start')}..{data.get('date_end')} cloud={data.get('max_cloud')}"
    if event_type == "tool_call_trace":
        return f"name={data.get('name')} args_keys={list((data.get('arguments') or {}).keys())}"
    if event_type == "ui_action":
        return f"command={data.get('command')}"
    if event_type == "token_metrics":
        return f"current={data.get('current_tokens')}/{data.get('max_tokens')}"
    if event_type == "updated_history":
        return f"messages={len(data) if isinstance(data, list) else '?'}"
    if event_type == "done":
        return "(end of stream)"
    if event_type == "error":
        return f"detail={data.get('detail') if isinstance(data, dict) else data}"
    return str(data)[:80]

if __name__ == "__main__":
    main()
