import asyncio
import json
import os
from datetime import datetime, timezone

import snowflake.connector
import uvicorn
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

DOCKER_WS_URL = "ws://localhost:3001/ws/sensing"
RECONNECT_DELAY = 3

HR_LOW_THRESHOLD = 50
BR_LOW_THRESHOLD = 8
FALL_BBOX_RATIO = 1.0  # width > height * ratio = fallen

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENTS: set[WebSocket] = set()
EVENT_HISTORY: list[dict] = []

ALERT_EVENTS = {"fall_detected", "low_heart_rate", "low_breathing_rate"}

def _sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )

def _insert_event(event: dict):
    try:
        with _sf_conn() as conn:
            conn.cursor().execute(
                "INSERT INTO events (event, timestamp, confidence, heart_rate_bpm, breathing_rate_bpm) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    event.get("event"),
                    event.get("timestamp"),
                    event.get("confidence"),
                    event.get("heart_rate_bpm"),
                    event.get("breathing_rate_bpm"),
                )
            )
    except Exception as e:
        print(f"Snowflake insert failed: {e}")

def _fetch_events() -> list[dict]:
    try:
        with _sf_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT event, timestamp, confidence, heart_rate_bpm, breathing_rate_bpm "
                "FROM events ORDER BY timestamp DESC LIMIT 100"
            )
            return [
                {
                    "event": r[0],
                    "timestamp": r[1].isoformat() if r[1] else None,
                    "confidence": r[2],
                    "heart_rate_bpm": r[3],
                    "breathing_rate_bpm": r[4],
                }
                for r in cur.fetchall()
            ]
    except Exception as e:
        print(f"Snowflake query failed: {e}")
        return EVENT_HISTORY

# Debounce state
_last_occupied: bool | None = None
_hr_was_low = False
_br_was_low = False
_fall_active = False
_vital_tick = 0

def now():
    return datetime.now(timezone.utc).isoformat()

def ts_from_unix(unix: float) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat()

async def broadcast(message: dict):
    EVENT_HISTORY.append(message)
    if message.get("event") in ALERT_EVENTS:
        await asyncio.to_thread(_insert_event, message)
    for client in list(CLIENTS):
        try:
            await client.send_text(json.dumps(message))
        except Exception:
            CLIENTS.discard(client)

def is_fallen(persons: list) -> tuple[bool, float]:
    for p in persons:
        bbox = p.get("bbox", {})
        w = bbox.get("width", 0)
        h = bbox.get("height", 1)
        if h > 0 and w / h >= FALL_BBOX_RATIO:
            return True, p.get("confidence", 0.9)
    return False, 0.0

async def process_sensing(data: dict):
    global _last_occupied, _hr_was_low, _br_was_low, _fall_active, _vital_tick

    ts = ts_from_unix(data.get("timestamp", 0)) if data.get("timestamp") else now()

    # Presence
    classification = data.get("classification", {})
    occupied = classification.get("presence", False)
    if occupied != _last_occupied:
        _last_occupied = occupied
        await broadcast({"event": "presence_update", "occupied": occupied, "timestamp": ts})

    # Vitals — subsample to ~1 per second (Docker ticks at 10Hz)
    _vital_tick += 1
    vitals = data.get("vital_signs", {})
    hr = vitals.get("heart_rate_bpm")
    br = vitals.get("breathing_rate_bpm")

    if hr is not None and br is not None and _vital_tick % 10 == 0:
        await broadcast({
            "event": "vital_signs",
            "heart_rate_bpm": round(hr, 1),
            "breathing_rate_bpm": round(br, 1),
            "timestamp": ts,
        })

        # Low heart rate alert (only on transition into low state)
        if hr < HR_LOW_THRESHOLD and not _hr_was_low:
            _hr_was_low = True
            await broadcast({"event": "low_heart_rate", "heart_rate_bpm": round(hr, 1), "timestamp": ts})
        elif hr >= HR_LOW_THRESHOLD:
            _hr_was_low = False

        # Low breathing rate alert
        if br < BR_LOW_THRESHOLD and not _br_was_low:
            _br_was_low = True
            await broadcast({"event": "low_breathing_rate", "breathing_rate_bpm": round(br, 1), "timestamp": ts})
        elif br >= BR_LOW_THRESHOLD:
            _br_was_low = False

    # Fall detection
    persons = data.get("persons", [])
    fallen, confidence = is_fallen(persons)
    if fallen and not _fall_active:
        _fall_active = True
        await broadcast({"event": "fall_detected", "confidence": round(confidence, 2), "timestamp": ts})
    elif not fallen:
        _fall_active = False

async def docker_consumer():
    while True:
        try:
            print(f"Connecting to Docker stream at {DOCKER_WS_URL}...")
            async with websockets.connect(DOCKER_WS_URL) as ws:
                print("Connected to Docker stream.")
                async for raw in ws:
                    try:
                        data = json.loads(raw)
                        await process_sensing(data)
                    except Exception as e:
                        print(f"Parse error: {e}")
        except Exception as e:
            print(f"Docker stream disconnected: {e}. Reconnecting in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)

@app.get("/events")
async def get_events():
    return await asyncio.to_thread(_fetch_events)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    CLIENTS.add(websocket)
    print(f"Frontend client connected ({len(CLIENTS)} total)")
    if _last_occupied is not None:
        try:
            await websocket.send_text(json.dumps({
                "event": "presence_update",
                "occupied": _last_occupied,
                "timestamp": now(),
            }))
        except Exception:
            pass
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        CLIENTS.discard(websocket)
        print(f"Frontend client disconnected ({len(CLIENTS)} total)")

@app.on_event("startup")
async def startup():
    asyncio.create_task(docker_consumer())

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
