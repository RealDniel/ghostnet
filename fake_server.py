import asyncio
import json
import math
import random
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIENTS: set[WebSocket] = set()
EVENT_HISTORY: list[dict] = []

HR_LOW_THRESHOLD = 50
BR_LOW_THRESHOLD = 8

def now():
    return datetime.now(timezone.utc).isoformat()

async def broadcast(message: dict):
    EVENT_HISTORY.append(message)
    for client in list(CLIENTS):
        try:
            await client.send_text(json.dumps(message))
        except Exception:
            CLIENTS.discard(client)

@app.get("/events")
async def get_events():
    return EVENT_HISTORY

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    CLIENTS.add(websocket)
    print(f"Client connected ({len(CLIENTS)} total)")
    try:
        await websocket.send_text(json.dumps({
            "event": "presence_update",
            "occupied": False,
            "timestamp": now()
        }))
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        CLIENTS.discard(websocket)
        print(f"Client disconnected ({len(CLIENTS)} total)")

async def event_loop():
    tick = 0
    occupied = False
    dip_hr = False
    dip_br = False

    await asyncio.sleep(1)

    while True:
        await asyncio.sleep(3)
        tick += 1

        # Presence toggle every ~15s
        if tick % 5 == 0:
            occupied = not occupied
            await broadcast({
                "event": "presence_update",
                "occupied": occupied,
                "timestamp": now()
            })
            print(f"[presence_update] occupied={occupied}")

        # Vitals — emit every tick
        if tick % 20 == 0:
            dip_hr = True
        if tick % 30 == 0:
            dip_br = True

        if dip_hr:
            hr = round(random.uniform(38, 48), 1)
            dip_hr = False
            await broadcast({
                "event": "low_heart_rate",
                "heart_rate_bpm": hr,
                "timestamp": now()
            })
            print(f"[low_heart_rate] {hr} bpm")
        else:
            hr = round(random.uniform(58, 90) + math.sin(tick * 0.3) * 5, 1)

        if dip_br:
            br = round(random.uniform(5, 7), 1)
            dip_br = False
            await broadcast({
                "event": "low_breathing_rate",
                "breathing_rate_bpm": br,
                "timestamp": now()
            })
            print(f"[low_breathing_rate] {br} bpm")
        else:
            br = round(random.uniform(12, 20) + math.sin(tick * 0.2) * 2, 1)

        await broadcast({
            "event": "vital_signs",
            "heart_rate_bpm": hr,
            "breathing_rate_bpm": br,
            "timestamp": now()
        })

        # Fall every ~30s when occupied
        if tick % 10 == 0 and occupied:
            confidence = round(random.uniform(0.80, 0.99), 2)
            await broadcast({
                "event": "fall_detected",
                "confidence": confidence,
                "timestamp": now()
            })
            print(f"[fall_detected] confidence={confidence}")

@app.on_event("startup")
async def startup():
    asyncio.create_task(event_loop())

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
