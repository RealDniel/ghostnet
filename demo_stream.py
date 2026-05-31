"""
demo_stream.py — replay a baked CSI data file into the visualizer.

Shared engine behind csi_position.py (walking demo) and csi_fall.py (fall + call).
Reads a *.csi.jsonl file (see tools/gen_csi.py), streams each frame over
ws://localhost:8000/ws at ~20 fps, and — when fall detection is on — fires a
fall_detected event + Twilio call once an impact is followed by stillness.

A slow lie-down never spikes the impact, so it never alerts; a real fall does.
"""

import asyncio
import json
import math
import os
import random
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from engine import (
    FPS, DT, ROOM_W, ROOM_D, DOOR, BOARDS,
    now_iso, place_caregiver_call, step_fall,
)


def load_session(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_app(session_path: str, detect_falls: bool) -> FastAPI:
    frames = load_session(session_path)
    print(f"[demo_stream] loaded {len(frames)} frames from {session_path} "
          f"(fall detection {'ON' if detect_falls else 'off'})")

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    clients: set[WebSocket] = set()
    fall = {"phase": "idle", "impact_at": None, "confirmed_at": None}
    force = {"fall": False}

    async def broadcast(msg: dict):
        for c in list(clients):
            try:
                await c.send_text(json.dumps(msg))
            except Exception:
                clients.discard(c)

    async def loop():
        i = 0
        fallen_at = None
        while True:
            # Fall scene: freeze on last frame so the alert stays visible.
            # Walk scene: stop broadcasting once done — blob stays at last position.
            idx = min(i, len(frames) - 1)
            fr = frames[idx]
            t = i * DT
            motion = fr["motion"]
            impact = fr["impact"]
            posture = fr["posture"]
            forced = force["fall"]
            force["fall"] = False
            if forced:
                impact, motion, posture = 3.0, 0.0, "fallen"

            if not detect_falls and i >= len(frames):
                await broadcast({"event": "session_end", "timestamp": now_iso()})
                await asyncio.sleep(0.5)
                for c in list(clients):
                    try:
                        await c.close()
                    except Exception:
                        pass
                await asyncio.sleep(0.5)
                os._exit(0)

            await broadcast({
                "event": "frame",
                "t": round(t, 2),
                "timestamp": now_iso(),
                "occupied": True,
                "posture": posture,
                "behind_wall": fr["behind_wall"],
                "position": fr["position"],
                "motion": motion,
                "impact": impact,
                "csi": fr["csi"],
                "boards": BOARDS,
                "room": {"w": ROOM_W, "d": ROOM_D, "door": DOOR},
            })

            if detect_falls:
                await step_fall(fall, t, impact, motion, forced,
                                broadcast_fn=broadcast, place_call_fn=place_caregiver_call)

            # Vitals ~1 Hz. Walk: HR climbs 70->90. Fall: spikes to 120 on impact,
            # stays elevated while fallen.
            if i % FPS == 0:
                total = len(frames)
                progress = min(1.0, i / total) if total > 0 else 0.0
                if posture == "falling":
                    fallen_at = t
                    hr = round(120.0 + random.uniform(-4, 4), 1)
                    br = round(24.0 + random.uniform(-1, 1), 1)
                elif posture == "fallen" and detect_falls:
                    time_down = (t - fallen_at) if fallen_at is not None else 0
                    hr = round(max(85.0, 120.0 - time_down * 2.5) + random.uniform(-2, 2), 1)
                    br = round(max(16.0, 24.0 - time_down * 0.4) + random.uniform(-0.5, 0.5), 1)
                elif posture not in ("falling", "fallen"):
                    base_hr = 72.0 + 18.0 * progress if detect_falls else 70.0 + 20.0 * progress
                    base_br = 14.0 + 4.0 * progress
                    hr = round(base_hr + 3.0 * math.sin(t / 4.0) + random.uniform(-1, 1), 1)
                    br = round(base_br + math.sin(t / 6.0) + random.uniform(-0.5, 0.5), 1)
                else:
                    hr = br = None
                if hr is not None:
                    await broadcast({"event": "vital_signs", "heart_rate_bpm": hr,
                                     "breathing_rate_bpm": br, "timestamp": now_iso()})

            i += 1
            await asyncio.sleep(DT)

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(loop())

    @app.get("/")
    async def root():
        return {"ok": True, "session": os.path.basename(session_path), "fall_detection": detect_falls}

    @app.get("/trigger/fall")
    async def trigger():
        force["fall"] = True
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        clients.add(websocket)
        print(f"[ws] client connected ({len(clients)})")
        try:
            while True:
                await websocket.receive_text()
        except (WebSocketDisconnect, Exception):
            pass
        finally:
            clients.discard(websocket)

    return app


def run(session_path: str, detect_falls: bool):
    if not os.path.isabs(session_path):
        session_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), session_path)
    if not os.path.exists(session_path):
        raise SystemExit(f"session file not found: {session_path}\n"
                         f"Generate it first:  python tools/gen_csi.py")
    uvicorn.run(build_app(session_path, detect_falls), host="localhost", port=8000)
