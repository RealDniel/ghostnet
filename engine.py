"""
mock_csi.py — GhostNet demo engine (synthetic CSI + scripted scenes).

Streams real-format frames over ws://localhost:8000/ws for the 3D visualizer:
  - synthetic CSI (64 subcarriers) whose variance tracks motion (drives the heatmap)
  - an authored person position (the blob's path)
  - a believable impact/acceleration metric that separates a slow lie-down from a fall

Two scenes (switchable for stage control):
  SCENE "walk" — run in a circle, then walk out the door behind the wall and stay sensed (loops).
  SCENE "fall" — lie down slowly (NO alert), get up, then fall (ALERT) -> Twilio call.

Run:
    python mock_csi.py                 # serves ws://localhost:8000/ws, loops the "walk" scene
    curl localhost:8000/scene/fall     # switch to the fall scene
    curl localhost:8000/scene/walk     # back to walk
    curl localhost:8000/trigger/fall   # force an immediate fall (stage backup)

Honesty note: the CSI and motion magnitude are synthetic-but-physically-shaped; the *path* is
authored. This is a record/replay-style demo — never claim the position is computed from WiFi.
"""

import asyncio
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ── Room geometry (metres) ──────────────────────────────────────────────────
ROOM_W, ROOM_D = 5.0, 4.0          # x in [0,5], y(depth) in [0,4]
CENTER = (2.5, 2.0)
DOOR = (5.0, 2.0)                  # door on the east wall; x > 5.0 is "behind the wall"
BOARDS = [(0.2, 0.2), (4.8, 0.2)]  # two ESP32-S3 positions
N_SUB = 64
FPS = 20
DT = 1.0 / FPS

# ── Fall discriminator ──────────────────────────────────────────────────────
# vertical speed (m/s): slow lie-down ~0.2, a real fall ~2+. Threshold cleanly between them.
IMPACT_FALL_THRESHOLD = 1.2
STILLNESS_CONFIRM_S = 0.8          # still this long after impact = a fall, not a stumble -> alert
# Grace/countdown before calling the caregiver: if they get back up, cancel — don't call.
# ~30s in production; short for the demo so the stage pause isn't awkward.
FALL_GRACE_S = float(os.environ.get("FALL_GRACE_S", "5"))
RECOVERY_MOTION = 0.3              # movement above this (during confirm/grace) = they got up

# ── Synthetic CSI base profile (fixed so the heatmap looks stable) ──────────
random.seed(42)
_BASE = [25.0 + 20.0 * math.sin(k / 6.0) + random.uniform(-3, 3) for k in range(N_SUB)]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def synth_csi(motion: float, behind_wall: bool, impact: float) -> list[float]:
    """64 subcarrier amplitudes; variance scales with motion, attenuated through walls."""
    atten = 0.4 if behind_wall else 1.0
    var = motion * 9.0 * atten
    spike = 22.0 if impact > IMPACT_FALL_THRESHOLD else 0.0   # fall shows a broadband jolt
    out = []
    for k in range(N_SUB):
        amp = _BASE[k] * atten
        amp += var * random.gauss(0, 1)
        amp += spike * random.gauss(0, 1) * (0.5 + 0.5 * math.sin(k / 3.0))
        out.append(round(max(0.0, amp), 2))
    return out


# ── Recorded walk session (replayed; the blob follows the captured path) ────
WALK_PATH = os.path.join(os.path.dirname(__file__), "demo", "walk-session.json")
MAX_WALK_SPEED = 1.2  # m/s -> mapped to motion 1.0


def _load_walk():
    try:
        with open(WALK_PATH) as f:
            d = json.load(f)
        wp = [(p["t"], p["x"], p["y"]) for p in d["waypoints"]]
        return d.get("loop_seconds", wp[-1][0]), wp
    except Exception as e:
        print(f"[walk] no session file ({e}); using fallback circle", flush=True)
        return None, None


_WALK_LOOP, _WALK_WP = _load_walk()


def _sample_path(t: float):
    """Interpolate the recorded waypoints; motion is derived from walking speed."""
    loop, wp = _WALK_LOOP, _WALK_WP
    t = t % loop
    for i in range(len(wp) - 1):
        t0, x0, y0 = wp[i]
        t1, x1, y1 = wp[i + 1]
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            x = x0 + f * (x1 - x0)
            y = y0 + f * (y1 - y0)
            dist = math.hypot(x1 - x0, y1 - y0)
            speed = dist / (t1 - t0) if t1 > t0 else 0.0
            motion = max(0.04, min(1.0, speed / MAX_WALK_SPEED))
            return (x, y, 1.0), motion, "walking" if motion > 0.15 else "standing"
    x, y = wp[-1][1], wp[-1][2]
    return (x, y, 1.0), 0.1, "standing"


def scene_walk(t: float) -> tuple[tuple[float, float, float], float, float, str]:
    """Replay the captured walk: wander the room, pause, then exit through the wall."""
    if _WALK_WP:
        pos, motion, posture = _sample_path(t)
        return pos, motion, 0.0, posture
    # Fallback if the session file is missing: a simple circle + exit.
    period = 20.0
    t = t % period
    if t < 13.0:
        ang = (t / 13.0) * (2 * math.pi * 1.5)
        x = CENTER[0] + 1.3 * math.cos(ang)
        y = CENTER[1] + 1.3 * math.sin(ang)
        return (x, y, 1.0), 0.9, 0.0, "running"
    f = (t - 13.0) / 7.0
    return (CENTER[0] + 1.3 + f * 4.0, DOOR[1], 1.0), 0.5, 0.0, "walking"


def scene_fall(t: float) -> tuple[tuple[float, float, float], float, float, str]:
    """0-5 stand, 6-10 LIE DOWN slow (no alert), 13-15 get up, ~18.2 FALL (alert)."""
    x, y = 2.0, 2.0
    if t < 5.0:
        return (x, y, 1.0 + 0.02 * math.sin(t * 3)), 0.18, 0.0, "standing"
    elif t < 6.0:
        return (x + (t - 5.0) * 0.5, y, 1.0), 0.5, 0.0, "walking"
    elif t < 10.0:                                    # slow lie-down over 4s -> ~0.2 m/s
        f = (t - 6.0) / 4.0
        z = 1.0 - 0.8 * f
        return (2.5, y, z), 0.3, 0.8 / 4.0, "lying-down"
    elif t < 13.0:
        return (2.5, y, 0.2), 0.05, 0.0, "lying"
    elif t < 15.0:                                    # get back up
        f = (t - 13.0) / 2.0
        return (2.5, y, 0.2 + 0.8 * f), 0.4, 0.0, "standing-up"
    elif t < 18.0:
        return (2.5 + (t - 15.0) * 0.2, y, 1.0), 0.3, 0.0, "standing"
    elif t < 18.4:                                    # FALL: 0.9m in 0.4s -> ~2.25 m/s
        f = (t - 18.0) / 0.4
        return (3.1, y, 1.0 - 0.9 * f), 1.0, 0.9 / 0.4, "falling"
    else:                                             # motionless on the floor
        return (3.1, y, 0.1), 0.02, 0.0, "fallen"


SCENES = {"walk": scene_walk, "fall": scene_fall}

# ── Server state ────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
CLIENTS: set[WebSocket] = set()
_state = {"scene": "walk", "t": 0.0, "force_fall": False}
# phase: idle -> confirming (impact, waiting for stillness) -> grace (alert shown, countdown)
#        -> alerted (call placed). Returns to idle if they move (get up).
_fall = {"phase": "idle", "impact_at": None, "confirmed_at": None}


async def broadcast(msg: dict):
    dead = []
    for c in list(CLIENTS):
        try:
            await c.send_text(json.dumps(msg))
        except Exception:
            dead.append(c)
    for c in dead:
        CLIENTS.discard(c)


def place_caregiver_call() -> dict:
    """Real Twilio call if creds + lib present, else a clearly-labelled mock."""
    sid = os.environ.get("TWILIO_ACCOUNT_SID")
    tok = os.environ.get("TWILIO_AUTH_TOKEN")
    frm = os.environ.get("TWILIO_FROM")
    to = os.environ.get("CAREGIVER_TO")
    if not all([sid, tok, frm, to]):
        print(f"[MOCK CALL] would call caregiver {to or '(CAREGIVER_TO unset)'} — fall detected", flush=True)
        return {"to": to, "status": "mock (set TWILIO_* + CAREGIVER_TO env to place a real call)"}
    try:
        from twilio.rest import Client  # optional dep; pip install twilio
        msg = os.environ.get(
            "TWILIO_MESSAGE",
            "GhostNet alert: a fall was detected. Please check on your loved one."
        )
        call = Client(sid, tok).calls.create(
            to=to, from_=frm,
            twiml=f"<Response><Say>{msg}</Say></Response>",
        )
        print(f"[TWILIO] call placed to {to}: {call.sid}", flush=True)
        return {"to": to, "status": f"placed ({call.sid})"}
    except Exception as e:
        print(f"[TWILIO] call failed: {e}", flush=True)
        return {"to": to, "status": f"failed: {e}"}


async def step_fall(state, t, impact, motion, forced, broadcast_fn=None, place_call_fn=None):
    """Shared fall state machine: confirm -> grace countdown -> call, cancel on recovery.

    `state` is a mutable dict with keys phase/impact_at/confirmed_at. Emits fall_detected
    (with grace_seconds), fall_cancelled, and call_placed via `broadcast_fn`.
    """
    bcast = broadcast_fn or broadcast
    place = place_call_fn or place_caregiver_call

    if forced:  # stage panic button: alert + call immediately
        ts = now_iso()
        await bcast({"event": "fall_detected", "confidence": 0.95, "grace_seconds": 0, "timestamp": ts})
        result = await asyncio.to_thread(place)
        await bcast({"event": "call_placed", "timestamp": ts, **result})
        state.update(phase="alerted", impact_at=None, confirmed_at=None)
        return

    phase = state["phase"]
    if phase == "idle":
        if impact >= IMPACT_FALL_THRESHOLD:
            state.update(phase="confirming", impact_at=t)
    elif phase == "confirming":
        # Wait out the confirm window (the fall itself is high-motion), then check stillness.
        if state["impact_at"] is not None and (t - state["impact_at"]) >= STILLNESS_CONFIRM_S:
            if motion < 0.15:                               # down and still -> it's a fall
                state.update(phase="grace", confirmed_at=t)
                await bcast({"event": "fall_detected", "confidence": 0.93,
                             "grace_seconds": FALL_GRACE_S, "timestamp": now_iso()})
            else:                                           # still moving after impact = stumble
                state.update(phase="idle", impact_at=None)
    elif phase == "grace":
        if motion >= RECOVERY_MOTION:                       # got back up -> cancel the call
            state["phase"] = "idle"
            await bcast({"event": "fall_cancelled", "reason": "recovered", "timestamp": now_iso()})
        elif state["confirmed_at"] is not None and (t - state["confirmed_at"]) >= FALL_GRACE_S:
            state["phase"] = "alerted"                      # stayed down -> place the call
            ts = now_iso()
            result = await asyncio.to_thread(place)
            await bcast({"event": "call_placed", "timestamp": ts, **result})
    elif phase == "alerted":
        if motion >= RECOVERY_MOTION:
            state.update(phase="idle", impact_at=None, confirmed_at=None)


async def stream_loop():
    while True:
        scene = _state["scene"]
        t = _state["t"]
        pos, motion, vspeed, posture = SCENES[scene](t)
        x, y, z = pos
        behind_wall = x > ROOM_W
        impact = vspeed

        forced = _state["force_fall"]
        if forced:
            impact = 3.0
            posture = "fallen"
            motion = 0.0
            _state["force_fall"] = False

        await broadcast({
            "event": "frame",
            "scene": scene,
            "t": round(t, 2),
            "timestamp": now_iso(),
            "occupied": True,
            "posture": posture,
            "behind_wall": behind_wall,
            "position": {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)},
            "motion": round(motion, 3),
            "impact": round(impact, 2),
            "csi": synth_csi(motion, behind_wall, impact),
            "boards": BOARDS,
            "room": {"w": ROOM_W, "d": ROOM_D, "door": DOOR},
        })

        # Fall handling: impact -> confirm (stillness) -> grace countdown -> call.
        # A slow lie-down never spikes the impact, so it never alerts. If they get
        # back up during confirm or grace, we cancel instead of calling.
        await step_fall(_fall, t, impact, motion, forced)

        # Vital signs ~1 Hz (best when still; suppressed mid-fall motion).
        if int(t * FPS) % FPS == 0 and motion < 0.95 and posture not in ("falling", "fallen"):
            br = round(14.0 + 2.0 * math.sin(t / 7.0) + random.uniform(-0.4, 0.4), 1)
            hr = round(70.0 + 6.0 * math.sin(t / 11.0) + random.uniform(-1.5, 1.5), 1)
            await broadcast({
                "event": "vital_signs", "heart_rate_bpm": hr,
                "breathing_rate_bpm": br, "timestamp": now_iso(),
            })

        _state["t"] = t + DT
        await asyncio.sleep(DT)


@app.on_event("startup")
async def _startup():
    asyncio.create_task(stream_loop())


@app.get("/")
async def root():
    return {"ok": True, "scene": _state["scene"], "scenes": list(SCENES), "ws": "ws://localhost:8000/ws"}


@app.get("/scene/{name}")
async def set_scene(name: str):
    if name not in SCENES:
        return {"error": f"unknown scene '{name}'", "scenes": list(SCENES)}
    _state.update(scene=name, t=0.0)
    _fall.update(phase="idle", impact_at=None, confirmed_at=None)
    print(f"[scene] -> {name}", flush=True)
    return {"ok": True, "scene": name}


@app.get("/trigger/fall")
async def trigger_fall():
    _state["force_fall"] = True
    return {"ok": True}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    CLIENTS.add(websocket)
    print(f"[ws] client connected ({len(CLIENTS)})", flush=True)
    try:
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        CLIENTS.discard(websocket)
        print(f"[ws] client disconnected ({len(CLIENTS)})", flush=True)


if __name__ == "__main__":
    print("GhostNet mock engine — ws://localhost:8000/ws  (scenes: walk, fall)")
    uvicorn.run(app, host="localhost", port=8000)
