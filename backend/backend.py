import asyncio
import json
import math
import os
import socket
import struct
from collections import deque
from datetime import datetime, timezone

import numpy as np
import snowflake.connector
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt, welch

load_dotenv()

UDP_PORT = 5005
HR_LOW_THRESHOLD = 50
BR_LOW_THRESHOLD = 8
SAMPLE_RATE = 10       # ~10 Hz from ESP32
WINDOW_SIZE = 300      # 30-second rolling window
MIN_SAMPLES = 60       # need 6 seconds before computing vitals

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

_amp_window: deque = deque(maxlen=WINDOW_SIZE)
_tick = 0
_last_occupied: bool | None = None
_hr_was_low = False
_br_was_low = False
_fall_active = False
_smooth_hr: float | None = None
_smooth_br: float | None = None
EMA_ALPHA = 0.2  # lower = smoother but slower to react

def now():
    return datetime.now(timezone.utc).isoformat()

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

def _insert_csi_frame(ts: str, amplitudes: list):
    try:
        import json
        with _sf_conn() as conn:
            conn.cursor().execute(
                "INSERT INTO csi_raw (timestamp, board_id, rssi, subcarriers, amplitudes) "
                "SELECT %s, %s, %s, %s, PARSE_JSON(%s)",
                (ts, "esp32", 0, len(amplitudes), json.dumps(amplitudes))
            )
    except Exception as e:
        print(f"Snowflake CSI insert failed: {e}")

def _insert_vitals_label(ts: str, hr: float, br: float):
    try:
        with _sf_conn() as conn:
            conn.cursor().execute(
                "INSERT INTO vitals_labels (timestamp, hr, br) VALUES (%s, %s, %s)",
                (ts, hr, br)
            )
    except Exception as e:
        print(f"Snowflake vitals label insert failed: {e}")

def _fetch_history(days: int = 60) -> dict:
    """Return daily avg HR/BR and fall counts for the past N days."""
    try:
        with _sf_conn() as conn:
            cur = conn.cursor()

            cur.execute(f"""
                SELECT
                    DATE_TRUNC('day', timestamp)::DATE AS day,
                    ROUND(AVG(hr), 1)                  AS avg_hr,
                    ROUND(AVG(br), 1)                  AS avg_br,
                    COUNT(*)                           AS n
                FROM vitals_labels
                WHERE timestamp > DATEADD(day, -{days}, CURRENT_TIMESTAMP())
                GROUP BY 1
                ORDER BY 1 ASC
            """)
            vitals_rows = [
                {
                    "day": str(r[0]),
                    "avg_hr": float(r[1]) if r[1] is not None else None,
                    "avg_br": float(r[2]) if r[2] is not None else None,
                    "n": int(r[3]),
                }
                for r in cur.fetchall()
            ]

            cur.execute(f"""
                SELECT
                    DATE_TRUNC('day', timestamp)::DATE AS day,
                    COUNT(*) AS falls
                FROM events
                WHERE event = 'fall_detected'
                  AND timestamp > DATEADD(day, -{days}, CURRENT_TIMESTAMP())
                GROUP BY 1
                ORDER BY 1 ASC
            """)
            fall_rows = [{"day": str(r[0]), "falls": int(r[1])} for r in cur.fetchall()]

            total_falls = sum(r["falls"] for r in fall_rows)

            return {
                "days": days,
                "vitals": vitals_rows,
                "falls_by_day": fall_rows,
                "total_falls": total_falls,
            }
    except Exception as e:
        print(f"Snowflake history query failed: {e}")
        return {"days": days, "vitals": [], "falls_by_day": [], "total_falls": 0}


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

async def broadcast(message: dict):
    EVENT_HISTORY.append(message)
    if message.get("event") in ALERT_EVENTS:
        await asyncio.to_thread(_insert_event, message)
    for client in list(CLIENTS):
        try:
            await client.send_text(json.dumps(message))
        except Exception:
            CLIENTS.discard(client)

def _bandpass(data, low, high, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

def _dominant_freq(signal, fs):
    nperseg = min(len(signal), 128)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs[np.argmax(psd)]

def _active_signal(window):
    arr = np.array(window)
    mask = np.mean(arr, axis=0) > 1.0  # drop null/DC subcarriers
    active = arr[:, mask]
    if active.shape[1] == 0:
        return None
    return np.mean(active, axis=1)

def compute_vitals(window):
    signal = _active_signal(window)
    if signal is None:
        return None, None
    try:
        br_signal = _bandpass(signal, 0.1, 0.5, SAMPLE_RATE)
        br_bpm = _dominant_freq(br_signal, SAMPLE_RATE) * 60

        hr_signal = _bandpass(signal, 0.8, 2.0, SAMPLE_RATE)
        hr_bpm = _dominant_freq(hr_signal, SAMPLE_RATE) * 60
    except Exception:
        return None, None
    return round(hr_bpm, 1), round(br_bpm, 1)

def detect_presence(window):
    signal = _active_signal(window)
    if signal is None:
        return False
    return float(np.var(signal)) > 2.0

def detect_fall(window):
    signal = _active_signal(window)
    if signal is None or len(signal) < 20:
        return False, 0.0
    # Short burst of high energy followed by stillness = fall
    recent = signal[-10:]
    prior = signal[-30:-10]
    recent_var = float(np.var(recent))
    prior_var = float(np.var(prior)) + 1e-6
    ratio = recent_var / prior_var
    if ratio > 8.0:
        confidence = min(ratio / 20.0, 1.0)
        return True, round(confidence, 2)
    return False, 0.0

async def process_csi(amplitudes: list, ts: str):
    global _tick, _last_occupied, _hr_was_low, _br_was_low, _fall_active, _smooth_hr, _smooth_br

    _amp_window.append(amplitudes)
    _tick += 1
    asyncio.create_task(asyncio.to_thread(_insert_csi_frame, ts, amplitudes))

    window = list(_amp_window)

    # Presence (every tick, needs 3 seconds)
    if len(window) >= 30:
        occupied = detect_presence(window[-30:])
        if occupied != _last_occupied:
            _last_occupied = occupied
            await broadcast({"event": "presence_update", "occupied": occupied, "timestamp": ts})

    # Fall detection (every tick, needs 3 seconds)
    if len(window) >= 30:
        fallen, confidence = detect_fall(window)
        if fallen and not _fall_active:
            _fall_active = True
            await broadcast({"event": "fall_detected", "confidence": confidence, "timestamp": ts})
        elif not fallen:
            _fall_active = False

    # Vitals (~1/sec, needs 6 seconds)
    if _tick % 10 == 0 and len(window) >= MIN_SAMPLES:
        hr, br = compute_vitals(window)
        if hr is not None and br is not None:
            # Clamp to physiological ranges
            hr = max(40.0, min(180.0, hr))
            br = max(4.0, min(40.0, br))
            # Exponential moving average smoothing
            _smooth_hr = hr if _smooth_hr is None else EMA_ALPHA * hr + (1 - EMA_ALPHA) * _smooth_hr
            _smooth_br = br if _smooth_br is None else EMA_ALPHA * br + (1 - EMA_ALPHA) * _smooth_br
            hr = round(_smooth_hr, 1)
            br = round(_smooth_br, 1)
            await asyncio.to_thread(_insert_vitals_label, ts, hr, br)
            await broadcast({
                "event": "vital_signs",
                "heart_rate_bpm": hr,
                "breathing_rate_bpm": br,
                "timestamp": ts,
            })

            if hr < HR_LOW_THRESHOLD and not _hr_was_low:
                _hr_was_low = True
                await broadcast({"event": "low_heart_rate", "heart_rate_bpm": hr, "timestamp": ts})
            elif hr >= HR_LOW_THRESHOLD:
                _hr_was_low = False

            if br < BR_LOW_THRESHOLD and not _br_was_low:
                _br_was_low = True
                await broadcast({"event": "low_breathing_rate", "breathing_rate_bpm": br, "timestamp": ts})
            elif br >= BR_LOW_THRESHOLD:
                _br_was_low = False

def _poll_snowflake(last_event_ts: str, last_vitals_ts: str) -> tuple[list[dict], list[dict], str, str]:
    """Fetch rows from events and vitals_labels newer than the given timestamps."""
    new_events, new_vitals = [], []
    try:
        with _sf_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT event, timestamp, confidence, heart_rate_bpm, breathing_rate_bpm "
                "FROM events WHERE timestamp > %s::TIMESTAMP_TZ ORDER BY timestamp ASC LIMIT 50",
                (last_event_ts,)
            )
            for r in cur.fetchall():
                ts_str = r[1].isoformat() if r[1] else now()
                new_events.append({
                    "event": r[0],
                    "timestamp": ts_str,
                    "confidence": r[2],
                    "heart_rate_bpm": r[3],
                    "breathing_rate_bpm": r[4],
                    "source": "snowflake",
                })
                last_event_ts = ts_str

            cur.execute(
                "SELECT timestamp, hr, br FROM vitals_labels "
                "WHERE timestamp > %s::TIMESTAMP_TZ ORDER BY timestamp ASC LIMIT 50",
                (last_vitals_ts,)
            )
            for r in cur.fetchall():
                ts_str = r[0].isoformat() if r[0] else now()
                new_vitals.append({
                    "event": "vital_signs",
                    "heart_rate_bpm": r[1],
                    "breathing_rate_bpm": r[2],
                    "timestamp": ts_str,
                    "source": "snowflake",
                })
                last_vitals_ts = ts_str
    except Exception as e:
        print(f"Snowflake poll failed: {e}")
    return new_events, new_vitals, last_event_ts, last_vitals_ts


async def snowflake_poller():
    """Poll Snowflake every 15s for new events/vitals and broadcast to WebSocket clients."""
    last_event_ts = "1970-01-01T00:00:00+00:00"
    last_vitals_ts = "1970-01-01T00:00:00+00:00"
    await asyncio.sleep(5)  # let startup settle
    while True:
        new_events, new_vitals, last_event_ts, last_vitals_ts = await asyncio.to_thread(
            _poll_snowflake, last_event_ts, last_vitals_ts
        )
        for msg in new_vitals:
            EVENT_HISTORY.append(msg)
            for client in list(CLIENTS):
                try:
                    await client.send_text(json.dumps(msg))
                except Exception:
                    CLIENTS.discard(client)
        for msg in new_events:
            EVENT_HISTORY.append(msg)
            for client in list(CLIENTS):
                try:
                    await client.send_text(json.dumps(msg))
                except Exception:
                    CLIENTS.discard(client)
        await asyncio.sleep(15)


async def udp_consumer():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", UDP_PORT))
    print(f"Listening for ESP32 CSI on UDP port {UDP_PORT}...")
    loop = asyncio.get_event_loop()
    _last_packet_log = 0

    while True:
        try:
            data = await loop.run_in_executor(None, lambda: sock.recvfrom(4096)[0])
            magic = struct.unpack_from("<I", data, 0)[0]
            if magic != 0xC5110001:
                continue
            n_sub = struct.unpack_from("<H", data, 6)[0]
            iq = data[20:]
            amplitudes = []
            for k in range(n_sub):
                i = struct.unpack_from("b", iq, k * 2)[0]
                q = struct.unpack_from("b", iq, k * 2 + 1)[0]
                amplitudes.append(round(math.sqrt(i * i + q * q), 2))
            _last_packet_log += 1
            if _last_packet_log % 100 == 0:
                print(f"ESP32 active — {_last_packet_log} packets received", flush=True)
            await process_csi(amplitudes, now())
        except Exception as e:
            print(f"UDP error: {e}")
            await asyncio.sleep(0.1)

class FallPayload(BaseModel):
    confidence: float
    label: str
    timestamp: str

@app.post("/fall")
async def receive_fall(payload: FallPayload):
    await broadcast({
        "event": "fall_detected",
        "confidence": payload.confidence,
        "timestamp": payload.timestamp,
    })
    return {"ok": True}

@app.get("/history")
async def get_history(days: int = 21):
    return await asyncio.to_thread(_fetch_history, days)

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
    asyncio.create_task(udp_consumer())
    asyncio.create_task(snowflake_poller())

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
