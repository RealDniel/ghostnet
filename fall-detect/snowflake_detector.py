#!/usr/bin/env python3
"""
Polls Snowflake csi_raw for new frames, feeds them into CSIStreamClassifier,
and POSTs a fall alert to backend.py when a fall is detected.

Run alongside csi_receiver.py and backend.py:
    python fall-detect/snowflake_detector.py
"""

import os
import sys
import time
import requests
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))  # so predict_stream can find csi_cnn_train

import numpy as np
import snowflake.connector
from dotenv import load_dotenv
from predict_stream import CSIStreamClassifier

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

CHECKPOINT      = os.path.join(os.path.dirname(__file__), "csi_cnn.pt")
BACKEND_URL     = "http://localhost:8000/fall"
POLL_INTERVAL   = 0.5   # seconds between Snowflake polls
STRIDE          = 5     # run inference every 5 new frames (~0.5s at 10Hz)
FALL_THRESHOLD  = 0.6
FALL_CONSECUTIVE = 2

def sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )

def fetch_new_frames(conn, since_ts):
    cur = conn.cursor()
    if since_ts is None:
        cur.execute(
            "SELECT timestamp, amplitudes FROM csi_raw "
            "WHERE board_id != 'synthetic' "
            "ORDER BY timestamp DESC LIMIT 40"
        )
        rows = list(reversed(cur.fetchall()))
    else:
        cur.execute(
            "SELECT timestamp, amplitudes FROM csi_raw "
            "WHERE board_id != 'synthetic' AND timestamp > %s "
            "ORDER BY timestamp ASC",
            (since_ts,)
        )
        rows = cur.fetchall()
    return rows

def parse_amplitudes(amps_raw):
    if isinstance(amps_raw, str):
        import json
        return json.loads(amps_raw)
    if hasattr(amps_raw, '__iter__'):
        return list(amps_raw)
    return amps_raw

def post_fall(confidence, label):
    ts = datetime.now(timezone.utc).isoformat()
    try:
        requests.post(BACKEND_URL, json={
            "confidence": round(confidence, 2),
            "label": label,
            "timestamp": ts,
        }, timeout=2)
        print(f"  Fall alert sent to backend (confidence={confidence:.2f})")
    except Exception as e:
        print(f"  Failed to notify backend: {e}")

def run():
    print(f"Loading model from {CHECKPOINT}...")
    clf = CSIStreamClassifier(
        CHECKPOINT,
        stride=STRIDE,
        fall_threshold=FALL_THRESHOLD,
        fall_consecutive=FALL_CONSECUTIVE,
    )
    print(f"Model loaded. Classes: {clf.classes}")
    print(f"Window: {clf.time_steps} frames ({clf.time_steps / 10:.0f}s @ 10Hz)")
    print(f"Polling Snowflake every {POLL_INTERVAL}s...\n")

    last_ts = None
    conn = sf_conn()

    try:
        while True:
            rows = fetch_new_frames(conn, last_ts)

            if rows:
                last_ts = rows[-1][0]

                for ts, amps_raw in rows:
                    amps = parse_amplitudes(amps_raw)

                    # Pad or trim to expected subcarrier count
                    n = clf.num_subcarriers
                    if len(amps) < n:
                        amps = amps + [0.0] * (n - len(amps))
                    elif len(amps) > n:
                        amps = amps[:n]

                    result = clf.push(amps)

                    if result is not None:
                        label = result["label"]
                        conf  = result["confidence"]
                        flag  = " <<< FALL ALARM" if result["fall_alarm"] else ""
                        print(f"[{str(ts)[11:19]}] {label:8s} ({conf:.2f}){flag}")

                        if result["fall_alarm"]:
                            post_fall(conf, label)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        conn.close()

if __name__ == "__main__":
    run()
