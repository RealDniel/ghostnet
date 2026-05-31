import json
import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import snowflake.connector

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

SAMPLE_RATE  = 10
WINDOW_SIZE  = 50
N_SUB        = 64
N_SAMPLES    = 200  # per group
START_TIME   = datetime(2026, 5, 30, 10, 0, 0, tzinfo=timezone.utc)

GROUPS = [
    {"label": "resting HR",  "hr": (60, 80),   "br": (12, 18), "offset_min": 0},
    {"label": "elevated HR", "hr": (80, 110),  "br": (14, 20), "offset_min": 400},
    {"label": "high HR",     "hr": (110, 145), "br": (16, 24), "offset_min": 800},
    {"label": "low BR",      "hr": (60, 90),   "br": (4, 12),  "offset_min": 1200},
    {"label": "high BR",     "hr": (60, 90),   "br": (40, 44), "offset_min": 1600},
]

def sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )

def generate_csi_window(hr_bpm, br_bpm, n_samples=WINDOW_SIZE, n_sub=N_SUB):
    t = np.linspace(0, n_samples / SAMPLE_RATE, n_samples)
    hr_freq = hr_bpm / 60.0
    br_freq = br_bpm / 60.0
    null_idx = list(range(27, 38))
    window = np.zeros((n_samples, n_sub), dtype=np.float32)
    for s in range(n_sub):
        if s in null_idx:
            continue
        amp   = np.random.uniform(8.0, 25.0)
        phase = np.random.uniform(0, 2 * np.pi)
        noise = np.random.normal(0, 0.5, n_samples)
        window[:, s] = amp \
            + 3.0 * np.sin(2 * np.pi * br_freq * t + phase) \
            + 1.0 * np.sin(2 * np.pi * hr_freq * t + phase) \
            + noise
    return window

def insert_synthetic():
    all_csi    = []
    all_labels = []

    for group in GROUPS:
        hr_range   = group["hr"]
        br_range   = group["br"]
        offset_min = group["offset_min"]
        label      = group["label"]
        print(f"Generating {N_SAMPLES} samples for: {label}")

        for i in range(N_SAMPLES):
            hr_bpm = float(np.random.uniform(*hr_range))
            br_bpm = float(np.random.uniform(*br_range))
            window = generate_csi_window(hr_bpm, br_bpm)
            label_ts = START_TIME + timedelta(
                minutes=offset_min,
                seconds=i * (WINDOW_SIZE / SAMPLE_RATE)
            )
            for j in range(WINDOW_SIZE):
                csi_ts = label_ts - timedelta(seconds=(WINDOW_SIZE - j) / SAMPLE_RATE)
                all_csi.append((csi_ts.isoformat(), "synthetic", 0, N_SUB, json.dumps(window[j].tolist())))
            all_labels.append((label_ts.isoformat(), round(hr_bpm, 1), round(br_bpm, 1)))

    total_csi    = len(all_csi)
    total_labels = len(all_labels)
    print(f"\nInserting {total_csi} CSI rows and {total_labels} label rows...")

    with sf_conn() as conn:
        cur = conn.cursor()
        batch = 100
        for start in range(0, total_csi, batch):
            chunk = all_csi[start:start + batch]
            for row in chunk:
                cur.execute(
                    "INSERT INTO csi_raw (timestamp, board_id, rssi, subcarriers, amplitudes) "
                    "SELECT %s, %s, %s, %s, PARSE_JSON(%s)",
                    row
                )
            print(f"  CSI {min(start + batch, total_csi)}/{total_csi}", end="\r", flush=True)

        print()
        for row in all_labels:
            cur.execute(
                "INSERT INTO vitals_labels (timestamp, hr, br) VALUES (%s, %s, %s)",
                row
            )
        print("  Labels inserted")

    print(f"\nDone. {total_labels} synthetic samples added.")

if __name__ == "__main__":
    insert_synthetic()
