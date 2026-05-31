"""
Generates pseudo HR+BR labels from raw CSI data in Snowflake csi_raw.
Uses the same bandpass+Welch approach as dev/backend.py.

Saves hr_br/pseudo_labels.npz  ->  { X: (N,50,64), y: (N,3) [hr, br, mask=1] }

Run once before training:
    python hr_br/pseudo_label.py
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from datetime import timezone
from dotenv import load_dotenv
import snowflake.connector
from scipy.signal import butter, filtfilt, welch

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

SAMPLE_RATE    = 10
WINDOW_SIZE    = 50       # 5 seconds at 10 Hz
STRIDE         = 5        # 0.5-second step
N_SUBCARRIERS  = 64
GAP_TOLERANCE  = 2.0      # seconds — gap larger than this = new segment
MIN_VAR        = 0.5      # drop flat/dead windows
HR_LOW, HR_HIGH = 40.0, 180.0
BR_LOW, BR_HIGH =  4.0,  40.0
OUT_PATH = os.path.join(os.path.dirname(__file__), "pseudo_labels.npz")


def sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )


def bandpass(data, low, high, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)


def dominant_freq(signal, fs):
    nperseg = min(len(signal), 128)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return freqs[np.argmax(psd)]


def compute_vitals(window):
    """window: (T, S) ndarray. Returns (hr_bpm, br_bpm) or (None, None)."""
    mask = np.mean(window, axis=0) > 1.0
    active = window[:, mask]
    if active.shape[1] == 0:
        return None, None
    signal = np.mean(active, axis=1)
    try:
        br_sig = bandpass(signal, 0.1, 0.5, SAMPLE_RATE)
        br_bpm = dominant_freq(br_sig, SAMPLE_RATE) * 60

        hr_sig = bandpass(signal, 0.8, 2.0, SAMPLE_RATE)
        hr_bpm = dominant_freq(hr_sig, SAMPLE_RATE) * 60
    except Exception:
        return None, None
    hr_bpm = float(np.clip(hr_bpm, HR_LOW, HR_HIGH))
    br_bpm = float(np.clip(br_bpm, BR_LOW, BR_HIGH))
    return round(hr_bpm, 1), round(br_bpm, 1)


def fetch_csi():
    print("Fetching real CSI rows from Snowflake (excluding synthetic)...")
    with sf_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT timestamp, amplitudes FROM csi_raw "
            "WHERE board_id != 'synthetic' "
            "ORDER BY timestamp ASC"
        )
        rows = cur.fetchall()
    print(f"  Fetched {len(rows)} rows")
    return rows


def parse_amps(raw):
    if isinstance(raw, str):
        return json.loads(raw)
    if hasattr(raw, '__iter__'):
        return list(raw)
    return raw


def split_segments(rows):
    """Group rows into continuous segments based on timestamp gaps."""
    segments = []
    current = []
    prev_ts = None

    for ts, amps_raw in rows:
        amps = parse_amps(amps_raw)
        if amps is None:
            continue

        # Pad/trim to N_SUBCARRIERS
        n = len(amps)
        if n < N_SUBCARRIERS:
            amps = amps + [0.0] * (N_SUBCARRIERS - n)
        else:
            amps = amps[:N_SUBCARRIERS]

        if prev_ts is not None:
            try:
                gap = (ts - prev_ts).total_seconds()
            except Exception:
                gap = float(ts) - float(prev_ts)
            if gap > GAP_TOLERANCE:
                if len(current) >= WINDOW_SIZE:
                    segments.append(current)
                current = []

        current.append(np.array(amps, dtype=np.float32))
        prev_ts = ts

    if len(current) >= WINDOW_SIZE:
        segments.append(current)

    print(f"  Split into {len(segments)} continuous segments")
    return segments


def build_pseudo_dataset(segments):
    X_list, y_list = [], []
    skipped_flat = skipped_vitals = 0

    for seg in segments:
        arr = np.stack(seg)  # (T, 64)
        n = len(arr)

        for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
            window = arr[start:start + WINDOW_SIZE]

            # Drop flat windows (no one present)
            if float(window.std()) < MIN_VAR:
                skipped_flat += 1
                continue

            hr, br = compute_vitals(window)
            if hr is None:
                skipped_vitals += 1
                continue

            norm = (window - window.mean()) / (window.std() + 1e-8)
            X_list.append(norm)
            y_list.append([hr, br, 1.0])  # mask=1 -> both HR and BR are "known"

    print(f"  Windows extracted: {len(X_list)}")
    print(f"  Skipped flat: {skipped_flat}  Skipped no-vitals: {skipped_vitals}")

    if not X_list:
        return None, None

    return np.stack(X_list).astype(np.float32), np.array(y_list, dtype=np.float32)


def run():
    rows = fetch_csi()
    if not rows:
        print("No real CSI data in Snowflake. Run dev/backend.py with ESP32 first.")
        return

    segments = split_segments(rows)
    if not segments:
        print("No continuous segments found. Data may be too sparse.")
        return

    X, y = build_pseudo_dataset(segments)
    if X is None:
        print("No valid windows generated.")
        return

    np.savez_compressed(OUT_PATH, X=X, y=y)
    print(f"\nSaved {X.shape[0]} pseudo-labeled windows to {OUT_PATH}")
    print(f"HR range: {y[:,0].min():.1f} - {y[:,0].max():.1f} BPM")
    print(f"BR range: {y[:,1].min():.1f} - {y[:,1].max():.1f} BPM")


if __name__ == "__main__":
    run()
