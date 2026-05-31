"""
infer.py

Live fall detection from a raw CSI amplitude stream.

Expected input: 64-subcarrier amplitude vectors arriving at ~200Hz.
Each vector is a 1D array of shape (64,) — float amplitudes per subcarrier.

The script maintains a rolling 100-frame window, runs the CNN every STRIDE
frames, applies a 3-frame debounce and a 5s cooldown before alerting.

Usage:
  python3 infer.py                  # HTTP polling mode (RuView server)
  python3 infer.py --source udp     # UDP stream mode (direct from ESP32)

Adapt get_csi_frame() to match your actual data source.
"""

import os
import time
import argparse
import numpy as np
import tensorflow as tf
from collections import deque

BASE        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, "models", "fall_cnn.keras")
SCALER_PATH = os.path.join(BASE, "models", "scaler.npy")

WINDOW      = 100      # frames per inference window (matches train.py)
STRIDE      = 10       # run inference every N new frames
N_SUB       = 64       # subcarriers
FALL_CLASS  = 1
THRESHOLD   = 0.90     # fall probability to count as a detection
DEBOUNCE    = 3        # consecutive detections needed before alert
COOLDOWN    = 5.0      # seconds between alerts


# ── Data source ───────────────────────────────────────────────────────────────

def iter_frames_http(host="localhost", port=3000):
    """
    Polls the RuView HTTP server and yields the 56-element subcarrier amplitude
    array, zero-padded to 64 dims. Replace with your actual 64-sub source.
    """
    import requests
    url = f"http://{host}:{port}/api/v1/sensing/latest"
    while True:
        try:
            r = requests.get(url, timeout=1.0)
            data = r.json()
            amps = np.array(data.get("subcarrier_amplitudes", [0.0] * 56),
                            dtype=np.float32)
            frame = np.zeros(N_SUB, dtype=np.float32)
            frame[:len(amps)] = amps
            yield frame
        except Exception:
            time.sleep(0.05)
        time.sleep(0.005)   # ~200Hz poll


def iter_frames_udp(host="0.0.0.0", port=5005):
    """
    Receives raw 64-float32 CSI amplitude packets over UDP.
    ESP32 should send exactly 64 * 4 = 256 bytes per packet.
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"Listening for UDP CSI packets on {host}:{port}")
    while True:
        data, _ = sock.recvfrom(4096)
        n = len(data) // 4
        frame = np.frombuffer(data[:n * 4], dtype=np.float32).copy()
        if len(frame) < N_SUB:
            frame = np.pad(frame, (0, N_SUB - len(frame)))
        yield frame[:N_SUB]


# ── Alert ─────────────────────────────────────────────────────────────────────

def trigger_alert():
    print("\n*** FALL DETECTED ***\n")
    # Add sound / notification / HA webhook here


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(source):
    scaler_data = np.load(SCALER_PATH, allow_pickle=True).item()
    mean  = scaler_data["mean"].astype(np.float32)
    scale = scaler_data["scale"].astype(np.float32)

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    window      = deque(maxlen=WINDOW)
    fall_count  = 0
    last_alert  = 0.0
    frames_since_infer = 0

    frame_iter = iter_frames_udp() if source == "udp" else iter_frames_http()

    for frame in frame_iter:
        # Normalize with training scaler
        frame_norm = (frame - mean) / scale
        window.append(frame_norm)
        frames_since_infer += 1

        if len(window) < WINDOW or frames_since_infer < STRIDE:
            continue
        frames_since_infer = 0

        clip = np.array(window, dtype=np.float32)[None]   # (1, WINDOW, 64)
        prob = model.predict(clip, verbose=0)[0][FALL_CLASS]

        print(f"\rFall probability: {prob:.3f}", end="", flush=True)

        if prob > THRESHOLD:
            fall_count += 1
        else:
            fall_count = 0

        if fall_count >= DEBOUNCE and (time.time() - last_alert) > COOLDOWN:
            trigger_alert()
            last_alert = time.time()
            fall_count = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["http", "udp"], default="http")
    args = parser.parse_args()
    run(args.source)
