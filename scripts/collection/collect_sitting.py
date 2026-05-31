import json
import math
import socket
import struct
import threading
import time
from datetime import datetime, timezone

import os as _os
UDP_PORT    = 5005
RECORD_SECS = 4
OUTPUT_FILE = _os.path.join(_os.path.dirname(__file__), '..', '..', 'data', 'sitting_data.jsonl')

recording = False
record_buf = []
record_lock = threading.Lock()
record_end_time = [0]

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", UDP_PORT))
    print(f"Listening on UDP port {UDP_PORT}...")

    while True:
        data, _ = sock.recvfrom(4096)
        magic = struct.unpack_from("<I", data, 0)[0]
        if magic != 0xC5110001:
            continue
        n_sub = struct.unpack_from("<H", data, 6)[0]
        iq    = data[20:]
        amps  = []
        for k in range(n_sub):
            i = struct.unpack_from("b", iq, k * 2)[0]
            q = struct.unpack_from("b", iq, k * 2 + 1)[0]
            amps.append(round(math.sqrt(i * i + q * q), 2))

        with record_lock:
            if recording and time.time() < record_end_time[0]:
                record_buf.append(amps)

def save_sample(window, idx):
    entry = {
        "label": "sitting",
        "sample_index": idx,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(window),
        "amplitudes": window,
    }
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  Saved sitting sample #{idx} ({len(window)} packets) to {OUTPUT_FILE}")

def main():
    global recording, record_buf

    t = threading.Thread(target=udp_listener, daemon=True)
    t.start()

    print("\nSitting data collector ready.")
    print(f"Sit near the ESP32, then press Enter to record {RECORD_SECS} seconds.")
    print("Press Ctrl+C to stop.\n")

    idx = 1
    while True:
        input(f"[Sample {idx}] Sit still, then press Enter to start recording...")

        with record_lock:
            recording = True
            record_buf = []
            record_end_time[0] = time.time() + RECORD_SECS

        print(f"  Recording {RECORD_SECS} seconds...", end="", flush=True)
        time.sleep(RECORD_SECS)

        with record_lock:
            recording = False
            window = list(record_buf)

        print(" done.")
        save_sample(window, idx)
        idx += 1

if __name__ == "__main__":
    main()
