#!/usr/bin/env python3
"""
Combine the labelled CSI clip files into one windowed data.json for training.

All sources share one format: each JSONL line is a clip
  {"label": "fall|lying|sitting", "amplitudes": [[..64..] x N], ...}
Clips are variable length, so we slide a fixed 40-frame window across each one.
Minority classes are then augmented up toward balance.

Output: data.json -> list of {"csi": [[..64..] x 40], "label": "fall|lay|sit"}
"""
import json
import numpy as np

# ---- CONFIG ----
SAMPLE_RATE = 10          # Hz (per your estimate; clips carry no per-frame time)
CLIP_SECONDS = 4
WINDOW = 40               # frames per training window (= shortest clip length)
STRIDE = 20               # 50% overlap
NUM_SUBCARRIERS = 64
SEED = 42

LABEL_MAP = {"fall": "fall", "lying": "lay", "sitting": "sit"}
CLASSES = ["fall", "lay", "sit"]      # no walk data in this batch
FILES = {
    "fall": "/mnt/user-data/uploads/fall_data.jsonl",
    "lay":  "/mnt/user-data/uploads/lying_data.jsonl",
    "sit":  "/mnt/user-data/uploads/sitting_data.jsonl",
}
rng = np.random.default_rng(SEED)


def load_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def windows_from_clips(recs):
    out = []
    for r in recs:
        amp = np.array(r["amplitudes"], dtype=np.float32)      # (N,64)
        lbl = LABEL_MAP.get(r["label"], r["label"])
        if amp.shape[1] != NUM_SUBCARRIERS or amp.shape[0] < WINDOW:
            if amp.shape[0] < WINDOW:
                continue
        for start in range(0, amp.shape[0] - WINDOW + 1, STRIDE):
            out.append((amp[start:start + WINDOW], lbl))
    return out


def augment(window):
    x = window.copy()
    x = x * (1.0 + rng.normal(0, 0.05))                          # amplitude scale
    x = x + rng.normal(0, 0.03 * (x.std() + 1e-6), x.shape)      # jitter
    shift = int(rng.integers(-3, 4))                             # small time shift
    if shift:
        x = np.roll(x, shift, axis=0)
    return x.astype(np.float32)


def balance(samples, max_factor=12):
    by_class = {c: [w for w, l in samples if l == c] for c in CLASSES}
    target = max(len(v) for v in by_class.values())
    out = list(samples)
    for c, wins in by_class.items():
        if not wins:
            continue
        need = min(target, len(wins) * max_factor) - len(wins)
        for _ in range(max(0, need)):
            out.append((augment(wins[rng.integers(len(wins))]), c))
    rng.shuffle(out)
    return out


def main():
    samples = []
    for c, path in FILES.items():
        samples += windows_from_clips(load_jsonl(path))

    raw = {c: sum(1 for _, l in samples if l == c) for c in CLASSES}
    balanced = balance(samples)
    final = {c: sum(1 for _, l in balanced if l == c) for c in CLASSES}

    print(f"Window = {WINDOW} frames (~{WINDOW/SAMPLE_RATE:.0f}s @ {SAMPLE_RATE}Hz), stride {STRIDE}\n")
    print("Real windows / after balancing:")
    for c in CLASSES:
        print(f"  {c:4s}: {raw[c]:4d}  ->  {final[c]}")
    print(f"\nTotal: {len(balanced)} samples")

    records = [{"csi": w.tolist(), "label": l} for w, l in balanced]
    json.dump(records, open("data.json", "w"))
    print("Wrote data.json")


if __name__ == "__main__":
    main()
