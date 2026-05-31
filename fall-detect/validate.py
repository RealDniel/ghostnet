#!/usr/bin/env python3
"""
Validate csi_cnn.pt against all collected real ESP32 data.

Usage:
    python fall-detect/validate.py
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from csi_cnn_train import CSICNN

CKPT    = os.path.join(os.path.dirname(__file__), "csi_cnn.pt")
BASE    = os.path.join(os.path.dirname(__file__), "..")
TIME_STEPS = 40
N_SUB   = 64

# label in file -> model class
LABEL_MAP = {
    "fall":    "fall",
    "falling": "fall",
    "lying":   "lay",
    "lay":     "lay",
    "sitting": "sit",
    "sit":     "sit",
    "walk":    "walk",
    "walking": "walk",
    "standing":"stand",
}

DATA_FILES = [
    os.path.join(BASE, "data", "fall_data.jsonl"),
    os.path.join(BASE, "data", "lying_data.jsonl"),
    os.path.join(BASE, "data", "sitting_data.jsonl"),
]


def load_clips():
    clips, labels = [], []
    for path in DATA_FILES:
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]

        for rec in records:
            raw_label = rec.get("label", "unknown").lower()
            cls = LABEL_MAP.get(raw_label)
            if cls is None:
                print(f"  Skipping unknown label: {raw_label}")
                continue

            amps = np.array(rec["amplitudes"], dtype=np.float32)  # (N, 64)

            # Pad or trim subcarriers to N_SUB
            if amps.shape[1] < N_SUB:
                pad = np.zeros((amps.shape[0], N_SUB - amps.shape[1]), dtype=np.float32)
                amps = np.hstack([amps, pad])
            else:
                amps = amps[:, :N_SUB]

            n = amps.shape[0]
            if n < TIME_STEPS:
                # Pad short clips with zeros
                pad = np.zeros((TIME_STEPS - n, N_SUB), dtype=np.float32)
                amps = np.vstack([amps, pad])
                clips.append(amps)
                labels.append(cls)
            else:
                # Slide TIME_STEPS windows with stride TIME_STEPS//2
                stride = max(1, TIME_STEPS // 2)
                for start in range(0, n - TIME_STEPS + 1, stride):
                    clips.append(amps[start:start + TIME_STEPS])
                    labels.append(cls)

    return clips, labels


def run():
    print(f"Loading checkpoint: {CKPT}")
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    mean = np.asarray(ckpt["mean"], dtype=np.float32).reshape(-1)
    std  = np.asarray(ckpt["std"],  dtype=np.float32).reshape(-1)
    print(f"Model classes: {classes}")

    model = CSICNN(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("\nLoading data...")
    clips, labels = load_clips()

    # Keep only labels the model knows
    filtered = [(c, l) for c, l in zip(clips, labels) if l in classes]
    if not filtered:
        print("No clips with known labels found.")
        return
    clips_f, labels_f = zip(*filtered)

    print(f"Total clips: {len(clips_f)}")
    for cls in classes:
        print(f"  {cls:6s}: {labels_f.count(cls)}")

    # Build tensors
    X = np.stack(clips_f).astype(np.float32)      # (N, 40, 64)
    X = (X - mean) / (std + 1e-6)                 # normalize same as training
    X_t = torch.from_numpy(X).unsqueeze(1)        # (N, 1, 40, 64)

    y_true = np.array([classes.index(l) for l in labels_f])

    # Inference
    with torch.no_grad():
        logits = model(X_t)
        probs  = F.softmax(logits, dim=1).numpy()
        y_pred = probs.argmax(axis=1)

    # Confusion matrix
    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "         " + "  ".join(f"{c:>6s}" for c in classes)
    print(header)
    for i, c in enumerate(classes):
        row = f"{c:>7s}  " + "  ".join(f"{cm[i,j]:6d}" for j in range(n_cls))
        print(row)

    # Per-class metrics
    print("\nPer-class metrics:")
    print(f"{'Class':>7s}  {'Precision':>10s}  {'Recall':>8s}  {'F1':>6s}  {'Support':>8s}")
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        support = cm[i, :].sum()
        flag = " <-- CRITICAL" if c == "fall" else ""
        print(f"{c:>7s}  {prec:>10.3f}  {recall:>8.3f}  {f1:>6.3f}  {support:>8d}{flag}")

    overall_acc = (y_pred == y_true).mean()
    print(f"\nOverall accuracy: {overall_acc:.3f}  ({(y_pred==y_true).sum()}/{len(y_true)})")

    # Fall confidence distribution
    fall_idx = classes.index("fall")
    fall_probs = probs[y_true == fall_idx, fall_idx]
    nonfail_probs = probs[y_true != fall_idx, fall_idx]
    if len(fall_probs):
        print(f"\nFall class confidence on TRUE falls:")
        print(f"  mean={fall_probs.mean():.3f}  min={fall_probs.min():.3f}  max={fall_probs.max():.3f}")
    if len(nonfail_probs):
        print(f"Fall class confidence on NON-falls (false alarm rate):")
        threshold = 0.6
        false_alarms = (nonfail_probs >= threshold).sum()
        print(f"  mean={nonfail_probs.mean():.3f}  false alarms at >{threshold}: {false_alarms}/{len(nonfail_probs)}")

    # Sample predictions
    print("\nSample predictions (first 10 clips):")
    print(f"{'True':>8s}  {'Pred':>8s}  {'Confidence':>12s}  {'Correct':>8s}")
    for i in range(min(10, len(y_true))):
        t = classes[y_true[i]]
        p = classes[y_pred[i]]
        conf = probs[i, y_pred[i]]
        print(f"{t:>8s}  {p:>8s}  {conf:>12.3f}  {'OK' if t==p else 'WRONG':>8s}")


if __name__ == "__main__":
    run()
