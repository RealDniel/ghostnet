"""
preprocess.py

Loads NTU-Fi HAR .mat files, slices to 64 subcarriers (antenna 0, subs 0-63),
applies a sliding window, normalizes, and splits into train/val/test.

Usage:
  from preprocess import build_dataset
  (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = build_dataset()
"""

import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE      = os.path.dirname(os.path.abspath(__file__))
NTUFI_DIR = os.path.join(BASE, "NTU-Fi_HAR", "NTU-Fi_HAR")

WINDOW  = 100   # timesteps per clip (~0.5s at 200Hz)
STRIDE  = 10    # overlap between clips
N_SUB   = 64    # subcarriers to keep (antenna 0, first 64 of 114)
ACTIVITIES = ["box", "circle", "clean", "fall", "run", "walk"]


def _load_sample(path):
    """Returns (2000, 64) float32 amplitude array from one .mat file."""
    mat = sio.loadmat(path)
    csi = mat["CSIamp"].astype(np.float32)   # (342, 2000)
    csi = csi.reshape(3, 114, 2000)          # (3 ant, 114 sub, 2000 t)
    return csi[0, :N_SUB, :].T               # (2000, 64)


def _window(csi, label):
    """Slide a window over (T, 64) → list of (WINDOW, 64) clips."""
    clips, labels = [], []
    for start in range(0, len(csi) - WINDOW + 1, STRIDE):
        clips.append(csi[start:start + WINDOW])
        labels.append(label)
    return clips, labels


def build_dataset(ntufi_dir=NTUFI_DIR):
    clips, labels = [], []

    for split in ("train_amp", "test_amp"):
        for activity in ACTIVITIES:
            act_dir = os.path.join(ntufi_dir, split, activity)
            if not os.path.isdir(act_dir):
                continue
            label = 1 if activity == "fall" else 0
            for fname in sorted(f for f in os.listdir(act_dir) if f.endswith(".mat")):
                csi = _load_sample(os.path.join(act_dir, fname))
                c, l = _window(csi, label)
                clips.extend(c)
                labels.extend(l)

    X = np.array(clips, dtype=np.float32)   # (N, WINDOW, 64)
    y = np.array(labels, dtype=np.int32)

    print(f"Total clips: {len(X)}  fall={np.sum(y==1)}  no_fall={np.sum(y==0)}")

    # Stratified split: 70 / 15 / 15
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=42)

    # Fit scaler on training clips only (per-subcarrier z-score)
    N_train, T, C = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N_train, T, C)
    X_val   = scaler.transform(X_val.reshape(-1, C)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, C)).reshape(X_test.shape)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


if __name__ == "__main__":
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), _ = build_dataset()
    print(f"Train: {X_tr.shape}  Val: {X_v.shape}  Test: {X_te.shape}")
