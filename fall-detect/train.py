#!/usr/bin/env python3
"""
Train a CNN to classify 4-second WiFi CSI clips into: fall, walk, lay, sit.

Data assumptions (edit CONFIG below to match your files):
  - Sampling rate: 10 Hz  ->  4 s clip = 40 timesteps
  - 64 subcarriers per timestep
  - Each clip is therefore a (40, 64) matrix, treated as a 2D "image"
  - Clips are stored as JSON (a single .json file containing a list of records,
    OR a directory of per-clip .json files).

Expected JSON record shapes (the loader tries to be forgiving):
  {"csi": [[...64 floats...], ... 40 rows ...], "label": "fall"}
  {"data": [...2560 floats flattened...], "activity": "walk"}
  Either (40,64) or (64,40) orientation is auto-detected and fixed.

Usage:
  python csi_cnn_train.py --data path/to/clips.json
  python csi_cnn_train.py --data path/to/clip_dir/ --epochs 50 --batch-size 32
"""

import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SAMPLE_RATE = 10          # CSI packets per second
CLIP_SECONDS = 4
TIME_STEPS = SAMPLE_RATE * CLIP_SECONDS   # 40
NUM_SUBCARRIERS = 64
CLASSES = ["fall", "walk", "lay", "sit"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# The loader will look for the CSI values under the first key it finds here,
# and the label under the first key it finds in LABEL_KEYS.
CSI_KEYS = ["csi", "data", "amplitude", "csi_data", "csi_amplitude"]
LABEL_KEYS = ["label", "activity", "class", "y", "target"]

SEED = 42


# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
def _find_key(record, candidates):
    for k in candidates:
        if k in record:
            return k
    raise KeyError(f"None of {candidates} found in record keys {list(record.keys())}")


def _normalize_label(raw):
    """Map a raw label (string or int) to a class index."""
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    key = str(raw).strip().lower()
    if key not in CLASS_TO_IDX:
        raise ValueError(f"Unknown label {raw!r}. Expected one of {CLASSES}.")
    return CLASS_TO_IDX[key]


def _to_clip_array(values):
    """Coerce a record's CSI values into a (TIME_STEPS, NUM_SUBCARRIERS) float array."""
    arr = np.asarray(values, dtype=np.float32)

    # Flattened vector -> reshape
    if arr.ndim == 1:
        if arr.size != TIME_STEPS * NUM_SUBCARRIERS:
            raise ValueError(
                f"Flat clip has {arr.size} values, expected "
                f"{TIME_STEPS * NUM_SUBCARRIERS} ({TIME_STEPS}x{NUM_SUBCARRIERS})."
            )
        arr = arr.reshape(TIME_STEPS, NUM_SUBCARRIERS)

    # 2D but transposed -> fix orientation
    elif arr.ndim == 2:
        if arr.shape == (NUM_SUBCARRIERS, TIME_STEPS):
            arr = arr.T
        elif arr.shape != (TIME_STEPS, NUM_SUBCARRIERS):
            raise ValueError(
                f"Clip shape {arr.shape} does not match "
                f"({TIME_STEPS}, {NUM_SUBCARRIERS}) or its transpose."
            )
    else:
        raise ValueError(f"Unexpected CSI ndim={arr.ndim}; expected 1 or 2.")

    return arr


def _iter_records(path):
    """Yield raw dict records from a single JSON file or a directory of JSON files."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
        if not files:
            raise FileNotFoundError(f"No .json files found in directory {path}")
        for fp in files:
            with open(fp) as f:
                obj = json.load(f)
            # A per-clip file may itself be a single record or a list.
            if isinstance(obj, list):
                yield from obj
            else:
                yield obj
    else:
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "clips" in obj:
            obj = obj["clips"]
        if not isinstance(obj, list):
            raise ValueError(
                "Expected a JSON list of clip records (or a dir of JSON files)."
            )
        yield from obj


def load_dataset(path):
    """Return X (N, TIME_STEPS, NUM_SUBCARRIERS) float32 and y (N,) int64."""
    clips, labels = [], []
    for rec in _iter_records(path):
        csi_key = _find_key(rec, CSI_KEYS)
        lbl_key = _find_key(rec, LABEL_KEYS)
        clips.append(_to_clip_array(rec[csi_key]))
        labels.append(_normalize_label(rec[lbl_key]))

    if not clips:
        raise RuntimeError("No clips loaded. Check your data path/format.")

    X = np.stack(clips).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def stratified_split(y, val_frac=0.2, seed=SEED):
    """Return (train_idx, val_idx) preserving class balance."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)


class CSIDataset(Dataset):
    def __init__(self, X, y, mean, std):
        # Standardize per-subcarrier using stats computed on the TRAIN split only.
        self.X = (X - mean) / std
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # Add channel dim -> (1, TIME_STEPS, NUM_SUBCARRIERS)
        x = torch.from_numpy(self.X[i]).unsqueeze(0).float()
        return x, int(self.y[i])


# ----------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------
class CSICNN(nn.Module):
    """Compact 2D CNN over (time x subcarrier) clips."""

    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 40x64 -> 20x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 20x32 -> 10x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),               # -> 128x1x1 (size-robust)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------------
# TRAIN / EVAL
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes=len(CLASSES)):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for x, y in loader:
        preds = model(x.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), preds):
            cm[t, p] += 1
    return cm


def main():
    parser = argparse.ArgumentParser(description="Train CSI activity CNN")
    parser.add_argument("--data", required=True, help="JSON file or directory of clips")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--out", default="csi_cnn.pt", help="checkpoint path")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="disable class weighting (useful if balanced)")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y = load_dataset(args.data)
    print(f"Loaded {len(y)} clips, shape per clip = {X.shape[1:]}")
    for c, cnt in zip(CLASSES, np.bincount(y, minlength=len(CLASSES))):
        print(f"  {c:5s}: {cnt}")

    tr_idx, va_idx = stratified_split(y, args.val_frac)

    # Standardization stats from TRAIN split only (per subcarrier), no leakage.
    mean = X[tr_idx].mean(axis=(0, 1), keepdims=True)
    std = X[tr_idx].std(axis=(0, 1), keepdims=True) + 1e-6

    train_ds = CSIDataset(X[tr_idx], y[tr_idx], mean, std)
    val_ds = CSIDataset(X[va_idx], y[va_idx], mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = CSICNN().to(device)

    # Class weighting helps because falls are usually rare vs. other activities.
    if args.no_class_weights:
        weight = None
    else:
        counts = np.bincount(y[tr_idx], minlength=len(CLASSES)).astype(np.float32)
        weight = torch.tensor((counts.sum() / (len(CLASSES) * np.maximum(counts, 1))),
                              dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                           patience=5)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        print(f"Epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc >= best_val:
            best_val = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "mean": mean, "std": std,
                "classes": CLASSES,
                "time_steps": TIME_STEPS,
                "num_subcarriers": NUM_SUBCARRIERS,
            }, args.out)

    print(f"\nBest val accuracy: {best_val:.3f}  (checkpoint -> {args.out})")
    # Report on the best saved model, not the final epoch's weights.
    model.load_state_dict(
        torch.load(args.out, map_location=device, weights_only=False)["model_state"]
    )
    cm = confusion_matrix(model, val_loader, device)
    print("\nConfusion matrix (rows = true, cols = pred):")
    print("        " + "  ".join(f"{c:>5s}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>6s}  " + "  ".join(f"{v:5d}" for v in cm[i]))


if __name__ == "__main__":
    main()
