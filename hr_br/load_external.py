"""
Loader for WiFi-CSI-MiningTool breathing rate dataset.
https://github.com/AlbanyArmenta0711/WiFi-CSI-MiningTool

Structure expected:
  <root>/Datasets/Subjects/Breathing/S*_CSI/<BPM>BPMAmp.csv

Each CSV: no header, rows = CSI frames at ~100Hz, cols = 90 subcarrier amplitudes.
Label is the BR in BPM, parsed from the filename.
HR is unknown for this dataset — placeholder 70.0 is used so the 2-output model
still trains; BR loss will dominate for these samples.

Usage:
    from load_external import load_mining_tool
    X, y = load_mining_tool("C:/Users/.../WiFi-CSI-MiningTool-main")
"""

import os
import re
import glob
import numpy as np

SAMPLE_RATE_ORIG = 100   # Hz (Intel CSI tool capture rate)
SAMPLE_RATE_TARGET = 10  # Hz (matches ESP32 backend)
DOWNSAMPLE = SAMPLE_RATE_ORIG // SAMPLE_RATE_TARGET  # take every 10th row

WINDOW_SIZE = 50         # samples at 10Hz = 5 seconds (matches train_model.py)
STRIDE = 10              # 1-second step between windows
HR_PLACEHOLDER = 70.0    # no HR ground truth in this dataset
N_SUBCARRIERS_TARGET = 64  # pad/trim to match ESP32 output


def load_mining_tool(root: str):
    """
    Returns:
        X : np.ndarray  shape (N, WINDOW_SIZE, N_SUBCARRIERS_TARGET)  float32
        y : np.ndarray  shape (N, 2)  [hr, br]  float32
    """
    pattern = os.path.join(root, "Datasets", "Subjects", "Breathing", "*_CSI", "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSI CSVs found under {root}. Check the path.")

    X_list, y_list = [], []

    for fp in files:
        fname = os.path.basename(fp)
        match = re.search(r"(\d+)BPMAmp", fname, re.IGNORECASE)
        if not match:
            continue
        br_bpm = float(match.group(1))

        try:
            data = np.loadtxt(fp, delimiter=",", dtype=np.float32)
        except Exception as e:
            print(f"  Skipping {fp}: {e}")
            continue

        if data.ndim != 2 or data.shape[1] < 10:
            continue

        # Downsample: take every DOWNSAMPLE-th row
        data = data[::DOWNSAMPLE]

        # Pad or trim subcarriers to N_SUBCARRIERS_TARGET
        n_sub = data.shape[1]
        if n_sub < N_SUBCARRIERS_TARGET:
            pad = np.zeros((data.shape[0], N_SUBCARRIERS_TARGET - n_sub), dtype=np.float32)
            data = np.hstack([data, pad])
        else:
            data = data[:, :N_SUBCARRIERS_TARGET]

        # Slide windows
        n_frames = data.shape[0]
        for start in range(0, n_frames - WINDOW_SIZE + 1, STRIDE):
            window = data[start:start + WINDOW_SIZE]

            # Drop windows where signal is flat/dead
            if window.std() < 0.1:
                continue

            # Normalize
            window = (window - window.mean()) / (window.std() + 1e-8)

            X_list.append(window)
            y_list.append([HR_PLACEHOLDER, br_bpm])

    if not X_list:
        raise RuntimeError("No valid windows extracted. Check dataset path and file format.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"Loaded {len(files)} files -> {X.shape[0]} windows  (shape {X.shape})")
    return X, y


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\tehla\Downloads\WiFi-CSI-MiningTool-main"
    X, y = load_mining_tool(root)
    print(f"X: {X.shape}  y: {y.shape}")
    print(f"BR range: {y[:,1].min():.0f} – {y[:,1].max():.0f} BPM")
    print(f"Sample window mean={X[0].mean():.3f} std={X[0].std():.3f}")
