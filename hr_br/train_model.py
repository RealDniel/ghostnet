import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import snowflake.connector

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# --- Config ---
WINDOW_SECONDS = 10       # seconds of CSI per sample
SAMPLE_RATE    = 10       # Hz
WINDOW_SIZE    = WINDOW_SECONDS * SAMPLE_RATE  # 100 samples
N_SUBCARRIERS  = 64
BATCH_SIZE     = 32
EPOCHS         = 50
LR             = 1e-3
TRAIN_SPLIT    = 0.8

# --- Snowflake ---
def sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )

SESSION_START = "2026-05-30 11:29:00"

def load_data():
    print("Loading data from Snowflake...")
    with sf_conn() as conn:
        cur = conn.cursor()

        cur.execute(
            "SELECT timestamp, amplitudes FROM csi_raw "
            "WHERE timestamp >= %s ORDER BY timestamp ASC",
            (SESSION_START,)
        )
        csi_rows = cur.fetchall()

        cur.execute(
            "SELECT timestamp, hr, br FROM vitals_labels "
            "WHERE timestamp >= %s ORDER BY timestamp ASC",
            (SESSION_START,)
        )
        label_rows = cur.fetchall()

    print(f"  CSI rows: {len(csi_rows)}")
    print(f"  Label rows: {len(label_rows)}")
    return csi_rows, label_rows

def load_ground_truth():
    gt_path = os.path.join(os.path.dirname(__file__), "hr.txt")
    readings = []
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            time_str, hr_str = line.split(",")
            readings.append((time_str.strip(), float(hr_str.strip())))
    return readings

def build_dataset(csi_rows, label_rows):
    # Build timestamp-indexed CSI lookup
    csi_by_ts = {}
    for ts, amps in csi_rows:
        if isinstance(amps, str):
            amps = json.loads(amps)
        elif hasattr(amps, '__iter__') and not isinstance(amps, list):
            amps = list(amps)
        csi_by_ts[ts] = np.array(amps, dtype=np.float32)

    csi_times = sorted(csi_by_ts.keys())

    X, y = [], []

    for label_ts, hr, br in label_rows:
        if hr is None or br is None:
            continue

        # Find the index of the closest CSI timestamp to this label
        idx = np.searchsorted(csi_times, label_ts)
        if idx < WINDOW_SIZE:
            continue
        if idx > len(csi_times):
            continue

        window_times = csi_times[idx - WINDOW_SIZE:idx]
        window = np.array([csi_by_ts[t] for t in window_times if t in csi_by_ts])

        if window.shape[0] != WINDOW_SIZE:
            continue

        # Filter null subcarriers (consistently near 0)
        active_mask = np.mean(window, axis=0) > 1.0
        if active_mask.sum() < 10:
            continue
        window = window[:, active_mask]

        # Normalize
        window = (window - window.mean()) / (window.std() + 1e-8)

        X.append(window)
        y.append([float(hr), float(br)])

    print(f"  Built {len(X)} training samples")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class VitalsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # HR, BR
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train():
    csi_rows, label_rows = load_data()
    X, y = build_dataset(csi_rows, label_rows)

    if len(X) < 10:
        print("Not enough data to train. Collect more CSI data first.")
        return

    # Train/val split
    n_train = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CSIDataset(X_val, y_val),   batch_size=BATCH_SIZE)

    input_size = X.shape[2]
    model = VitalsLSTM(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"\nTraining LSTM — input_size={input_size}, samples={len(X)}, epochs={EPOCHS}")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}\n")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(X_val)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "vitals_model.pt"))

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print("Model saved to hr_br/vitals_model.pt")

    # Sanity check on val set
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "vitals_model.pt")))
    model.eval()
    print("\nSample predictions vs signal-processing labels (first 5 val samples):")
    print(f"{'Pred HR':>8} {'Pred BR':>8} {'True HR':>8} {'True BR':>8}")
    with torch.no_grad():
        xb = torch.tensor(X_val[:5])
        preds = model(xb).numpy()
        for p, t in zip(preds, y_val[:5]):
            print(f"{p[0]:8.1f} {p[1]:8.1f} {t[0]:8.1f} {t[1]:8.1f}")

    # Validate against InstantHeartRate ground truth
    gt = load_ground_truth()
    label_times = [r[0] for r in label_rows]
    label_hrs   = [r[1] for r in label_rows]

    print("\nGround truth validation (InstantHeartRate vs model):")
    print(f"{'Time':>8} {'GT HR':>8} {'Model HR':>10} {'Diff':>8}")
    errors = []
    with torch.no_grad():
        for time_str, gt_hr in gt:
            # Find label index closest to this time (match by HH:MM)
            matches = [i for i, ts in enumerate(label_times)
                       if hasattr(ts, 'strftime') and ts.strftime("%H:%M") == time_str
                       or str(ts)[11:16] == time_str]
            if not matches:
                continue
            idx = matches[len(matches) // 2]
            if idx < WINDOW_SIZE or idx >= len(X):
                continue
            xb = torch.tensor(X[idx:idx+1])
            pred_hr = model(xb).numpy()[0][0]
            diff = abs(pred_hr - gt_hr)
            errors.append(diff)
            print(f"{time_str:>8} {gt_hr:>8.1f} {pred_hr:>10.1f} {diff:>8.1f}")

    if errors:
        print(f"\nMean absolute error vs ground truth: {np.mean(errors):.1f} bpm")


if __name__ == "__main__":
    train()
