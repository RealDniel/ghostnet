import argparse
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
WINDOW_SECONDS = 5
SAMPLE_RATE    = 10
WINDOW_SIZE    = WINDOW_SECONDS * SAMPLE_RATE  # 50
N_SUBCARRIERS  = 64
BATCH_SIZE     = 32
EPOCHS         = 50
LR             = 1e-3
TRAIN_SPLIT    = 0.8
HR_PLACEHOLDER = 70.0  # used for external-dataset samples that have no HR label


# --- Snowflake ---
def sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )

def load_snowflake():
    print("Loading data from Snowflake...")
    try:
        with sf_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT timestamp, amplitudes FROM csi_raw ORDER BY timestamp ASC")
            csi_rows = cur.fetchall()
            cur.execute("SELECT timestamp, hr, br FROM vitals_labels ORDER BY timestamp ASC")
            label_rows = cur.fetchall()
        print(f"  CSI rows: {len(csi_rows)}  Label rows: {len(label_rows)}")
        return csi_rows, label_rows
    except Exception as e:
        print(f"  Snowflake unavailable: {e}")
        return [], []

def build_snowflake_dataset(csi_rows, label_rows):
    if not csi_rows or not label_rows:
        return np.empty((0, WINDOW_SIZE, N_SUBCARRIERS), dtype=np.float32), \
               np.empty((0, 3), dtype=np.float32)

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
        idx = np.searchsorted(csi_times, label_ts)
        if idx < WINDOW_SIZE or idx > len(csi_times):
            continue
        nearest = csi_times[min(idx, len(csi_times) - 1)]
        try:
            diff = abs((nearest - label_ts).total_seconds())
        except Exception:
            diff = abs(float(nearest) - float(label_ts))
        if diff > 60:
            continue

        window_times = csi_times[idx - WINDOW_SIZE:idx]
        window = np.array([csi_by_ts[t] for t in window_times if t in csi_by_ts])
        if window.shape[0] != WINDOW_SIZE:
            continue

        active_mask = np.mean(window, axis=0) > 1.0
        if active_mask.sum() < 10:
            continue
        window = window[:, active_mask]

        # Pad/trim to N_SUBCARRIERS
        n = window.shape[1]
        if n < N_SUBCARRIERS:
            pad = np.zeros((WINDOW_SIZE, N_SUBCARRIERS - n), dtype=np.float32)
            window = np.hstack([window, pad])
        else:
            window = window[:, :N_SUBCARRIERS]

        window = (window - window.mean()) / (window.std() + 1e-8)
        X.append(window)
        # mask=1 means both HR and BR are known
        y.append([float(hr), float(br), 1.0])

    print(f"  Built {len(X)} Snowflake windows")
    if not X:
        return np.empty((0, WINDOW_SIZE, N_SUBCARRIERS), dtype=np.float32), \
               np.empty((0, 3), dtype=np.float32)
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


def load_external(root: str):
    """Load WiFi-CSI-MiningTool BR dataset. Returns X, y[:,3] with mask col."""
    from load_external import load_mining_tool
    X, y2 = load_mining_tool(root)          # y2: [hr_placeholder, br]
    # mask=0 means HR is unknown (placeholder), only BR loss applies
    mask = np.zeros((len(y2), 1), dtype=np.float32)
    y3 = np.hstack([y2, mask])              # [hr, br, hr_known]
    return X, y3


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


def masked_mse(pred, target_with_mask):
    """MSE loss. HR loss is skipped when hr_known mask == 0."""
    target = target_with_mask[:, :2]   # [hr, br]
    hr_known = target_with_mask[:, 2]  # 1 = known, 0 = placeholder

    br_loss = ((pred[:, 1] - target[:, 1]) ** 2).mean()
    hr_mask = hr_known.bool()
    if hr_mask.any():
        hr_loss = ((pred[:, 0][hr_mask] - target[:, 0][hr_mask]) ** 2).mean()
    else:
        hr_loss = torch.tensor(0.0)

    return br_loss + hr_loss


def load_ground_truth():
    base = os.path.dirname(__file__)
    hr_readings, br_readings = {}, {}
    try:
        with open(os.path.join(base, "hr.txt")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                time_str, val = line.split(",")
                parts = time_str.strip().split(":")
                key = f"2026-05-30 {int(parts[0]):02d}:{int(parts[1]):02d}"
                hr_readings.setdefault(key, []).append(float(val.strip()))
        with open(os.path.join(base, "br.txt")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                time_str, val = line.split(",")
                parts = time_str.strip().split(":")
                key = f"2026-05-31 {int(parts[0]):02d}:{int(parts[1]):02d}"
                br_readings.setdefault(key, []).append(float(val.strip()))
    except Exception as e:
        print(f"  Ground truth files not found: {e}")
        return {}, {}
    return {k: sum(v)/len(v) for k, v in hr_readings.items()}, \
           {k: sum(v)/len(v) for k, v in br_readings.items()}


def train(external_root=None):
    # --- Load Snowflake data ---
    csi_rows, label_rows = load_snowflake()
    X_sf, y_sf = build_snowflake_dataset(csi_rows, label_rows)

    # --- Load pseudo-labeled data from Snowflake CSI ---
    pseudo_path = os.path.join(os.path.dirname(__file__), "pseudo_labels.npz")
    X_pseudo, y_pseudo = np.empty((0, WINDOW_SIZE, N_SUBCARRIERS), dtype=np.float32), \
                         np.empty((0, 3), dtype=np.float32)
    if os.path.exists(pseudo_path):
        print(f"\nLoading pseudo-labeled HR+BR from {pseudo_path}...")
        data = np.load(pseudo_path)
        X_pseudo, y_pseudo = data["X"], data["y"]
        print(f"  {X_pseudo.shape[0]} pseudo windows  "
              f"HR {y_pseudo[:,0].min():.0f}-{y_pseudo[:,0].max():.0f}  "
              f"BR {y_pseudo[:,1].min():.0f}-{y_pseudo[:,1].max():.0f}")
    else:
        print("\nNo pseudo_labels.npz found. Run: python hr_br/pseudo_label.py")

    # --- Load external BR dataset ---
    X_ext, y_ext = np.empty((0, WINDOW_SIZE, N_SUBCARRIERS), dtype=np.float32), \
                   np.empty((0, 3), dtype=np.float32)
    if external_root:
        print(f"\nLoading external dataset from {external_root}...")
        X_ext, y_ext = load_external(external_root)

    if len(X_sf) == 0 and len(X_ext) == 0 and len(X_pseudo) == 0:
        print("No training data found. Run pseudo_label.py and/or pass --external.")
        return

    # --- Stratified split per source so each always appears in val ---
    def split(X, y):
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        n = max(1, int(len(X) * TRAIN_SPLIT))
        return X[:n], y[:n], X[n:], y[n:]

    parts_train_X, parts_train_y, parts_val_X, parts_val_y = [], [], [], []

    if len(X_sf) > 0:
        xt, yt, xv, yv = split(X_sf, y_sf)
        parts_train_X.append(xt); parts_train_y.append(yt)
        parts_val_X.append(xv);   parts_val_y.append(yv)
        print(f"  Snowflake split:  train={len(xt)}  val={len(xv)}")

    if len(X_pseudo) > 0:
        xt, yt, xv, yv = split(X_pseudo, y_pseudo)
        parts_train_X.append(xt); parts_train_y.append(yt)
        parts_val_X.append(xv);   parts_val_y.append(yv)
        print(f"  Pseudo split:     train={len(xt)}  val={len(xv)}")

    if len(X_ext) > 0:
        xt, yt, xv, yv = split(X_ext, y_ext)
        parts_train_X.append(xt); parts_train_y.append(yt)
        parts_val_X.append(xv);   parts_val_y.append(yv)
        print(f"  External split:   train={len(xt)}  val={len(xv)}")

    X_train = np.concatenate(parts_train_X)
    y_train = np.concatenate(parts_train_y)
    X_val   = np.concatenate(parts_val_X)
    y_val   = np.concatenate(parts_val_y)

    # shuffle combined splits
    for arr_pair in [(X_train, y_train), (X_val, y_val)]:
        pass  # arrays already shuffled above per-source

    train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CSIDataset(X_val,   y_val),   batch_size=BATCH_SIZE)

    input_size = X_train.shape[2]
    model = VitalsLSTM(input_size=input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total = len(X_train) + len(X_val)

    print(f"\nTraining LSTM: input_size={input_size}, total={total} "
          f"(SF={len(X_sf)} pseudo={len(X_pseudo)} ext={len(X_ext)}), epochs={EPOCHS}")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}\n")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = masked_mse(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += masked_mse(pred, yb).item() * len(xb)
        val_loss /= len(X_val)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), "vitals_model.pt"))

    print(f"\nBest val_loss: {best_val_loss:.4f}")
    print("Model saved to hr_br/vitals_model.pt")

    # --- Sample predictions ---
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "vitals_model.pt"),
                                     weights_only=True))
    model.eval()

    # Show Snowflake val samples (HR known) separately
    sf_mask = y_val[:, 2] > 0
    ext_mask = ~sf_mask

    print("\nVal samples with real HR+BR (Snowflake):")
    print(f"{'Pred HR':>8} {'Pred BR':>8} {'True HR':>8} {'True BR':>8}")
    with torch.no_grad():
        sf_idx = np.where(sf_mask)[0][:10]
        if len(sf_idx):
            xb = torch.tensor(X_val[sf_idx])
            preds = model(xb).numpy()
            for p, t in zip(preds, y_val[sf_idx]):
                print(f"{p[0]:8.1f} {p[1]:8.1f} {t[0]:8.1f} {t[1]:8.1f}")
        else:
            print("  (none in val set)")

    print("\nVal samples BR-only (external dataset):")
    print(f"{'Pred BR':>8} {'True BR':>8} {'Error':>8}")
    with torch.no_grad():
        ext_idx = np.where(ext_mask)[0][:10]
        if len(ext_idx):
            xb = torch.tensor(X_val[ext_idx])
            preds = model(xb).numpy()
            for p, t in zip(preds, y_val[ext_idx]):
                print(f"{p[1]:8.1f} {t[1]:8.1f} {abs(p[1]-t[1]):8.1f}")

    # BR MAE across all val
    with torch.no_grad():
        all_preds = model(torch.tensor(X_val)).numpy()
    br_mae = np.mean(np.abs(all_preds[:, 1] - y_val[:, 1]))
    print(f"\nVal BR MAE (all {len(X_val)} samples): {br_mae:.2f} BPM")

    # --- Ground truth validation vs hr.txt / br.txt ---
    hr_gt, br_gt = load_ground_truth()
    if not hr_gt and not br_gt:
        return

    # Map Snowflake label timestamps to X indices for ground truth lookup
    label_times = [r[0] for r in label_rows]

    def get_minute_key(ts, date_prefix):
        if hasattr(ts, 'strftime'):
            return ts.strftime(f"{date_prefix} %H:%M")
        return str(ts)[:16].replace("T", " ")

    # Rebuild X_sf index map (pre-split, so we can look up any timestamp)
    _, y_sf_full = build_snowflake_dataset(csi_rows, label_rows) if csi_rows else (None, None)

    def validate_gt(gt_dict, metric_idx, metric_name, date_prefix, X_full, label_times_list):
        print(f"\nGround truth validation -- {metric_name}:")
        print(f"{'Time':>17} {'GT':>8} {'Model':>8} {'Diff':>8}")
        errors = []
        with torch.no_grad():
            for key, gt_val in sorted(gt_dict.items()):
                matches = [i for i, ts in enumerate(label_times_list)
                           if get_minute_key(ts, date_prefix) == key]
                if not matches or matches[len(matches)//2] >= len(X_full):
                    continue
                idx = matches[len(matches)//2]
                xb = torch.tensor(X_full[idx:idx+1])
                pred = model(xb).numpy()[0][metric_idx]
                diff = abs(pred - gt_val)
                errors.append(diff)
                print(f"{key:>17} {gt_val:>8.1f} {pred:>8.1f} {diff:>8.1f}")
        if errors:
            print(f"  Mean absolute error: {np.mean(errors):.1f} BPM")
        else:
            print("  No matching timestamps found.")

    # Build index map: label_row_index -> X_sf_index
    sf_label_indices = []
    csi_by_ts2 = {}
    for ts, amps in csi_rows:
        if isinstance(amps, str):
            amps = json.loads(amps)
        elif hasattr(amps, '__iter__') and not isinstance(amps, list):
            amps = list(amps)
        csi_by_ts2[ts] = np.array(amps, dtype=np.float32)
    csi_times2 = sorted(csi_by_ts2.keys())
    xi = 0
    for li, (label_ts, hr, br) in enumerate(label_rows):
        if hr is None or br is None:
            continue
        idx2 = np.searchsorted(csi_times2, label_ts)
        if idx2 < WINDOW_SIZE or idx2 > len(csi_times2):
            continue
        nearest2 = csi_times2[min(idx2, len(csi_times2)-1)]
        try:
            diff2 = abs((nearest2 - label_ts).total_seconds())
        except Exception:
            diff2 = abs(float(nearest2) - float(label_ts))
        if diff2 > 60:
            continue
        window_times2 = csi_times2[idx2 - WINDOW_SIZE:idx2]
        window2 = np.array([csi_by_ts2[t] for t in window_times2 if t in csi_by_ts2])
        if window2.shape[0] != WINDOW_SIZE:
            continue
        active2 = np.mean(window2, axis=0) > 1.0
        if active2.sum() < 10:
            continue
        sf_label_indices.append((li, xi))
        xi += 1

    sf_lt = [label_times[li] for li, _ in sf_label_indices]
    sf_xi = [xi for _, xi in sf_label_indices]

    def validate_gt2(gt_dict, metric_idx, metric_name, date_prefix):
        print(f"\nGround truth validation -- {metric_name}:")
        print(f"{'Time':>17} {'GT':>8} {'Model':>8} {'Diff':>8}")
        errors = []
        with torch.no_grad():
            for key, gt_val in sorted(gt_dict.items()):
                matches = [i for i, ts in enumerate(sf_lt)
                           if get_minute_key(ts, date_prefix) == key]
                if not matches:
                    continue
                xidx = sf_xi[matches[len(matches)//2]]
                if xidx >= len(X_sf):
                    continue
                xb = torch.tensor(X_sf[xidx:xidx+1])
                pred = model(xb).numpy()[0][metric_idx]
                diff = abs(pred - gt_val)
                errors.append(diff)
                print(f"{key:>17} {gt_val:>8.1f} {pred:>8.1f} {diff:>8.1f}")
        if errors:
            print(f"  Mean absolute error: {np.mean(errors):.1f} BPM")
        else:
            print("  No matching timestamps found.")

    if len(X_sf) > 0:
        validate_gt2(hr_gt, 0, "Heart Rate (bpm)",        "2026-05-30")
        validate_gt2(br_gt, 1, "Breathing Rate (br/min)", "2026-05-31")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", default=None,
                        help="Path to WiFi-CSI-MiningTool-main folder")
    args = parser.parse_args()
    train(external_root=args.external)
