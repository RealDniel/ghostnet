# GhostNet

WiFi-based, camera-free elderly monitoring system built on two ESP32-S3 boards reading Channel State Information (CSI). Detects presence, estimates heart rate and breathing rate, detects falls, and stores all data in Snowflake.

**Stack:** ESP32-S3 hardware → Python FastAPI backend → Snowflake → React/Vite frontend

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.13 recommended |
| Node.js | 18+ | for the frontend |
| Snowflake account | — | free trial works |
| ESP32-S3-WROOM-1 | — | two boards for TX/RX |

---

## Quick Start

### 1. Clone and configure secrets

```bash
git clone <repo-url>
cd ghostnet
cp .env.example .env
```

Edit `.env` with your Snowflake credentials:

```
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=byiulwt-af43260
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch** must be installed separately — pick the right build for your hardware:
>
> - **Windows CUDA:** `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> - **Windows CPU:** `pip install torch --index-url https://download.pytorch.org/whl/cpu`
> - **macOS (Apple Silicon / Intel):** `pip install torch`

### 3. Set up Snowflake

Run the setup script once. This creates the stream, stored procedure, and scheduled tasks in Snowflake:

```bash
python scripts/snowflake_tasks.py
```

This creates:
- `csi_raw_stream` — detects new rows written to `csi_raw`
- `ghostnet_process()` — stored procedure that computes vitals and detects falls using numpy/scipy inside Snowflake
- `ghostnet_task` — runs `ghostnet_process()` every minute when new CSI data is present
- `ghostnet_cleanup()` — stored procedure that deletes data older than 22 days
- `ghostnet_cleanup_task` — runs the cleanup daily at 03:00 UTC

### 4. Start the backend

```bash
python backend/backend.py
```

The backend listens on `http://localhost:8000`. It:
- Receives CSI frames from the ESP32 over UDP port 5005
- Inserts every frame into Snowflake `csi_raw`
- Runs real-time signal processing (fall detection, HR/BR estimation) locally
- Polls Snowflake every 15 seconds for cloud-computed results
- Broadcasts all events to connected WebSocket clients

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

---

## Project Structure

```
ghostnet/
├── backend/
│   └── backend.py          # FastAPI server — UDP receiver, signal processing, WebSocket
├── frontend/               # React + Vite + Tailwind dashboard
│   └── src/
│       ├── components/     # VitalsDisplay, VitalsGraph, HistoryPanel, FallAlert, EventLog, ...
│       └── hooks/
├── fall-detect/            # Fall detection CNN (PyTorch)
│   ├── csi_cnn_train.py    # Model architecture + training
│   ├── train.py            # Training entry point
│   ├── validate.py         # Validation against collected data
│   ├── snowflake_detector.py  # Real-time fall detection → POST /fall
│   └── csi_cnn.pt          # Trained model checkpoint (gitignored)
├── hr_br/                  # Heart rate / breathing rate LSTM (PyTorch)
│   ├── train_model.py      # Training with masked MSE loss
│   ├── pseudo_label.py     # Generate HR/BR labels from Snowflake CSI data
│   ├── load_external.py    # Load WiFi-CSI-MiningTool dataset (BR labels)
│   └── vitals_model.pt     # Trained model checkpoint (gitignored)
├── data/                   # Raw sensor recordings (gitignored)
│   ├── fall_data.jsonl
│   ├── lying_data.jsonl
│   ├── sitting_data.jsonl
│   └── demo/
│       └── walk-session.json
├── scripts/
│   ├── snowflake_tasks.py  # One-time Snowflake setup (run this first)
│   ├── collection/         # ESP32 data collection scripts
│   │   ├── collect_falls.py
│   │   ├── collect_lying.py
│   │   ├── collect_sitting.py
│   │   ├── collect_walking.py
│   │   └── upload_to_snowflake.py
│   ├── gen_csi.py          # Synthetic CSI generator for testing
│   └── legacy/             # Pre-refactor scripts (archived)
├── .env.example            # Credentials template
├── requirements.txt        # Python dependencies
└── PRD.md                  # Product requirements document
```

---

## Data Pipeline

```
ESP32 (UDP 5005)
    │
    ▼
backend/backend.py
    ├── real-time: signal processing → WebSocket → frontend   (~100ms)
    └── async:    INSERT into Snowflake csi_raw
                      │
                      ▼
              csi_raw_stream (Snowflake)
                      │  triggers when new rows arrive
                      ▼
              ghostnet_task (every 1 min)
                      │
                      ▼
              ghostnet_process() stored procedure
              (numpy/scipy inside Snowflake)
                      │
              ┌───────┴───────┐
              ▼               ▼
        vitals_labels       events
              │               │
              └───────┬───────┘
                      ▼
              backend poller (every 15s)
                      │
                      ▼
              WebSocket → frontend
```

Data older than 22 days is automatically deleted daily by `ghostnet_cleanup_task`.

---

## ML Models

### Fall Detection CNN (`fall-detect/`)

Classifies 4-second CSI clips (40 timesteps × 64 subcarriers) into: `fall`, `lay`, `sit`.

**Collect training data** (ESP32 must be running):
```bash
python scripts/collection/collect_falls.py
python scripts/collection/collect_lying.py
python scripts/collection/collect_sitting.py
```

**Train:**
```bash
python fall-detect/train.py
```

**Validate:**
```bash
python fall-detect/validate.py
```

**Run live inference** (alongside the backend):
```bash
python fall-detect/snowflake_detector.py
```

---

### Vitals LSTM (`hr_br/`)

Predicts heart rate and breathing rate from 5-second CSI windows (50 timesteps × 64 subcarriers).

**Generate pseudo-labels** from Snowflake CSI (requires backend to have run with ESP32):
```bash
python hr_br/pseudo_label.py
```

**Train:**
```bash
# With pseudo-labels only:
python hr_br/train_model.py

# With external WiFi-CSI-MiningTool dataset for breathing rate (download separately):
python hr_br/train_model.py --external path/to/WiFi-CSI-MiningTool-main
```

---

## Snowflake Tables

| Table | Contents |
|---|---|
| `csi_raw` | Raw CSI frames from ESP32 (timestamp, board_id, amplitudes) |
| `vitals_labels` | Heart rate + breathing rate estimates (timestamp, hr, br) |
| `events` | Alert events: `fall_detected`, `low_heart_rate`, `low_breathing_rate` |
| `pose_data` | Labeled pose data uploaded for training |
| `ground_truth` | Manual HR/BR ground truth labels |

---

## Backend API

| Endpoint | Description |
|---|---|
| `WS /ws` | WebSocket — live event stream to frontend |
| `GET /events` | Last 100 alert events from Snowflake |
| `GET /history?days=21` | Daily avg HR/BR + fall counts for the past N days |
| `POST /fall` | Receive fall alert from `snowflake_detector.py` |

---

## Security

- Snowflake credentials are in `.env` only — never committed
- The frontend never connects to Snowflake directly; all data flows through the backend
- `.env`, `*.pt`, `*.npz`, and `data/*.jsonl` are gitignored
