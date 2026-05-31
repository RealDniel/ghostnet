# Fall Detection — Hackathon Project

## Project Overview

We are building a **WiFi-based fall detection system** for a hackathon. The system uses RuView (https://github.com/ruvnet/RuView) to convert WiFi Channel State Information (CSI) into 128-dimensional embeddings, which are then fed into a CNN to classify fall vs. non-fall events in real time.

No cameras. No wearables. Just a $9 ESP32-S3 and existing WiFi.

---

## How the Full Stack Works

```
ESP32-S3 hardware (or Docker simulator)
  → captures raw CSI (56 subcarriers × antennas × timestep)
  → RuView sensing server (port 3000)
  → signal processing pipeline (Hampel, SpotFi, BVP, phase unwrapping)
  → pretrained contrastive encoder (transformer + GNN)
  → 128-dim float embedding per frame   ← our CNN input
  → Fall Detection CNN
  → binary alert: FALL / NO_FALL
```

---

## Key Technical Decisions

### Input: 128-dim CSI Embeddings (not COCO skeleton)
- The RuView pretrained model ships on HuggingFace at `ruvnet/wifi-densepose-pretrained`
- It produces 128-dim embeddings at ~164K/sec — no latency concern
- The 17-keypoint COCO skeleton model weights are still pending (issue #509, ADR-079) — do NOT rely on them
- Embeddings are opaque but highly discriminative for fall vs. non-fall classification

### Model: 1D CNN over sliding window
- Input shape: `[Batch × T × 128]` where T = ~30 frames (~1 second at 30Hz)
- Architecture: Conv1D → Conv1D → MaxPool → Conv1D → GlobalAvgPool → Dense → Softmax
- Optimize for **recall over precision** — missing a fall is worse than a false alarm
- Post-processing: 3-frame debounce + 5s cooldown (mirrors RuView's own heuristic)

### Hardware Status
- Using Docker image (`ruvnet/wifi-densepose:latest`) for pipeline validation
- Docker simulates CSI but does NOT simulate real fall signatures — not suitable for training data
- Real training data must come from: ESP32-S3 hardware OR a public CSI fall dataset

---

## Project Steps

### Step 1 — Environment Setup (DONE)
- [x] Identified RuView as CSI-to-pose pipeline
- [x] Decided on 128-dim embeddings as CNN input
- [x] Docker Desktop installed and running
- [ ] `docker pull ruvnet/wifi-densepose:latest && docker run -p 3000:3000 ruvnet/wifi-densepose:latest`
- [ ] `pip install "ruview[client]"` — verify embedding stream works

### Step 2 — Data Collection
Collect labeled sequences of shape `[T × 128]`:

**Classes to record:**
- `fall` — forward, backward, sideways falls
- `no_fall` — walking, sitting, crouching, stumbling-but-recovering, standing still

**Target:** 50–100 sequences per action type minimum for hackathon baseline

**Recording script** (`record.py`):
```python
import numpy as np
import time
from ruview.client import SensingClient

client = SensingClient(host="localhost", port=3000)

def record_sequence(label, duration_sec=3, fps=30):
    frames = []
    start = time.time()
    for embedding in client.stream_embeddings():
        frames.append(embedding)
        if time.time() - start >= duration_sec:
            break
    return {"label": label, "data": np.array(frames)}  # shape: [T x 128]

sequences = []

for _ in range(50):
    input("Press enter, then FALL...")
    sequences.append(record_sequence(label="fall"))

for _ in range(50):
    input("Press enter, then walk normally...")
    sequences.append(record_sequence(label="no_fall"))

np.save("dataset.npy", sequences)
```

**If no ESP32 hardware:** Use a public CSI fall dataset (UTD-MHAD, SiFall, or FallDeFi),
run through RuView encoder offline to get embeddings, then proceed as normal.

### Step 3 — Preprocessing
- Load `dataset.npy`
- Normalize embeddings per-dimension (StandardScaler or per-sequence z-score)
- Sliding window over sequences: window size T=30, stride=5
- One-hot encode labels
- Train/val/test split: 70/15/15
- Handle class imbalance: weighted loss (fall weight 5–10×) or oversample fall class

### Step 4 — Model
```python
import tensorflow as tf

def build_model(T=30, C=128, n_classes=2):
    inputs = tf.keras.Input(shape=(T, C))
    x = tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)
```

### Step 5 — Training
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)
model.fit(
    train_ds,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_recall', mode='max'),
        tf.keras.callbacks.ReduceLROnPlateau()
    ]
)
```

**Target metrics:**
- Recall (sensitivity) > 95% — safety critical
- Specificity > 90% — avoid alert fatigue
- End-to-end latency < 500ms

### Step 6 — Inference & Post-processing
```python
# 3-frame debounce + 5s cooldown — mirrors RuView's own heuristic
fall_counter = 0
last_alert_time = 0
COOLDOWN = 5.0

for embedding in client.stream_embeddings():
    window.append(embedding)
    if len(window) == 30:
        pred = model.predict(np.array(window)[None])[0]
        if pred[FALL_CLASS] > 0.8:
            fall_counter += 1
        else:
            fall_counter = 0

        if fall_counter >= 3 and (time.time() - last_alert_time) > COOLDOWN:
            trigger_alert()
            last_alert_time = time.time()
            fall_counter = 0

        window.pop(0)
```

### Step 7 — Demo
- Live visualization of embedding stream + fall probability score
- Alert trigger (sound, notification, or HA webhook)
- Show recall/precision metrics on test set

---

## File Structure
```
fall-detect/
├── CLAUDE.md              ← this file
├── record.py              ← data collection script
├── preprocess.py          ← windowing, normalization, splitting
├── model.py               ← CNN definition
├── train.py               ← training loop
├── infer.py               ← live inference + debounce
├── dataset.npy            ← collected labeled sequences
└── models/
    └── fall_cnn.keras     ← saved trained model
```

---

## RuView Reference

- Repo: https://github.com/ruvnet/RuView
- Pretrained weights: https://huggingface.co/ruvnet/wifi-densepose-pretrained
- Python install: `pip install "ruview[client]"`
- Docker: `docker run -p 3000:3000 ruvnet/wifi-densepose:latest`
- Embeddings: 128-dim float32, ~164K/sec throughput, 4-bit quantized variant fits 8KB
- Built-in fall heuristic: phase-acceleration threshold + 3-frame debounce + 5s cooldown (<200ms)
- Do NOT rely on 17-keypoint pose output — weights still pending (issue #509)

## Key Constraints

- Optimize for recall — a missed fall is a safety failure
- Keep end-to-end latency under 500ms
- The Docker image is for pipeline validation only, not training data
- CSI embeddings are environment-sensitive — test in the same room you train in
