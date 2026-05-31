# RuView Server Data Reader

This script connects to the RuView WiFi-based fall detection server running on localhost:3000 and streams sensing data for real-time fall detection research.

## What It Does

- **Streams sensing data** from the RuView server's HTTP API (`/api/v1/sensing/latest`)
- **Records labeled sequences** of motion data for training datasets
- **Extracts features** from raw sensing frames into feature vectors
- **Saves data** in both JSON (raw) and NumPy (features) formats

## Server Requirements

The RuView Docker container must be running:

```bash
docker run -p 3000:3000 ruvnet/wifi-densepose:latest
```

The server provides:
- **HTTP endpoint**: `http://localhost:3000/api/v1/sensing/latest` — Latest CSI frame with classification, vital signs, and signal features
- **Update rate**: ~10 Hz (varies with network conditions)
- **Data**: Motion classification, presence detection, breathing/heart rate, spectral features, node amplitude data

## Usage

### Stream Data for 10 Seconds
```bash
python3 read_server.py --duration 10
```

### Record a Labeled Sequence
Record 3 seconds of a "fall" event:
```bash
python3 read_server.py --record --label fall --duration 3 --save fall_1.json
```

### Record and Extract Features (for Training)
```bash
python3 read_server.py --record --label no_fall --duration 3 \
  --save no_fall_1.npy --save-features
```

This creates:
- `no_fall_1.npy` — Feature matrix of shape `(T, 19)` where T=number of frames
- `no_fall_1_meta.json` — Metadata (label, frame count, timestamp)

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--host HOST` | Server hostname (default: `localhost`) |
| `--port PORT` | Server port (default: `3000`) |
| `--duration SEC` | Stream/record duration in seconds (default: `10`) |
| `--record` | Record a labeled sequence (default: stream only) |
| `--label LABEL` | Label for recording (e.g., "fall", "no_fall") |
| `--save FILE` | Save data to file (JSON if no `--save-features`, else numpy metadata) |
| `--save-features` | Extract and save 19-dimensional feature vectors as numpy array |

## Data Format

### Raw Data (JSON)
```json
{
  "label": "fall",
  "data": [
    {
      "classification": {
        "confidence": 0.54,
        "motion_level": "present_moving",
        "presence": true
      },
      "features": {
        "breathing_band_power": 8.9,
        "motion_band_power": 28.5,
        "spectral_power": 230.0,
        "variance": 11.9,
        "mean_rssi": -44.0,
        "dominant_freq_hz": 0.2,
        "change_points": 4
      },
      "nodes": [{"amplitude": [...]}],
      "estimated_persons": 1,
      ...
    },
    ...
  ],
  "timestamp": 1685462400.123
}
```

### Features (NumPy)
- **Shape**: `(T, 19)` where T = number of frames
- **19 features per frame**:
  1. Classification confidence
  2. Presence (binary)
  3-9. Signal features (breathing, motion, spectral power, variance, RSSI, dominant frequency, change points)
  10-19. First 10 subcarrier amplitudes

## Examples

### Collect Training Data
Record 10 fall sequences:
```bash
for i in {1..10}; do
  echo "Recording fall $i..."
  python3 read_server.py --record --label fall --duration 3 \
    --save data/fall_$i.npy --save-features
  sleep 1
done
```

Record 10 no-fall (normal activity) sequences:
```bash
for i in {1..10}; do
  echo "Recording no-fall $i..."
  python3 read_server.py --record --label no_fall --duration 3 \
    --save data/no_fall_$i.npy --save-features
  sleep 1
done
```

### View Raw Server Output
```bash
python3 read_server.py --duration 3 --save output.json
# Then view the JSON
cat output.json | python3 -m json.tool | head -100
```

## Integration with Training Pipeline

The extracted features (19-dim vectors) can be fed into the CNN defined in `CLAUDE.md`:
- Input shape: `[Batch × T × 19]` where T = number of frames in sequence
- Sliding window: Use T≈30 frames (~3 seconds at 10 Hz) with stride of 5-10
- See `preprocess.py` for windowing and normalization

## Troubleshooting

### "Connection refused" on localhost:3000
- Check Docker container is running: `docker ps | grep wifi-densepose`
- Verify port mapping: `netstat -an | grep 3000`
- Start container: `docker run -p 3000:3000 ruvnet/wifi-densepose:latest`

### Very low frame rate (< 1 FPS)
- Normal behavior during polling-based HTTP streaming
- Server updates at ~10 Hz but duplicates are filtered
- For higher-speed streaming, modify polling interval (see `poll_interval` parameter)

### Inconsistent feature count
- Feature extraction depends on available data in each frame
- If `nodes` list is empty, amplitude features are skipped
- Check sample frame keys: data includes 'nodes', 'features', etc.

## Next Steps

1. **Collect training data** using `--record --save-features` for falls and normal activities
2. **Preprocess** sequences with windowing (see `preprocess.py`)
3. **Train CNN** on batches of windowed sequences (see `model.py`, `train.py`)
4. **Deploy inference** with debouncing (see `infer.py`)
