#!/usr/bin/env python3
"""
Read sensing data from RuView server on localhost:3000.

The server should already be running via:
  docker run -p 3000:3000 ruvnet/wifi-densepose:latest

Streams CSI sensing data including motion classification, vital signs, and features.
"""

import numpy as np
import time
import sys
import json
import requests


def stream_sensing_data(host="localhost", port=3000, duration_sec=10, poll_interval=0.05):
    """
    Stream sensing data from the RuView HTTP server and print statistics.

    Args:
        host: Server hostname
        port: Server port (default 3000)
        duration_sec: How long to stream for (None = infinite)
        poll_interval: Time between polls in seconds
    """
    url = f"http://{host}:{port}/api/v1/sensing/latest"

    try:
        print(f"Connected to RuView server at http://{host}:{port}")
        print(f"Streaming sensing data for {duration_sec}s...")
        print("-" * 60)

        start_time = time.time()
        frame_count = 0
        data_batch = []
        last_data_hash = None

        while True:
            try:
                response = requests.get(url, timeout=2)
                response.raise_for_status()
                data = response.json()

                # Avoid counting duplicate frames (same data returned)
                data_hash = hash(json.dumps(data, sort_keys=True, default=str))
                if data_hash != last_data_hash:
                    frame_count += 1
                    data_batch.append(data)
                    last_data_hash = data_hash

                    # Print stats every 30 frames (~1 second)
                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed
                        motion = data.get('classification', {}).get('motion_level', 'N/A')
                        presence = data.get('classification', {}).get('presence', False)
                        print(f"Frame {frame_count} | FPS: {fps:.1f} | "
                              f"Motion: {motion} | Presence: {presence}")

                # Check if duration exceeded
                if duration_sec is not None and (time.time() - start_time) >= duration_sec:
                    break

                time.sleep(poll_interval)

            except requests.RequestException as e:
                print(f"Warning: Request error: {e}")
                time.sleep(0.1)

        print("-" * 60)
        elapsed = time.time() - start_time
        print(f"Streamed {frame_count} frames in {elapsed:.2f}s")
        if frame_count > 0:
            print(f"Average FPS: {frame_count / elapsed:.1f}")

        return data_batch

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def record_labeled_sequence(label, duration_sec=3, host="localhost", port=3000, poll_interval=0.05):
    """
    Record a single labeled sequence of sensing data.

    Args:
        label: Label for this sequence (e.g., "fall" or "no_fall")
        duration_sec: Duration to record
        host: Server hostname
        port: Server port
        poll_interval: Time between polls in seconds

    Returns:
        dict with keys "label", "data" (list of dicts), and "timestamp"
    """
    url = f"http://{host}:{port}/api/v1/sensing/latest"

    try:
        print(f"Recording '{label}' for {duration_sec}s...")

        frames = []
        start = time.time()
        last_data_hash = None

        while (time.time() - start) < duration_sec:
            try:
                response = requests.get(url, timeout=2)
                response.raise_for_status()
                data = response.json()

                # Avoid counting duplicate frames
                data_hash = hash(json.dumps(data, sort_keys=True, default=str))
                if data_hash != last_data_hash:
                    frames.append(data)
                    last_data_hash = data_hash

                time.sleep(poll_interval)

            except requests.RequestException as e:
                print(f"Warning: Request error: {e}")
                time.sleep(0.1)

        print(f"Recorded {len(frames)} frames")
        if frames:
            print(f"Sample frame keys: {list(frames[0].keys())}")

        return {"label": label, "data": frames, "timestamp": time.time()}

    except Exception as e:
        print(f"ERROR recording sequence: {e}")
        sys.exit(1)


def extract_features(data_point):
    """
    Extract a feature vector from a single sensing data point.

    Args:
        data_point: Dict with sensing data from the server

    Returns:
        numpy array of features
    """
    features = []

    # Motion classification features
    classification = data_point.get('classification', {})
    features.extend([
        float(classification.get('confidence', 0)),
        float(classification.get('presence', False)),
    ])

    # Signal features
    signal_features = data_point.get('features', {})
    features.extend([
        signal_features.get('breathing_band_power', 0),
        signal_features.get('motion_band_power', 0),
        signal_features.get('spectral_power', 0),
        signal_features.get('variance', 0),
        signal_features.get('mean_rssi', 0),
        signal_features.get('dominant_freq_hz', 0),
        signal_features.get('change_points', 0),
    ])

    # Node amplitude features (first 10 subcarriers)
    nodes = data_point.get('nodes', [])
    if nodes:
        amplitude = nodes[0].get('amplitude', [])
        features.extend(amplitude[:10])

    return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Read sensing data from RuView server")
    parser.add_argument("--host", default="localhost", help="Server hostname (default: localhost)")
    parser.add_argument("--port", type=int, default=3000, help="Server port (default: 3000)")
    parser.add_argument("--duration", type=int, default=10,
                       help="Stream duration in seconds (default: 10)")
    parser.add_argument("--record", action="store_true",
                       help="Record a labeled sequence instead of streaming")
    parser.add_argument("--label", default="test",
                       help="Label for recording (used with --record)")
    parser.add_argument("--save", type=str,
                       help="Save recorded data to JSON file")
    parser.add_argument("--save-features", action="store_true",
                       help="Extract and save features as numpy array instead of raw data")

    args = parser.parse_args()

    if args.record:
        sequence = record_labeled_sequence(args.label, args.duration, args.host, args.port)
        print(f"\nSequence recorded with {len(sequence['data'])} frames")

        if args.save:
            if args.save_features:
                # Extract features from each frame
                features = np.array([extract_features(frame) for frame in sequence['data']])
                print(f"Extracted features shape: {features.shape}")
                np.save(args.save.replace('.npy', '') + '.npy', features)
                # Also save metadata
                metadata = {"label": sequence['label'], "num_frames": len(sequence['data']), "timestamp": sequence['timestamp']}
                import json
                with open(args.save.replace('.npy', '') + '_meta.json', 'w') as f:
                    json.dump(metadata, f)
                print(f"Saved features to {args.save.replace('.npy', '')}.npy")
                print(f"Saved metadata to {args.save.replace('.npy', '')}_meta.json")
            else:
                import json
                with open(args.save, 'w') as f:
                    json.dump(sequence, f, indent=2, default=str)
                print(f"Saved to {args.save}")
    else:
        data_batch = stream_sensing_data(args.host, args.port, args.duration)
        print(f"\nCollected {len(data_batch)} frames of sensing data")

        if args.save:
            import json
            with open(args.save, 'w') as f:
                json.dump(data_batch, f, indent=2, default=str)
            print(f"Saved to {args.save}")
