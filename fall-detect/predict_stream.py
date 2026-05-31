#!/usr/bin/env python3
"""
Live (streaming) inference for the CSI activity CNN.

While the training script (csi_cnn_train.py) works on pre-cut 4 s clips, a real
deployment receives CSI frames one at a time off a continuous stream. This module
maintains a sliding window of the most recent TIME_STEPS frames and re-runs the
model as new frames arrive, applying the SAME normalization that was used in
training (loaded from the checkpoint).

Core piece is `CSIStreamClassifier`. It is source-agnostic: feed it one 64-dim
frame at a time via .push(frame). It returns a prediction once the window is full,
respecting the configured stride, and debounces fall detections.

Wire it to your real feed like:

    clf = CSIStreamClassifier("csi_cnn.pt")
    for frame in my_csi_source():          # frame: list/array of 64 floats
        result = clf.push(frame)
        if result and result["fall_alarm"]:
            trigger_alert()

Run this file directly for a simulated-stream demo:
    python predict_stream.py --checkpoint csi_cnn.pt
"""

import argparse
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

# Import the model definition so the checkpoint's weights map onto the same class.
from csi_cnn_train import CSICNN


class CSIStreamClassifier:
    """Sliding-window classifier over a live CSI frame stream."""

    def __init__(self, checkpoint_path, device=None, stride=5,
                 fall_threshold=0.6, fall_consecutive=2):
        """
        stride            : run inference every `stride` new frames once the
                            window is full (1 = every frame). At 10 Hz, 5 -> 0.5 s.
        fall_threshold    : min softmax prob on 'fall' to count a window as a fall.
        fall_consecutive  : number of consecutive fall windows required to raise
                            an alarm (debounce against single-window false positives).
        """
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)

        self.classes = ckpt["classes"]
        self.time_steps = ckpt["time_steps"]
        self.num_subcarriers = ckpt["num_subcarriers"]
        # Stored as (1,1,S) in training; flatten to (S,) so it broadcasts over (T,S).
        self.mean = np.asarray(ckpt["mean"], dtype=np.float32).reshape(-1)
        self.std = np.asarray(ckpt["std"], dtype=np.float32).reshape(-1)

        self.model = CSICNN(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.stride = max(1, int(stride))
        self.fall_threshold = fall_threshold
        self.fall_consecutive = max(1, int(fall_consecutive))
        try:
            self.fall_idx = self.classes.index("fall")
        except ValueError:
            self.fall_idx = None

        self._buffer = deque(maxlen=self.time_steps)
        self._since_pred = 0
        self._fall_streak = 0

    def reset(self):
        """Clear the window and debounce state (e.g. after a stream gap)."""
        self._buffer.clear()
        self._since_pred = 0
        self._fall_streak = 0

    @torch.no_grad()
    def _infer(self):
        window = np.asarray(self._buffer, dtype=np.float32)   # (T, S)
        window = (window - self.mean) / self.std
        x = torch.from_numpy(window).unsqueeze(0).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=1).cpu().numpy()[0]
        return probs

    def push(self, frame):
        """
        Add one frame (length == num_subcarriers) to the window.

        Returns None while warming up or between strides, otherwise a dict:
          {label, confidence, probs, fall_alarm, n_frames}
        """
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.size != self.num_subcarriers:
            raise ValueError(
                f"Frame has {frame.size} values, expected {self.num_subcarriers}."
            )
        self._buffer.append(frame)

        # Warming up: not enough history for a full 4 s window yet.
        if len(self._buffer) < self.time_steps:
            return None

        # Respect stride: only predict every `stride`-th new frame.
        self._since_pred += 1
        if self._since_pred < self.stride:
            return None
        self._since_pred = 0

        probs = self._infer()
        top = int(probs.argmax())

        fall_alarm = False
        if self.fall_idx is not None:
            if probs[self.fall_idx] >= self.fall_threshold:
                self._fall_streak += 1
            else:
                self._fall_streak = 0
            fall_alarm = self._fall_streak >= self.fall_consecutive

        return {
            "label": self.classes[top],
            "confidence": float(probs[top]),
            "probs": {c: float(p) for c, p in zip(self.classes, probs)},
            "fall_alarm": fall_alarm,
            "n_frames": len(self._buffer),
        }


# ----------------------------------------------------------------------------
# Demo: simulate a 10 Hz stream so you can see the windowing behavior.
# Replace `simulated_stream` with your real frame source.
# ----------------------------------------------------------------------------
def simulated_stream(n_frames=120, num_subcarriers=64, seed=1):
    rng = np.random.default_rng(seed)
    # walk for a bit, then a fall, to show the alarm debounce trigger.
    schedule = [("walk", 60), ("fall", 60)]
    class_idx = {"fall": 0, "walk": 1, "lay": 2, "sit": 3}
    for activity, count in schedule:
        base = rng.normal(0, 1, num_subcarriers) + class_idx[activity]
        for _ in range(count):
            yield base + rng.normal(0, 0.5, num_subcarriers)


def main():
    ap = argparse.ArgumentParser(description="Streaming CSI inference demo")
    ap.add_argument("--checkpoint", default="csi_cnn.pt")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--fall-threshold", type=float, default=0.6)
    ap.add_argument("--fall-consecutive", type=int, default=2)
    ap.add_argument("--realtime", action="store_true",
                    help="sleep 0.1 s per frame to mimic a true 10 Hz feed")
    args = ap.parse_args()

    clf = CSIStreamClassifier(
        args.checkpoint, stride=args.stride,
        fall_threshold=args.fall_threshold, fall_consecutive=args.fall_consecutive,
    )
    print(f"Classes: {clf.classes} | window = {clf.time_steps} frames "
          f"({clf.time_steps / 10:.0f}s @10Hz) | stride = {clf.stride}")

    for i, frame in enumerate(simulated_stream(num_subcarriers=clf.num_subcarriers)):
        if args.realtime:
            time.sleep(0.1)
        res = clf.push(frame)
        if res is None:
            continue
        flag = "  <<< FALL ALARM" if res["fall_alarm"] else ""
        print(f"frame {i:3d} | {res['label']:5s} "
              f"({res['confidence']:.2f}){flag}")


if __name__ == "__main__":
    main()
