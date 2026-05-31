"""
gen_csi.py — bake the demo scenes into CSI data files (run once, commit the output).

Produces JSONL "capture" files where each line is one frame:
  { "t", "node_id", "csi": [64 amplitudes], "position": {x,y,z},
    "motion", "posture", "impact", "behind_wall" }

These are the CSI data files the demo streams (csi_position.py / csi_fall.py).
The CSI amplitudes are synthetic-but-physically-shaped (variance tracks motion); the
position is the captured/authored walk path. See PRD.md for the honesty framing.

    python tools/gen_csi.py     # writes data/walk-csi.jsonl and data/fall-csi.jsonl
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as m  # reuse the scene math + CSI synthesis

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def bake(name: str, duration: float, sampler):
    """sampler(t) -> (position(x,y,z), motion, vertical_speed, posture)."""
    path = os.path.join(OUT, name)
    n = int(duration * m.FPS)
    with open(path, "w") as f:
        for i in range(n):
            t = i * m.DT
            (x, y, z), motion, vspeed, posture = sampler(t)
            behind = x > m.ROOM_W
            impact = vspeed
            frame = {
                "t": round(t, 3),
                "node_id": 1,
                "csi": m.synth_csi(motion, behind, impact, pos_x=x),
                "position": {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)},
                "motion": round(motion, 3),
                "posture": posture,
                "impact": round(impact, 3),
                "behind_wall": behind,
            }
            f.write(json.dumps(frame) + "\n")
    print(f"wrote {path}  ({n} frames, {duration:.0f}s)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    # Walk: replays demo/walk-session.json (wander -> pause -> exit through wall).
    bake("walk-csi.jsonl", m._WALK_LOOP or 24.0, m.scene_walk)
    # Fall: stand -> slow lie-down (no alert) -> get up -> fall -> stays down through the
    # grace countdown so the caregiver call fires. 32 s leaves room for the default 5 s grace.
    bake("fall-csi.jsonl", 32.0, m.scene_fall)
