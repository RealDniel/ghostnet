"""
csi_position.py — WALKING DEMO.

Streams the captured CSI walk session into the visualizer; the blob follows the
recorded path (wander the room, pause, then walk out the door and stay sensed
through the wall). No fall alerts in this scene.

    python csi_position.py            # streams data/walk-csi.jsonl on ws://localhost:8000/ws

Then open the frontend (cd frontend && npm run dev).
"""

from demo_stream import run

if __name__ == "__main__":
    print("GhostNet — walking demo (CSI position stream)")
    run("data/walk-csi.jsonl", detect_falls=False)
