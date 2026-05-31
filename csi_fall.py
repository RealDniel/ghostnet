"""
csi_fall.py — FALL DEMO.

Streams the captured CSI fall session: the person stands, then LIES DOWN slowly
(gradual descent -> NO alert), gets up, then FALLS (sudden impact + stillness ->
ALERT). On the confirmed fall it fires `fall_detected` and places the caregiver
call (Twilio if TWILIO_* + CAREGIVER_TO env are set, otherwise a [MOCK CALL] log).

    python csi_fall.py                # streams data/fall-csi.jsonl, runs fall detection
    curl localhost:8000/trigger/fall  # stage backup: force an immediate confirmed fall
"""

from demo_stream import run

if __name__ == "__main__":
    print("GhostNet — fall demo (CSI fall stream + caregiver call)")
    run("data/fall-csi.jsonl", detect_falls=True)
