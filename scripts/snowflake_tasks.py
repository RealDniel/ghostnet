"""
Sets up Snowflake stored procedures + Tasks for GhostNet.

Creates:
  - STREAM    csi_raw_stream        (detects new rows in csi_raw)
  - PROCEDURE ghostnet_process()    (computes vitals + fall detection via numpy/scipy)
  - TASK      ghostnet_task         (calls the procedure every minute when stream has data)
  - PROCEDURE ghostnet_cleanup()    (deletes data older than 22 days)
  - TASK      ghostnet_cleanup_task (runs daily at 03:00 UTC)

Run once:
    python scripts/snowflake_tasks.py
"""

import os
from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

PROCEDURE_BODY = '''
import numpy as np
from scipy.signal import butter, filtfilt, welch
from datetime import datetime, timezone
import json

def run(session):
    SAMPLE_RATE   = 10
    WINDOW_VITALS = 50
    WINDOW_FALL   = 30
    HR_LOW        = 50.0
    BR_LOW        = 8.0

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

    # ── Fetch most recent real frames ────────────────────────────────────────
    rows = session.sql("""
        SELECT timestamp, amplitudes
        FROM csi_raw
        WHERE board_id != 'synthetic'
        ORDER BY timestamp DESC
        LIMIT 60
    """).collect()

    if len(rows) < 10:
        return "insufficient_data"

    frames = []
    for row in reversed(rows):
        try:
            raw = row["AMPLITUDES"]
            if isinstance(raw, str):
                amps = json.loads(raw)
            else:
                amps = list(raw) if raw is not None else []
            if len(amps) > 0:
                frames.append(np.array(amps, dtype=np.float32))
        except Exception:
            continue

    if len(frames) < 10:
        return "parse_error"

    results = []

    # ── Active subcarrier signal ─────────────────────────────────────────────
    def active_signal(window_arr):
        mask = np.mean(window_arr, axis=0) > 1.0
        active = window_arr[:, mask]
        if active.shape[1] == 0:
            return None
        return np.mean(active, axis=1)

    def bandpass(data, low, high, fs, order=4):
        nyq = fs / 2
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, data)

    def dominant_freq(signal, fs):
        nperseg = min(len(signal), 128)
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
        return float(freqs[np.argmax(psd)])

    # ── Vitals ───────────────────────────────────────────────────────────────
    if len(frames) >= WINDOW_VITALS:
        arr = np.stack(frames[-WINDOW_VITALS:])
        sig = active_signal(arr)
        if sig is not None:
            try:
                br_bpm = round(float(np.clip(
                    dominant_freq(bandpass(sig, 0.1, 0.5, SAMPLE_RATE), SAMPLE_RATE) * 60,
                    4, 40)), 1)
                hr_bpm = round(float(np.clip(
                    dominant_freq(bandpass(sig, 0.8, 2.0, SAMPLE_RATE), SAMPLE_RATE) * 60,
                    40, 180)), 1)

                # Dedup: skip if a vitals row was written in the last 25 seconds
                recent = session.sql(f"""
                    SELECT COUNT(*) AS cnt FROM vitals_labels
                    WHERE timestamp > DATEADD(second, -25, '{ts}'::TIMESTAMP_TZ)
                """).collect()
                if recent[0]["CNT"] == 0:
                    session.sql(f"""
                        INSERT INTO vitals_labels (timestamp, hr, br)
                        VALUES ('{ts}'::TIMESTAMP_TZ, {hr_bpm}, {br_bpm})
                    """).collect()
                    results.append(f"vitals HR={hr_bpm} BR={br_bpm}")

                    # Low HR alert
                    if hr_bpm < HR_LOW:
                        session.sql(f"""
                            INSERT INTO events (event, timestamp, confidence, heart_rate_bpm, breathing_rate_bpm)
                            VALUES ('low_heart_rate', '{ts}'::TIMESTAMP_TZ, 1.0, {hr_bpm}, {br_bpm})
                        """).collect()
                        results.append("alert:low_hr")

                    # Low BR alert
                    if br_bpm < BR_LOW:
                        session.sql(f"""
                            INSERT INTO events (event, timestamp, confidence, heart_rate_bpm, breathing_rate_bpm)
                            VALUES ('low_breathing_rate', '{ts}'::TIMESTAMP_TZ, 1.0, {hr_bpm}, {br_bpm})
                        """).collect()
                        results.append("alert:low_br")

            except Exception as e:
                results.append(f"vitals_err:{e}")

    # ── Fall detection ───────────────────────────────────────────────────────
    if len(frames) >= WINDOW_FALL:
        arr = np.stack(frames[-WINDOW_FALL:])
        sig = active_signal(arr)
        if sig is not None:
            recent_var = float(np.var(sig[-10:]))
            prior_var  = float(np.var(sig[-30:-10])) + 1e-6
            ratio = recent_var / prior_var
            if ratio > 8.0:
                confidence = round(min(ratio / 20.0, 1.0), 2)
                # Dedup: no fall alert in last 10 seconds
                recent_fall = session.sql(f"""
                    SELECT COUNT(*) AS cnt FROM events
                    WHERE event = 'fall_detected'
                    AND timestamp > DATEADD(second, -10, '{ts}'::TIMESTAMP_TZ)
                """).collect()
                if recent_fall[0]["CNT"] == 0:
                    session.sql(f"""
                        INSERT INTO events (event, timestamp, confidence, heart_rate_bpm, breathing_rate_bpm)
                        VALUES ('fall_detected', '{ts}'::TIMESTAMP_TZ, {confidence}, NULL, NULL)
                    """).collect()
                    results.append(f"fall:conf={confidence}")

    return "; ".join(results) if results else "ok_no_action"
'''


def setup():
    conn = snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )
    cur = conn.cursor()

    steps = [
        # Stream — detects new appended rows in csi_raw
        ("Creating stream on csi_raw", """
            CREATE STREAM IF NOT EXISTS csi_raw_stream
            ON TABLE csi_raw
            APPEND_ONLY = TRUE
        """),

        # Stored procedure
        ("Creating stored procedure ghostnet_process", f"""
            CREATE OR REPLACE PROCEDURE ghostnet_process()
            RETURNS STRING
            LANGUAGE PYTHON
            RUNTIME_VERSION = '3.9'
            PACKAGES = ('numpy', 'scipy', 'snowflake-snowpark-python')
            HANDLER = 'run'
            AS $$
{PROCEDURE_BODY}
            $$
        """),

        # Task — runs every minute when stream has new data
        ("Creating task ghostnet_task", """
            CREATE OR REPLACE TASK ghostnet_task
                WAREHOUSE = 'COMPUTE_WH'
                SCHEDULE  = '1 MINUTE'
                WHEN SYSTEM$STREAM_HAS_DATA('csi_raw_stream')
            AS
                CALL ghostnet_process()
        """),

        # Tasks start suspended — resume it
        ("Resuming task", "ALTER TASK ghostnet_task RESUME"),

        # Cleanup procedure — deletes rows older than 22 days from all tables
        ("Creating stored procedure ghostnet_cleanup", """
            CREATE OR REPLACE PROCEDURE ghostnet_cleanup()
            RETURNS STRING
            LANGUAGE PYTHON
            RUNTIME_VERSION = '3.9'
            PACKAGES = ('snowflake-snowpark-python')
            HANDLER = 'run'
            AS $$
def run(session):
    # csi_raw grows at ~10 rows/sec — keep 22 days only
    # vitals_labels and events are small — keep 61 days (frontend shows 60)
    plan = [
        ("csi_raw",       22),
        ("vitals_labels", 61),
        ("events",        61),
    ]
    deleted = {}
    for tbl, days in plan:
        cutoff = f"DATEADD(day, -{days}, CURRENT_TIMESTAMP())"
        result = session.sql(
            f"DELETE FROM {tbl} WHERE timestamp < {cutoff}"
        ).collect()
        deleted[tbl] = result[0][0] if result else 0
    summary = ", ".join(f"{t}:{n}" for t, n in deleted.items())
    return f"cleanup ok — {summary} rows deleted"
            $$
        """),

        # Cleanup task — daily at 03:00 UTC
        ("Creating task ghostnet_cleanup_task", """
            CREATE OR REPLACE TASK ghostnet_cleanup_task
                WAREHOUSE = 'COMPUTE_WH'
                SCHEDULE  = 'USING CRON 0 3 * * * UTC'
            AS
                CALL ghostnet_cleanup()
        """),

        ("Resuming cleanup task", "ALTER TASK ghostnet_cleanup_task RESUME"),
    ]

    for label, sql in steps:
        print(f"{label}...", end=" ", flush=True)
        try:
            cur.execute(sql.strip())
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    # Verify both tasks
    print()
    for task_name in ("ghostnet_task", "ghostnet_cleanup_task"):
        cur.execute(f"SHOW TASKS LIKE '{task_name}'")
        rows = cur.fetchall()
        if rows:
            cols = [d[0] for d in cur.description]
            row = dict(zip(cols, rows[0]))
            state    = row.get("state",    row.get("STATE",    "unknown"))
            schedule = row.get("schedule", row.get("SCHEDULE", "unknown"))
            print(f"  {task_name}: {state}  ({schedule})")
        else:
            print(f"  {task_name}: not found")

    cur.close()
    conn.close()
    print("\nSetup complete.")
    print("  ghostnet_task       — runs every minute when new CSI data arrives")
    print("  ghostnet_cleanup_task — runs daily at 03:00 UTC, deletes data >22 days old")
    print("\nTo test manually:")
    print("  CALL ghostnet_process();")
    print("  CALL ghostnet_cleanup();")


if __name__ == "__main__":
    setup()
