"""
populate_snowflake.py

Uploads all 1,200 NTU-Fi HAR sequences to Snowflake as the baseline
training dataset. Uses a gzipped JSON Lines file + Snowflake stage for
bulk loading (much faster than individual inserts).

Runtime: ~5-15 min depending on network speed.

Usage:
  export $(cat .env | xargs) && python3 populate_snowflake.py
"""

import os, json, gzip, uuid, tempfile
import numpy as np
import scipy.io as sio
import snowflake.connector

BASE      = os.path.dirname(os.path.abspath(__file__))
NTUFI_DIR = os.path.join(BASE, "NTU-Fi_HAR", "NTU-Fi_HAR")

N_ANT       = 3
N_SUB_TOTAL = 114
N_TIME      = 2000
N_SUB       = 64    # subcarriers to keep (antenna 0, first 64 of 114)

ACTIVITIES = {
    "box":    "no_fall",
    "circle": "no_fall",
    "clean":  "no_fall",
    "fall":   "fall",
    "run":    "no_fall",
    "walk":   "no_fall",
}


def load_csi(path):
    mat = sio.loadmat(path)
    csi = mat["CSIamp"].astype(np.float32)           # (342, 2000)
    csi = csi.reshape(N_ANT, N_SUB_TOTAL, N_TIME)
    return csi[0, :N_SUB, :].T                       # (2000, 64)


def get_conn():
    return snowflake.connector.connect(
        account   = os.environ["SNOWFLAKE_ACCOUNT"],
        user      = os.environ["SNOWFLAKE_USER"],
        password  = os.environ["SNOWFLAKE_PASSWORD"],
        warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", "FALL_DETECTION"),
        database  = "ghostnet",
        schema    = "fall_detect",
    )


def main():
    conn = get_conn()
    cur  = conn.cursor()

    # ── 1. Insert sessions (one per activity) ────────────────────────────────
    print("=== Step 1: Insert recording sessions ===")
    session_ids = {}
    session_rows = []
    for activity, label in ACTIVITIES.items():
        sid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"ntufi-{activity}"))
        session_ids[activity] = sid
        n_files = sum(
            len([f for f in os.listdir(os.path.join(NTUFI_DIR, s, activity))
                 if f.endswith(".mat")])
            for s in ("train_amp", "test_amp")
            if os.path.isdir(os.path.join(NTUFI_DIR, s, activity))
        )
        session_rows.append((
            sid, label,
            "NTU-Fi HAR lab environment",
            "NTU-Fi public dataset",
            f"NTU-Fi HAR activity: {activity}",
            n_files,
        ))

    # Delete existing NTU-Fi sessions to make re-runs idempotent
    ntufi_sids = [r[0] for r in session_rows]
    for sid in ntufi_sids:
        cur.execute("DELETE FROM csi_recordings    WHERE session_id = %s", (sid,))
        cur.execute("DELETE FROM recording_sessions WHERE session_id = %s", (sid,))

    cur.executemany(
        "INSERT INTO recording_sessions "
        "(session_id, label, location, subject, notes, n_sequences) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        session_rows,
    )
    print(f"  Inserted {len(session_rows)} sessions: {list(ACTIVITIES.keys())}")

    # ── 2. Write sequences to gzipped JSON Lines file ────────────────────────
    tmp_path = os.path.join(tempfile.gettempdir(), "ntufi_data.jsonl.gz")
    print(f"\n=== Step 2: Writing sequences to {tmp_path} ===")
    print("  (this writes ~1 GB uncompressed; gzip will reduce it 3-5x)")

    count = 0
    with gzip.open(tmp_path, "wt", encoding="utf-8") as f:
        for split in ("train_amp", "test_amp"):
            for activity, label in ACTIVITIES.items():
                act_dir = os.path.join(NTUFI_DIR, split, activity)
                if not os.path.isdir(act_dir):
                    continue
                sid = session_ids[activity]
                for fname in sorted(x for x in os.listdir(act_dir) if x.endswith(".mat")):
                    csi = load_csi(os.path.join(act_dir, fname))
                    record = {
                        "session_id":    sid,
                        "label":         label,
                        "n_frames":      N_TIME,
                        "n_subcarriers": N_SUB,
                        "frames":        csi.tolist(),
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1
                    if count % 100 == 0:
                        size_mb = os.path.getsize(tmp_path) / 1e6
                        print(f"  {count}/1200 sequences  ({size_mb:.0f} MB compressed so far)")

    size_mb = os.path.getsize(tmp_path) / 1e6
    print(f"  Done. {count} sequences, {size_mb:.1f} MB compressed.")

    # ── 3. Create internal stage + PUT ───────────────────────────────────────
    print("\n=== Step 3: Upload to Snowflake stage ===")
    cur.execute("CREATE STAGE IF NOT EXISTS ntufi_stage")
    cur.execute(f"PUT 'file://{tmp_path}' @ntufi_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
    rows = cur.fetchall()
    print(f"  PUT result: {rows[0][6] if rows else 'unknown'}")   # status column

    # ── 4. COPY INTO csi_recordings ──────────────────────────────────────────
    print("\n=== Step 4: COPY INTO csi_recordings ===")
    cur.execute("""
        COPY INTO csi_recordings
            (session_id, label, n_frames, n_subcarriers, frames)
        FROM (
            SELECT
                $1:session_id::VARCHAR(36),
                $1:label::VARCHAR(10),
                $1:n_frames::INTEGER,
                $1:n_subcarriers::INTEGER,
                $1:frames
            FROM @ntufi_stage/ntufi_data.jsonl.gz
        )
        FILE_FORMAT = (TYPE = 'JSON' COMPRESSION = 'GZIP')
    """)
    copy_result = cur.fetchall()
    print(f"  COPY result: {copy_result}")

    # ── 5. Verify ─────────────────────────────────────────────────────────────
    print("\n=== Verification ===")
    cur.execute("SELECT * FROM dataset_summary")
    for row in cur.fetchall():
        print(f"  label={row[0]}  n_sequences={row[1]}  avg_frames={row[2]}")

    cur.close()
    conn.close()
    os.remove(tmp_path)
    print("\nDone. Snowflake is populated with NTU-Fi baseline data.")


if __name__ == "__main__":
    main()
