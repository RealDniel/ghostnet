"""
setup_snowflake.py

Creates the ghostnet database, schema, tables, and views in Snowflake.
Safe to re-run — all statements use IF NOT EXISTS / CREATE OR REPLACE.

Usage:
  export SNOWFLAKE_ACCOUNT=... SNOWFLAKE_USER=... SNOWFLAKE_PASSWORD=...
  python3 setup_snowflake.py
"""

import snowflake.connector
import os

ACCOUNT   = os.environ["SNOWFLAKE_ACCOUNT"]
USER      = os.environ["SNOWFLAKE_USER"]
PASSWORD  = os.environ["SNOWFLAKE_PASSWORD"]
WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "compute_wh")

STATEMENTS = [
    # ── Database & schema ─────────────────────────────────────────────────────
    "CREATE DATABASE IF NOT EXISTS ghostnet",
    "USE DATABASE ghostnet",
    "CREATE SCHEMA IF NOT EXISTS ghostnet.fall_detect",
    "USE SCHEMA ghostnet.fall_detect",

    # ── Sessions table ────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS recording_sessions (
        session_id   VARCHAR(36)   NOT NULL,
        label        VARCHAR(10)   NOT NULL,
        recorded_at  TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
        location     VARCHAR(200),
        subject      VARCHAR(100),
        notes        VARCHAR(1000),
        n_sequences  INTEGER       DEFAULT 0,
        CONSTRAINT pk_sessions PRIMARY KEY (session_id),
        CONSTRAINT chk_label CHECK (label IN ('fall', 'no_fall'))
    )
    """,

    # ── CSI recordings table ──────────────────────────────────────────────────
    # One row per labeled sequence.
    # frames: JSON array of arrays  [[f0..f63], [f0..f63], ...]  (T x 64)
    """
    CREATE TABLE IF NOT EXISTS csi_recordings (
        recording_id  INTEGER       NOT NULL AUTOINCREMENT PRIMARY KEY,
        session_id    VARCHAR(36)   NOT NULL,
        label         VARCHAR(10)   NOT NULL,
        recorded_at   TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
        n_frames      INTEGER       NOT NULL,
        n_subcarriers INTEGER       DEFAULT 64,
        frames        VARIANT       NOT NULL,
        CONSTRAINT fk_session  FOREIGN KEY (session_id)
            REFERENCES recording_sessions (session_id),
        CONSTRAINT chk_rec_label CHECK (label IN ('fall', 'no_fall'))
    )
    """,

    # Cluster by label so training queries scan less data
    "ALTER TABLE csi_recordings CLUSTER BY (label)",

    # ── Training view ─────────────────────────────────────────────────────────
    """
    CREATE OR REPLACE VIEW training_data AS
    SELECT
        recording_id,
        session_id,
        label,
        CASE label WHEN 'fall' THEN 1 ELSE 0 END AS label_int,
        recorded_at,
        n_frames,
        frames
    FROM csi_recordings
    ORDER BY recorded_at
    """,

    # ── Summary view ──────────────────────────────────────────────────────────
    """
    CREATE OR REPLACE VIEW dataset_summary AS
    SELECT
        label,
        COUNT(*)          AS n_sequences,
        ROUND(AVG(n_frames), 1) AS avg_frames,
        MIN(recorded_at)  AS first_recorded,
        MAX(recorded_at)  AS last_recorded
    FROM csi_recordings
    GROUP BY label
    """,
]


def main():
    print(f"Connecting to Snowflake ({ACCOUNT})...")
    conn = snowflake.connector.connect(
        account=ACCOUNT,
        user=USER,
        password=PASSWORD,
        warehouse=WAREHOUSE,
    )
    cur = conn.cursor()

    for stmt in STATEMENTS:
        stmt = stmt.strip()
        if not stmt:
            continue
        label = stmt.split("\n")[0][:60]
        print(f"  {label}...")
        cur.execute(stmt)

    # Verify
    cur.execute("SHOW TABLES IN SCHEMA ghostnet.fall_detect")
    tables = [row[1] for row in cur.fetchall()]
    print(f"\nTables created: {tables}")

    cur.execute("SELECT * FROM ghostnet.fall_detect.dataset_summary")
    rows = cur.fetchall()
    print(f"dataset_summary: {rows if rows else '(empty — no data yet)'}")

    cur.close()
    conn.close()
    print("\nSetup complete.")


if __name__ == "__main__":
    main()
