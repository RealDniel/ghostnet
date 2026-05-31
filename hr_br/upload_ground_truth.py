import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timezone
from dotenv import load_dotenv
import snowflake.connector

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

def sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )

def parse_file(filepath, date_str, metric):
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            time_str, val_str = line.split(",")
            time_str = time_str.strip()
            val = float(val_str.strip())

            # Parse time — handle H:MM and HH:MM
            parts = time_str.split(":")
            hour   = int(parts[0])
            minute = int(parts[1])

            ts = datetime.strptime(f"{date_str} {hour:02d}:{minute:02d}:00", "%Y-%m-%d %H:%M:%S")
            ts = ts.replace(tzinfo=timezone.utc)
            rows.append((ts.isoformat(), metric, val))
    return rows

def upload():
    hr_rows = parse_file(
        os.path.join(os.path.dirname(__file__), "hr.txt"),
        "2026-05-30", "hr"
    )
    br_rows = parse_file(
        os.path.join(os.path.dirname(__file__), "br.txt"),
        "2026-05-31", "br"
    )

    all_rows = hr_rows + br_rows
    print(f"Uploading {len(hr_rows)} HR readings and {len(br_rows)} BR readings...")

    with sf_conn() as conn:
        cur = conn.cursor()
        for row in all_rows:
            cur.execute(
                "INSERT INTO ground_truth (timestamp, metric, value) VALUES (%s, %s, %s)",
                row
            )

    print("Done. Verify with:")
    print("  SELECT metric, COUNT(*), MIN(timestamp), MAX(timestamp)")
    print("  FROM ghostnet.public.ground_truth GROUP BY metric;")

if __name__ == "__main__":
    upload()
