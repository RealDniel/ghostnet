import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
import snowflake.connector

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

_DATA = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
FILES = [
    os.path.join(_DATA, "fall_data.jsonl"),
    os.path.join(_DATA, "lying_data.jsonl"),
    os.path.join(_DATA, "sitting_data.jsonl"),
    os.path.join(_DATA, "walking_data.jsonl"),
    os.path.join(_DATA, "fall-csi.jsonl"),
    os.path.join(_DATA, "walk-csi.jsonl"),
]

def sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )

def upload():
    base = os.path.join(os.path.dirname(__file__), '..', '..')
    all_rows = []

    for filename in FILES:
        path = os.path.join(base, filename)
        if not os.path.exists(path):
            print(f"  Skipping {filename} — not found")
            continue
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                # Our format: label, timestamp, n_samples, amplitudes
                if "amplitudes" in entry:
                    amps = entry["amplitudes"]
                    # amplitudes is a list of lists (one per timestep)
                    if isinstance(amps[0], list):
                        for frame in amps:
                            all_rows.append((
                                entry["label"],
                                entry["timestamp"],
                                len(frame),
                                json.dumps(frame),
                            ))
                    else:
                        all_rows.append((
                            entry["label"],
                            entry["timestamp"],
                            len(amps),
                            json.dumps(amps),
                        ))

                # Teammate format: csi, posture, t
                elif "csi" in entry:
                    label = entry.get("posture", "unknown")
                    ts    = entry.get("t", 0)
                    amps  = entry["csi"]
                    all_rows.append((
                        label,
                        str(ts),
                        len(amps),
                        json.dumps(amps),
                    ))

                count += 1
        print(f"  Loaded {filename} ({count} entries)")

    if not all_rows:
        print("No data found to upload.")
        return

    print(f"\nUploading {len(all_rows)} samples to Snowflake...")

    with sf_conn() as conn:
        cur = conn.cursor()
        for i, row in enumerate(all_rows):
            cur.execute(
                "INSERT INTO pose_data (label, timestamp, n_samples, amplitudes) "
                "SELECT %s, %s, %s, PARSE_JSON(%s)",
                row
            )
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(all_rows)} uploaded")

    print(f"\nDone. {len(all_rows)} samples uploaded.")
    print("Verify with:")
    print("  SELECT label, COUNT(*) FROM ghostnet.public.pose_data GROUP BY label;")

if __name__ == "__main__":
    upload()
