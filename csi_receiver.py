import json
import math
import os
import socket
import struct
from datetime import datetime, timezone

import snowflake.connector
from dotenv import load_dotenv

load_dotenv()

def sf_conn():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        database="ghostnet",
        schema="public",
    )

def insert_csi(ts, board_id, rssi, n_sub, amplitudes):
    try:
        with sf_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO csi_raw (timestamp, board_id, rssi, subcarriers, amplitudes) "
                "SELECT %s, %s, %s, %s, PARSE_JSON(%s)",
                (ts, str(board_id), rssi, n_sub, json.dumps(amplitudes))
            )
    except Exception as e:
        print(f"Snowflake insert failed: {e}", flush=True)

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp.bind(('', 5005))
print("Receiving CSI...")

while True:
    data, addr = udp.recvfrom(4096)
    magic = struct.unpack_from('<I', data, 0)[0]
    if magic != 0xC5110001:
        continue
    node_id = data[4]
    n_sub = struct.unpack_from('<H', data, 6)[0]
    rssi = struct.unpack_from('b', data, 16)[0]
    iq = data[20:]
    amplitudes = []
    for k in range(n_sub):
        i = struct.unpack_from('b', iq, k*2)[0]
        q = struct.unpack_from('b', iq, k*2+1)[0]
        amplitudes.append(round(math.sqrt(i*i + q*q), 2))
    ts = datetime.now(timezone.utc).isoformat()
    print(f"Board {node_id}  rssi={rssi}  subcarriers={n_sub}  amp[0..4]={[f'{a:.1f}' for a in amplitudes[:4]]}", flush=True)
    insert_csi(ts, node_id, rssi, n_sub, amplitudes)