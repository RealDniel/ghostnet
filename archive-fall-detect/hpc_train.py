"""
hpc_train.py — Self-contained fall detection training script for HPC.

Pulls CSI recordings from Snowflake, preprocesses, trains a 1D CNN,
and saves the model + scaler locally.

Setup on HPC (run once):
  pip install --user tensorflow snowflake-connector-python scikit-learn scipy numpy

Run:
  python3 hpc_train.py \\
      --account  UDLCYTH-GDB50567 \\
      --user     martid24 \\
      --password 'your-password' \\
      --warehouse FALL_DETECTION

After training, copy results back:
  scp hpc:/path/to/hpc_train/models/* ./models/
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
import snowflake.connector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Snowpark Container Services injects an OAuth token — detect it at startup
_IN_CONTAINER = os.path.exists("/snowflake/session/token")

# ── Config ────────────────────────────────────────────────────────────────────

WINDOW     = 100   # timesteps per clip
STRIDE     = 10    # stride between clips
N_SUB      = 64    # subcarriers
FALL_CLASS = 1
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ── Snowflake fetch ───────────────────────────────────────────────────────────

CHUNK_SIZE  = 100   # rows per Snowflake fetch — keeps memory low
_HERE       = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE  = os.path.join(_HERE, "snowflake_cache.npy")
PARTIAL_DIR = os.path.join(_HERE, "cache_chunks")


def fetch_sequences(args):
    # Full cache present → done
    if os.path.exists(CACHE_FILE):
        print(f"Loading from full cache: {CACHE_FILE}")
        return list(np.load(CACHE_FILE, allow_pickle=True))

    os.makedirs(PARTIAL_DIR, exist_ok=True)

    # Count total rows so we know how many chunks to expect
    print("Connecting to Snowflake...")
    if _IN_CONTAINER:
        token = open("/snowflake/session/token").read()
        conn = snowflake.connector.connect(
            host          = os.environ["SNOWFLAKE_HOST"],
            account       = os.environ["SNOWFLAKE_ACCOUNT"],
            authenticator = "oauth",
            token         = token,
            warehouse     = os.environ.get("SNOWFLAKE_WAREHOUSE", "FALL_DETECTION"),
            database      = "ghostnet",
            schema        = "fall_detect",
        )
    else:
        conn = snowflake.connector.connect(
            account   = args.account,
            user      = args.user,
            password  = args.password,
            warehouse = args.warehouse,
            database  = "ghostnet",
            schema    = "fall_detect",
        )
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM training_data")
    total = cur.fetchone()[0]
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"  {total} sequences → {n_chunks} chunks of {CHUNK_SIZE}")

    sequences = []
    for chunk_idx in range(n_chunks):
        chunk_path = os.path.join(PARTIAL_DIR, f"chunk_{chunk_idx:04d}.npy")

        # Resume: if this chunk was already saved, load it instead of fetching
        if os.path.exists(chunk_path):
            chunk = list(np.load(chunk_path, allow_pickle=True))
            sequences.extend(chunk)
            print(f"  Chunk {chunk_idx+1}/{n_chunks} — loaded from disk ({len(chunk)} rows)")
            continue

        offset = chunk_idx * CHUNK_SIZE
        cur.execute(
            f"SELECT label, n_frames, frames FROM training_data "
            f"LIMIT {CHUNK_SIZE} OFFSET {offset}"
        )
        chunk = []
        for label, n_frames, frames_raw in cur:
            frames = np.array(json.loads(frames_raw), dtype=np.float32)
            chunk.append({"label": label, "data": frames})

        np.save(chunk_path, np.array(chunk, dtype=object))
        sequences.extend(chunk)
        print(f"  Chunk {chunk_idx+1}/{n_chunks} — fetched and saved ({len(chunk)} rows)")

    cur.close()
    conn.close()

    # Merge all chunks into single cache, then clean up
    np.save(CACHE_FILE, np.array(sequences, dtype=object))
    import shutil
    shutil.rmtree(PARTIAL_DIR)
    print(f"  Full cache saved → {CACHE_FILE}  ({len(sequences)} sequences)")
    return sequences


# ── Preprocessing ─────────────────────────────────────────────────────────────

def build_dataset(sequences):
    clips, labels = [], []
    for s in sequences:
        csi   = s["data"]          # (T, 64)
        label = 1 if s["label"] == "fall" else 0
        for start in range(0, len(csi) - WINDOW + 1, STRIDE):
            clips.append(csi[start:start + WINDOW])
            labels.append(label)

    X = np.array(clips,  dtype=np.float32)   # (N, WINDOW, 64)
    y = np.array(labels, dtype=np.int32)
    print(f"Clips: {len(X)}  fall={np.sum(y==1)}  no_fall={np.sum(y==0)}")

    # Stratified 70 / 15 / 15 split
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15 / 0.85, stratify=y_tmp, random_state=42)

    # Per-subcarrier z-score normalization (fit on train only)
    N_tr, T, C = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, C)).reshape(N_tr, T, C)
    X_val   = scaler.transform(X_val.reshape(-1, C)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, C)).reshape(X_test.shape)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(T=100, C=64):
    inputs = tf.keras.Input(shape=(T, C))
    x = tf.keras.layers.Conv1D(64,  7, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


# ── Training ──────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    model_path  = os.path.join(OUT_DIR, "fall_cnn.keras")
    scaler_path = os.path.join(OUT_DIR, "scaler.npy")

    sequences = fetch_sequences(args)

    print("\n=== Preprocessing ===")
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te), scaler = build_dataset(sequences)
    np.save(scaler_path, {"mean": scaler.mean_, "scale": scaler.scale_})
    print(f"Scaler saved → {scaler_path}")

    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    weights[FALL_CLASS] *= 3.0
    class_weight = dict(zip(classes, weights))
    print(f"Class weights: {class_weight}")

    print(f"\n=== Training ({args.epochs} epochs, batch {args.batch}) ===")
    model = build_model(T=X_tr.shape[1], C=X_tr.shape[2])
    model.compile(
        optimizer = tf.keras.optimizers.Adam(1e-3),
        loss      = "sparse_categorical_crossentropy",
        metrics   = ["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=15, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min",
            factor=0.5, patience=7, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger("training_log.csv"),
    ]

    model.fit(
        X_tr, y_tr,
        validation_data = (X_v, y_v),
        epochs          = args.epochs,
        batch_size      = args.batch,
        class_weight    = class_weight,
        callbacks       = callbacks,
    )

    print("\n=== Test set evaluation ===")
    model = tf.keras.models.load_model(model_path)
    preds = np.argmax(model.predict(X_te, verbose=0), axis=1)

    tp = np.sum((preds == 1) & (y_te == 1))
    fn = np.sum((preds == 0) & (y_te == 1))
    fp = np.sum((preds == 1) & (y_te == 0))
    tn = np.sum((preds == 0) & (y_te == 0))

    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  Recall:      {recall:.3f}  {'PASS' if recall > 0.95 else 'FAIL'}")
    print(f"  Specificity: {specificity:.3f}  {'PASS' if specificity > 0.90 else 'FAIL'}")

    # When running inside Snowpark Container Services the local filesystem is
    # ephemeral — upload the model files to a Snowflake stage so they persist.
    if _IN_CONTAINER:
        print("\nUploading model files to @model_stage...")
        if _IN_CONTAINER:
            token = open("/snowflake/session/token").read()
            upload_conn = snowflake.connector.connect(
                host=os.environ["SNOWFLAKE_HOST"],
                account=os.environ["SNOWFLAKE_ACCOUNT"],
                authenticator="oauth",
                token=token,
                warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "FALL_DETECTION"),
                database="ghostnet",
                schema="fall_detect",
            )
        cur = upload_conn.cursor()
        cur.execute("CREATE STAGE IF NOT EXISTS model_stage")
        for fpath in [model_path, scaler_path]:
            cur.execute(f"PUT 'file://{fpath}' @model_stage OVERWRITE=TRUE")
            print(f"  Uploaded {os.path.basename(fpath)} → @model_stage")
        cur.close()
        upload_conn.close()
        print("  Download with:  GET @ghostnet.fall_detect.model_stage ./models/")
    else:
        print(f"\nFiles saved to ./{OUT_DIR}/")
        print("  Copy back with:  scp <hpc>:<job-dir>/models/* ./models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--account",   default=None,            help="Snowflake account (not needed inside container)")
    parser.add_argument("--user",      default=None,            help="Snowflake username (not needed inside container)")
    parser.add_argument("--password",  default=None,            help="Snowflake password (not needed inside container)")
    parser.add_argument("--warehouse", default="FALL_DETECTION",help="Snowflake warehouse")
    parser.add_argument("--epochs",    type=int, default=100)
    parser.add_argument("--batch",     type=int, default=128,   help="Larger batch = faster on GPU")
    args = parser.parse_args()
    main(args)
