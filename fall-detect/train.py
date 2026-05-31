"""
train.py

Trains the fall detection CNN on NTU-Fi raw CSI data (64 subcarriers).
Saves the best model to models/fall_cnn.keras and the scaler to models/scaler.npy.

Usage:
  python3 train.py
  python3 train.py --epochs 150 --batch 64
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from preprocess import build_dataset
from model import build_model

BASE       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "fall_cnn.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.npy")

FALL_CLASS = 1


def main(epochs, batch_size):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=== Loading and preprocessing NTU-Fi data ===")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = build_dataset()

    # Save scaler params for inference
    np.save(SCALER_PATH, {"mean": scaler.mean_, "scale": scaler.scale_})
    print(f"Scaler saved → {SCALER_PATH}")

    # Class weights: fall is rare (~1:5), penalize misses heavily
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    # Extra penalty on fall (safety critical)
    weights[FALL_CLASS] *= 3.0
    class_weight = dict(zip(classes, weights))
    print(f"Class weights: {class_weight}")

    print(f"\n=== Training ({epochs} epochs, batch {batch_size}) ===")
    model = build_model(T=X_train.shape[1], C=X_train.shape[2])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(class_id=FALL_CLASS, name="recall"),
        ],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_recall", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall", mode="max",
            patience=15, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_recall", mode="max",
            factor=0.5, patience=7, verbose=1,
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    print("\n=== Test set evaluation ===")
    model = tf.keras.models.load_model(MODEL_PATH)
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)

    tp = np.sum((preds == 1) & (y_test == 1))
    fn = np.sum((preds == 0) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    tn = np.sum((preds == 0) & (y_test == 0))

    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    print(f"  Recall (sensitivity): {recall:.3f}  {'PASS' if recall > 0.95 else 'FAIL'} (target > 0.95)")
    print(f"  Specificity:          {specificity:.3f}  {'PASS' if specificity > 0.90 else 'FAIL'} (target > 0.90)")
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch",  type=int, default=64)
    args = parser.parse_args()
    main(args.epochs, args.batch)
