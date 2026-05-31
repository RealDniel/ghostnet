"""
model.py

1D CNN for fall detection from raw CSI amplitude.
Input: [Batch x WINDOW x 64] — normalized subcarrier amplitudes.
"""

import tensorflow as tf


def build_model(T=100, C=64, n_classes=2):
    inputs = tf.keras.Input(shape=(T, C))

    x = tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu")(inputs)
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

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


if __name__ == "__main__":
    m = build_model()
    m.summary()
