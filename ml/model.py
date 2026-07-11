"""The anomaly-detection autoencoder (DCASE 2020 Task 2 baseline topology).

A symmetric dense autoencoder: FEATURE_DIM -> [HIDDEN]*DEPTH -> BOTTLENECK ->
[HIDDEN]*DEPTH -> FEATURE_DIM, trained to reconstruct log-mel vectors of *normal*
machine sound. Reconstruction MSE is the anomaly score.
"""
import config
import tensorflow as tf
from tensorflow.keras import Model, layers


def _dense_block(x, units):
    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def build_autoencoder() -> Model:
    inp = layers.Input(shape=(config.FEATURE_DIM,), name="log_mel")
    x = inp
    for _ in range(config.DEPTH):
        x = _dense_block(x, config.HIDDEN)
    x = _dense_block(x, config.BOTTLENECK)          # bottleneck
    for _ in range(config.DEPTH):
        x = _dense_block(x, config.HIDDEN)
    out = layers.Dense(config.FEATURE_DIM, name="recon")(x)

    model = Model(inp, out, name="machinesense_ae")
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LR), loss="mse")
    return model
