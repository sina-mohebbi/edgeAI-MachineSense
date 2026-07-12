"""Quantize each per-machine-ID autoencoder to int8 (mirrors export_tflite.py).

Reads ml/artifacts/per_id/<id>/{model.keras,rep_vectors.npy} (written by
train_per_id.py) and writes, per ID:
    model_int8.tflite   -- the deployable model for that specific machine
    model_data.cc / .h  -- C byte array, variable name g_model_data_<id>

A real deployment binds one physical ESP32 to one physical machine, so it only ever
needs the .tflite/.cc for that machine's ID -- these per-ID exports are what actually
ship, unlike the pooled model which exists mainly for the evaluation comparison.
"""
from __future__ import annotations

import config
import numpy as np
import tensorflow as tf
from export_tflite import tflite_to_c_header, to_int8_tflite


def main() -> None:
    if not config.PER_ID_ARTIFACTS.exists():
        raise FileNotFoundError("Run train_per_id.py first (no artifacts/per_id/).")

    id_dirs = sorted(d for d in config.PER_ID_ARTIFACTS.iterdir() if d.is_dir())
    if not id_dirs:
        raise FileNotFoundError(f"No per-ID model directories under {config.PER_ID_ARTIFACTS}")

    for out_dir in id_dirs:
        model_path = out_dir / "model.keras"
        rep_path = out_dir / "rep_vectors.npy"
        if not model_path.exists() or not rep_path.exists():
            print(f"[skip] {out_dir.name}: missing model.keras or rep_vectors.npy")
            continue

        model = tf.keras.models.load_model(model_path)
        rep = np.load(rep_path)
        tflite_bytes = to_int8_tflite(model, rep)

        (out_dir / "model_int8.tflite").write_bytes(tflite_bytes)
        var_name = f"{config.C_VAR_NAME}_{out_dir.name}"
        tflite_to_c_header(tflite_bytes, var_name, out_dir / "model_data.cc")

        print(f"[saved] {out_dir.name}: model_int8.tflite "
              f"({len(tflite_bytes) / 1024:.1f} KiB), model_data.cc ({var_name})")

    print("[next] python evaluate_per_id_tflite.py")


if __name__ == "__main__":
    main()
