"""Evaluate the exported full-int8 model on MIMII WAV files."""

from __future__ import annotations

import argparse
import json

import config
import data
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from train import limit_test_files, normalize


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the int8 TFLite model")
    parser.add_argument(
        "--max-test-per-class", type=int, default=None,
        help="Limit normal and abnormal clips per machine ID",
    )
    return parser.parse_args()


class Int8Autoencoder:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        if self.input["dtype"] != np.int8 or self.output["dtype"] != np.int8:
            raise TypeError("Expected an int8-input/int8-output TFLite model")

    @staticmethod
    def _quantize(values, details):
        scale, zero_point = details["quantization"]
        if scale <= 0:
            raise ValueError("Tensor has invalid quantization scale")
        quantized = np.rint(values / scale + zero_point)
        return np.clip(quantized, -128, 127).astype(np.int8)

    @staticmethod
    def _dequantize(values, details):
        scale, zero_point = details["quantization"]
        return (values.astype(np.float32) - zero_point) * scale

    def reconstruct(self, vectors):
        reconstructions = np.empty_like(vectors, dtype=np.float32)
        for index, vector in enumerate(vectors):
            quantized = self._quantize(vector[None, :], self.input)
            self.interpreter.set_tensor(self.input["index"], quantized)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output["index"])
            reconstructions[index] = self._dequantize(output, self.output)[0]
        return reconstructions


def file_score(model, path, mean, std):
    vectors = data.cached_file_to_vectors(path)
    if len(vectors) == 0:
        return 0.0
    vectors = normalize(vectors, mean, std)
    reconstruction = model.reconstruct(vectors)
    return float(np.mean((vectors - reconstruction) ** 2))


def main():
    args = parse_args()
    if not config.TFLITE_INT8.exists() or not config.NORMALIZATION.exists():
        raise FileNotFoundError("Run train.py and export_tflite.py first")

    _, test_files = data.build_dataset()
    test_files = limit_test_files(test_files, args.max_test_per_class)
    stats = np.load(config.NORMALIZATION)
    model = Int8Autoencoder(config.TFLITE_INT8)

    scores = np.array([
        file_score(model, path, stats["mean"], stats["std"])
        for _, path, _ in test_files
    ])
    labels = np.array([label for _, _, label in test_files])
    ids = np.array([machine_id for machine_id, _, _ in test_files])
    overall_auc = float(roc_auc_score(labels, scores))
    per_id = {}
    for machine_id in sorted(set(ids)):
        selected = ids == machine_id
        if len(set(labels[selected])) == 2:
            per_id[machine_id] = round(
                float(roc_auc_score(labels[selected], scores[selected])), 4
            )

    results = {
        "machine": config.MACHINE,
        "overall_auc": round(overall_auc, 4),
        "per_id_auc": per_id,
        "test_clips": len(test_files),
        "model_size_bytes": config.TFLITE_INT8.stat().st_size,
    }
    output = config.ARTIFACTS / "metrics_int8.json"
    output.write_text(json.dumps(results, indent=2))

    print(f"[result] int8 overall AUC = {overall_auc:.4f}")
    for machine_id, auc in per_id.items():
        print(f"[result]   {machine_id}: AUC = {auc:.4f}")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
