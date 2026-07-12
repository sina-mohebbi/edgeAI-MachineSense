"""Evaluate the exported per-machine-ID int8 models (mirrors evaluate_tflite.py).

Confirms quantization is ~lossless per ID, and reports the macro-average int8 AUC to
compare against train_per_id.py's macro-average float32 AUC. See train_per_id.py's
macro_average_auc() docstring for why macro-average (not a pooled cross-ID ranking)
is the correct headline metric here.
"""
from __future__ import annotations

import json

import config
import data
import numpy as np
from evaluate_tflite import Int8Autoencoder, file_score
from sklearn.metrics import roc_auc_score
from train_per_id import combine, group_by_id, macro_average_auc


def main() -> None:
    _, test_files = data.build_dataset()
    test_by_id = group_by_id(test_files)

    scores_by_id, labels_by_id, per_id = {}, {}, {}
    for machine_id, id_test_files in test_by_id.items():
        id_dir = config.PER_ID_ARTIFACTS / machine_id
        tflite_path = id_dir / "model_int8.tflite"
        normalization_path = id_dir / "normalization.npz"
        if not tflite_path.exists() or not normalization_path.exists():
            print(f"[skip] {machine_id}: run train_per_id.py + export_per_id.py first")
            continue

        stats = np.load(normalization_path)
        model = Int8Autoencoder(tflite_path)
        scores = [file_score(model, path, stats["mean"], stats["std"])
                  for _, path, _ in id_test_files]
        labels = [label for _, _, label in id_test_files]
        auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")

        scores_by_id[machine_id] = scores
        labels_by_id[machine_id] = labels
        per_id[machine_id] = round(float(auc), 4)
        print(f"[result] {machine_id}: int8 AUC = {auc:.4f}")

    if not scores_by_id:
        raise FileNotFoundError("No exported per-ID int8 models found.")

    macro_auc = macro_average_auc(per_id)
    pooled_ranking_auc = combine(scores_by_id, labels_by_id)
    print(f"\n[result] macro-average int8 AUC (headline metric)      = {macro_auc:.4f}")
    print(f"[result] pooled-ranking int8 AUC (reference only)       = {pooled_ranking_auc:.4f}")

    result = {
        "machine": config.MACHINE,
        "macro_average_auc": round(macro_auc, 4),
        "pooled_ranking_auc": round(pooled_ranking_auc, 4),
        "per_id_auc": per_id,
    }
    output = config.PER_ID_ARTIFACTS / "metrics_int8.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
