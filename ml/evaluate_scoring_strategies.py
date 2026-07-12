"""Compare clip-level anomaly scoring strategies for per-ID autoencoders.

The trained autoencoder produces one reconstruction error per log-mel context
vector. Existing evaluation uses the clip mean. This script tests whether other
aggregations, such as p95 or top-k mean, separate normal/abnormal clips better.
It does not retrain models or change firmware artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
import data
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from train import limit_test_files, normalize
from train_per_id import group_by_id, macro_average_auc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate different clip scoring strategies per machine ID"
    )
    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=None,
        help="Limit normal and abnormal clips per machine ID for a quick run",
    )
    parser.add_argument(
        "--output",
        default=config.PER_ID_ARTIFACTS / "scoring_strategies.json",
        type=Path,
        help="Path for the JSON result file",
    )
    return parser.parse_args()


def vector_errors(model, path, mean, std) -> np.ndarray:
    vectors = data.cached_file_to_vectors(path)
    if len(vectors) == 0:
        return np.array([0.0], dtype=np.float32)
    vectors = normalize(vectors, mean, std)
    reconstruction = model.predict(vectors, verbose=0, batch_size=4096)
    return np.mean((vectors - reconstruction) ** 2, axis=1).astype(np.float32)


def top_fraction_mean(values: np.ndarray, fraction: float = 0.10) -> float:
    count = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.mean(np.partition(values, -count)[-count:]))


def aggregate(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(errors)),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
        "p99": float(np.percentile(errors, 99)),
        "max": float(np.max(errors)),
        "top10_mean": top_fraction_mean(errors, 0.10),
        "top5_mean": top_fraction_mean(errors, 0.05),
    }


def evaluate_machine_id(machine_id: str, test_files):
    model_path = config.PER_ID_ARTIFACTS / machine_id / "model.keras"
    normalization_path = config.PER_ID_ARTIFACTS / machine_id / "normalization.npz"
    if not model_path.exists() or not normalization_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts for {machine_id}. Run train_per_id.py first."
        )

    model = tf.keras.models.load_model(model_path)
    stats = np.load(normalization_path)
    labels = np.array([label for _, _, label in test_files])
    scores_by_strategy: dict[str, list[float]] = {}

    for _, path, _ in test_files:
        errors = vector_errors(model, path, stats["mean"], stats["std"])
        for strategy, score in aggregate(errors).items():
            scores_by_strategy.setdefault(strategy, []).append(score)

    auc_by_strategy = {}
    for strategy, scores in scores_by_strategy.items():
        if len(set(labels)) == 2:
            auc_by_strategy[strategy] = round(
                float(roc_auc_score(labels, np.array(scores))), 4
            )

    return auc_by_strategy


def main() -> None:
    args = parse_args()
    _, test_files = data.build_dataset()
    test_files = limit_test_files(test_files, args.max_test_per_class)
    test_by_id = group_by_id(test_files)

    by_id = {}
    for machine_id, id_test_files in test_by_id.items():
        by_id[machine_id] = evaluate_machine_id(machine_id, id_test_files)
        best_strategy, best_auc = max(
            by_id[machine_id].items(),
            key=lambda item: item[1],
        )
        print(f"[result] {machine_id}: best={best_strategy} AUC={best_auc:.4f}")
        for strategy, auc in sorted(by_id[machine_id].items()):
            print(f"  {strategy:>10}: {auc:.4f}")

    strategies = sorted({strategy for scores in by_id.values() for strategy in scores})
    macro_by_strategy = {
        strategy: round(
            macro_average_auc(
                {
                    machine_id: scores[strategy]
                    for machine_id, scores in by_id.items()
                    if strategy in scores
                }
            ),
            4,
        )
        for strategy in strategies
    }
    best_macro_strategy, best_macro_auc = max(
        macro_by_strategy.items(),
        key=lambda item: item[1],
    )

    results = {
        "machine": config.MACHINE,
        "by_id": by_id,
        "macro_average_auc": macro_by_strategy,
        "best_macro_strategy": best_macro_strategy,
        "best_macro_auc": best_macro_auc,
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))

    print("\n[result] macro-average AUC by strategy")
    for strategy, auc in sorted(macro_by_strategy.items()):
        print(f"  {strategy:>10}: {auc:.4f}")
    print(f"[result] best macro strategy: {best_macro_strategy} ({best_macro_auc:.4f})")
    print(f"[saved] {output}")


if __name__ == "__main__":
    main()
