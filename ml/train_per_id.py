"""Train one autoencoder PER machine ID instead of a single pooled model.

Motivation
----------
train.py trains one shared autoencoder across all machine IDs of a type. Different
physical units of the same machine type have different baseline sounds, so a pooled
model is forced to compromise. On the fan dataset this showed up clearly: pooled AUC
was 0.713 overall, but id_00 alone scored only 0.576 (barely better than random)
while id_02/04/06 scored 0.70-0.80. This script trains a separate autoencoder per ID
and reports:
  * each ID's AUC using only ITS OWN model (no cross-machine compromise)
  * a combined AUC over all test files, each scored by its own-ID model -- the fair
    number to compare against the pooled model's overall AUC.

Uses the exact same train/validation/test file split as train.py (both call
data.build_dataset()), so the comparison is apples-to-apples.

Artifacts per ID written to ml/artifacts/per_id/<id>/:
    model.keras, normalization.npz, rep_vectors.npy, history.json
Combined summary: ml/artifacts/per_id/metrics.json
Run `python export_per_id.py` afterwards to quantize each ID's model for the ESP32.
"""
from __future__ import annotations

import json

import config
import data
import numpy as np
import tensorflow as tf
from model import build_autoencoder
from sklearn.metrics import roc_auc_score
from train import (
    file_score,
    limit_test_files,
    limit_train_files,
    normalize,
    parse_args,
    split_train_validation,
)


def group_by_id(items):
    """Group (machine_id, ...) tuples into {machine_id: [items]}, id-sorted."""
    grouped = {}
    for item in items:
        grouped.setdefault(item[0], []).append(item)
    return {mid: grouped[mid] for mid in sorted(grouped)}


def combine(scores_by_id, labels_by_id):
    """Concatenate per-ID scores/labels and compute one pooled-ranking AUC.

    CAUTION: each per-ID model is trained independently, so reconstruction-error
    scales differ across IDs (e.g. id_02's baseline loss is ~0.59 vs id_06's ~0.45).
    Ranking raw scores globally across models is not meaningful -- it penalizes IDs
    whose model happens to produce larger absolute errors, regardless of how well
    that model separates its own normal/anomalous clips. Kept for reference only;
    use macro_average_auc() as the headline metric (see its docstring).
    """
    scores = np.concatenate([np.asarray(s) for s in scores_by_id.values()])
    labels = np.concatenate(
        [np.asarray(label_values) for label_values in labels_by_id.values()]
    )
    return float(roc_auc_score(labels, scores))


def macro_average_auc(per_id_auc: dict) -> float:
    """Unweighted mean of each ID's own AUC -- the deployment-relevant metric.

    A real deployment binds one physical device to one physical machine, so a
    device only ever ranks its own machine's scores against its own threshold --
    it never needs to rank its scores against another machine's. Averaging each
    ID's own (correctly-scaled) AUC reflects that, unlike combine() above.
    """
    values = [v for v in per_id_auc.values() if not np.isnan(v)]  # drop NaN
    return float(np.mean(values)) if values else float("nan")


def train_one_id(machine_id, id_train_files, id_test_files, epochs, patience):
    """Fit one autoencoder on a single machine ID; return everything needed to save."""
    fitting, validation = split_train_validation(id_train_files)
    x_train = data.stack_train_vectors(fitting)
    x_validation = data.stack_train_vectors(validation)

    feature_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    x_train_n = normalize(x_train, feature_mean, feature_std)
    x_validation_n = normalize(x_validation, feature_mean, feature_std)

    model = build_autoencoder()
    history = model.fit(
        x_train_n, x_train_n,
        epochs=epochs, batch_size=config.BATCH,
        validation_data=(x_validation_n, x_validation_n), shuffle=True, verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=max(2, patience // 2),
                min_lr=1e-5,
            ),
        ],
    )

    scores = [file_score(model, path, feature_mean, feature_std)
              for _, path, _ in id_test_files]
    labels = [label for _, _, label in id_test_files]
    auc = roc_auc_score(labels, scores) if len(set(labels)) == 2 else float("nan")

    return {
        "model": model, "mean": feature_mean, "std": feature_std,
        "x_train_n": x_train_n, "history": history, "scores": scores,
        "labels": labels, "auc": auc,
        "train_clips": len(fitting), "validation_clips": len(validation),
        "test_clips": len(id_test_files),
    }


def save_id_artifacts(machine_id, result):
    out_dir = config.PER_ID_ARTIFACTS / machine_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result["model"].save(out_dir / "model.keras")
    np.savez(out_dir / "normalization.npz", mean=result["mean"], std=result["std"])
    rep_idx = np.random.default_rng(0).choice(
        len(result["x_train_n"]), size=min(1000, len(result["x_train_n"])), replace=False)
    np.save(out_dir / "rep_vectors.npy", result["x_train_n"][rep_idx])
    history_data = {k: [float(v) for v in vs] for k, vs in result["history"].history.items()}
    (out_dir / "history.json").write_text(json.dumps(history_data, indent=2))


def main() -> None:
    args = parse_args()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)
    config.PER_ID_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    train_files, test_files = data.build_dataset()
    train_files = limit_train_files(train_files, args.max_train_files)
    test_files = limit_test_files(test_files, args.max_test_per_class)

    train_by_id = group_by_id(train_files)
    test_by_id = group_by_id(test_files)
    machine_ids = [mid for mid in train_by_id if mid in test_by_id]
    print(f"[data] machine={config.MACHINE}  per-id training for: {machine_ids}")

    per_id_metrics = {}
    scores_by_id, labels_by_id = {}, {}

    for machine_id in machine_ids:
        print(f"\n=== {machine_id} "
              f"({len(train_by_id[machine_id])} train clips, "
              f"{len(test_by_id[machine_id])} test clips) ===")
        result = train_one_id(
            machine_id, train_by_id[machine_id], test_by_id[machine_id],
            epochs=args.epochs, patience=args.patience,
        )
        save_id_artifacts(machine_id, result)
        scores_by_id[machine_id] = result["scores"]
        labels_by_id[machine_id] = result["labels"]
        per_id_metrics[machine_id] = {
            "auc": round(float(result["auc"]), 4),
            "train_clips": result["train_clips"],
            "validation_clips": result["validation_clips"],
            "test_clips": result["test_clips"],
            "epochs_completed": len(result["history"].history["loss"]),
        }
        print(f"[result] {machine_id}: AUC = {result['auc']:.4f}")

    per_id_auc = {mid: m["auc"] for mid, m in per_id_metrics.items()}
    macro_auc = macro_average_auc(per_id_auc)
    pooled_ranking_auc = combine(scores_by_id, labels_by_id)

    print(f"\n[result] macro-average AUC (per-ID models, headline metric) = {macro_auc:.4f}")
    print(f"[result] pooled-ranking AUC (scale-sensitive, reference only) = "
          f"{pooled_ranking_auc:.4f}")

    pooled_path = config.ARTIFACTS / "metrics.json"
    pooled_model_auc = None
    if pooled_path.exists():
        pooled_model_auc = json.loads(pooled_path.read_text()).get("overall_auc")
        print(f"[result] pooled MODEL overall AUC (train.py, for comparison)  = "
              f"{pooled_model_auc}")

    summary = {
        "machine": config.MACHINE,
        "macro_average_auc": round(macro_auc, 4),
        "pooled_ranking_auc": round(pooled_ranking_auc, 4),
        "per_id_auc": per_id_metrics,
        "pooled_model_overall_auc": pooled_model_auc,
    }
    (config.PER_ID_ARTIFACTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {config.PER_ID_ARTIFACTS / 'metrics.json'}")
    print("[next] python export_per_id.py")


if __name__ == "__main__":
    main()
