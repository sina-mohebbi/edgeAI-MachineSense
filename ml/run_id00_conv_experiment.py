"""Train a small convolutional autoencoder for one difficult machine ID.

This is an isolated experiment for the weak `id_00` result. It keeps the main
dense-autoencoder artifacts intact and writes outputs to:
    ml/artifacts/experiments/id00_conv/
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
from tensorflow.keras import Model, layers
from train import (
    limit_test_files,
    limit_train_files,
    normalize,
    split_train_validation,
)
from train_per_id import group_by_id


def parse_args():
    parser = argparse.ArgumentParser(description="Run id_00 Conv2D AE experiment")
    parser.add_argument("--machine-id", default="id_00")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.ARTIFACTS / "experiments" / "id00_conv",
    )
    return parser.parse_args()


def to_image(vectors: np.ndarray) -> np.ndarray:
    """Convert flat context vectors to (frames, mel, channel) tensors."""
    return vectors.reshape((-1, config.FRAMES, config.N_MELS, 1)).astype(np.float32)


def build_conv_autoencoder() -> Model:
    inp = layers.Input(
        shape=(config.FRAMES, config.N_MELS, 1),
        name="log_mel_context",
    )
    x = layers.Conv2D(8, (3, 5), padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D((1, 2), padding="same")(x)
    x = layers.Conv2D(4, (3, 5), padding="same", activation="relu")(x)
    x = layers.MaxPooling2D((1, 2), padding="same")(x)

    shape_before_flatten = x.shape[1:]
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu", name="bottleneck")(x)
    x = layers.Dense(int(np.prod(shape_before_flatten)), activation="relu")(x)
    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.UpSampling2D((1, 2))(x)
    x = layers.Conv2D(4, (3, 5), padding="same", activation="relu")(x)
    x = layers.UpSampling2D((1, 2))(x)
    x = layers.Conv2D(8, (3, 5), padding="same", activation="relu")(x)
    out = layers.Conv2D(1, (3, 5), padding="same", name="recon")(x)

    model = Model(inp, out, name="machinesense_conv_ae")
    model.compile(optimizer=tf.keras.optimizers.Adam(config.LR), loss="mse")
    return model


def stack_vectors(files) -> np.ndarray:
    parts = [data.cached_file_to_vectors(path) for _, path in files]
    parts = [part for part in parts if len(part)]
    return np.concatenate(parts, axis=0).astype(np.float32)


def file_score(model: Model, path: Path, mean, std) -> float:
    vectors = data.cached_file_to_vectors(path)
    if len(vectors) == 0:
        return 0.0
    vectors = normalize(vectors, mean, std)
    images = to_image(vectors)
    recon = model.predict(images, verbose=0, batch_size=4096)
    per_vector = np.mean((images - recon) ** 2, axis=(1, 2, 3))
    return float(np.mean(per_vector))


def main() -> None:
    args = parse_args()
    tf.random.set_seed(config.SEED)
    np.random.seed(config.SEED)

    train_files, test_files = data.build_dataset()
    train_files = limit_train_files(train_files, args.max_train_files)
    test_files = limit_test_files(test_files, args.max_test_per_class)
    train_by_id = group_by_id(train_files)
    test_by_id = group_by_id(test_files)

    if args.machine_id not in train_by_id or args.machine_id not in test_by_id:
        raise ValueError(f"{args.machine_id} not found in dataset split")

    fitting, validation = split_train_validation(train_by_id[args.machine_id])
    x_train = stack_vectors(fitting)
    x_validation = stack_vectors(validation)

    feature_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std < 1e-6] = 1.0
    x_train = normalize(x_train, feature_mean, feature_std)
    x_validation = normalize(x_validation, feature_mean, feature_std)

    x_train_img = to_image(x_train)
    x_validation_img = to_image(x_validation)

    print(
        f"[data] {args.machine_id}: train_clips={len(fitting)} "
        f"validation_clips={len(validation)} "
        f"test_clips={len(test_by_id[args.machine_id])} "
        f"train_vectors={x_train_img.shape[0]} input={x_train_img.shape[1:]}"
    )

    model = build_conv_autoencoder()
    model.summary()
    history = model.fit(
        x_train_img,
        x_train_img,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_data=(x_validation_img, x_validation_img),
        shuffle=True,
        verbose=2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=max(2, args.patience // 2),
                min_lr=1e-5,
            ),
        ],
    )

    scores = np.array(
        [
            file_score(model, path, feature_mean, feature_std)
            for _, path, _ in test_by_id[args.machine_id]
        ]
    )
    labels = np.array([label for _, _, label in test_by_id[args.machine_id]])
    auc = float(roc_auc_score(labels, scores))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "model.keras")
    np.savez(args.output_dir / "normalization.npz", mean=feature_mean, std=feature_std)
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    (args.output_dir / "history.json").write_text(json.dumps(history_data, indent=2))
    metrics = {
        "machine": config.MACHINE,
        "machine_id": args.machine_id,
        "model": "conv2d_autoencoder",
        "frames": config.FRAMES,
        "feature_dim": config.FEATURE_DIM,
        "auc": round(auc, 4),
        "train_clips": len(fitting),
        "validation_clips": len(validation),
        "test_clips": len(test_by_id[args.machine_id]),
        "epochs_completed": len(history.history["loss"]),
        "parameters": int(model.count_params()),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[result] {args.machine_id}: conv AE AUC = {auc:.4f}")
    print(f"[saved] {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
