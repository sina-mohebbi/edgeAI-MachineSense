"""Run a per-ID model experiment with a different log-mel context length.

This keeps the main artifacts intact by redirecting outputs to:
    ml/artifacts/experiments/frames<FRAMES>/

Example:
    python ml/run_frames_experiment.py --frames 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run a FRAMES experiment safely")
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-test-per-class", type=int, default=None)
    return parser.parse_args()


def configure_experiment(frames: int):
    import config

    config.FRAMES = frames
    config.FEATURE_DIM = config.N_MELS * config.FRAMES

    feature_key = (
        f"sr{config.SR}_nfft{config.N_FFT}_hop{config.HOP}_"
        f"mel{config.N_MELS}_pow{config.POWER}_frames{config.FRAMES}"
    )
    config.FEATURE_SIG = hashlib.md5(feature_key.encode()).hexdigest()[:8]

    experiment_root = config.ARTIFACTS / "experiments" / f"frames{frames}"
    config.FEATURE_CACHE = experiment_root / "feature_cache" / config.FEATURE_SIG
    config.PER_ID_ARTIFACTS = experiment_root / "per_id"
    config.MODEL_KERAS = experiment_root / "model.keras"
    config.REP_VECTORS = experiment_root / "rep_vectors.npy"
    config.NORMALIZATION = experiment_root / "normalization.npz"
    config.HISTORY = experiment_root / "history.json"
    config.TFLITE_INT8 = experiment_root / "model_int8.tflite"
    config.C_HEADER = experiment_root / "model_data.cc"

    experiment_root.mkdir(parents=True, exist_ok=True)
    (experiment_root / "experiment_config.json").write_text(
        json.dumps(
            {
                "frames": config.FRAMES,
                "feature_dim": config.FEATURE_DIM,
                "feature_sig": config.FEATURE_SIG,
                "per_id_artifacts": str(config.PER_ID_ARTIFACTS),
            },
            indent=2,
        )
    )
    return config, experiment_root


def set_argv(program: str, args) -> None:
    argv = [program]
    if args.epochs is not None:
        argv += ["--epochs", str(args.epochs)]
    if args.patience is not None:
        argv += ["--patience", str(args.patience)]
    if args.max_train_files is not None:
        argv += ["--max-train-files", str(args.max_train_files)]
    if args.max_test_per_class is not None:
        argv += ["--max-test-per-class", str(args.max_test_per_class)]
    sys.argv = argv


def main() -> None:
    args = parse_args()
    config, experiment_root = configure_experiment(args.frames)

    import evaluate_per_id_tflite
    import export_per_id
    import train_per_id

    print(
        f"[experiment] FRAMES={config.FRAMES} FEATURE_DIM={config.FEATURE_DIM} "
        f"out={experiment_root}"
    )

    set_argv("train_per_id.py", args)
    train_per_id.main()

    sys.argv = ["export_per_id.py"]
    export_per_id.main()

    sys.argv = ["evaluate_per_id_tflite.py"]
    evaluate_per_id_tflite.main()

    print(f"[experiment] done: {experiment_root}")


if __name__ == "__main__":
    main()
