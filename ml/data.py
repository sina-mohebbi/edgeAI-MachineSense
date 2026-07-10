"""MIMII loading + log-mel feature extraction.

A WAV file -> a matrix of (T, FEATURE_DIM) feature vectors, where each vector is
FRAMES consecutive log-mel frames concatenated (the DCASE 2020 Task 2 baseline input).
The per-file anomaly score used for AUC is the mean reconstruction error over its vectors.
"""
from __future__ import annotations

from pathlib import Path

import config
import librosa
import numpy as np


def file_to_vectors(path: str | Path) -> np.ndarray:
    """Load a WAV and return its (T, FEATURE_DIM) log-mel context vectors."""
    y, _ = librosa.load(str(path), sr=config.SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.SR, n_fft=config.N_FFT, hop_length=config.HOP,
        n_mels=config.N_MELS, power=config.POWER,
    )
    # log-mel in dB
    log_mel = 20.0 / config.POWER * np.log10(np.maximum(mel, 1e-10))  # (N_MELS, frames)

    n = config.FRAMES
    T = log_mel.shape[1] - n + 1
    if T < 1:
        return np.empty((0, config.FEATURE_DIM), dtype=np.float32)

    vectors = np.empty((T, config.FEATURE_DIM), dtype=np.float32)
    for t in range(n):
        vectors[:, t * config.N_MELS:(t + 1) * config.N_MELS] = log_mel[:, t:t + T].T
    return vectors


def cached_file_to_vectors(path: str | Path) -> np.ndarray:
    """Load features from disk, extracting and caching them on first use."""
    path = Path(path)
    relative = path.resolve().relative_to(config.DATA_DIR.resolve())
    cache_path = (config.FEATURE_CACHE / relative).with_suffix(".npy")

    if cache_path.exists():
        return np.load(cache_path)

    vectors = file_to_vectors(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors


def list_machine_files(machine_id: str) -> tuple[list[Path], list[Path]]:
    """Return (normal_wavs, abnormal_wavs) for one machine id."""
    base = config.DATA_DIR / machine_id
    normal = sorted((base / "normal").glob("*.wav"))
    abnormal = sorted((base / "abnormal").glob("*.wav"))
    return normal, abnormal


def build_dataset(seed: int = config.SEED):
    """Split into train (normal only) and test (held-out normal + all abnormal).

    Returns
    -------
    train_files : list[(machine_id, path)]
    test_files  : list[(machine_id, path, label)]   label: 0 normal, 1 anomaly
    """
    rng = np.random.default_rng(seed)
    train_files, test_files = [], []
    for mid in config.MACHINE_IDS:
        if not (config.DATA_DIR / mid).exists():
            continue
        normal, abnormal = list_machine_files(mid)
        normal = list(normal)
        rng.shuffle(normal)
        n_test = max(1, int(config.TEST_NORMAL_FRACTION * len(normal)))
        test_norm, train_norm = normal[:n_test], normal[n_test:]

        train_files += [(mid, f) for f in train_norm]
        test_files += [(mid, f, 0) for f in test_norm]
        test_files += [(mid, f, 1) for f in abnormal]

    if not train_files:
        raise FileNotFoundError(
            f"No WAV files under {config.DATA_DIR}. Download a MIMII machine type "
            f"and unzip it so that e.g. {config.DATA_DIR / 'id_00' / 'normal'} exists. "
            f"See ml/README.md."
        )
    return train_files, test_files


def stack_train_vectors(train_files) -> np.ndarray:
    """Concatenate feature vectors from all training (normal) clips."""
    parts = [cached_file_to_vectors(f) for _, f in train_files]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts, axis=0).astype(np.float32)
