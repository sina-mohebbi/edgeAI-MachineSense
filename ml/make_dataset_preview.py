"""Generate the dataset preview figure used in the top-level README.

Renders log-mel spectrograms of one normal and one abnormal fan clip, using the
same feature parameters the training pipeline uses (config.py), so the figure
shows exactly what the autoencoder sees. Output: docs/images/dataset_preview.png.

Needs matplotlib (not a core dependency): pip install matplotlib
Run from the repo root: python ml/make_dataset_preview.py
"""
from pathlib import Path

import librosa
import matplotlib
import numpy as np

matplotlib.use("Agg")
import config
import matplotlib.pyplot as plt

MACHINE_ID = "id_02"
OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "dataset_preview.png"


def log_mel(path):
    """Same log-mel as ml/data.py.file_to_vectors, kept as a (n_mels, T) image."""
    y, _ = librosa.load(str(path), sr=config.SR, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.SR, n_fft=config.N_FFT, hop_length=config.HOP,
        n_mels=config.N_MELS, power=config.POWER,
    )
    return 20.0 / config.POWER * np.log10(np.maximum(mel, 1e-10))


def main():
    base = config.DATA_DIR / MACHINE_ID
    normal_wav = min((base / "normal").glob("*.wav"))
    abnormal_wav = min((base / "abnormal").glob("*.wav"))
    normal, abnormal = log_mel(normal_wav), log_mel(abnormal_wav)

    vmin = min(normal.min(), abnormal.min())
    vmax = max(normal.max(), abnormal.max())
    extent = [0, normal.shape[1] * config.HOP / config.SR, 0, config.N_MELS]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for ax, data, title in ((axes[0], normal, "Normal"), (axes[1], abnormal, "Anomalous")):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(f"{title} fan ({MACHINE_ID})")
        ax.set_xlabel("Time (s)")
    axes[0].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes, label="Log-mel energy (dB)", shrink=0.9)
    fig.suptitle("MIMII fan: log-mel spectrogram the model receives (128 mels, 16 kHz)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"[saved] {OUT}  ({OUT.stat().st_size // 1024} KiB)")
    print(f"[used] normal={normal_wav.name}  abnormal={abnormal_wav.name}")


if __name__ == "__main__":
    main()
