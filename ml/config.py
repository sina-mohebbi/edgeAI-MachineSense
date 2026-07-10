"""Central configuration for the edgeAI-MachineSense ML pipeline (Phase 0).

All audio/feature parameters follow the DCASE 2020 Task 2 baseline so results are
comparable to published numbers on the MIMII dataset.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # ml/
ARTIFACTS = ROOT / "artifacts"

# --- Dataset -----------------------------------------------------------------
# Unzip one MIMII machine type here, e.g. for "fan":
#   ml/data/fan/id_00/normal/*.wav
#   ml/data/fan/id_00/abnormal/*.wav
MACHINE = "fan"                                  # fan | pump | valve | slider
DATA_DIR = ROOT / "data" / MACHINE
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
TEST_NORMAL_FRACTION = 0.2                       # held-out normal clips used for AUC

# --- Audio / log-mel features (DCASE 2020 Task 2 baseline) -------------------
SR = 16000
N_FFT = 1024
HOP = 512
N_MELS = 128
POWER = 2.0
FRAMES = 5                                       # context frames concatenated
FEATURE_DIM = N_MELS * FRAMES                    # = 640

# The feature cache is namespaced by a signature of the parameters above, so changing
# any of them writes to a fresh directory instead of serving stale cached vectors.
_FEATURE_KEY = f"sr{SR}_nfft{N_FFT}_hop{HOP}_mel{N_MELS}_pow{POWER}_frames{FRAMES}"
FEATURE_SIG = hashlib.md5(_FEATURE_KEY.encode()).hexdigest()[:8]
FEATURE_CACHE = ARTIFACTS / "feature_cache" / FEATURE_SIG

# --- Autoencoder -------------------------------------------------------------
HIDDEN = 128
BOTTLENECK = 8
DEPTH = 4                                         # dense layers per half
EPOCHS = 100
BATCH = 512
LR = 1e-3
VAL_SPLIT = 0.1
PATIENCE = 10
SEED = 42

# --- Export ------------------------------------------------------------------
MODEL_KERAS = ARTIFACTS / "model.keras"
REP_VECTORS = ARTIFACTS / "rep_vectors.npy"      # representative set for int8 calibration
NORMALIZATION = ARTIFACTS / "normalization.npz"  # training-set mean/std for ESP32 input
HISTORY = ARTIFACTS / "history.json"
TFLITE_INT8 = ARTIFACTS / "model_int8.tflite"
C_HEADER = ARTIFACTS / "model_data.cc"           # copy into firmware/main/ for Phase 1
C_VAR_NAME = "g_model_data"

# --- Per-machine-ID models (see train_per_id.py) ------------------------------
# One autoencoder per machine ID instead of a single pooled model -- pooling forces
# a compromise across machines with different baseline sounds (observed: id_00
# AUC 0.576 pooled vs 0.70-0.80 for the others).
PER_ID_ARTIFACTS = ARTIFACTS / "per_id"
