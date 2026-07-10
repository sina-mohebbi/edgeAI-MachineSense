# ml: train the anomaly detector

Trains a dense autoencoder on **normal** machine sound (MIMII) and evaluates anomaly-detection
**AUC**, then quantizes to **int8** and exports a C header for TensorFlow Lite for Microcontrollers.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Get the dataset

MIMII (DCASE 2020 Task 2) is public on Zenodo: https://zenodo.org/record/3384388

Download **one** machine type to start (fan is a good first pick) at the cleanest SNR
(the `+6 dB` archive), and unzip so the layout is:

```
ml/data/fan/id_00/normal/*.wav
ml/data/fan/id_00/abnormal/*.wav
ml/data/fan/id_02/...
```

Set `MACHINE` / `MACHINE_IDS` in `config.py` if you use a different type or ids.

## 3. Train + evaluate

```bash
python train.py
```

Prints overall + per-id AUC and writes `artifacts/model.keras` and `artifacts/metrics.json`.
Extracted log-mel vectors are cached under `artifacts/feature_cache/`, so later runs do
not process the same WAV files again.

Feature-wise mean and standard deviation are learned from normal training data only and
saved to `artifacts/normalization.npz`. The same transformation must be applied before
inference on the ESP32.

Validation is split by complete WAV files, preventing overlapping feature windows from
the same recording appearing in both training and validation. Early stopping restores
the weights with the best validation loss, and `artifacts/history.json` stores the loss
curves.

Start with a small end-to-end experiment before the full run:

```bash
python train.py --epochs 5 --max-train-files 200 --max-test-per-class 20
```

These limits are for pipeline validation only; do not report their AUC as the final result.

## 4. Quantize + export for the ESP32

```bash
python export_tflite.py
python evaluate_tflite.py
```

Writes `artifacts/model_int8.tflite` and `artifacts/model_data.cc` (+ `.h`). In Phase 1
these get copied into `firmware/main/` and compiled into the ESP32 image. The evaluation
command writes `artifacts/metrics_int8.json` so quantization can be compared with float32.

## 5. (Optional but recommended) Per-machine-ID models

The pooled model above trains one autoencoder shared across all machine IDs. Machines
of the same type still sound different from unit to unit, so pooling forces a
compromise: on the fan dataset, pooled overall AUC was 0.713, but **id_00 alone
scored only 0.576** (barely above random) while id_02/04/06 scored 0.70-0.80.

Training one autoencoder **per machine ID** removes that compromise:

```bash
python train_per_id.py
python export_per_id.py
python evaluate_per_id_tflite.py
```

Uses the same train/validation/test split as `train.py`. The headline metric is the
macro-average AUC, meaning the unweighted mean of each ID's own AUC, rather than a
pooled ranking across models. Each per-ID autoencoder is trained independently, so
their reconstruction-error scales differ, and ranking raw scores from different
models against each other is not meaningful. A naive pooled-ranking AUC is also
reported for reference, but it can look worse even when every individual model
improved, purely because of that scale mismatch.

Macro-average is also the metric that matches real deployment. One physical ESP32
monitors one physical machine, so it only ever ranks its own scores against its own
threshold, never against another machine's.

Artifacts land under `artifacts/per_id/<id>/` (own `model.keras`,
`normalization.npz`, `model_int8.tflite`, `model_data.cc` with variable name
`g_model_data_<id>`), plus a summary at `artifacts/per_id/metrics.json` (and
`metrics_int8.json` after export). A real deployment only ever needs the
`.tflite`/`.cc` for its own machine's ID.

## Files

| File | Role |
|---|---|
| `config.py` | all audio/feature/model/export parameters (DCASE baseline) |
| `data.py` | MIMII loading + log-mel context-vector extraction |
| `model.py` | the autoencoder topology |
| `train.py` | train pooled model + AUC evaluation + save artifacts |
| `export_tflite.py` | int8 quantization + C-header emit (pooled model) |
| `evaluate_tflite.py` | host int8 AUC evaluation before device deployment (pooled) |
| `train_per_id.py` | train one autoencoder per machine ID |
| `export_per_id.py` | int8 quantization + C-header emit, per machine ID |
| `evaluate_per_id_tflite.py` | host int8 AUC evaluation, per machine ID |
| `compute_threshold.py` | pick an anomaly threshold from normal-train scores; emit `firmware/main/threshold.h` + report precision/recall/F1 |
| `make_dataset_preview.py` | render the normal-vs-anomalous spectrogram figure in the root README |
| `conftest.py` | puts `ml/` on `sys.path` so the tests can import these modules |
| `tests/test_smoke.py` | dataset-free tests (also run in CI) |
| `tests/test_training_utils.py` | dataset-free tests for dataset limiting + normalization |

### `id_00` follow-up experiments

These reproduce the numbers in the root README's `id_00` limitation section. Each
writes to `artifacts/experiments/...` and leaves the main deployment artifacts alone.

| File | Role |
|---|---|
| `run_frames_experiment.py` | retrain a per-ID model with a longer log-mel context (`--frames 10`) |
| `run_id00_conv_experiment.py` | train a small Conv2D autoencoder for `id_00` |
| `evaluate_scoring_strategies.py` | compare clip-scoring aggregations (mean vs p95 vs top-k) |

## How the anomaly score works

Each 10 s clip becomes many 640-dim log-mel context vectors. The autoencoder is trained
only on normal clips, so it reconstructs normal sound with low error. A clip's anomaly
score is the **mean reconstruction MSE** over its vectors; higher = more anomalous. AUC
measures how well that score separates normal from abnormal test clips.

## Reproduce the smoke tests locally

```bash
cd ml && pytest -q          # no dataset required
```
