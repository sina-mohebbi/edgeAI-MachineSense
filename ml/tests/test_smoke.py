"""Dataset-free smoke tests so CI stays green without downloading MIMII.

These exercise the model + export path on random data: build the autoencoder, train
one step, int8-quantize, and emit the C header. They catch shape/topology/export
regressions in seconds.
"""
import config
import export_tflite
import numpy as np
import tensorflow as tf
from model import build_autoencoder
from train_per_id import combine, group_by_id, macro_average_auc


def test_autoencoder_io_shape():
    model = build_autoencoder()
    x = np.random.rand(4, config.FEATURE_DIM).astype(np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (4, config.FEATURE_DIM)


def test_train_one_step():
    model = build_autoencoder()
    x = np.random.rand(64, config.FEATURE_DIM).astype(np.float32)
    hist = model.fit(x, x, epochs=1, batch_size=32, verbose=0)
    assert np.isfinite(hist.history["loss"][-1])


def test_int8_export_and_header(tmp_path):
    model = build_autoencoder()
    rep = np.random.rand(64, config.FEATURE_DIM).astype(np.float32)
    tflite_bytes = export_tflite.to_int8_tflite(model, rep)
    assert len(tflite_bytes) > 0

    # input/output tensors must be int8 for TFLite-Micro
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    assert interp.get_input_details()[0]["dtype"] == np.int8
    assert interp.get_output_details()[0]["dtype"] == np.int8

    out = tmp_path / "model_data.cc"
    export_tflite.tflite_to_c_header(tflite_bytes, "g_model_data", out)
    text = out.read_text()
    assert "g_model_data[]" in text
    assert "g_model_data_len" in text
    assert out.with_suffix(".h").exists()


def test_group_by_id_is_sorted_and_complete():
    items = [("id_02", "b"), ("id_00", "a"), ("id_00", "c")]
    grouped = group_by_id(items)
    assert list(grouped.keys()) == ["id_00", "id_02"]
    assert grouped["id_00"] == [("id_00", "a"), ("id_00", "c")]
    assert grouped["id_02"] == [("id_02", "b")]


def test_combine_matches_pooled_auc():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    scores_by_id = {"id_00": rng.random(20), "id_02": rng.random(20)}
    labels_by_id = {
        "id_00": rng.integers(0, 2, 20),
        "id_02": rng.integers(0, 2, 20),
    }
    combined_auc = combine(scores_by_id, labels_by_id)

    pooled_scores = np.concatenate(list(scores_by_id.values()))
    pooled_labels = np.concatenate(list(labels_by_id.values()))
    expected = roc_auc_score(pooled_labels, pooled_scores)
    assert combined_auc == expected


def test_macro_average_auc_ignores_nan_and_averages():
    per_id_auc = {"id_00": 0.5, "id_02": 0.9, "id_04": float("nan")}
    assert macro_average_auc(per_id_auc) == 0.7


def test_macro_average_can_beat_pooled_ranking_under_scale_mismatch():
    """Regression guard for the scale-mismatch pitfall this metric was added to fix:
    per-ID models can each rank their own clips well while a naive pooled ranking
    across models (different score scales) makes the ensemble look worse."""
    from sklearn.metrics import roc_auc_score

    # id_A: well-separated but on a low score scale; id_B: well-separated, high scale.
    scores_by_id = {
        "id_A": np.array([0.01, 0.02, 0.03, 0.09, 0.10, 0.11]),
        "id_B": np.array([0.50, 0.51, 0.52, 0.90, 0.91, 0.92]),
    }
    labels_by_id = {
        "id_A": np.array([0, 0, 0, 1, 1, 1]),
        "id_B": np.array([0, 0, 0, 1, 1, 1]),
    }
    per_id_auc = {
        mid: roc_auc_score(labels_by_id[mid], scores_by_id[mid])
        for mid in scores_by_id
    }
    assert macro_average_auc(per_id_auc) == 1.0  # each model perfectly separates its own clips
    assert combine(scores_by_id, labels_by_id) < 1.0  # pooled ranking is corrupted by scale
