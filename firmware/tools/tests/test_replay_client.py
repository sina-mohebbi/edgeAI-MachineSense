"""Dataset-free, hardware-free tests for replay_client.py's protocol/aggregation
logic. Uses a FakeDevice instead of real serial or a loaded TFLite model, so
these run in CI with no dataset and no ESP32 attached.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_client import Device, anomaly_metrics, score_files


class FakeDevice(Device):
    """Returns one (score, anomaly) per call in sequence. If anomalies are not
    given, derives the flag from a threshold so the integrity check passes."""

    def __init__(self, scores, threshold=0.0, anomalies=None):
        self._scores = list(scores)
        self._anoms = (list(anomalies) if anomalies is not None
                       else [int(s > threshold) for s in scores])
        self._i = 0

    def query(self, vector):
        s, a = self._scores[self._i], self._anoms[self._i]
        self._i += 1
        return s, a


def test_score_files_averages_per_vector_scores_into_a_clip_score():
    # File 1: 2 vectors (scores 1.0, 3.0 -> clip score 2.0), normal;
    # File 2: 1 vector (score 10.0), anomalous.
    device = FakeDevice([1.0, 3.0, 10.0], threshold=5.0)
    files = [
        (0, [np.zeros(4), np.zeros(4)]),
        (1, [np.zeros(4)]),
    ]
    scores, labels, mism = score_files(device, files, threshold=5.0)

    assert scores == [2.0, 10.0]
    assert labels == [0, 1]
    assert mism == 0  # device flags were derived from the same threshold


def test_score_files_skips_empty_vector_lists():
    device = FakeDevice([5.0], threshold=1.0)
    files = [(0, []), (1, [np.zeros(4)])]
    scores, labels, mism = score_files(device, files, threshold=1.0)

    assert scores == [5.0]
    assert labels == [1]
    assert mism == 0


def test_score_files_counts_device_flag_mismatches():
    # Device claims anomaly=0 for a score of 9.0 that IS above the threshold.
    device = FakeDevice([9.0], threshold=5.0, anomalies=[0])
    files = [(1, [np.zeros(4)])]
    _, _, mism = score_files(device, files, threshold=5.0)
    assert mism == 1


def test_anomaly_metrics_precision_recall_f1():
    # scores > 5 predicted anomalous; labels: two anomalies (10, 6), two normal (1, 8).
    # pred: [F, T, T, T] vs labels [0, 1, 1, 0] -> TP=2 (6,10), FP=1 (8), FN=0, TN=1
    scores = [1.0, 10.0, 6.0, 8.0]
    labels = [0, 1, 1, 0]
    m = anomaly_metrics(scores, labels, threshold=5.0)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 0 and m["tn"] == 1
    assert m["recall"] == 1.0
    assert m["precision"] == round(2 / 3, 4)
