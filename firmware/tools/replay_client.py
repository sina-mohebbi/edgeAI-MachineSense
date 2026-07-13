"""Phase 2 replay client: stream held-out MIMII test vectors to the ESP32 over
UART, reconstruct an on-device AUC, and evaluate anomaly detection at the
device's threshold -- with no sensor attached.

Protocol (must match firmware/main/main.cc):
    host  -> device : 640 float32 LE (2560 bytes) -- one normalized log-mel vector
    device -> host  : 12 bytes -- float32 score, uint32 anomaly flag (0/1),
                                  uint32 byte-sum checksum of the received request

The device flags anomalies PER VECTOR (that drives its LED live). The rigorous
anomaly DECISION is per CLIP: we average a clip's per-vector scores and compare
that to the threshold, matching ml/compute_threshold.py. So this client reports
AUC + per-clip precision/recall/F1 at the threshold, and separately checks that
the device's per-vector flag equals (device score > threshold) as an on-device
integrity check.

Usage:
    python replay_client.py --port COM5 --machine-id id_02
    python replay_client.py --mock --machine-id id_02       # no hardware needed
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

# ml/ is a sibling of firmware/; reuse its data loading + normalization exactly
# so the vectors sent over the wire match what the ML scripts used.
ML_DIR = Path(__file__).resolve().parents[2] / "ml"
sys.path.insert(0, str(ML_DIR))

import config
import data
from evaluate_tflite import Int8Autoencoder
from train import normalize
from train_per_id import group_by_id

READY_MARKER = "MACHINESENSE_READY"


class Device:
    """Answers one normalized float32 vector with (score, anomaly_flag)."""

    def query(self, vector: np.ndarray) -> tuple[float, int]:
        raise NotImplementedError


class SerialDevice(Device):
    """Talks to the real ESP32 over UART."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 5.0,
                 ready_timeout: float = 15.0):
        import time

        import serial  # local import: only required for real-hardware runs

        self._serial = serial.Serial(port, baud, timeout=timeout)

        # pyserial's default DTR/RTS state on open() is not a reliable reset on
        # every USB-serial chipset; do the explicit EN-pin reset esptool uses.
        self._serial.dtr = False
        self._serial.rts = True
        time.sleep(0.1)
        self._serial.rts = False
        time.sleep(0.1)

        self._wait_for_ready(ready_timeout)

        # Drop any boot-log bytes still buffered from the readline() wait so the
        # first raw read(12) starts on a clean frame boundary.
        time.sleep(0.2)
        self._serial.reset_input_buffer()

    def _wait_for_ready(self, ready_timeout: float):
        import time

        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            line = self._serial.readline().decode(errors="replace").strip()
            if not line:
                continue  # a quiet gap between boot-log lines, not a failure
            print(f"[device] {line}")
            if line.startswith(READY_MARKER):
                return
        raise TimeoutError(
            f"Timed out after {ready_timeout}s waiting for {READY_MARKER} "
            "-- is the board freshly flashed/reset and on the right port?"
        )

    def query(self, vector: np.ndarray, max_attempts: int = 3) -> tuple[float, int]:
        # Strictly synchronous: one 12-byte reply per 2560-byte request. On a
        # bad exchange (wrong length, checksum mismatch = corrupted request, or
        # a non-finite score) flush and resend the SAME vector -- safe because
        # the device is idle awaiting its next request and the score is
        # deterministic. The checksum caught the Phase 1 UART CR<->LF bug.
        payload = vector.astype("<f4").tobytes()
        expected_checksum = sum(payload) & 0xFFFFFFFF
        last_error = None
        for attempt in range(max_attempts):
            if attempt > 0:
                self._serial.reset_input_buffer()
            self._serial.write(payload)
            response = self._serial.read(12)
            if len(response) != 12:
                last_error = f"expected 12 response bytes, got {len(response)}"
                continue
            score, anomaly, checksum = struct.unpack("<fII", response)
            if checksum != expected_checksum:
                last_error = (f"checksum mismatch (device received corrupted "
                              f"request: got {checksum:#010x}, expected "
                              f"{expected_checksum:#010x})")
                continue
            if not np.isfinite(score):
                last_error = f"non-finite score decoded ({score!r})"
                continue
            return score, int(anomaly)
        raise OSError(f"query failed after {max_attempts} attempts: {last_error}")


class LoopbackDevice(Device):
    """Runs the exact same math as firmware/main/inference.cc, in Python.

    Validates the client + methodology with zero hardware. Delegates to
    Int8Autoencoder.reconstruct() (quantize once, invoke, dequantize output),
    scoring true-float input vs dequantized output -- identical to inference.cc's
    RunOnFloatVector -- then applies the same per-vector threshold the device does.
    """

    def __init__(self, tflite_path, threshold: float):
        self._model = Int8Autoencoder(tflite_path)
        self._threshold = threshold

    def query(self, vector: np.ndarray) -> tuple[float, int]:
        reconstruction = self._model.reconstruct(vector[None, :])[0]
        score = float(np.mean((vector - reconstruction) ** 2))
        return score, int(score > self._threshold)


def score_files(device: Device, files: Sequence[tuple], threshold: float,
                progress: bool = False):
    """Score (label, vectors) tuples; return (clip_scores, labels, flag_mismatches).

    clip score = mean of the clip's per-vector scores. flag_mismatches counts
    per-vector cases where the device's returned anomaly flag != (score >
    threshold) -- should be 0, an on-device integrity check. Pure aggregation
    logic (works with any Device), so tests can exercise it with a fake Device.
    """
    import time

    scores, labels = [], []
    flag_mismatches = 0
    start = time.time()
    total_vectors = sum(len(v) for _, v in files)
    vectors_done = 0

    for i, (label, vectors) in enumerate(files):
        if len(vectors) == 0:
            continue
        clip = [device.query(v) for v in vectors]
        for s, a in clip:
            if a != int(s > threshold):
                flag_mismatches += 1
        scores.append(float(np.mean([s for s, _ in clip])))
        labels.append(label)

        if progress:
            vectors_done += len(vectors)
            elapsed = time.time() - start
            rate = vectors_done / elapsed if elapsed > 0 else 0
            remaining = (total_vectors - vectors_done) / rate if rate > 0 else float("inf")
            print(f"[progress] file {i + 1}/{len(files)}  "
                  f"vectors {vectors_done}/{total_vectors}  "
                  f"({rate:.1f} vec/s, ~{remaining / 60:.1f} min left)")

    return scores, labels, flag_mismatches


def anomaly_metrics(scores, labels, threshold):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred = (scores > threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def build_normalized_test_files(machine_id: str, limit_files=None, seed: int = 0):
    """Load + normalize (not quantize) held-out test clips; balanced sample if limited."""
    stats = np.load(config.PER_ID_ARTIFACTS / machine_id / "normalization.npz")
    _, test_files = data.build_dataset()
    id_test_files = group_by_id(test_files).get(machine_id, [])

    if limit_files is not None:
        rng = np.random.default_rng(seed)
        normal = [f for f in id_test_files if f[2] == 0]
        anomalous = [f for f in id_test_files if f[2] == 1]
        rng.shuffle(normal)
        rng.shuffle(anomalous)
        per_class = max(1, limit_files // 2)
        id_test_files = normal[:per_class] + anomalous[:per_class]

    out = []
    for _, path, label in id_test_files:
        vectors = data.cached_file_to_vectors(path)
        vectors = normalize(vectors, stats["mean"], stats["std"])
        out.append((label, list(vectors)))
    return out


def load_threshold(machine_id: str) -> float:
    path = config.PER_ID_ARTIFACTS / machine_id / "threshold.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run ml/compute_threshold.py first")
    return float(json.loads(path.read_text())["threshold"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine-id", default="id_02")
    parser.add_argument("--port", help="Serial port, e.g. COM5 (required unless --mock)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mock", action="store_true",
                         help="Score in Python instead of over serial (no hardware)")
    parser.add_argument("--limit-files", type=int, default=None,
                         help="Balanced sample of ~this many clips instead of the full set")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = config.PER_ID_ARTIFACTS / args.machine_id
    tflite_path = model_dir / "model_int8.tflite"
    if not tflite_path.exists():
        raise FileNotFoundError(f"{tflite_path} missing -- run export_per_id.py first")
    threshold = load_threshold(args.machine_id)

    files = build_normalized_test_files(args.machine_id, args.limit_files)
    print(f"[data] {args.machine_id}: {len(files)} held-out test clips  "
          f"threshold={threshold:.6f}")

    if args.mock:
        device: Device = LoopbackDevice(tflite_path, threshold)
        print("[device] mock mode (Python loopback, no hardware)")
    else:
        if not args.port:
            raise SystemExit("--port is required unless --mock is passed")
        device = SerialDevice(args.port, args.baud)

    scores, labels, flag_mismatches = score_files(
        device, files, threshold, progress=not args.mock)
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    m = anomaly_metrics(scores, labels, threshold)

    print(f"\n[result] {args.machine_id}: on-device AUC = {auc:.4f}  ({len(scores)} clips)")
    print(f"[result] anomaly detection @ threshold {threshold:.4f}: "
          f"precision={m['precision']} recall={m['recall']} F1={m['f1']}  "
          f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")
    print(f"[check] per-vector device-flag mismatches vs (score>threshold): "
          f"{flag_mismatches} (should be 0)")

    result = {"machine_id": args.machine_id, "on_device_auc": round(auc, 4),
              "threshold": threshold, "anomaly": m,
              "flag_mismatches": flag_mismatches, "clips": len(scores),
              "mock": args.mock}
    out_path = model_dir / "metrics_on_device.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
