"""Diagnostic: compare per-vector (score, anomaly) between LoopbackDevice (Python)
and the real ESP32 for the same vectors -- isolates whether the on-device C++
inference/threshold diverges from the verified Python math, independent of
AUC/aggregation. Handy when bringing up new firmware.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from replay_client import (
    LoopbackDevice,
    SerialDevice,
    build_normalized_test_files,
    load_threshold,
)


def main():
    machine_id = "id_02"
    port = sys.argv[1] if len(sys.argv) > 1 else "COM7"
    threshold = load_threshold(machine_id)
    files = build_normalized_test_files(machine_id, limit_files=2)
    _, vectors = files[0]
    print(f"[data] {machine_id} threshold={threshold:.6f}; comparing first 5 vectors")

    tflite_path = config.PER_ID_ARTIFACTS / machine_id / "model_int8.tflite"
    mock = LoopbackDevice(tflite_path, threshold)
    real = SerialDevice(port)

    for i, v in enumerate(vectors[:5]):
        ms, ma = mock.query(v)
        rs, ra = real.query(v)
        print(f"vec {i}: mock=({ms:.6f},{ma})  real=({rs:.6f},{ra})  "
              f"score_diff={rs - ms:+.6f}  flag_match={ma == ra}")


if __name__ == "__main__":
    main()
